"""Operator endpoints: spend, limits, routing policy, and a dry-run router.

``/admin/route/preview`` is the one worth knowing about — it runs the full
classify-and-score path and returns the decision *without spending anything*.
It is how you sanity-check a policy change before it touches traffic.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..errors import GatewayError
from ..pipeline import canonicalise
from ..routing.policy import INTENT_POLICY
from ..schemas import ChatCompletionRequest
from ..tokens import estimate_request_tokens

router = APIRouter(prefix="/admin", tags=["admin"])


class LimitUpdate(BaseModel):
    daily_usd: float | None = None
    rpm: int | None = None


@router.get("/usage/{tenant}")
async def usage(tenant: str, request: Request) -> dict:
    app = request.app
    spend = await app.state.ledger.spend_today(tenant)
    limit = await app.state.budget.limit_for(tenant)
    return {
        "tenant": tenant,
        "spend_usd_today": round(spend, 6),
        "limit_usd_daily": limit,
        "utilisation": round(spend / limit, 4) if limit else 0.0,
        "rpm_limit": await app.state.limiter.limit_for(tenant),
    }


@router.post("/limits/{tenant}")
async def set_limits(tenant: str, body: LimitUpdate, request: Request) -> dict:
    app = request.app
    if body.daily_usd is not None:
        await app.state.budget.set_limit(tenant, body.daily_usd)
    if body.rpm is not None:
        await app.state.store.set(f"rpm:{tenant}:limit", str(body.rpm))
    return await usage(tenant, request)


@router.get("/fleet")
async def fleet(request: Request, flow_limit: int = 25) -> dict:
    """Where the enterprise's traffic actually goes.

    A hop trace answers "what happened to this request". This answers the
    question asked second and cared about longer: across everything we send,
    which vendors carry the load, what do they cost, how do they behave.

    Rolling in-memory window — live and cheap to poll. The JSONL record remains
    the durable history.
    """
    return request.app.state.fleet.snapshot(flow_limit=flow_limit)


class EffortReport(BaseModel):
    """Effort only the caller can see.

    The gateway measures what happened inside one call. It cannot know that the
    user asked the same thing three different ways, gave up, or rewrote the
    answer before using it — and those are the expensive parts. An orchestrator
    that reports them makes the router aware of costs it is otherwise blind to.
    """

    model: str
    intent: str
    turns_to_goal: int | None = None
    user_reasked: bool | None = None
    user_rejected: bool | None = None
    manual_escalation: bool | None = None
    edit_distance: float | None = Field(default=None, ge=0.0, le=1.0)
    extras: dict[str, float] = Field(default_factory=dict)


@router.post("/effort")
async def report_effort(body: EffortReport, request: Request) -> dict:
    """Report caller-side effort for work already served.

    Returns the itemised score so the caller can see exactly what its report
    was worth — an effort penalty that cannot be itemised is indistinguishable
    from a grudge.
    """
    reputation = request.app.state.reputation
    if reputation is None:
        raise GatewayError(503, "reputation tracking is disabled", code="no_reputation")

    from ..routing.effort import EffortObservation

    scored = reputation.record_effort(
        EffortObservation(
            model_key=body.model,
            intent=body.intent,
            turns_to_goal=body.turns_to_goal,
            user_reasked=body.user_reasked,
            user_rejected=body.user_rejected,
            manual_escalation=body.manual_escalation,
            edit_distance=body.edit_distance,
            extras=body.extras,
        )
    )
    return {
        "recorded": True,
        "scored": scored,
        "multiplier_now": round(reputation.multiplier(body.model, body.intent), 3),
    }


@router.get("/effort/signals")
async def effort_signals(request: Request) -> dict:
    """The open table: every signal, its weight, and what it is waiting on.

    Published because a routing penalty nobody can enumerate is not auditable.
    Signals marked `awaiting data` are registered and inert — an empty column
    that names what is missing beats a signal that quietly is not there.
    """
    reputation = request.app.state.reputation
    if reputation is None:
        return {"signals": [], "enabled": False}
    return {"signals": reputation.effort.table(), "enabled": True}


@router.get("/cache/effectiveness")
async def cache_effectiveness(request: Request) -> dict:
    """Whether each model actually delivers the cache the router prices it for.

    Published because the router discounts input cost by ~90% on the strength of
    an assumed warm read. A model that never returns cached tokens is favoured by
    a discount it does not earn, and that is invisible unless the assumption is
    checked against what the vendor actually returned.
    """
    from ..cache.effectiveness import DEAD_CACHE_RATE, MIN_SAMPLES

    return {
        "min_samples": MIN_SAMPLES,
        "dead_cache_rate": DEAD_CACHE_RATE,
        "models": request.app.state.cache_effectiveness.snapshot(),
    }


@router.get("/alerts")
async def alerts(request: Request) -> dict:
    """System alerts — the gateway saying it needs a human.

    Polled by the console to decide whether to light the red flag and sound the
    alarm. `ok` is the single boolean that answers "is anything wrong"; each
    active alert carries `detail` for the operator and `user_message` in plain
    language for whoever is on the other end of the agent.
    """
    return request.app.state.alerts.snapshot()


@router.post("/alerts/clear")
async def clear_alerts(request: Request) -> dict:
    """Acknowledge and clear. Alerts also clear on their own when a request
    succeeds; this is for the case where you have fixed it and want the flag
    down without waiting for traffic."""
    cleared = request.app.state.alerts.clear()
    return {"cleared": cleared, **request.app.state.alerts.snapshot()}


@router.get("/baselines")
async def baselines(request: Request) -> dict:
    """What the console is colouring against.

    Published deliberately. A colour you cannot check is a colour you have to
    trust blindly — this returns every segment's mean, sigma, p50/p95, sample
    count and the exact thresholds a `warn` or `critical` was measured against,
    so any band on screen can be verified rather than believed.
    """
    return request.app.state.baselines.snapshot()


@router.post("/fleet/reset")
async def reset_fleet(request: Request) -> dict:
    request.app.state.fleet.reset()
    return {"reset": True}


class SwitchUpdate(BaseModel):
    enabled: bool


@router.get("/reputation")
async def reputation(request: Request) -> dict:
    """Observed quality per (model, intent) — what quality-adjusted routing sees.

    A model is only penalised once there is enough evidence, and a fraction of
    requests ignore the penalty so a model that has been fixed can recover.
    """
    s = request.app.state.settings
    return {
        "enabled": s.quality_routing_enabled,
        "min_samples": s.quality_min_samples,
        "max_penalty": s.quality_max_penalty,
        "exploration_rate": s.quality_exploration_rate,
        "window": s.quality_window,
        "rows": request.app.state.reputation.snapshot(),
    }


@router.post("/reputation/reset")
async def reset_reputation(request: Request) -> dict:
    request.app.state.reputation.reset()
    return {"reset": True}


@router.get("/switchboard")
async def switchboard(request: Request) -> dict:
    """Which models and vendors are switched on."""
    return request.app.state.switchboard.state()


@router.post("/switchboard/model/{model_key}")
async def switch_model(model_key: str, body: SwitchUpdate, request: Request) -> dict:
    """Turn one model on or off. Takes effect on the next request."""
    request.app.state.switchboard.set_model(model_key, body.enabled)
    # Turning something back on directly invalidates a "there is nothing left"
    # alert, so the flag comes down immediately rather than waiting for the next
    # served request. Health-caused alerts are untouched — a switch says nothing
    # about whether the vendor is reachable.
    request.app.state.alerts.clear("no_models_available:all_switched_off")
    request.app.state.alerts.clear("no_models_available:pinned_model_switched_off")
    return request.app.state.switchboard.state()


@router.post("/switchboard/provider/{provider}")
async def switch_provider(provider: str, body: SwitchUpdate, request: Request) -> dict:
    """Turn a whole vendor on or off — the fastest way to test failover."""
    request.app.state.switchboard.set_provider(provider.lower(), body.enabled)
    # Turning something back on directly invalidates a "there is nothing left"
    # alert, so the flag comes down immediately rather than waiting for the next
    # served request. Health-caused alerts are untouched — a switch says nothing
    # about whether the vendor is reachable.
    request.app.state.alerts.clear("no_models_available:all_switched_off")
    request.app.state.alerts.clear("no_models_available:pinned_model_switched_off")
    return request.app.state.switchboard.state()


@router.post("/switchboard/reset")
async def switch_reset(request: Request) -> dict:
    request.app.state.switchboard.reset()
    request.app.state.alerts.clear("no_models_available:all_switched_off")
    request.app.state.alerts.clear("no_models_available:pinned_model_switched_off")
    return request.app.state.switchboard.state()


@router.get("/pool")
async def pool(request: Request) -> dict:
    """Model pool: health, circuit-breaker state, and observed performance.

    Health comes from two sources — periodic free probes against each provider,
    and the outcome of real traffic. The second is the one that matters, and it
    is what drives the breaker.
    """
    health = request.app.state.health
    models = health.snapshot()
    return {
        "probe_interval_seconds": request.app.state.settings.health_probe_interval_seconds,
        "breaker_failure_threshold": request.app.state.settings.breaker_failure_threshold,
        "breaker_cooldown_seconds": request.app.state.settings.breaker_cooldown_seconds,
        "pools": sorted({m["rate_limit_pool"] for m in models}),
        "models": models,
    }


@router.post("/pool/probe")
async def probe_now(request: Request) -> dict:
    """Force a health probe instead of waiting for the next interval."""
    await request.app.state.health.probe()
    return {"probed": True, "models": request.app.state.health.snapshot()}


@router.post("/pool/reset")
async def reset_breakers(request: Request, model: str | None = None) -> dict:
    """Close circuit breakers manually — after a known fix, or for a demo."""
    request.app.state.health.reset(model)
    return {"reset": model or "all", "models": request.app.state.health.snapshot()}


@router.get("/policy")
async def policy() -> dict:
    return {
        "intents": [
            {
                "intent": p.intent,
                "min_tier": p.min_tier.name.lower(),
                "effort": p.effort,
                "escalate_on_tools": p.escalate_on_tools,
                "notes": p.notes,
            }
            for p in INTENT_POLICY.values()
        ]
    }


@router.post("/route/preview")
async def route_preview(
    body: ChatCompletionRequest, request: Request, include_unavailable: bool = True
) -> dict:
    """Dry-run the router. Classifies and scores; never calls a model.

    ``include_unavailable`` defaults to true so the routing logic can be
    inspected before any credentials are entered. Set it false to see what
    would actually be servable right now.
    """
    app = request.app
    enabled = app.state.registry.enabled
    canonical = canonicalise(body)
    prefix_tokens, volatile_tokens = estimate_request_tokens(canonical)
    intent = await app.state.classifier.classify(canonical, prefix_tokens, volatile_tokens)

    # Apply the tenant's budget ceiling, exactly as the serving path would. A
    # preview that ignored it would not be previewing the routing you actually
    # get — the same argument that makes the switchboard apply on dry runs. The
    # check is read-only: it never records spend.
    principal = await app.state.auth.authenticate(request)
    dry = await app.state.router.route(
        canonical, intent.intent, require_available=not include_unavailable
    )
    verdict = await app.state.budget.check(principal.tenant_id, dry.estimated_cost_usd)
    decision = dry
    if verdict.tier_ceiling is not None:
        decision = await app.state.router.route(
            canonical,
            intent.intent,
            cost_ceiling_tier=verdict.tier_ceiling,
            require_available=not include_unavailable,
        )

    from ..routing import explain as explain_decision

    return {
        "servable_now": decision.model.provider in enabled,
        "explain": explain_decision(decision, intent.confidence, intent.source),
        "intent": {
            "resolved": intent.intent,
            "confidence": intent.confidence,
            "source": intent.source,
            "rationale": intent.rationale,
        },
        "tokens": {"prefix_est": prefix_tokens, "volatile_est": volatile_tokens},
        "decision": {
            "model": decision.model.key,
            "provider": decision.model.provider,
            "tier": decision.tier.name.lower(),
            "effort": decision.effort,
            "reason": decision.reason,
            "cache_state": decision.cache_state,
            "cache_plan": decision.cache_plan.reason,
            "estimated_cost_usd": round(decision.estimated_cost_usd, 6),
            # How the decision was reached, not just what it was. Omitting these
            # made a dry run unable to show a pin, a degrade or a sticky
            # session — the three cases where the answer is *not* "cheapest
            # capable model" and you most want to know why.
            "pinned": decision.pinned,
            "degraded": decision.degraded,
            "sticky": decision.sticky,
            "required_tier": decision.required_tier.name.lower(),
            "escalated_from": decision.escalated_from,
        },
        "considered": [
            {
                "model": c.model.key,
                "tier": c.model.tier.name.lower(),
                "estimated_cost_usd": round(c.cost_usd, 6),
                "cache_state": c.cache_state,
                "available": c.model.provider in enabled,
            }
            for c in sorted(decision.considered, key=lambda c: c.cost_usd)
        ],
    }
