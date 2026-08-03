"""Operator endpoints: spend, limits, routing policy, and a dry-run router.

``/admin/route/preview`` is the one worth knowing about — it runs the full
classify-and-score path and returns the decision *without spending anything*.
It is how you sanity-check a policy change before it touches traffic.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

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


@router.post("/fleet/reset")
async def reset_fleet(request: Request) -> dict:
    request.app.state.fleet.reset()
    return {"reset": True}


class SwitchUpdate(BaseModel):
    enabled: bool


@router.get("/switchboard")
async def switchboard(request: Request) -> dict:
    """Which models and vendors are switched on."""
    return request.app.state.switchboard.state()


@router.post("/switchboard/model/{model_key}")
async def switch_model(model_key: str, body: SwitchUpdate, request: Request) -> dict:
    """Turn one model on or off. Takes effect on the next request."""
    request.app.state.switchboard.set_model(model_key, body.enabled)
    return request.app.state.switchboard.state()


@router.post("/switchboard/provider/{provider}")
async def switch_provider(provider: str, body: SwitchUpdate, request: Request) -> dict:
    """Turn a whole vendor on or off — the fastest way to test failover."""
    request.app.state.switchboard.set_provider(provider.lower(), body.enabled)
    return request.app.state.switchboard.state()


@router.post("/switchboard/reset")
async def switch_reset(request: Request) -> dict:
    request.app.state.switchboard.reset()
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
    decision = await app.state.router.route(
        canonical, intent.intent, require_available=not include_unavailable
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
