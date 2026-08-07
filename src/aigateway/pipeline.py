"""The request pipeline.

Order matters and is deliberate:

  1. rate limit        — cheapest rejection first
  2. canonicalise      — OpenAI shape -> neutral shape
  3. classify intent   — layered, cheapest layer first
  4. route (dry)       — produces a cost estimate to budget against
  5. budget check      — may impose a tier ceiling
  6. route (final)     — re-routed under the ceiling if degraded
  7. cache plan        — breakpoints for the chosen model, not before
  8. cache pilot       — serialise the first request of an unseen prefix
  9. invoke + fallback — same vendor first
 10. price and record  — actual usage, not the estimate

Steps 4 and 6 look redundant and are not: the budget needs a cost, the cost
needs a model, and the model depends on the budget. Routing twice is free
(it is arithmetic over a small catalog); guessing is not.

Steps 1-8 are one shared phase (``_prepare``) and step 10 one shared epilogue
(``_settle``), used identically by the unary and streaming paths. They were
two hand-maintained copies once, and the copies drifted exactly as copies do:
streaming skipped the pin-scope check (an authz hole), never wrote a record,
and lost the ledger write when a client disconnected mid-stream. The transport
is allowed to differ; what the gateway *means* by serving a request is not.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .auth import Principal
from .cache.hints import plan_cache
from .cache.pilot import CachePilot, PilotRole
from .catalog import ModelSpec, Tier
from .config import Settings
from .errors import GatewayError, NoModelsAvailable, ProviderRefusal, UpstreamError
from .governance import BudgetGuard, CostLedger, RateLimiter, price_usage
from .observability import (
    AlertCentre,
    LatencyBaselines,
    RecordSink,
    RequestRecord,
    TraceContext,
    no_models_available,
)
from .observability.baselines import segment_key
from .quality import (
    REASONING_FLOOR_BY_EFFORT,
    Check,
    QualityReport,
    assess,
    budget_starves_the_answer,
    judge,
)
from .routing import IntentClassifier, Router, explain
from .routing.effort import EffortObservation
from .routing.policy import policy_for
from .schemas import (
    CanonicalRequest,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    GatewayMeta,
    ProviderResponse,
    ToolDef,
    Usage,
)
from .tokens import estimate_request_tokens, estimate_tokens

log = logging.getLogger(__name__)


@dataclass
class Prepared:
    """Everything decided before a byte goes upstream — transport-independent."""

    canonical: CanonicalRequest
    intent: Any
    decision: Any
    verdict: Any
    plan: Any
    role: PilotRole
    # Set once the reserved estimate has been settled against actuals, so a
    # later failure does not refund money that was already reconciled.
    settled: bool = False


def canonicalise(request: ChatCompletionRequest) -> CanonicalRequest:
    """OpenAI-shaped request -> neutral representation.

    System messages are hoisted out of ``messages`` into their own field: they
    are the most stable part of the prompt and therefore the natural home for
    the first cache breakpoint. Leaving them inline would bury the boundary.
    """
    ext = request.x_gateway

    system: list[str] = []
    messages: list[ChatMessage] = []
    for msg in request.messages:
        if msg.role == "system" and not messages:
            # Only leading system messages are hoisted. A system message that
            # appears mid-conversation is a deliberate operator instruction and
            # must keep its position, or the cached prefix ahead of it breaks.
            system.append(msg.content if isinstance(msg.content, str) else str(msg.content))
        else:
            messages.append(msg)

    tools = [
        ToolDef(
            name=t.get("function", {}).get("name", t.get("name", "")),
            description=t.get("function", {}).get("description", ""),
            parameters=t.get("function", {}).get("parameters", {}),
            strict=bool(t.get("function", {}).get("strict", False)),
        )
        for t in (request.tools or [])
    ]

    schema = None
    if request.response_format and request.response_format.get("type") == "json_schema":
        schema = request.response_format.get("json_schema", {}).get("schema")

    pin = ext.pin_model
    if request.model and request.model not in ("auto", "gateway", "default"):
        pin = request.model  # an explicit model id is a pin

    return CanonicalRequest(
        system=system,
        messages=messages,
        tools=tools,
        tool_choice=request.tool_choice,
        response_schema=schema,
        max_tokens=request.resolved_max_tokens(),
        max_tokens_explicit=request.chose_max_tokens(),
        effort=ext.effort,
        stream=request.stream,
        temperature=request.temperature,
        top_p=request.top_p,
        session_id=ext.session_id,
        intent_hint=ext.intent,
        cache_hints=ext.cache_hints or ["system", "tools"],
        pin_model=pin,
        max_tier=ext.max_tier,
        vendor_overrides=ext.vendor_overrides,
    )


class GatewayPipeline:
    def __init__(
        self,
        settings: Settings,
        store,
        registry,
        router: Router,
        classifier: IntentClassifier,
        budget: BudgetGuard,
        limiter: RateLimiter,
        ledger: CostLedger,
        sink: RecordSink,
        health=None,
        reputation=None,
        baselines: LatencyBaselines | None = None,
        switchboard=None,
        alerts: AlertCentre | None = None,
        cache_effectiveness=None,
        outputs=None,
    ):
        self._s = settings
        self._store = store
        self._registry = registry
        self._router = router
        self._classifier = classifier
        self._budget = budget
        self._limiter = limiter
        self._ledger = ledger
        self._sink = sink
        self._health = health
        self._switchboard = switchboard
        self._reputation = reputation
        # Learned expectations, so the console can say whether *this* stage was
        # slow rather than just how long it took.
        self.baselines = baselines or LatencyBaselines()
        # Raised when there is nothing left to serve with; cleared by the next
        # request that succeeds. Recovery is detected from real traffic rather
        # than a probe, like the circuit breaker.
        self.alerts = alerts or AlertCentre()
        self._cache_effectiveness = cache_effectiveness
        # Learned completion volume per intent; fed here, consumed by the
        # router's cost forecast.
        self._outputs = outputs
        self.pilot = CachePilot(
            store, enabled=settings.cache_pilot_enabled, wait_ms=settings.cache_pilot_wait_ms
        )

    async def handle(
        self,
        request: ChatCompletionRequest,
        principal: Principal,
        emit=None,
    ) -> ChatCompletionResponse:
        """Run a request.

        ``emit`` is an optional async callback ``(stage, payload)`` invoked as
        each pipeline stage completes. It exists so the console can render the
        route as it actually happens rather than animating a guess — the stage
        timings it reports are real.
        """
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        trace = TraceContext(
            trace_id,
            origin={
                "tenant": principal.tenant_id,
                "agent": principal.agent_id,
                "session": request.x_gateway.session_id,
                "endpoint": "POST /v1/chat/completions",
                "client": "agent",
            },
        )
        record = RequestRecord(
            trace_id=trace_id,
            tenant=principal.tenant_id,
            agent=principal.agent_id,
            session_id=request.x_gateway.session_id,
            declared_intent=request.x_gateway.intent,
        )

        try:
            return await self._run(
                request,
                principal,
                record,
                started,
                emit,
                trace if self._s.hop_trace_enabled else None,
            )
        except Exception as exc:
            self._note_failure(record, exc)
            raise
        finally:
            record.latency_total_ms = int((time.perf_counter() - started) * 1000)
            self._sink.write(record)

    def _note_failure(self, record: RequestRecord, exc: Exception) -> None:
        """Classify a failure onto the record — one mapping for both transports."""
        if isinstance(exc, NoModelsAvailable):
            record.outcome = "no_models_available"
            record.error = str(exc.detail)
            self.alerts.raise_alert(
                no_models_available(
                    cause=exc.cause,
                    detail=exc.detail["error"]["message"],
                    remedy=exc.remedy,
                )
            )
        elif isinstance(exc, ProviderRefusal):
            record.outcome = "refusal"
            record.error = str(exc.detail)
        elif isinstance(exc, GatewayError):
            record.outcome = (
                "budget_exceeded"
                if exc.status_code == 402
                else "rate_limited"
                if exc.status_code == 429
                else "error"
            )
            record.error = str(exc.detail)
        else:
            record.outcome = "error"
            record.error = repr(exc)

    async def _run(
        self,
        request: ChatCompletionRequest,
        principal: Principal,
        record: RequestRecord,
        started: float,
        emit=None,
        trace: TraceContext | None = None,
    ) -> ChatCompletionResponse:
        last_stage_at = started

        async def stage(name: str, payload: dict) -> None:
            """Emit one stage, timed and judged against its baseline.

            Two clocks, because they answer different questions: `elapsed_ms` is
            cumulative and says where in the request you are; `stage_ms` is this
            step alone and is the only one worth comparing to a baseline.

            The upstream call is judged on its own measured latency rather than
            on wall-clock between stages — the two differ by whatever the cache
            pilot spent waiting, and blaming the vendor for our own wait would
            be both wrong and unfalsifiable.
            """
            nonlocal last_stage_at
            now = time.perf_counter()
            if not emit:
                last_stage_at = now
                return

            payload["elapsed_ms"] = int((now - started) * 1000)
            # Sub-millisecond resolution, because most gateway-local work is
            # sub-millisecond: truncating to whole ms pinned canonicalise,
            # classify and route at a flat 0 and threw away the only signal
            # those three stages have.
            payload["stage_ms"] = round(max(0.0, (now - last_stage_at) * 1000), 3)
            last_stage_at = now

            if name == "served":
                key = segment_key(
                    "served",
                    model=payload.get("model", ""),
                    cache_state=payload.get("cache_state", ""),
                )
                measured = payload.get("latency_ms", payload["stage_ms"])
            else:
                key, measured = name, payload["stage_ms"]

            if self._s.latency_baselines_enabled:
                payload["baseline"] = self.baselines.observe_and_judge(key, measured)
                payload["baseline"]["segment"] = key
                payload["baseline"]["measured_ms"] = measured
            await emit(name, payload)

        def restart_stage_clock() -> None:
            """Stop charging the next stage for the upstream call.

            Stage durations are wall-clock between emissions, and the model call
            sits between `cache` and `quality` without being a stage of its own —
            so `quality` was being billed the entire vendor round-trip and
            learned a 3.7-second baseline for work that takes microseconds. The
            upstream call is measured separately and reported on `served`.
            """
            nonlocal last_stage_at
            last_stage_at = time.perf_counter()

        prepared = await self._prepare(request, principal, record, stage, trace)
        canonical = prepared.canonical
        decision = prepared.decision
        verdict = prepared.verdict
        plan = prepared.plan
        role = prepared.role

        # 9. invoke, with same-vendor-first fallback. The pilot lock stays
        # heartbeat-alive for exactly as long as the call is in flight.
        try:
            async with self.pilot.holding(plan.fingerprint, role):
                response, model_used, chain = await self._invoke_with_fallback(
                    canonical, decision.model, decision.effort, plan, record, trace
                )
        except Exception:
            if role is PilotRole.PILOT:
                await self.pilot.release_failed(plan.fingerprint)
            # The money reserved for this request never became spend.
            await self._budget.release(principal.tenant_id, verdict)
            raise
        restart_stage_clock()

        try:
            if role is PilotRole.PILOT and plan.cacheable:
                await self.pilot.mark_warm(plan.fingerprint, self._s.session_ttl_seconds)

            await self._router.remember(canonical.session_id, model_used, canonical)
            await self._limiter.note_upstream(model_used.rate_limit_pool)

            # 10. price from actual usage, settling any hard-mode reservation.
            priced = price_usage(response.usage, model_used, self._s.cache_ttl)
            await self._ledger.record(
                principal.tenant_id, principal.agent_id, model_used.key, priced,
                reserved_usd=verdict.reserved_usd,
            )
            prepared.settled = True
        except Exception:
            if not prepared.settled:
                await self._budget.release(principal.tenant_id, verdict)
            raise

        return await self._finish(
            prepared, response, model_used, chain, priced, record, started,
            stage, trace,
        )

    async def _prepare(
        self,
        request: ChatCompletionRequest,
        principal: Principal,
        record: RequestRecord,
        stage,
        trace: TraceContext | None = None,
    ) -> Prepared:
        """Steps 1-8: everything decided before a byte goes upstream.

        Shared verbatim by the unary and streaming paths — the transport is
        allowed to differ, the meaning of accepting a request is not.
        """
        # 1. inbound rate limit
        await self._limiter.check_tenant(principal.tenant_id)
        await stage("accepted", {"tenant": principal.tenant_id, "agent": principal.agent_id})

        # 2. canonicalise
        canonical = canonicalise(request)
        if canonical.pin_model and not principal.may_pin_model():
            canonical.pin_model = None
            log.info("pin ignored: %s lacks model:pin scope", principal.agent_id)

        prefix_tokens, volatile_tokens = estimate_request_tokens(canonical)
        record.prefix_tokens_est = prefix_tokens
        record.volatile_tokens_est = volatile_tokens
        await stage(
            "canonicalised",
            {
                "prefix_tokens": prefix_tokens,
                "volatile_tokens": volatile_tokens,
                "tools": len(canonical.tools),
                "system_blocks": len(canonical.system),
            },
        )

        # 3. intent
        intent = await self._classifier.classify(canonical, prefix_tokens, volatile_tokens)
        record.resolved_intent = intent.intent
        record.intent_confidence = intent.confidence
        record.intent_source = intent.source
        await stage(
            "classified",
            {
                "intent": intent.intent,
                "confidence": intent.confidence,
                "source": intent.source,
                "rationale": intent.rationale,
                "declared": canonical.intent_hint,
            },
        )

        # 3b. size the output budget to the work, unless the caller chose one.
        #
        # Output budget is what a request actually spends its wall-clock on: the
        # same prompt measured 18s at 1,200 tokens and 58s at 8,000, while every
        # gateway-local stage together took 1.3ms. A single global default is
        # therefore either slow for classification or starving for review, and
        # it cannot be both right. The intent knows which this is.
        if not canonical.max_tokens_explicit and self._s.auto_size_max_tokens:
            budget = policy_for(intent.intent).max_tokens
            if budget != canonical.max_tokens:
                canonical.max_tokens = budget
                await stage(
                    "budgeted",
                    {
                        "max_tokens": budget,
                        "intent": intent.intent,
                        "source": "intent policy",
                        "note": (
                            f"No max_tokens given, so '{intent.intent}' work was "
                            f"sized to {budget:,} tokens. Send max_tokens to override."
                        ),
                    },
                )

        # 4. dry route -> cost estimate
        decision = await self._router.route(canonical, intent.intent)

        # 5. budget, which may impose a ceiling
        verdict = await self._budget.check(principal.tenant_id, decision.estimated_cost_usd)
        await stage(
            "budget",
            {
                "spend_usd": round(verdict.spend_usd, 6),
                "limit_usd": verdict.limit_usd,
                "utilisation": round(verdict.utilisation, 4),
                "degraded": verdict.tier_ceiling is not None,
                "message": verdict.message,
            },
        )
        # Anything that fails between the reservation above and the pricing of
        # a real response must hand the reserved money back — otherwise a
        # rejected or crashed request spends budget it never used.
        try:
            if verdict.tier_ceiling is not None:
                # 6. re-route under the ceiling
                decision = await self._router.route(
                    canonical, intent.intent, cost_ceiling_tier=verdict.tier_ceiling
                )
                log.info("tenant %s degraded: %s", principal.tenant_id, verdict.message)

            record.chosen_model = decision.model.key
            record.provider = decision.model.provider
            record.tier = decision.tier.name.lower()
            record.effort = decision.effort
            record.routing_reason = decision.reason
            record.degraded = decision.degraded
            record.estimated_cost_usd = decision.estimated_cost_usd
            record.considered = [
                {"model": c.model.key, "cost_usd": round(c.cost_usd, 6), "cache": c.cache_state}
                for c in decision.considered
            ]

            await stage(
                "routed",
                {
                    "model": decision.model.key,
                    "provider": decision.model.provider,
                    "tier": decision.tier.name.lower(),
                    "effort": decision.effort,
                    # Warn before the call, not after. Below the lowest reasoning
                    # floor there is no effort left to step down to, so the request
                    # will most likely burn its whole allowance thinking and return
                    # nothing — and the caller pays for it either way.
                    "budget_warning": (
                        f"max_tokens={canonical.max_tokens} is below the "
                        f"{REASONING_FLOOR_BY_EFFORT['low']} tokens a reasoning model "
                        f"typically needs before any visible answer appears on work "
                        f"this demanding. Expect an empty reply; raise max_tokens."
                        if budget_starves_the_answer(canonical.max_tokens)
                        and decision.tier is not Tier.LIGHT
                        else None
                    ),
                    "reason": decision.reason,
                    # The plain-language account of the same decision. The console
                    # leads with this; `reason` is kept for logs and debugging.
                    "explain": explain(decision, intent.confidence, intent.source),
                    "degraded": decision.degraded,
                    "estimated_cost_usd": round(decision.estimated_cost_usd, 6),
                    "considered": [
                        {
                            "model": c.model.key,
                            "provider": c.model.provider,
                            "tier": c.model.tier.name.lower(),
                            "estimated_cost_usd": round(c.cost_usd, 6),
                            "cache_state": c.cache_state,
                            "raw_cost_usd": round(c.raw_cost_usd, 6),
                            "quality_multiplier": round(c.quality_multiplier, 3),
                            "quality_samples": c.quality_samples,
                            "quality_success_rate": c.quality_success_rate,
                            "chosen": c.model.key == decision.model.key,
                        }
                        for c in sorted(decision.considered, key=lambda c: c.cost_usd)
                    ],
                },
            )

            # 7. full cache plan for the model we actually picked
            plan = plan_cache(canonical, decision.model, ttl=self._s.cache_ttl)
            record.cache_state = decision.cache_state
            record.cache_plan = plan.reason

            # 8. cache pilot — the fan-out fix
            # Only engage the pilot when there is a cache to protect. Serialising
            # requests whose prefix is too short to cache makes followers wait out
            # the full pilot timeout for an entry that can never exist — pure added
            # latency for zero saving, which is worse than not having the pilot.
            pilot_started = time.perf_counter()
            if plan.cacheable:
                # Follower patience scaled to the pilot's observed latency: a
                # flat wait shorter than the pilot's time-to-warm times every
                # follower out, and they all pay the cold write anyway — which
                # is the fan-out bug this module exists to fix.
                wait_ms = None
                if self._health and (p50 := self._health.p50_of(decision.model.key)):
                    wait_ms = max(
                        self._s.cache_pilot_wait_ms, min(int(p50 * 1.5), 60_000)
                    )
                role = await self.pilot.acquire(
                    plan.fingerprint, self._s.session_ttl_seconds, wait_ms=wait_ms
                )
            else:
                role = PilotRole.DISABLED
            record.pilot_role = role.value
            if trace is not None and role in (PilotRole.FOLLOWER, PilotRole.TIMEOUT):
                # Only record a hop when the pilot actually made us wait — a hop
                # that always fires with 0ms is noise in the waterfall.
                trace.add(
                    kind="cache_wait",
                    label="waited for cache pilot",
                    latency_ms=int((time.perf_counter() - pilot_started) * 1000),
                    status="ok" if role is PilotRole.FOLLOWER else "timeout",
                    detail=(
                        "another request was warming this prefix; waited so this one "
                        "could read the cache instead of writing it again"
                    ),
                )
            await stage(
                "cache",
                {
                    "state": decision.cache_state,
                    "plan": plan.reason,
                    "breakpoints": plan.breakpoints,
                    "pilot_role": role.value,
                    "fingerprint": plan.fingerprint[:12],
                },
            )
        except Exception:
            await self._budget.release(principal.tenant_id, verdict)
            raise

        return Prepared(
            canonical=canonical,
            intent=intent,
            decision=decision,
            verdict=verdict,
            plan=plan,
            role=role,
        )

    async def _finish(
        self,
        prepared: Prepared,
        response: ProviderResponse,
        model_used: ModelSpec,
        chain: list[str],
        priced,
        record: RequestRecord,
        started: float,
        stage,
        trace: TraceContext | None = None,
    ) -> ChatCompletionResponse:
        """Steps 11+: quality, feedback loops, and the response envelope."""
        canonical = prepared.canonical
        intent = prepared.intent
        decision = prepared.decision
        plan = prepared.plan
        role = prepared.role

        record.fallback_chain = chain
        record.chosen_model = model_used.key
        record.provider = model_used.provider
        record.prompt_tokens = response.usage.prompt_tokens
        record.completion_tokens = response.usage.completion_tokens
        record.cache_read_tokens = response.usage.cache_read_tokens
        record.cache_write_tokens = response.usage.cache_write_tokens
        record.actual_cost_usd = priced.total_usd
        record.cache_savings_usd = priced.cache_savings_usd
        if trace is not None:
            summary = trace.summary()
            record.hosts_contacted = summary["hosts_contacted"]
            record.hop_count = summary["hop_count"]
            record.upstream_ms = summary["upstream_ms"]
            record.gateway_overhead_ms = summary["gateway_overhead_ms"]

        # 11. did the routing decision actually work out? Deterministic checks
        #     are free; the LLM grader is opt-in because it is a billable call.
        report = (
            assess(canonical, response, decision)
            if self._s.quality_checks_enabled
            else QualityReport()
        )
        if self._s.quality_judge_enabled and self._registry.enabled:
            last_user = next(
                (m.content for m in reversed(canonical.messages) if m.role == "user"), ""
            )
            try:
                report.judge = await judge(
                    self._registry, self._s.quality_judge_model, str(last_user),
                    response.text,
                )
            except Exception as exc:
                # The grader is advisory. The response is already served and
                # billed, so a judge outage must not turn success into failure.
                log.warning("quality judge failed: %s", exc)
                report.judge = None
            if report.judge and report.judge.get("adequate") is False:
                report.checks.append(
                    Check(
                        "judge_inadequate",
                        "fail",
                        f"Grader scored the answer {report.judge.get('score')}/5",
                        report.judge.get("reason", ""),
                    )
                )
        # Feed the outcome back into routing. This is the loop that lets the
        # router learn a cheap model is bad at *this* task, rather than being
        # told so by configuration.
        # Did the cache the router priced actually turn up? Only requests that
        # expected a hit are evidence — a cold write returning nothing cached is
        # correct, and counting it would condemn every model on first use.
        if self._cache_effectiveness is not None:
            self._cache_effectiveness.record(
                model_used.key,
                expected_hit=decision.cache_state == "warm_read",
                cached_tokens=response.usage.cache_read_tokens,
            )

        # A served request is proof the fleet is reachable again. Recovery is
        # observed, not probed — the same principle as the circuit breaker.
        self.alerts.clear("no_models_available")

        if self._reputation:
            self._reputation.record(model_used.key, intent.intent, report.routing_ok)

        # Teach the output forecast what this intent actually produced.
        if self._outputs is not None:
            self._outputs.record(intent.intent, response.usage.completion_tokens)

        if self._reputation and self._s.effort_tracking_enabled:
            # What this answer actually cost beyond one clean call. Everything
            # here is measured on this request; the signals that need a human
            # in the loop arrive later via POST /admin/effort.
            text = response.text or ""
            effort = self._reputation.record_effort(
                EffortObservation(
                    model_key=model_used.key,
                    intent=intent.intent,
                    usable=report.routing_ok,
                    attempts=len(chain) + 1,
                    completion_tokens=response.usage.completion_tokens,
                    visible_tokens=estimate_tokens(text),
                    truncated=response.finish_reason == "length",
                    empty=not text.strip() and not response.tool_calls,
                    # Upstream time, not total request time: the model is
                    # answerable for its own call, not for the gateway's
                    # overhead or the cache pilot's wait.
                    latency_ms=float(record.upstream_ms),
                    cost_usd=priced.total_usd,
                )
            )
            record.extra_effort = effort["extra_effort"]
            await stage("effort", effort)
        record.quality_verdict = report.verdict
        record.quality_failures = [c.id for c in report.failures]
        record.routing_ok = report.routing_ok
        if not report.routing_ok:
            log.info(
                "routing miss: %s on %s produced %s",
                intent.intent, model_used.key, record.quality_failures,
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        await stage("quality", report.summary())
        await stage(
            "served",
            {
                "model": model_used.key,
                "provider": model_used.provider,
                "actual_cost_usd": round(priced.total_usd, 6),
                "cache_read_tokens": response.usage.cache_read_tokens,
                "cache_write_tokens": response.usage.cache_write_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "fallback_chain": chain,
                "latency_ms": latency_ms,
                "cache_state": decision.cache_state,
            },
        )
        if trace is not None:
            await stage("hops", trace.summary())

        return ChatCompletionResponse(
            model=model_used.key,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(
                        role="assistant",
                        content=response.text or None,
                        tool_calls=response.tool_calls or None,
                    ),
                    finish_reason=response.finish_reason,
                )
            ],
            usage=response.usage,
            x_gateway=GatewayMeta(
                trace_id=record.trace_id,
                session_id=canonical.session_id,
                declared_intent=canonical.intent_hint,
                resolved_intent=intent.intent,
                intent_confidence=intent.confidence,
                intent_source=intent.source,
                chosen_model=model_used.key,
                provider=model_used.provider,
                routing_reason=decision.reason,
                tier=model_used.tier.name.lower(),
                cache_state=f"{decision.cache_state} ({role.value}); {plan.reason}",
                cache_read_tokens=response.usage.cache_read_tokens,
                cache_write_tokens=response.usage.cache_write_tokens,
                estimated_cost_usd=round(decision.estimated_cost_usd, 6),
                actual_cost_usd=round(priced.total_usd, 6),
                cache_savings_usd=round(priced.cache_savings_usd, 6),
                fallback_chain=chain,
                degraded=decision.degraded,
                latency_ms=latency_ms,
                considered=[
                    {
                        "model": c.model.key,
                        "provider": c.model.provider,
                        "tier": c.model.tier.name.lower(),
                        "estimated_cost_usd": round(c.cost_usd, 6),
                        "cache_state": c.cache_state,
                        "chosen": c.model.key == model_used.key,
                    }
                    for c in sorted(decision.considered, key=lambda c: c.cost_usd)
                ],
                pilot_role=role.value,
                quality=report.summary(),
                trace=trace.summary() if trace else {},
                prefix_tokens_est=decision.prefix_tokens,
                volatile_tokens_est=decision.volatile_tokens,
            ),
        )

    async def _invoke_with_fallback(
        self,
        canonical: CanonicalRequest,
        model: ModelSpec,
        effort: str,
        plan,
        record: RequestRecord,
        trace: TraceContext | None = None,
    ) -> tuple[ProviderResponse, ModelSpec, list[str]]:
        chain: list[str] = []
        candidates = [model] + self._registry.fallback_chain(
            model, switchboard=self._switchboard, health=self._health
        )
        last_error: Exception | None = None

        for attempt, spec in enumerate(candidates):
            try:
                provider = self._registry.get(spec.provider)
            except GatewayError as exc:
                last_error = exc
                continue

            # The cache plan is model-specific: a fallback to another model
            # means a different prefix minimum and, on a cross-vendor hop, a
            # different caching mechanism entirely. Recompute rather than reuse.
            attempt_plan = plan if spec.key == model.key else plan_cache(
                canonical, spec, ttl=self._s.cache_ttl
            )

            call_started = time.perf_counter()
            hop = None
            if trace is not None:
                hop = trace.add(
                    kind="model",
                    label=f"gateway → {spec.provider}",
                    host=getattr(provider, "host", spec.provider),
                    endpoint="/v1/messages" if spec.provider == "anthropic"
                    else "/chat/completions",
                    model=spec.key,
                    provider=spec.provider,
                    attempt=attempt + 1,
                )
            try:
                # Gateway-owned deadline: the breaker stops repeat offenders,
                # but only after they return. This bounds the single hung call
                # the breaker cannot see, and turns it into a normal fallback.
                response = await asyncio.wait_for(
                    provider.invoke(canonical, spec.key, effort, attempt_plan),
                    timeout=self._s.upstream_timeout_seconds,
                )
                elapsed = int((time.perf_counter() - call_started) * 1000)
                if self._health:
                    self._health.record_success(spec.key, elapsed)
                if hop is not None:
                    hop.latency_ms = elapsed
                    hop.tokens_in = response.usage.prompt_tokens
                    hop.tokens_out = response.usage.completion_tokens
                    hop.cached_tokens = response.usage.cache_read_tokens
                    hop.detail = f"stop={response.raw_stop_reason or 'end_turn'}"
                if attempt:
                    chain.append(f"{spec.key}(recovered)")
                return response, spec, chain

            except ProviderRefusal:
                if hop is not None:
                    hop.latency_ms = int((time.perf_counter() - call_started) * 1000)
                    hop.status = "refused"
                    hop.detail = "declined on policy grounds"
                # A policy refusal is a content outcome, not an outage. Retrying
                # it on another model is both futile and a way to launder a
                # refusal, so it propagates.
                raise
            except TimeoutError:
                if hop is not None:
                    hop.latency_ms = int((time.perf_counter() - call_started) * 1000)
                    hop.status = "error"
                    hop.detail = f"timeout after {self._s.upstream_timeout_seconds:.0f}s"
                if self._health:
                    self._health.record_failure(spec.key, "gateway deadline exceeded")
                chain.append(f"{spec.key}(failed:timeout)")
                last_error = UpstreamError(
                    f"{spec.key} exceeded the {self._s.upstream_timeout_seconds:.0f}s "
                    f"gateway deadline", 504,
                )
                log.warning("fallback: %s timed out", spec.key)
                continue
            except (UpstreamError, GatewayError) as exc:
                status = getattr(exc, "status_code", 500)
                if hop is not None:
                    hop.latency_ms = int((time.perf_counter() - call_started) * 1000)
                    hop.status = "error"
                    hop.http_status = status
                    hop.detail = f"HTTP {status}"
                if status in (400, 403, 422):
                    raise  # our request is wrong; another model will not fix it
                # Only upstream faults count against the model's health. A 4xx
                # we caused would open a breaker on a perfectly healthy model.
                if self._health:
                    self._health.record_failure(spec.key, f"HTTP {status}")
                chain.append(f"{spec.key}(failed:{status})")
                last_error = exc
                log.warning("fallback: %s failed with %s", spec.key, status)
                continue

        raise last_error or UpstreamError("all providers failed", 503)

    async def stream(
        self, request: ChatCompletionRequest, principal: Principal
    ) -> Any:
        """SSE passthrough in the OpenAI chunk format.

        Same ``_prepare`` phase as the unary path — pin scopes, budget
        (including hard-mode reservation), the cache pilot, all of it — so
        streaming cannot drift into a side door. Only the transport differs:

        * fallback happens *before the first chunk* — once a byte has been
          sent the model identity is committed;
        * the pilot marks the prefix warm on the **first chunk**, because the
          provider cache is readable from first token — this is the earliest
          truthful moment, and it is what lets followers stop waiting;
        * settlement (ledger, record, feedback loops) runs when the stream
          closes — including on client disconnect, where usage is estimated
          from what was actually sent rather than silently unbilled.
        """
        started = time.perf_counter()
        record = RequestRecord(
            trace_id=uuid.uuid4().hex,
            tenant=principal.tenant_id,
            agent=principal.agent_id,
            session_id=request.x_gateway.session_id,
            declared_intent=request.x_gateway.intent,
        )

        async def no_stage(name: str, payload: dict) -> None:
            return None  # SSE data chunks are the transport; no side-channel

        try:
            prepared = await self._prepare(request, principal, record, no_stage, None)
        except Exception as exc:
            self._note_failure(record, exc)
            record.latency_total_ms = int((time.perf_counter() - started) * 1000)
            self._sink.write(record)
            raise

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def chunk_for(model_key: str, delta: dict, finish: str | None) -> str:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_key,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload)}\n\n"

        async def generate():
            canonical, decision, plan, role = (
                prepared.canonical, prepared.decision, prepared.plan, prepared.role
            )
            usage: Usage | None = None
            text_parts: list[str] = []
            model_used = decision.model
            chain: list[str] = []
            call_started: float | None = None
            streamed = False       # at least one upstream chunk arrived
            disconnected = False
            failed: Exception | None = None
            try:
                try:
                    async with self.pilot.holding(plan.fingerprint, role):
                        agen, first, spec, used_plan, call_started = (
                            await self._open_stream(canonical, decision, plan, chain)
                        )
                    if spec.key != model_used.key and role is PilotRole.PILOT:
                        # The pilot's model fell over; its prefix will not be
                        # warmed. Free the seat so a follower can take it.
                        await self.pilot.release_failed(plan.fingerprint)
                    model_used = spec
                    streamed = True
                    # First token = the provider cache is readable. Release the
                    # followers now, not when the stream finishes minutes later.
                    if used_plan.cacheable:
                        await self.pilot.mark_warm(
                            used_plan.fingerprint, self._s.session_ttl_seconds
                        )
                except Exception as exc:
                    if role is PilotRole.PILOT:
                        await self.pilot.release_failed(plan.fingerprint)
                    failed = exc
                    raise

                opener = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_used.key,
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                    # Routing transparency arrives in the first chunk so a
                    # streaming caller learns which model it got without
                    # waiting for the end.
                    "x_gateway": {
                        "trace_id": record.trace_id,
                        "chosen_model": model_used.key,
                        "resolved_intent": prepared.intent.intent,
                        "routing_reason": decision.reason,
                        "cache_state": decision.cache_state,
                        "pilot_role": role.value,
                        "fallback_chain": chain,
                    },
                }
                yield f"data: {json.dumps(opener)}\n\n"

                async def relay(chunk: dict):
                    nonlocal usage
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    delta = chunk.get("delta", {})
                    if isinstance(delta.get("content"), str):
                        text_parts.append(delta["content"])
                    return chunk_for(model_used.key, delta, chunk.get("finish_reason"))

                if first is not None:
                    yield await relay(first)
                try:
                    async for chunk in agen:
                        yield await relay(chunk)
                except (UpstreamError, GatewayError) as exc:
                    # Mid-stream failure: the model identity is committed, so
                    # there is nothing to fall back to. Close the stream
                    # honestly rather than leaving it hanging.
                    failed = exc
                    log.warning("stream from %s died mid-flight: %s", model_used.key, exc)
                    yield chunk_for(model_used.key, {}, "error")
                yield "data: [DONE]\n\n"
            except GeneratorExit:
                disconnected = True
                raise
            finally:
                settle = self._settle_stream(
                    prepared, record, principal, model_used, chain, usage,
                    "".join(text_parts), call_started, streamed, disconnected,
                    failed, started,
                )
                if disconnected:
                    # Awaiting inside GeneratorExit handling is not allowed;
                    # the books still have to balance, so settle out-of-band.
                    asyncio.get_running_loop().create_task(settle)
                else:
                    await settle

        return generate()

    async def _open_stream(self, canonical, decision, plan, chain: list[str]):
        """Start a provider stream, falling back until the first chunk arrives.

        Fallback is only sound *before* any byte reaches the caller — after
        that the response is committed to one model. So each candidate is held
        to a first-chunk deadline, and failures roll to the next exactly like
        the unary chain.
        """
        candidates = [decision.model] + self._registry.fallback_chain(
            decision.model, switchboard=self._switchboard, health=self._health
        )
        last_error: Exception | None = None

        for attempt, spec in enumerate(candidates):
            try:
                provider = self._registry.get(spec.provider)
            except GatewayError as exc:
                last_error = exc
                continue

            attempt_plan = plan if spec.key == decision.model.key else plan_cache(
                canonical, spec, ttl=self._s.cache_ttl
            )
            agen = provider.stream(canonical, spec.key, decision.effort, attempt_plan)
            call_started = time.perf_counter()
            try:
                first = await asyncio.wait_for(
                    agen.__anext__(), timeout=self._s.upstream_timeout_seconds
                )
            except StopAsyncIteration:
                first = None  # empty stream is still a served stream
            except ProviderRefusal:
                raise  # a content outcome, not an outage — never retried
            except TimeoutError:
                if self._health:
                    self._health.record_failure(spec.key, "no first token in time")
                chain.append(f"{spec.key}(failed:timeout)")
                last_error = UpstreamError(
                    f"{spec.key} sent no first token within "
                    f"{self._s.upstream_timeout_seconds:.0f}s", 504,
                )
                continue
            except (UpstreamError, GatewayError) as exc:
                status = getattr(exc, "status_code", 500)
                if status in (400, 403, 422):
                    raise  # our request is wrong; another model will not fix it
                if self._health:
                    self._health.record_failure(spec.key, f"HTTP {status}")
                chain.append(f"{spec.key}(failed:{status})")
                last_error = exc
                continue

            if attempt:
                chain.append(f"{spec.key}(recovered)")
            return agen, first, spec, attempt_plan, call_started

        raise last_error or UpstreamError("all providers failed", 503)

    async def _settle_stream(
        self,
        prepared: Prepared,
        record: RequestRecord,
        principal: Principal,
        model_used: ModelSpec,
        chain: list[str],
        usage: Usage | None,
        text: str,
        call_started: float | None,
        streamed: bool,
        disconnected: bool,
        failed: Exception | None,
        started: float,
    ) -> None:
        """Close the books on a stream, however it ended.

        Ledger, health, record, and the feedback loops all run here — the
        parts of serving a request that must not depend on whether the client
        stayed for the whole answer.
        """
        try:
            canonical = prepared.canonical
            decision = prepared.decision

            if failed is not None:
                self._note_failure(record, failed)
                if self._health and streamed and call_started is not None:
                    self._health.record_failure(model_used.key, "died mid-stream")
            elif disconnected:
                record.outcome = "client_disconnected"

            if streamed and failed is None and call_started is not None:
                elapsed = int((time.perf_counter() - call_started) * 1000)
                record.upstream_ms = elapsed
                if self._health:
                    self._health.record_success(model_used.key, elapsed)

            if usage is None and streamed and failed is None:
                # The stream ended before the usage chunk (disconnect, or a
                # provider that never sends one). Estimate from what actually
                # went over the wire — an approximate bill beats a silent zero,
                # and the estimate is marked as such on the record.
                prompt = record.prefix_tokens_est + record.volatile_tokens_est
                usage = Usage(
                    prompt_tokens=prompt,
                    completion_tokens=estimate_tokens(text),
                    total_tokens=prompt + estimate_tokens(text),
                )
                record.error = (record.error or "") + " [usage estimated]"

            if usage is not None:
                priced = price_usage(usage, model_used, self._s.cache_ttl)
                await self._ledger.record(
                    principal.tenant_id, principal.agent_id, model_used.key, priced,
                    reserved_usd=prepared.verdict.reserved_usd,
                )
                prepared.settled = True
                record.prompt_tokens = usage.prompt_tokens
                record.completion_tokens = usage.completion_tokens
                record.cache_read_tokens = usage.cache_read_tokens
                record.cache_write_tokens = usage.cache_write_tokens
                record.actual_cost_usd = priced.total_usd
                record.cache_savings_usd = priced.cache_savings_usd
            else:
                # Nothing was served, nothing can be priced — the reservation
                # goes back.
                await self._budget.release(principal.tenant_id, prepared.verdict)
                prepared.settled = True

            if streamed:
                await self._router.remember(canonical.session_id, model_used, canonical)
                await self._limiter.note_upstream(model_used.rate_limit_pool)
                self.alerts.clear("no_models_available")

            if self._cache_effectiveness is not None and usage is not None and not disconnected:
                self._cache_effectiveness.record(
                    model_used.key,
                    expected_hit=decision.cache_state == "warm_read",
                    cached_tokens=usage.cache_read_tokens,
                )

            # Quality and the learning loops, on the reconstructed response.
            # A disconnect says nothing about the model, so it teaches nothing.
            if streamed and failed is None and not disconnected and usage is not None:
                response = ProviderResponse(
                    text=text,
                    finish_reason="stop",
                    model=model_used.key,
                    usage=usage,
                )
                report = (
                    assess(canonical, response, decision)
                    if self._s.quality_checks_enabled
                    else QualityReport()
                )
                record.quality_verdict = report.verdict
                record.quality_failures = [c.id for c in report.failures]
                record.routing_ok = report.routing_ok
                if self._reputation:
                    self._reputation.record(
                        model_used.key, prepared.intent.intent, report.routing_ok
                    )
                if self._outputs is not None:
                    self._outputs.record(
                        prepared.intent.intent, usage.completion_tokens
                    )

            record.fallback_chain = chain
            record.chosen_model = model_used.key
            record.provider = model_used.provider
        except Exception as exc:  # settling must never break the transport
            log.warning("stream settlement failed: %s", exc)
        finally:
            record.latency_total_ms = int((time.perf_counter() - started) * 1000)
            self._sink.write(record)
