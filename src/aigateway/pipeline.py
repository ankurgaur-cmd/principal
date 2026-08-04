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
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from .auth import Principal
from .cache.hints import plan_cache
from .cache.pilot import CachePilot, PilotRole
from .catalog import ModelSpec, Tier
from .config import Settings
from .errors import GatewayError, ProviderRefusal, UpstreamError
from .governance import BudgetGuard, CostLedger, RateLimiter, price_usage
from .observability import LatencyBaselines, RecordSink, RequestRecord, TraceContext
from .observability.baselines import segment_key
from .quality import (
    REASONING_FLOOR_BY_EFFORT,
    Check,
    assess,
    budget_starves_the_answer,
    judge,
)
from .routing import IntentClassifier, Router, explain
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
from .tokens import estimate_request_tokens

log = logging.getLogger(__name__)


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
        self._reputation = reputation
        # Learned expectations, so the console can say whether *this* stage was
        # slow rather than just how long it took.
        self.baselines = baselines or LatencyBaselines()
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
            return await self._run(request, principal, record, started, emit, trace)
        except ProviderRefusal as exc:
            record.outcome = "refusal"
            record.error = str(exc.detail)
            raise
        except GatewayError as exc:
            record.outcome = (
                "budget_exceeded"
                if exc.status_code == 402
                else "rate_limited"
                if exc.status_code == 429
                else "error"
            )
            record.error = str(exc.detail)
            raise
        except Exception as exc:
            record.outcome = "error"
            record.error = repr(exc)
            raise
        finally:
            record.latency_total_ms = int((time.perf_counter() - started) * 1000)
            self._sink.write(record)

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
            role = await self.pilot.acquire(plan.fingerprint, self._s.session_ttl_seconds)
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

        # 9. invoke, with same-vendor-first fallback
        try:
            response, model_used, chain = await self._invoke_with_fallback(
                canonical, decision.model, decision.effort, plan, record, trace
            )
        except Exception:
            if role is PilotRole.PILOT:
                await self.pilot.release_failed(plan.fingerprint)
            raise
        restart_stage_clock()

        if role is PilotRole.PILOT and plan.cacheable:
            await self.pilot.mark_warm(plan.fingerprint, self._s.session_ttl_seconds)

        await self._router.remember(canonical.session_id, model_used)
        await self._limiter.note_upstream(model_used.rate_limit_pool)

        # 10. price from actual usage
        priced = price_usage(response.usage, model_used, self._s.cache_ttl)
        await self._ledger.record(
            principal.tenant_id, principal.agent_id, model_used.key, priced
        )

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
        report = assess(canonical, response, decision)
        if self._s.quality_judge_enabled and self._registry.enabled:
            last_user = next(
                (m.content for m in reversed(canonical.messages) if m.role == "user"), ""
            )
            report.judge = await judge(
                self._registry, self._s.quality_judge_model, str(last_user), response.text
            )
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
        if self._reputation:
            self._reputation.record(model_used.key, intent.intent, report.routing_ok)
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
        candidates = [model] + self._registry.fallback_chain(model)
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
                response = await provider.invoke(canonical, spec.key, effort, attempt_plan)
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

        Routing, budget, and cache handling are identical; only the response
        shape differs. Usage arrives in the final chunk, so the ledger is
        written when the stream closes rather than up front.
        """
        await self._limiter.check_tenant(principal.tenant_id)
        canonical = canonicalise(request)
        prefix_tokens, volatile_tokens = estimate_request_tokens(canonical)
        intent = await self._classifier.classify(canonical, prefix_tokens, volatile_tokens)
        decision = await self._router.route(canonical, intent.intent)
        verdict = await self._budget.check(principal.tenant_id, decision.estimated_cost_usd)
        if verdict.tier_ceiling is not None:
            decision = await self._router.route(
                canonical, intent.intent, cost_ceiling_tier=verdict.tier_ceiling
            )

        plan = plan_cache(canonical, decision.model, ttl=self._s.cache_ttl)
        provider = self._registry.get(decision.model.provider)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        async def generate():
            meta = {
                "chosen_model": decision.model.key,
                "resolved_intent": intent.intent,
                "routing_reason": decision.reason,
            }
            opener = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": decision.model.key,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
                # Routing transparency arrives in the first chunk so a streaming
                # caller learns which model it got without waiting for the end.
                "x_gateway": meta,
            }
            yield f"data: {json.dumps(opener)}\n\n"

            usage: Usage | None = None
            async for chunk in provider.stream(
                canonical, decision.model.key, decision.effort, plan
            ):
                if chunk.get("usage"):
                    usage = chunk["usage"]
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": decision.model.key,
                    "choices": [
                        {
                            "index": 0,
                            "delta": chunk.get("delta", {}),
                            "finish_reason": chunk.get("finish_reason"),
                        }
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"

            if usage:
                priced = price_usage(usage, decision.model, self._s.cache_ttl)
                await self._ledger.record(
                    principal.tenant_id, principal.agent_id, decision.model.key, priced
                )
            await self._router.remember(canonical.session_id, decision.model)
            yield "data: [DONE]\n\n"

        return generate()
