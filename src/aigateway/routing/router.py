"""Cache-aware, session-sticky model router.

The design decision that shapes everything else: **prompt caches are
model-scoped**. Routing turn 3 of a workflow to a cheaper model than turns 1-2
throws away a warm prefix and pays a cold write on the new model. For the
repetitive workflows a multi-agent system produces, a naively "optimal"
per-request choice routinely costs *more* than pinning one model per session.

So:

* the routing unit is the **session**, not the request;
* the cost function **prices in the cache transition**, not just the sticker
  rate; and
* mid-session tier changes are **escalation-only** — going down almost never
  recovers the write you just paid for.

Everything is a scored comparison, and every decision carries a human-readable
``reason`` into the response and the record. A router you cannot audit is a
router you cannot tune.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..cache.hints import CachePlan, plan_cache
from ..catalog import Capability, ModelSpec, Tier, get_model
from ..config import Settings
from ..quality import effort_that_fits
from ..tokens import estimate_request_tokens
from .policy import policy_for, tier_from_name

log = logging.getLogger(__name__)

# Expected output volume by effort level, for cost scoring only. max_tokens is a
# ceiling, not a forecast — scoring against the ceiling would make every model
# look output-dominated and flatten the comparison.
_EXPECTED_OUTPUT = {"low": 600, "medium": 1200, "high": 2500, "xhigh": 5000, "max": 8000}


@dataclass
class Candidate:
    model: ModelSpec
    cost_usd: float  # the score the router ranks on (may include adjustments)
    cache_state: str
    plan: CachePlan
    note: str = ""
    # Sticker cost before quality/vendor adjustment, so the two are never
    # confused — one is what you pay, the other is how we rank.
    raw_cost_usd: float = 0.0
    quality_multiplier: float = 1.0
    quality_samples: int = 0
    quality_success_rate: float | None = None


@dataclass
class RoutingDecision:
    model: ModelSpec
    tier: Tier
    effort: str
    reason: str
    cache_state: str
    cache_plan: CachePlan
    estimated_cost_usd: float
    prefix_tokens: int
    volatile_tokens: int
    considered: list[Candidate] = field(default_factory=list)
    # Models that did NOT qualify, and why. Surfaced because "who was ruled out
    # and for what reason" explains a routing decision at least as well as the
    # winner does.
    excluded: list[dict] = field(default_factory=list)
    degraded: bool = False
    escalated_from: str | None = None
    sticky: bool = False
    pinned: bool = False
    required_tier: Tier = Tier.LIGHT
    intent: str = ""


class Router:
    def __init__(
        self,
        settings: Settings,
        store,
        providers,
        health=None,
        switchboard=None,
        reputation=None,
        cache_effectiveness=None,
    ):
        """``providers`` is a live source of enabled provider names.

        Accepts either a plain set (tests, fixed deployments) or anything with
        an ``.enabled`` property — in practice the ProviderRegistry. It must be
        read per request rather than snapshotted, because credentials can be
        added at runtime and a snapshot would keep routing to a provider that
        is no longer configured, or ignore one that just became available.
        """
        self._s = settings
        self._store = store
        self._providers_source = providers
        # Optional: when present, models whose circuit breaker is open are not
        # candidates. Health is a routing input, not an error path.
        self._health = health
        # Optional: operator on/off switches. Checked before health, because an
        # operator's decision outranks an observation.
        self._switchboard = switchboard
        # Optional: observed quality per (model, intent). Makes a model that
        # keeps failing a given task more expensive to choose.
        self._reputation = reputation
        # Optional: whether a model actually delivers the cache we price it for.
        # Without this the router discounts a warm read that never arrives, and
        # since it picks the cheapest candidate, a model that fails to cache is
        # favoured by the very discount it does not earn.
        self._cache_effectiveness = cache_effectiveness

    @property
    def _providers(self) -> set[str]:
        source = self._providers_source
        return source.enabled if hasattr(source, "enabled") else source

    # -- session stickiness -------------------------------------------------
    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"session:{session_id}:model"

    async def warm_model_for(self, session_id: str | None) -> ModelSpec | None:
        if not session_id:
            return None
        key = await self._store.get(self._session_key(session_id))
        return get_model(key) if key else None

    async def remember(self, session_id: str | None, model: ModelSpec) -> None:
        if not session_id:
            return
        # TTL tracks the prompt-cache window: once the provider cache expires,
        # stickiness is no longer buying anything.
        await self._store.set(
            self._session_key(session_id), model.key, ttl=self._s.session_ttl_seconds
        )

    # -- scoring ------------------------------------------------------------
    def _cost(
        self,
        model: ModelSpec,
        prefix_tokens: int,
        volatile_tokens: int,
        expected_output: int,
        is_warm: bool,
        intent: str = "",
        apply_quality: bool = True,
    ) -> tuple[float, str, CachePlan, float, float]:
        plan = plan_cache_for(model, prefix_tokens, volatile_tokens)
        if (
            plan.cacheable
            and self._cache_effectiveness is not None
            and not self._cache_effectiveness.delivers(model.key)
        ):
            # Observed: this model does not return cached tokens. Price it for
            # what it actually does, not for what the catalog says it supports.
            plan.cacheable = False
            plan.reason = "observed: this model does not return cached tokens"
        # Rates can step up past a context threshold (OpenAI roughly doubles
        # above 272K; Anthropic has no such premium). Using the headline rate
        # for a large-context request under-prices it by 2x, which is exactly
        # backwards for the workloads where the bill is biggest.
        rate_in, rate_out = model.rates_for(prefix_tokens + volatile_tokens)
        price_in = rate_in / 1_000_000
        price_out = rate_out / 1_000_000

        if not plan.cacheable:
            cache_state = "uncached"
            input_cost = (prefix_tokens + volatile_tokens) * price_in
        elif is_warm:
            cache_state = "warm_read"
            input_cost = (
                prefix_tokens * price_in * model.cache_read_multiplier
                + volatile_tokens * price_in
            )
        else:
            cache_state = "cold_write"
            input_cost = (
                prefix_tokens * price_in * model.cache_write_multiplier(self._s.cache_ttl)
                + volatile_tokens * price_in
            )

        raw = input_cost + expected_output * price_out

        # Vendor preference is a deliberate thumb on the scale, not a price.
        # Applied to the *score*, never to the ledger — what you are billed
        # stays the real number.
        weight = self._s.vendor_weights.get(model.provider, 1.0)

        # Observed quality, priced as expected cost: a model that succeeds a
        # fraction s of the time needs 1/s attempts, so it really does cost
        # more than its sticker rate.
        quality_mult = 1.0
        if apply_quality and self._reputation and intent and self._s.quality_routing_enabled:
            quality_mult = self._reputation.multiplier(model.key, intent)

        return raw * weight * quality_mult, cache_state, plan, raw, quality_mult

    def _capable(
        self, model: ModelSpec, canonical, total_tokens: int, require_available: bool = True
    ) -> str | None:
        """Return a rejection reason, or None if the model can serve this."""
        if self._switchboard:
            # Checked even on a dry run: a preview that ignores your switches
            # would not be previewing the routing you actually configured.
            if off := self._switchboard.reason(model.key, model.provider):
                return off
        if require_available and model.provider not in self._providers:
            return "provider not configured"
        if require_available and self._health and not self._health.is_available(model.key):
            return "circuit open (unhealthy)"
        if canonical.tools and not model.supports(Capability.TOOLS):
            return "no tool support"
        if canonical.response_schema and not model.supports(Capability.STRUCTURED_OUTPUTS):
            return "no structured outputs"
        headroom = total_tokens + canonical.max_tokens
        if headroom > model.context_window:
            return f"context {model.context_window} < required {headroom}"
        if canonical.max_tokens > model.max_output_tokens:
            return f"max_output {model.max_output_tokens} < requested {canonical.max_tokens}"
        return None

    # -- main entry point ---------------------------------------------------
    async def route(
        self,
        canonical,
        intent: str,
        *,
        cost_ceiling_tier: Tier | None = None,
        require_available: bool = True,
    ) -> RoutingDecision:
        """Score the catalog and pick a model.

        ``require_available=False`` scores models whose provider has no
        credentials configured. Only ever correct for a dry run — the preview
        endpoint uses it so the routing logic can be inspected before any key
        is entered. The serving path must leave it True.
        """
        prefix_tokens, volatile_tokens = estimate_request_tokens(canonical)
        total = prefix_tokens + volatile_tokens
        policy = policy_for(intent)
        # Report the effort that will actually be used, not the one asked for.
        # The provider steps effort down when the budget cannot sustain it, so a
        # decision claiming `high` while `medium` was sent is a trace that
        # disagrees with the request it describes — and it is the field the
        # reasoning-starvation message quotes back at you.
        effort = effort_that_fits(canonical.effort or policy.effort, canonical.max_tokens)
        expected_output = min(canonical.max_tokens, _EXPECTED_OUTPUT.get(effort, 1500))

        # 1. Explicit pin bypasses the *scoring*, but not the operator.
        #
        # A pin is a caller instruction; a switch is an operator decision about
        # what this deployment is allowed to talk to right now. The operator
        # wins. Anything else means "switched off" does not actually mean off,
        # and an operator who disabled a model to take it out of service would
        # still be sending it traffic — which is the one moment the switch has
        # to be trustworthy.
        #
        # Health is deliberately NOT checked here. A breaker is an observation
        # that heals itself, and a caller who names a model explicitly is
        # entitled to try it.
        if canonical.pin_model:
            model = get_model(canonical.pin_model)
            if model is None:
                from ..errors import NoCapableModel

                raise NoCapableModel(f"unknown model '{canonical.pin_model}'")
            if self._switchboard:
                if off := self._switchboard.reason(model.key, model.provider):
                    from ..errors import NoModelsAvailable

                    # 503, not 422: the model exists and the request is fine —
                    # it is temporarily unavailable, so retrying is sensible.
                    raise NoModelsAvailable(
                        message=(
                            f"'{model.key}' is pinned by the request but {off}. An "
                            f"operator switch overrides a pin."
                        ),
                        cause="pinned_model_switched_off",
                        remedy=(
                            f"Turn '{model.key}' back on, or drop pin_model and let "
                            f"the router choose."
                        ),
                    )
            # Whether the session is already warm on this model matters just as
            # much for a pin as for a routed request. Hardcoding False here made
            # every pinned request report `cold_write` however many times it had
            # run — so the reported cache state was wrong, the cost estimate was
            # inflated by a write premium already paid, and the effectiveness
            # learner never saw a single pinned request because it only counts
            # requests where a hit was expected.
            pinned_warm = await self.warm_model_for(canonical.session_id)
            is_warm = bool(
                self._s.cache_aware_routing
                and pinned_warm is not None
                and pinned_warm.key == model.key
            )
            cost, cache_state, plan, raw, _ = self._cost(
                model, prefix_tokens, volatile_tokens, expected_output, is_warm,
                apply_quality=False,
            )
            return RoutingDecision(
                model=model,
                tier=model.tier,
                effort=effort,
                reason=f"pinned by caller (router bypassed); intent={intent}",
                cache_state=cache_state,
                cache_plan=plan,
                estimated_cost_usd=cost,
                prefix_tokens=prefix_tokens,
                volatile_tokens=volatile_tokens,
                pinned=True,
                required_tier=model.tier,
                intent=intent,
            )

        # 2. Required tier: policy floor, plus signal-based escalation.
        required = policy.min_tier
        reason_bits = [f"intent={intent} floor={required.name.lower()}"]
        if policy.escalate_on_tools and canonical.tools:
            required = max(required, Tier(min(required + 1, Tier.HEAVY)))
            reason_bits.append("escalated (tools present)")

        # 3. Caller and budget ceilings. The budget ceiling wins — it is the
        #    degraded path and must be able to pull the floor down.
        degraded = False
        if caller_cap := tier_from_name(canonical.max_tier):
            if required > caller_cap:
                required = caller_cap
                reason_bits.append(f"capped by caller max_tier={caller_cap.name.lower()}")
        if cost_ceiling_tier is not None and required > cost_ceiling_tier:
            required = cost_ceiling_tier
            degraded = True
            reason_bits.append(f"DEGRADED to {cost_ceiling_tier.name.lower()} by budget")

        warm = await self.warm_model_for(canonical.session_id)

        # Exploration is decided once for the whole request. Deciding it per
        # candidate would compare some models on adjusted cost and others on
        # raw cost, which is not a comparison at all.
        exploring = bool(
            self._reputation
            and self._s.quality_routing_enabled
            and self._reputation.should_explore()
        )
        if exploring:
            reason_bits.append("exploring (quality penalties ignored this request)")

        # 4. Build the candidate set.
        from ..catalog import CATALOG

        candidates: list[Candidate] = []
        rejected: list[str] = []
        excluded: list[dict] = []
        for model in CATALOG.values():
            if model.tier < required:
                excluded.append(
                    {
                        "model": model.key,
                        "provider": model.provider,
                        "tier": model.tier.name.lower(),
                        "required_tier": required.name.lower(),
                        "reason": "below the tier this task needs",
                        "kind": "tier",
                    }
                )
                continue
            if reject := self._capable(model, canonical, total, require_available):
                rejected.append(f"{model.key}: {reject}")
                excluded.append(
                    {
                        "model": model.key,
                        "provider": model.provider,
                        "tier": model.tier.name.lower(),
                        "required_tier": required.name.lower(),
                        "reason": reject,
                        "kind": _exclusion_kind(reject),
                    }
                )
                continue
            is_warm = bool(
                self._s.cache_aware_routing and warm is not None and warm.key == model.key
            )
            cost, cache_state, plan, raw, qmult = self._cost(
                model, prefix_tokens, volatile_tokens, expected_output, is_warm,
                intent=intent, apply_quality=not exploring,
            )
            candidates.append(
                Candidate(
                    model=model,
                    cost_usd=cost,
                    cache_state=cache_state,
                    plan=plan,
                    raw_cost_usd=raw,
                    quality_multiplier=qmult,
                    quality_samples=(
                        self._reputation.sample_count(model.key, intent)
                        if self._reputation else 0
                    ),
                    quality_success_rate=(
                        self._reputation.success_rate(model.key, intent)
                        if self._reputation else None
                    ),
                )
            )

        if not candidates:
            from ..errors import NoModelsAvailable

            # *Why* nothing is left decides both the remedy and whether a human
            # needs waking. "You switched everything off" and "every breaker has
            # tripped" produce identical empty candidate sets and could not be
            # more different to act on.
            cause, remedy = _diagnose_empty(excluded, required)
            raise NoModelsAvailable(
                message=(
                    f"no model satisfies intent '{intent}' at tier "
                    f"{required.name.lower()}: {'; '.join(rejected) or 'catalog empty'}"
                ),
                cause=cause,
                remedy=remedy,
            )

        # 5. Escalation-only stickiness — but ONLY when there is a warm cache to
        #    protect. The whole justification for holding a session on an
        #    expensive model is that switching discards a cached prefix and pays
        #    to rebuild it. If the prefix is not cacheable on that model, there
        #    is no cache, nothing is discarded, and holding would pin every
        #    cheap follow-up ("classify this", "translate that") to whatever
        #    heavyweight the session happened to open with. Fall through to
        #    price in that case.
        warm_candidate = next(
            (c for c in candidates if warm is not None and c.model.key == warm.key), None
        )
        if (
            self._s.escalate_only
            and warm is not None
            and warm.tier >= required
            and warm_candidate is not None
            and warm_candidate.plan.cacheable
        ):
            chosen = warm_candidate
            reason_bits.append(
                f"sticky: session warm on {warm.key} (escalate-only, no de-escalation)"
            )
            return RoutingDecision(
                model=chosen.model,
                tier=chosen.model.tier,
                effort=effort,
                reason="; ".join(reason_bits),
                cache_state=chosen.cache_state,
                cache_plan=chosen.plan,
                estimated_cost_usd=chosen.cost_usd,
                prefix_tokens=prefix_tokens,
                volatile_tokens=volatile_tokens,
                considered=candidates,
                excluded=excluded,
                degraded=degraded,
                sticky=True,
                required_tier=required,
                intent=intent,
            )

        # 6. Cheapest capable model, cache transition already priced in.
        candidates.sort(key=lambda c: (c.cost_usd, c.model.tier))
        chosen = candidates[0]

        if warm is not None and warm.key != chosen.model.key:
            delta = next(
                (c.cost_usd for c in candidates if c.model.key == warm.key), None
            )
            if delta is not None:
                reason_bits.append(
                    f"switching off warm {warm.key} "
                    f"(${delta:.5f}) to {chosen.model.key} (${chosen.cost_usd:.5f}) "
                    f"— cache write already priced in"
                )
            escalated_from = warm.key
        else:
            escalated_from = None
            reason_bits.append(
                f"cheapest capable: {chosen.model.key} @ ${chosen.cost_usd:.5f} "
                f"({chosen.cache_state})"
            )

        return RoutingDecision(
            model=chosen.model,
            tier=chosen.model.tier,
            effort=effort,
            reason="; ".join(reason_bits),
            cache_state=chosen.cache_state,
            cache_plan=chosen.plan,
            estimated_cost_usd=chosen.cost_usd,
            prefix_tokens=prefix_tokens,
            volatile_tokens=volatile_tokens,
            considered=candidates,
            excluded=excluded,
            degraded=degraded,
            escalated_from=escalated_from,
            required_tier=required,
            intent=intent,
        )


def _diagnose_empty(excluded: list[dict], required: Tier) -> tuple[str, str]:
    """Work out why the candidate set is empty, and what would fix it.

    Ordered by what an operator can act on fastest. A switch is one click; a
    credential is a config change; a tripped breaker needs the vendor to
    recover; a genuine capability gap needs the request to change.
    """
    kinds = [e.get("kind") for e in excluded if e.get("kind") != "tier"]
    if not kinds:
        return (
            "no_capable_model",
            f"nothing in the catalog is at {required.name.lower()} tier or above. "
            f"Add a model at that tier, or lower the floor for this intent.",
        )
    if all(k == "switched_off" for k in kinds):
        return (
            "all_switched_off",
            "Turn a model back on in the console, or POST /admin/switchboard/reset.",
        )
    if all(k == "no_credentials" for k in kinds):
        return (
            "no_credentials",
            "Add a key in the console, or set ANTHROPIC_API_KEY / OPENAI_API_KEY.",
        )
    if all(k in ("unhealthy", "switched_off", "no_credentials") for k in kinds):
        return (
            "all_unhealthy",
            "Breakers heal on their own. POST /admin/pool/reset to force them "
            "closed if the vendor has recovered.",
        )
    return (
        "no_capable_model",
        "Check context window, output length, tool support and structured outputs.",
    )


def _exclusion_kind(reason: str) -> str:
    """Classify a rejection so the explanation can group and word it correctly.

    The distinction that matters: a model kept out by the tier floor is a
    *policy* decision about this task, not a statement that the model is
    incapable. Conflating the two produces explanations that read as slurs on
    perfectly good models.
    """
    if "not configured" in reason:
        return "no_credentials"
    if "circuit open" in reason:
        return "unhealthy"
    if "switched off" in reason:
        return "switched_off"
    if reason.startswith("context ") or reason.startswith("max_output "):
        return "capacity"
    return "capability"


def plan_cache_for(model: ModelSpec, prefix_tokens: int, volatile_tokens: int) -> CachePlan:
    """Lightweight cacheability check used during scoring.

    The full plan (which regions get markers) is computed once, after a model is
    chosen — see ``cache.hints.plan_cache``. Here we only need to know whether
    this model *would* cache this prefix at all, which is exactly the
    ``min_cacheable_tokens`` question that varies non-monotonically across models.
    """
    plan = CachePlan(prefix_tokens=prefix_tokens, volatile_tokens=volatile_tokens)
    supports = model.supports(Capability.EXPLICIT_CACHE_BREAKPOINTS) or model.supports(
        Capability.AUTO_PREFIX_CACHE
    )
    plan.cacheable = supports and prefix_tokens >= model.min_cacheable_tokens
    plan.reason = (
        "cacheable"
        if plan.cacheable
        else f"prefix {prefix_tokens} < min {model.min_cacheable_tokens}"
    )
    return plan


__all__ = ["Router", "RoutingDecision", "Candidate", "plan_cache", "plan_cache_for"]
