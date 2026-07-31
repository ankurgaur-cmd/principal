"""Cache breakpoint planning and prefix fingerprinting.

The asymmetry this module exists to absorb: Anthropic requires **explicit**
``cache_control`` breakpoints (max 4 per request, prefix-match semantics, a
per-model minimum prefix size), while OpenAI caches prefixes **automatically**
with no markers at all. A vendor-neutral request has nowhere natural to say
"cache here", so the gateway decides — callers supply intent-level hints
(``system``, ``tools``, ``history``, ``last_turn``) and the Anthropic adapter
compiles them into markers. The OpenAI adapter ignores them.

Placement rule: markers go at *stability boundaries*, on the last block of the
stable region. Anything volatile — timestamps, per-request ids — must sit after
the final breakpoint or the whole cache is dead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ..catalog import Capability, ModelSpec
from ..tokens import estimate_request_tokens

MAX_BREAKPOINTS = 4


@dataclass
class CachePlan:
    """Where breakpoints go and what we expect them to be worth."""

    breakpoints: list[str] = field(default_factory=list)
    ttl: str = "5m"
    prefix_tokens: int = 0
    volatile_tokens: int = 0
    cacheable: bool = False
    reason: str = ""
    fingerprint: str = ""

    @property
    def enabled(self) -> bool:
        return self.cacheable and bool(self.breakpoints)


def prefix_fingerprint(canonical, model_key: str) -> str:
    """Stable identity for the cacheable prefix.

    Deliberately includes the model key: **caches are model-scoped**, so the
    same prefix on two models is two distinct cache entries. It deliberately
    excludes the final turn, which is the volatile part.

    Everything is serialised with sorted keys — non-deterministic JSON ordering
    is one of the most common silent cache invalidators there is.
    """
    parts = {
        "model": model_key,
        "tools": [
            {"n": t.name, "d": t.description, "p": t.parameters}
            for t in canonical.sorted_tools()
        ],
        "system": canonical.system,
        "history": [
            {"role": m.role, "content": m.content} for m in canonical.messages[:-1]
        ],
    }
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def plan_cache(canonical, model: ModelSpec, ttl: str = "5m") -> CachePlan:
    prefix_tokens, volatile_tokens = estimate_request_tokens(canonical)
    plan = CachePlan(
        ttl=ttl,
        prefix_tokens=prefix_tokens,
        volatile_tokens=volatile_tokens,
        fingerprint=prefix_fingerprint(canonical, model.key),
    )

    if model.supports(Capability.AUTO_PREFIX_CACHE):
        # Nothing to place. The provider caches prefixes on its own; our job is
        # limited to keeping the prefix byte-stable, which sorted_tools() and
        # the frozen-system-prompt rule already do.
        plan.cacheable = prefix_tokens >= model.min_cacheable_tokens
        plan.reason = "provider caches prefixes automatically"
        return plan

    if not model.supports(Capability.EXPLICIT_CACHE_BREAKPOINTS):
        plan.reason = "model does not support prompt caching"
        return plan

    if prefix_tokens < model.min_cacheable_tokens:
        # Silent failure mode: the API accepts the marker and simply never
        # caches. Say so out loud instead.
        plan.reason = (
            f"prefix {prefix_tokens} tok below {model.key} minimum "
            f"{model.min_cacheable_tokens} tok — markers would be a no-op"
        )
        return plan

    requested = canonical.cache_hints or ["system", "tools"]
    ordered: list[str] = []
    # Render order is tools -> system -> messages. Breakpoints must follow it.
    for region in ("tools", "system", "history", "last_turn"):
        if region not in requested:
            continue
        if region == "tools" and not canonical.tools:
            continue
        if region == "system" and not canonical.system:
            continue
        if region == "history" and len(canonical.messages) <= 1:
            continue
        ordered.append(region)

    if not ordered:
        plan.reason = "no stable region to mark"
        return plan

    # A breakpoint on the last system block already covers tools, which render
    # first. Spending a separate marker on tools wastes one of only four.
    if "tools" in ordered and "system" in ordered:
        ordered.remove("tools")
        plan.reason = "tools covered by the system breakpoint; "

    plan.breakpoints = ordered[:MAX_BREAKPOINTS]
    plan.cacheable = True
    plan.reason += f"breakpoints at {', '.join(plan.breakpoints)}"
    return plan
