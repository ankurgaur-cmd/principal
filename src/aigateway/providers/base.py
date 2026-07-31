"""Provider adapter contract.

The neutral core plus typed vendor extensions, rather than a lowest-common-
denominator schema. A gateway that only exposes the intersection of two vendors
throws away precisely the features that make each one cheaper or better —
explicit cache breakpoints and adaptive thinking on one side, automatic prefix
caching on the other.

So adapters are allowed to know things. What they are *not* allowed to do is
leak a vendor error for something the caller did nothing wrong to trigger:
unsupported sampling params get stripped, refusals get normalised, and thinking
configuration is derived from the neutral ``effort`` value.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from ..cache.hints import CachePlan
from ..schemas import CanonicalRequest, ProviderResponse


@runtime_checkable
class Provider(Protocol):
    name: str

    async def invoke(
        self,
        canonical: CanonicalRequest,
        model_key: str,
        effort: str,
        cache_plan: CachePlan,
    ) -> ProviderResponse: ...

    async def stream(
        self,
        canonical: CanonicalRequest,
        model_key: str,
        effort: str,
        cache_plan: CachePlan,
    ) -> AsyncIterator[dict[str, Any]]: ...

    async def classify(
        self, model_key: str, system: str, text: str, schema: dict[str, Any]
    ) -> dict[str, Any] | None: ...

    async def count_tokens(
        self, canonical: CanonicalRequest, model_key: str
    ) -> int | None:
        """Exact pre-flight count, or None when the provider offers no endpoint."""
        ...
