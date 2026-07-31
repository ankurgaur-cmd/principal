"""Model discovery, in the OpenAI ``/v1/models`` shape plus a catalog view."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from ..catalog import CATALOG

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models(request: Request) -> dict:
    enabled = request.app.state.registry.enabled
    return {
        "object": "list",
        "data": [
            {
                "id": spec.key,
                "object": "model",
                "created": int(time.time()),
                "owned_by": spec.provider,
                "available": spec.provider in enabled,
            }
            for spec in CATALOG.values()
        ]
        # 'auto' is the intended default: it hands model selection to the router.
        + [{"id": "auto", "object": "model", "created": int(time.time()), "owned_by": "gateway"}],
    }


@router.get("/catalog")
async def catalog(request: Request) -> dict:
    """Full capability and pricing matrix — what the router scores against."""
    enabled = request.app.state.registry.enabled
    return {
        "models": [
            {
                "key": s.key,
                "provider": s.provider,
                "available": s.provider in enabled,
                "tier": s.tier.name.lower(),
                "price_in_per_mtok": s.price_in_per_mtok,
                "price_out_per_mtok": s.price_out_per_mtok,
                "context_window": s.context_window,
                "max_output_tokens": s.max_output_tokens,
                "min_cacheable_tokens": s.min_cacheable_tokens,
                "cache_read_multiplier": s.cache_read_multiplier,
                "cache_write_multiplier_5m": s.cache_write_multiplier_5m,
                "supports_sampling_params": s.supports_sampling_params,
                "rate_limit_pool": s.rate_limit_pool,
                "capabilities": sorted(s.capabilities),
            }
            for s in CATALOG.values()
        ]
    }
