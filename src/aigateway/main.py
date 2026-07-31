"""Application wiring.

Everything is constructed once at startup and hung off ``app.state`` so the
request path does no discovery. Run with::

    uvicorn aigateway.main:app --reload

Note the worker warning below: with the in-memory store, more than one worker
does not split a budget — it duplicates it.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from .api import admin, chat, credentials, demo, models
from .auth import Authenticator
from .config import get_settings
from .errors import GatewayError
from .governance import BudgetGuard, CostLedger, RateLimiter
from .observability import FleetStats, RecordSink
from .pipeline import GatewayPipeline
from .providers import ProviderRegistry
from .providers.health import HealthMonitor
from .routing import IntentClassifier, Router
from .state import build_store

log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    store = build_store(settings.redis_url)

    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        # Probing needs a running loop, so it starts here rather than at
        # construction time.
        app_.state.health.start(settings.health_probe_interval_seconds)
        yield
        await app_.state.health.stop()
        await store.close()

    app = FastAPI(
        lifespan=lifespan,
        title="Moon AI Gateway",
        version="0.1.0",
        description=(
            "Vendor-agnostic AI gateway: intent-based model routing with "
            "cross-vendor prompt-cache orchestration."
        ),
    )

    registry = ProviderRegistry(settings)
    health = HealthMonitor(
        registry,
        failure_threshold=settings.breaker_failure_threshold,
        cooldown_seconds=settings.breaker_cooldown_seconds,
    )
    # The registry is passed live, not snapshotted — credentials can be added
    # at runtime via the console, and routing must pick that up immediately.
    router_ = Router(settings, store, registry, health=health)
    classifier = IntentClassifier(
        store,
        registry,
        enabled=settings.llm_classifier_enabled,
        model_key=settings.classifier_model,
        min_confidence=settings.classifier_min_confidence,
    )
    ledger = CostLedger(store)
    budget = BudgetGuard(settings, store, ledger)
    limiter = RateLimiter(settings, store)
    fleet = FleetStats()
    sink = RecordSink(settings.record_path, fleet=fleet)

    app.state.settings = settings
    app.state.store = store
    app.state.registry = registry
    app.state.health = health
    app.state.router = router_
    app.state.classifier = classifier
    app.state.ledger = ledger
    app.state.budget = budget
    app.state.limiter = limiter
    app.state.sink = sink
    app.state.fleet = fleet
    app.state.auth = Authenticator(settings)
    app.state.pipeline = GatewayPipeline(
        settings, store, registry, router_, classifier, budget, limiter, ledger, sink,
        health=health,
    )

    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(admin.router)
    app.include_router(credentials.router)
    app.include_router(demo.router)

    static_dir = Path(__file__).parent / "static"

    @app.get("/", include_in_schema=False)
    async def console():
        """Self-contained demo console. No CDN, no build step.

        Served with no-store: the page is edited constantly during development
        and a cached copy silently hides new panels, which looks exactly like a
        feature not working.
        """
        return FileResponse(
            static_dir / "index.html",
            headers={"cache-control": "no-store, must-revalidate"},
        )

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers=getattr(exc, "headers", None),
        )

    @app.get("/health", tags=["ops"])
    async def health():
        return {
            "status": "ok",
            "providers": sorted(registry.enabled),
            "state_backend": type(store).__name__,
            "cache_pilot": settings.cache_pilot_enabled,
            "cache_aware_routing": settings.cache_aware_routing,
            "escalate_only": settings.escalate_only,
            "budget_mode": settings.budget_mode,
        }

    if not settings.redis_url:
        log.warning(
            "Running with in-memory state. Do not start more than one worker: "
            "budgets and rate limits are per-process, so N workers means N "
            "times the intended limit."
        )

    return app


app = create_app()
