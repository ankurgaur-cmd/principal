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
from .cache import CacheEffectiveness
from .config import get_settings
from .errors import GatewayError
from .governance import BudgetGuard, CostLedger, RateLimiter
from .observability import AlertCentre, FleetStats, LatencyBaselines, build_sink
from .pipeline import GatewayPipeline
from .prices import PriceFeed
from .providers import ProviderRegistry
from .providers.health import HealthMonitor
from .providers.switchboard import Switchboard
from .routing import IntentClassifier, Reputation, Router
from .routing.outputs import OutputEstimator
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
        # Prices first restore what an earlier feed pull established, then
        # keep themselves current on a daily timer.
        await app_.state.prices.restore()
        app_.state.prices.start()
        yield
        await app_.state.prices.stop()
        await app_.state.health.stop()
        await store.close()
        if close := getattr(app_.state.sink, "close", None):
            close()

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
    # Backed by the store so every worker sees the same switches — an "off"
    # that only one process honours is not off.
    switchboard = Switchboard(store)
    # Checked by the router before it discounts anything for a cache.
    cache_effectiveness = CacheEffectiveness()
    reputation = Reputation(
        window=settings.quality_window,
        min_samples=settings.quality_min_samples,
        max_penalty=settings.quality_max_penalty,
        exploration_rate=settings.quality_exploration_rate,
    )
    ledger = CostLedger(store)
    budget = BudgetGuard(settings, store, ledger)
    limiter = RateLimiter(settings, store)
    # Learned completion volume per intent — written by the pipeline after
    # every response, read by the router's cost forecast.
    outputs = OutputEstimator()
    router_ = Router(
        settings, store, registry,
        health=health, switchboard=switchboard, reputation=reputation,
        cache_effectiveness=cache_effectiveness,
        limiter=limiter, outputs=outputs,
    )
    classifier = IntentClassifier(
        store,
        registry,
        enabled=settings.llm_classifier_enabled,
        model_key=settings.classifier_model,
        min_confidence=settings.classifier_min_confidence,
    )
    fleet = FleetStats()
    baselines = LatencyBaselines()
    alerts = AlertCentre()
    sink = build_sink(settings.record_path, fleet=fleet)

    app.state.settings = settings
    app.state.store = store
    app.state.registry = registry
    app.state.health = health
    app.state.switchboard = switchboard
    app.state.reputation = reputation
    app.state.router = router_
    app.state.classifier = classifier
    app.state.ledger = ledger
    app.state.budget = budget
    app.state.limiter = limiter
    app.state.sink = sink
    app.state.fleet = fleet
    app.state.baselines = baselines
    app.state.alerts = alerts
    app.state.cache_effectiveness = cache_effectiveness
    app.state.outputs = outputs
    app.state.prices = PriceFeed(
        store, settings.price_feed_url, settings.price_refresh_hours
    )
    app.state.auth = Authenticator(settings)
    app.state.pipeline = GatewayPipeline(
        settings, store, registry, router_, classifier, budget, limiter, ledger, sink,
        health=health, reputation=reputation, baselines=baselines,
        switchboard=switchboard, alerts=alerts,
        cache_effectiveness=cache_effectiveness, outputs=outputs,
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

    @app.get("/analytics", include_in_schema=False)
    async def analytics_page():
        """Analytics dashboard over the durable record database.

        Same conventions as the console: self-contained, no CDN, no-store.
        """
        return FileResponse(
            static_dir / "analytics.html",
            headers={"cache-control": "no-store, must-revalidate"},
        )

    @app.get("/transactions", include_in_schema=False)
    async def transactions_page():
        """Per-request drill-down over the record database. Same rules as the
        other pages: self-contained, no CDN, no-store."""
        return FileResponse(
            static_dir / "transactions.html",
            headers={"cache-control": "no-store, must-revalidate"},
        )

    @app.get("/db", include_in_schema=False)
    async def db_page():
        """Read-only SQL explorer over the record database."""
        return FileResponse(
            static_dir / "db.html",
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
            # Switchable work. Measure before changing any of these: together
            # they cost ~1.3ms against an upstream call of tens of seconds.
            "switches": {
                "auto_size_max_tokens": settings.auto_size_max_tokens,
                "latency_baselines": settings.latency_baselines_enabled,
                "hop_trace": settings.hop_trace_enabled,
                "quality_checks": settings.quality_checks_enabled,
                "effort_tracking": settings.effort_tracking_enabled,
                "quality_judge": settings.quality_judge_enabled,
                "llm_classifier": settings.llm_classifier_enabled,
            },
        }

    from .catalog import catalog_warnings, stale_prices, unverified_prices

    if placeholders := unverified_prices():
        log.warning(
            "placeholder pricing on %s — the router selects on price, so these "
            "decide which vendor gets traffic. Verify before trusting the ledger.",
            ", ".join(placeholders),
        )
    for lapsed in stale_prices():
        log.warning(
            "%s promotional pricing expired on %s: catalog says %s, actual is %s. "
            "Routing is being decided by a stale rate.",
            lapsed["model"], lapsed["expired_on"],
            lapsed["catalog_price"], lapsed["actual_price"],
        )

    for issue in catalog_warnings():
        log.warning("catalog: %s", issue)

    if not settings.redis_url:
        log.warning(
            "Running with in-memory state. Do not start more than one worker: "
            "budgets and rate limits are per-process, so N workers means N "
            "times the intended limit."
        )

    return app


app = create_app()
