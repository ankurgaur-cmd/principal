"""Live price refresh: pull current per-token rates into the catalog.

The catalog's prices are verified by hand and dated (see catalog.py) — and the
router selects on them, so a lapsed rate does not cause a small billing error,
it re-routes whole tiers. This module keeps them current without waiting for a
human to notice.

The uncomfortable fact this design starts from: **vendors do not publish a
pricing API.** Rates live on marketing pages. So the refresh pulls from a
machine-readable *feed* instead — by default the community-maintained LiteLLM
price table, which tracks vendor list prices in JSON — and the feed URL is a
setting, so an enterprise can point it at its own blessed price file with the
same shape. Every applied change carries provenance (source URL + date) into
the same ``price_source``/``price_checked`` fields the hand-verified entries
use, because a price you cannot trace is a price you cannot trust.

Guard rails, because a feed is an input, not an authority:

* A swing bigger than 8x in either direction is *reported, not applied* — that
  is far more likely a feed bug or a unit error than a real repricing, and one
  bad row must not re-route production traffic.
* Models absent from the feed are left exactly as they were, and named in the
  report, so "refreshed" never quietly means "half-refreshed".
* Applied overrides persist in the store and are re-applied on startup —
  otherwise a restart silently reverts to code-time prices and the ledger
  disagrees with yesterday's decisions.

Refresh runs on demand (the console button, ``POST /admin/prices/refresh``)
and on a daily timer (``price_refresh_hours``), which is the cadence list
prices actually change on.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time

from .catalog import CATALOG

log = logging.getLogger(__name__)

_STORE_KEY = "prices:overrides"
_REPORT_KEY = "prices:last_report"

# Beyond this ratio, assume the feed is wrong before assuming the vendor is.
_MAX_SWING = 8.0


class PriceFeed:
    def __init__(self, store, url: str, refresh_hours: float = 24.0):
        self._store = store
        self._url = url
        self._refresh_hours = refresh_hours
        self._task: asyncio.Task | None = None

    # -- fetch ---------------------------------------------------------------
    async def fetch(self) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(self._url)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _rates_from(entry: dict) -> tuple[float, float] | None:
        """Per-MTok rates from a feed entry (LiteLLM shape: cost per token)."""
        cost_in = entry.get("input_cost_per_token")
        cost_out = entry.get("output_cost_per_token")
        if not isinstance(cost_in, (int, float)) or not isinstance(cost_out, (int, float)):
            return None
        return round(cost_in * 1e6, 6), round(cost_out * 1e6, 6)

    # -- apply ---------------------------------------------------------------
    def apply(self, feed: dict, source: str) -> dict:
        """Fold a feed into the catalog; returns the full accounting.

        Never silent: every model lands in exactly one of `updated`,
        `confirmed` (same price, freshness date bumped), `suspicious`
        (swing too large, NOT applied), or `missing` (not in the feed).
        """
        today = time.strftime("%Y-%m-%d")
        updated, confirmed, suspicious, missing = [], [], [], []

        for key, spec in list(CATALOG.items()):
            entry = feed.get(spec.vendor_model_id) or feed.get(spec.key)
            rates = self._rates_from(entry) if isinstance(entry, dict) else None
            if rates is None:
                missing.append(key)
                continue
            new_in, new_out = rates

            if new_in == spec.price_in_per_mtok and new_out == spec.price_out_per_mtok:
                # Confirmation is information: the catalog's "last checked"
                # date is the whole defence against silent list-price drift.
                CATALOG[key] = dataclasses.replace(
                    spec, price_verified=True, price_checked=today
                )
                confirmed.append(key)
                continue

            if _swing(new_in, spec.price_in_per_mtok) > _MAX_SWING or _swing(
                new_out, spec.price_out_per_mtok
            ) > _MAX_SWING:
                suspicious.append(
                    {
                        "model": key,
                        "catalog": [spec.price_in_per_mtok, spec.price_out_per_mtok],
                        "feed": [new_in, new_out],
                        "note": "swing exceeds the sanity bound; not applied",
                    }
                )
                continue

            updated.append(
                {
                    "model": key,
                    "from": [spec.price_in_per_mtok, spec.price_out_per_mtok],
                    "to": [new_in, new_out],
                }
            )
            CATALOG[key] = dataclasses.replace(
                spec,
                price_in_per_mtok=new_in,
                price_out_per_mtok=new_out,
                price_verified=True,
                price_source=source,
                price_checked=today,
                # A live rate supersedes any recorded promo-expiry schedule.
                price_expires=None,
                price_after_expiry=None,
                price_note=f"live feed {today}; was "
                f"{spec.price_in_per_mtok}/{spec.price_out_per_mtok}",
            )

        return {
            "checked_at": today,
            "source": source,
            "updated": updated,
            "confirmed": sorted(confirmed),
            "suspicious": suspicious,
            "missing": sorted(missing),
        }

    async def refresh(self) -> dict:
        feed = await self.fetch()
        report = self.apply(feed, source=self._url)
        await self._persist(report)
        if report["updated"]:
            log.warning(
                "prices refreshed from feed: %s",
                "; ".join(
                    f"{u['model']} {u['from']} -> {u['to']}" for u in report["updated"]
                ),
            )
        return report

    # -- persistence ---------------------------------------------------------
    async def _persist(self, report: dict) -> None:
        overrides = {
            key: {
                "in": CATALOG[key].price_in_per_mtok,
                "out": CATALOG[key].price_out_per_mtok,
                "source": CATALOG[key].price_source,
                "checked": CATALOG[key].price_checked,
            }
            for key in [u["model"] for u in report["updated"]]
            + report["confirmed"]
        }
        existing = json.loads(await self._store.get(_STORE_KEY) or "{}")
        existing.update(overrides)
        await self._store.set(_STORE_KEY, json.dumps(existing))
        await self._store.set(
            _REPORT_KEY,
            json.dumps(
                {
                    "at": time.time(),
                    "source": report["source"],
                    "updated": len(report["updated"]),
                    "confirmed": len(report["confirmed"]),
                    "suspicious": len(report["suspicious"]),
                    "missing": len(report["missing"]),
                }
            ),
        )

    async def restore(self) -> int:
        """Re-apply persisted overrides after a restart.

        Without this, a restart quietly reverts to code-time prices while the
        ledger keeps yesterday's feed-priced records — an inconsistency nobody
        would spot until a chargeback dispute.
        """
        raw = await self._store.get(_STORE_KEY)
        if not raw:
            return 0
        try:
            overrides = json.loads(raw)
        except json.JSONDecodeError:
            return 0
        applied = 0
        for key, entry in overrides.items():
            spec = CATALOG.get(key)
            if spec is None:
                continue
            CATALOG[key] = dataclasses.replace(
                spec,
                price_in_per_mtok=entry["in"],
                price_out_per_mtok=entry["out"],
                price_verified=True,
                price_source=entry.get("source", ""),
                price_checked=entry.get("checked", ""),
                price_expires=None,
                price_after_expiry=None,
            )
            applied += 1
        if applied:
            log.info("restored %d feed-sourced prices from the store", applied)
        return applied

    async def last_report(self) -> dict | None:
        raw = await self._store.get(_REPORT_KEY)
        return json.loads(raw) if raw else None

    # -- daily timer ---------------------------------------------------------
    def start(self) -> None:
        """Daily pull. List prices change on day scale; poll on day scale."""
        if self._refresh_hours <= 0 or not self._url or self._task is not None:
            return

        async def loop() -> None:
            while True:
                await asyncio.sleep(self._refresh_hours * 3600)
                try:
                    await self.refresh()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # a feed outage must not kill the loop
                    log.warning("scheduled price refresh failed: %s", exc)

        self._task = asyncio.create_task(loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None


def _swing(a: float, b: float) -> float:
    lo, hi = sorted((abs(a), abs(b)))
    return hi / max(lo, 1e-9)
