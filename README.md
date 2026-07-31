# Moon — AI Gateway

A vendor-agnostic gateway for multi-agent systems. It picks the cheapest model
that can actually do the job, and it does that **without throwing away your
prompt cache** — which is the part naive routers get wrong.

- **OpenAI-compatible.** Agents point `base_url` here and change nothing else.
- **Intent-based routing.** Layered classification (declared → rules → embeddings
  → small-model), mapped to capability tiers.
- **Cache-aware.** Routing is session-sticky and escalation-only, and the cost
  function prices the cache transition, not just the sticker rate.
- **Cross-vendor.** Anthropic and OpenAI behind one schema, with vendor quirks
  absorbed rather than leaked.
- **Governed.** Per-tenant budgets, rate limits, cost attribution, and a JSONL
  record per request that feeds an offline replay harness.

Design rationale, decision records, failure modes, and known gaps:
**[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Quickstart

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/uvicorn aigateway.main:app --reload
```

Then open **<http://localhost:8000/>** — the demo console. No keys needed to
start; paste them into the **API keys** panel at the top left.

## Providing credentials

Intent classification, cost scoring, and `/admin/route/preview` work with **no
credentials at all** — they are pure computation over the catalog, and preview
labels every candidate `available: false` plus a top-level `servable_now: false`
so nothing is misrepresented. Serving an actual request needs a key. Three ways
to supply one:

| Method | How |
|---|---|
| **Console panel** (easiest) | Paste into **API keys** → *Validate & use*. Held in memory; tick the box to also write it to `.env`. |
| **Environment** | `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in `.env` or the shell. |
| **Ambient** | Click *Use ambient* to use whatever the SDK already resolves — including an OAuth profile from `ant auth login`. Nothing is pasted or stored. |

Keys are validated against each vendor's **free models endpoint**, so a wrong key
fails immediately and costs nothing. A bad key is discarded rather than left in
place, and the API never returns a key — only a mask (`sk-…a1b2`).

> **A Claude.ai or ChatGPT subscription is not API access.** Claude Pro/Max and
> ChatGPT Plus do not include API keys. You need API credits from
> [console.anthropic.com](https://console.anthropic.com) or
> [platform.openai.com](https://platform.openai.com). Either provider alone is
> enough to run the demo — the router simply scores a smaller catalog.

The credential endpoints are **dev-mode only** (`GATEWAY_AUTH_MODE=dev`, which is
localhost-only by design). Under `jwt` mode they return `403` and keys must come
from the environment.

## The demo console

A single self-contained page (no CDN, no build step) that drives the real public
API, so what you see is what an agent gets.

**The four things worth showing:**

1. **Routing, live — and why.** Type a query, hit Send. The pipeline diagram
   lights up stage by stage — **Query → Intent classifier → Router → Model
   pool** — from a real SSE event stream (`POST /demo/trace`). Routing takes
   single-digit milliseconds, so **Watch** mode holds each stage for a beat to
   make it followable; the ms figure on each node is always the real
   measurement. Switch to **Instant** for true speed.

   Below it, **"Why this model?"** answers the actual question in five plain
   steps — what you asked for, what that needs, who qualified, why this one
   over the runner-up (with the price ratio), and what it costs — plus every
   model that was ruled out and why. The router's own log line is still there,
   collapsed under *Technical detail*. Try "classify this ticket" versus
   "review this diff for race conditions" and watch the tier move.

2. **Caching.** Put a large block in *Shared context* — the panel tells you which
   models it is big enough to cache on, since the minimum is not the same across
   models. Send once (`↑ cold write`), then send again on the same session
   (`✓ warm read`, ~90% off the prefix). **Reset session** to go cold again.

3. **Fan-out.** Run the fan-out panel with the pilot **enabled**: one agent
   writes the cache, the rest read it. Switch it to **disabled** and re-run — all
   N pay a cold write, which is the default behaviour of every gateway that does
   not do this. Compare the totals.

4. **The transaction trace.** A waterfall of every hop — origination, the
   classifier's own model call, each model attempt (including ones that failed
   and triggered a fallback), and any cache-pilot wait. Each row names the
   server actually contacted, the endpoint, tokens, and latency; bar position
   and length place it on a shared timeline. The summary splits **upstream
   time** from **gateway overhead**, so "we are slow" and "their servers are
   slow" are separate numbers. Also on `x_gateway.trace` for any agent.

5. **The model pool.** Every model with a live health dot, circuit-breaker
   state, observed p50 latency, success rate, price, and cache minimum. The
   chosen model is highlighted on each request. Health comes from two sources —
   periodic **free** probes of each provider, and the outcome of real traffic —
   and the second drives the breaker: after 3 consecutive upstream failures a
   model leaves the rotation entirely, and the router stops scoring it. *Probe
   now* and *Reset breakers* let you drive it by hand during a demo.

### Call it like OpenAI

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused-in-dev-mode")

resp = client.chat.completions.create(
    model="auto",                      # 'auto' delegates to the router
    messages=[
        {"role": "system", "content": LARGE_STABLE_SYSTEM_PROMPT},
        {"role": "user", "content": "Classify this ticket."},
    ],
    extra_body={"x_gateway": {
        "session_id": "workflow-42",   # groups turns; enables cache-aware routing
        "intent": "classify",          # optional hint; verified by the gateway
    }},
)

print(resp.model)                      # which model actually served it
print(resp.x_gateway.routing_reason)   # and why
```

`session_id` is the one field worth always sending. Without it every request is
a fresh routing decision with a cold cache, and you lose most of the benefit.

### See the router's reasoning without spending anything

```bash
curl -s localhost:8000/admin/route/preview -H 'content-type: application/json' -d '{
  "model": "auto",
  "messages": [{"role":"user","content":"Review this diff for security issues"}]
}' | jq .
```

```json
{
  "intent": { "resolved": "code_review", "confidence": 0.75, "source": "rules" },
  "decision": {
    "model": "gpt-5",
    "tier": "heavy",
    "reason": "intent=code_review floor=heavy; cheapest capable: gpt-5 @ $0.03440 (cold_write)",
    "estimated_cost_usd": 0.034397
  },
  "considered": [
    { "model": "gpt-5",        "estimated_cost_usd": 0.034397 },
    { "model": "claude-opus-5","estimated_cost_usd": 0.109473 }
  ]
}
```

### Prove the routing policy is worth it

```bash
./.venv/bin/python -m aigateway.replay.harness var/records.jsonl
```

Re-scores recorded traffic against alternative policies — always-frontier,
always-cheapest, stickiness on/off — with no model calls. Read the caveat in the
module docstring before acting on the numbers.

---

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET  /` | Demo console |
| `POST /v1/chat/completions` | OpenAI-compatible, streaming and unary |
| `GET  /v1/models` | OpenAI-shaped model list |
| `GET  /v1/catalog` | Full capability/pricing matrix the router scores against |
| `POST /admin/route/preview` | Dry-run the router; classifies and scores, never calls a model |
| `GET  /admin/usage/{tenant}` | Spend, limits, utilisation |
| `POST /admin/limits/{tenant}` | Set daily USD / RPM |
| `GET  /admin/policy` | Current intent → tier table |
| `GET  /admin/credentials` | Masked credential status (never a key) |
| `POST /admin/credentials` | Set + validate a key, or use ambient credentials |
| `DELETE /admin/credentials/{provider}` | Remove a key |
| `GET  /admin/pool` | Model pool: health, breaker state, latency, success rate |
| `POST /admin/pool/probe` | Force a health probe now |
| `POST /admin/pool/reset` | Close circuit breakers (`?model=` for one) |
| `POST /demo/trace` | SSE: stream each pipeline stage as it completes |
| `POST /demo/fanout` | N sub-agents on one shared prefix; pilot on/off |
| `POST /demo/reset/{session}` | Forget a session's warm model |
| `GET  /health` | Liveness plus the active feature flags |

## Gateway extensions

Sent as `x_gateway` in the request body; invisible to vendors.

| Field | Purpose |
|---|---|
| `session_id` | Groups turns of one workflow. Enables cache-aware, sticky routing. |
| `intent` | Caller-declared intent (layer 0). Verified and overridable. |
| `effort` | `low`…`max`, mapped per vendor. |
| `cache_hints` | Which regions get breakpoints: `system`, `tools`, `history`, `last_turn`. |
| `pin_model` | Bypass the router. Logged as a routing bypass. |
| `max_tier` | Caller-imposed ceiling (`light`/`standard`/`heavy`). |
| `vendor_overrides` | Escape hatch, keyed by provider. |

## Layout

```
src/aigateway/
  catalog.py         capability matrix, pricing, cache economics
  pipeline.py        request lifecycle
  routing/           intent classification, policy table, cache-aware router
  cache/             breakpoint planning, prefix fingerprints, fan-out pilot
  providers/         Anthropic + OpenAI adapters, fallback chain
  governance/        budgets, rate limits, cost ledger
  observability/     per-request record
  providers/health   model pool: probes, breaker, latency
  replay/            offline policy comparison
```

## Tests

```bash
./.venv/bin/python -m pytest -q      # 86 tests, no credentials needed
```

`tests/test_e2e.py` drives the real app through a stub provider that simulates a
model-scoped prompt cache — including the await between cache lookup and write,
without which the fan-out race cannot be reproduced.

## Operational notes

- **Do not run multiple workers without Redis.** Budgets and rate limits are
  per-process; N workers means N times the intended limit. The gateway warns
  about this on start.
- **`GATEWAY_AUTH_MODE=dev` trusts request headers** for tenant identity. Local
  use only. Switch to `jwt` for anything reachable.
- **Provider keys live only in the gateway.** Agents authenticate to the gateway;
  the gateway authenticates to vendors.
- **Verify the OpenAI catalog entries** — ids and prices are placeholders. The
  Anthropic figures are current first-party rates.
