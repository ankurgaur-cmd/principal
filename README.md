# Moon — AI Gateway

A vendor-agnostic gateway for multi-agent systems. It picks the cheapest model
that can actually do the job, and it does that **without throwing away your
prompt cache** — which is the part naive routers get wrong.

- **OpenAI-compatible.** Agents point `base_url` here and change nothing else.
- **Intent-based routing.** Layered classification (declared → rules → embeddings
  → small-model), mapped to capability tiers.
- **Cache-aware.** Routing is session-sticky and escalation-only, and the cost
  function prices the cache transition, not just the sticker rate.
- **Quality- and effort-adjusted.** A model that keeps failing — or keeps costing
  four turns to get one answer — gets more expensive in the router's eyes.
- **Cross-vendor.** Anthropic and OpenAI behind one schema, with vendor quirks
  absorbed rather than leaked.
- **Governed.** Per-tenant budgets, rate limits, cost attribution, and a JSONL
  record per request that feeds an offline replay harness.

Design rationale, decision records (D1–D26), failure modes, and known gaps:
**[ARCHITECTURE.md](ARCHITECTURE.md)**. How the gateway decides what you asked
for: **[CLASSIFICATION.md](CLASSIFICATION.md)**. How it then picks a model:
**[ROUTING.md](ROUTING.md)**. Ready-made scenarios for checking any of it
yourself: **[PLAYBOOK.md](PLAYBOOK.md)**.

---

## Contents

1. [Quickstart](#quickstart)
2. [Read this before your first request](#read-this-before-your-first-request) ← the one that bites
3. [Providing credentials](#providing-credentials)
4. [Calling it from an agent](#calling-it-from-an-agent)
5. [Multi-agent patterns: fan-out and fan-in](#multi-agent-patterns-fan-out-and-fan-in)
6. [Getting the cache to actually work](#getting-the-cache-to-actually-work)
7. [The console, tab by tab](#the-console-tab-by-tab)
8. [Closing the quality loop](#closing-the-quality-loop)
9. [Making it faster](#making-it-faster)
10. [Operating it](#operating-it)
11. [Endpoint reference](#endpoint-reference)
12. [Gateway extensions](#gateway-extensions-x_gateway)
13. [Layout, tests, and notes](#layout)

---

## Quickstart

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/uvicorn aigateway.main:app --reload
```

Then open **<http://localhost:8000/>** — the demo console, on **port 8000**. No
keys needed to start; paste them into the **API keys** panel (tab 4).

```bash
make dev     # same thing
make test    # 208 tests, no credentials needed
make lint
```

---

## Read this before your first request

**Reasoning models bill hidden reasoning against the same output budget as the
visible answer.** Set `max_tokens` too low and the model spends the entire
allowance thinking, returns nothing, and you are billed for it. It looks exactly
like a gateway fault and is not one.

This is not a small effect. Measured on one heavy code-review prompt, tokens
consumed *before a single visible character*:

| effort | gpt-5 | claude-opus-5 |
|---|---|---|
| low | 1,937 | 2,828 |
| medium | 2,742 | 4,445 |
| high | 6,080 | 3,886 |

**So the safest thing you can do is send no `max_tokens` at all.** The gateway
then sizes the budget from the classified intent — 600 for a `classify`, 8,000
for a `code_review`, 14,000 for `architecture`. A budget you send always wins.

That column is also the main **latency** control, because output budget is what a
request actually spends its wall-clock on:

| Request | budget | wall clock | answer |
|---|---|---|---|
| "Classify this ticket" (auto) | 600 | **2.8s** | 320 chars |
| "Review this middleware" (auto) | 8,000 | 56s | 6,486 chars |
| Same, Fast profile | 2,000 @ low effort | **23s** | 5,969 chars |

Note the third row: `effort=low` with a tighter budget returned 92% of the content
in 40% of the time. If a request feels slow, those two knobs are the answer — see
[Making it faster](#making-it-faster).

The gateway does three further things about starvation, but none of them can
conjure budget from nothing:

1. **Steps effort down to fit.** `high` on a 5,000-token budget becomes `medium`
   — your budget is your decision, how much goes to reasoning is the gateway's.
2. **Warns before spending.** Below the lowest floor there is nothing left to
   trade, so the `routed` stage carries a `budget_warning` and the console shows
   it before the call.
3. **Flags it after.** The quality check reports `reasoning_starved` with the
   floor for the effort actually used.

---

## Providing credentials

Intent classification, cost scoring, and `/admin/route/preview` work with **no
credentials at all** — they are pure computation over the catalog, and preview
labels every candidate `available: false` plus a top-level `servable_now: false`
so nothing is misrepresented. Serving a real request needs a key.

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
> enough — the router simply scores a smaller catalog.

Credential endpoints are **dev-mode only** (`GATEWAY_AUTH_MODE=dev`, which is
localhost-only by design). Under `jwt` mode they return `403` and keys must come
from the environment.

---

## Calling it from an agent

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused-in-dev-mode")

resp = client.chat.completions.create(
    model="auto",                      # 'auto' delegates to the router
    # Omit max_tokens and the gateway sizes it from the intent. Send one to override.
    messages=[
        {"role": "system", "content": LARGE_STABLE_SYSTEM_PROMPT},
        {"role": "user", "content": "Review this diff for race conditions."},
    ],
    extra_body={"x_gateway": {
        "session_id": "workflow-42",   # groups turns; enables cache-aware routing
        "intent": "code_review",       # optional hint; verified by the gateway
    }},
)

print(resp.model)                                # which model actually served it
print(resp.x_gateway.routing_reason)             # and why
print(resp.x_gateway.actual_cost_usd)            # what it cost
print(resp.x_gateway.cache_read_tokens)          # what the cache saved
print(resp.x_gateway.quality["verdict"])         # pass | warn | fail
print(resp.x_gateway.trace["hosts_contacted"])   # every server touched
```

**`session_id` is the one field worth always sending.** Without it every request
is a fresh routing decision with a cold cache, and you lose most of the benefit.

### Let the router decide, or tell it what you know

The router works with nothing but the messages. Three optional levers, in
increasing order of "you are overriding a decision":

```python
"x_gateway": {
    "intent": "code_review",   # a hint — the gateway verifies and may override
    "max_tier": "standard",    # a ceiling — never route above this
    "pin_model": "gpt-5",      # bypass the router entirely, logged as a bypass
}
```

### See the reasoning without spending anything

```bash
curl -s localhost:8000/admin/route/preview -H 'content-type: application/json' -d '{
  "model": "auto",
  "messages": [{"role":"user","content":"Review this diff for security issues"}]
}' | jq '.explain'
```

Returns the plain-language explanation the console renders: the five decision
steps, the **qualified** set across both vendors with the reason each one lost,
the five **dimensions** weighed (intelligence required, availability, cost, cache
economics, observed quality), which one actually **decided** it, and every model
ruled out with the rule that applied.

Add `"include_unavailable": true` to score models you have no key for — useful
for "what *would* this cost on Anthropic?"

---

## Multi-agent patterns: fan-out and fan-in

Your orchestrator does not pick models. It describes each unit of work, sends it
with `model: "auto"`, and the gateway routes each one independently.

### Fan-out — N agents, one shared prefix

```bash
curl -s localhost:8000/demo/fanout -H 'content-type: application/json' -d '{
  "prompt": "Review the capture retry change.",
  "shared_context": "<your large standing instruction>",
  "agents": 6,
  "subtasks": [
    "Check the retry loop for duplicate-capture risk.",
    "Check whether the idempotency key is enforced in the database.",
    "Check the ordering assumptions between capture and refund."
  ],
  "max_tokens": 4000,
  "pilot_enabled": true
}' | jq '.agents[] | {agent, model, pilot_role, cache_read_tokens, quality}'
```

`subtasks` gives each agent its own slice of the work — they cycle if you ask for
more agents than tasks. The **shared prefix stays byte-identical either way**,
which is what makes the cache shareable, so distinct tasks cost nothing and make
the parallel output actually worth comparing. Omit `subtasks` and every agent
gets the same prompt.

**The cache pilot** is the point of this endpoint. A cache entry is only readable
once the first response starts, so without help all N agents pay a cold write.
With `pilot_enabled: true` exactly one agent writes and the rest read it. Run it
both ways and compare `total_cache_write_tokens`.

> **If you see more than one cold write with the pilot on**, check whether the
> agents landed on different models. A prompt cache belongs to one model, so two
> models genuinely need two writes. The console names the split when it happens.
> Pin `intent` to hold every agent on one model.

### Fan-in — scatter, then gather

```bash
curl -s localhost:8000/demo/fanin -H 'content-type: application/json' -d '{
  "task": "Should we adopt prompt caching?",
  "subtasks": ["Cost impact?", "Latency impact?", "Operational risk?"],
  "shared_context": "<standing instruction>",
  "max_tokens": 4000
}' | jq '{tier_split, workers: [.workers[] | {model, tier}], synthesis: .synthesis.model}'
```

This is the shape most multi-agent systems actually take, and where a router
earns its keep: the workers do narrow jobs a small model handles fine, while the
synthesis step has to hold all their output at once and reason across it. Routing
them identically either wastes money on the workers or under-powers the
synthesis. Every leg is routed independently and traced.

---

## Getting the cache to actually work

Three things have to be true, and the console tells you when they are not:

1. **The prefix must be big enough.** Minimums are *not* uniform and *not*
   monotonic: `claude-opus-5` caches from 512 tokens, most models from 1,024,
   `claude-haiku-4-5` needs 4,096. A 1,400-token prefix caches on 11 of the 12
   models in the catalog and not on Haiku.
2. **The prefix must be byte-identical.** A timestamp, a reordered tool
   definition, or an edited system prompt invalidates it. The quality check
   reports `cache_missed` when the router priced a warm read and got none.
3. **The session must be the same.** Send `session_id`.

```python
# Turn 1 — cold write, ~25% surcharge on the prefix
# Turn 2 — warm read, ~90% off the prefix
```

`POST /demo/reset/{session}` forgets a session's warm model so you can run the
same query cold then warm, back to back, without waiting out the TTL.

---

## The console, tab by tab

A single self-contained page (no CDN, no build step) that drives the real public
API, so what you see is what an agent gets.

### 1 · Route — one request, end to end

Type a query and hit **Send**. Three presets fill the form: *Easy task*, *Hard
task*, and *Big cached prefix* (a real ~1,400-token review standard, sized to
cache on 11 of 12 models and deliberately short of Haiku's 4,096 floor so you can
see the cache economics differ on identical input).

- **Pipeline — live.** Stages light up from a real SSE stream (`POST
  /demo/trace`), not an animation. Each node is **coloured against its own
  learned baseline**: green faster than usual, blue within 1σ, amber over 1σ, red
  over 2σ, grey while still learning. A gauge shows measured-vs-expected with the
  baseline at the midpoint, a sparkline keeps the last 12 runs, and the arrow into
  each node carries the same verdict — its dashes travel faster when the stage is
  fast, so the signal survives a colour-vision difference. Hover any stage for
  mean, σ, p50/p95, sample count and segment.
  **Watch** mode holds each stage for a beat; the ms figure is always real. Switch
  to **Instant** for true speed.
- **Why this model?** Five plain steps, then the **qualified** set across both
  vendors with the reason each lost, the five **dimensions** weighed, and which
  one decided it. Models ruled out are grouped by the rule that applied — a tier
  floor and a missing API key read differently, because they are different
  situations.
- **Request & response.** What was sent, what came back, and a **quality check**
  on it: truncation, empty answers, schema violations, malformed tool arguments,
  a cache that was supposed to be warm and wasn't.
- **Transaction trace.** A waterfall of every hop — origination, the classifier's
  own model call, each model attempt *including ones that failed and triggered a
  fallback*, and any cache-pilot wait. The summary splits **upstream time** from
  **gateway overhead**, so "we are slow" and "their servers are slow" are separate
  numbers.

### 2 · Agent patterns

Diagrams of both shapes, then live runners for each. Fan-out has a per-agent task
box; every agent's answer is shown expanded next to the task it answered, with
its model, pilot role, cache reads, cost and quality verdict.

### 3 · Fleet & intelligence

- **Latency baselines** — every segment's mean, σ, p50/p95, best, sample count
  and the exact ms thresholds a warn or critical was measured against. Published
  because a colour you cannot check is a colour you have to trust blindly.
- **Fleet** — where traffic actually goes, per LLM server, and the
  `tenant → intent → tier → model → host` flows behind it.
- **Reputation** — per (model, intent) success rate, mean extra effort, and the
  resulting cost multiplier.

### 4 · Models & keys

Every model with a live health dot, circuit-breaker state, observed p50, success
rate, price and cache minimum. Health comes from periodic **free** probes and
from the outcome of real traffic; the second drives the breaker (3 consecutive
failures and a model leaves rotation). **Vendor and model switches** let you turn
any of them off by hand and watch the routing change — deliberately separate from
health, because a switch never self-heals and a breaker does.

---

## Closing the quality loop

The gateway can prove it picked a cheaper model. It cannot, on its own, tell you
the cheaper model was *good enough* — and without that every claimed saving is
unfalsifiable. Two mechanisms close it.

**Deterministic checks** run free on every response and feed routing: truncation,
empty output, invalid JSON against a requested schema, malformed tool arguments,
a missed cache. A failure is recorded as evidence the router was wrong.

**Effort** measures what reaching the goal actually cost. The retry model
(`1/success_rate`) counts one kind of effort; a task that "succeeded" after four
turns, two truncated drafts and an escalation cost far more than one call. Both
are expressed in **extra ideal calls** so they compose:

```
multiplier = (1 + mean_extra_effort) / success_rate
```

With no effort evidence the left factor is exactly 1.0 — this changes no routing
decision until there is evidence to change it.

Six signals are measured from the request itself (retries, wasted call,
truncation, token overrun, invisible reasoning work, latency overrun). Five need
a human in the loop and are registered but inert until your orchestrator reports
them:

```bash
curl -s localhost:8000/admin/effort -H 'content-type: application/json' -d '{
  "model": "gpt-5-nano", "intent": "code_review",
  "turns_to_goal": 5, "user_reasked": true, "manual_escalation": true
}' | jq '{extra_effort: .scored.extra_effort, multiplier_now}'
```

Every score comes back itemised — an effort penalty that cannot be enumerated is
indistinguishable from a grudge. `GET /admin/effort/signals` publishes the whole
table, including what each inert signal is waiting for.

**Adding your own signal** is a row, not a scorer edit:

```python
model.register(EffortSignal(
    "tool_thrash", "Same tool repeatedly, no progress",
    weight=1.0, attribution="session",
    measure=lambda obs, norms: obs.extras.get("tool_repeats", 0) / 3,
))
```

Every signal is normalised against the *same intent's* norm, never an absolute
threshold — otherwise you punish whichever model gets handed the hard problems,
which is exactly backwards.

### Prove the policy is worth it

```bash
./.venv/bin/python -m aigateway.replay.harness var/records.jsonl
```

Re-scores recorded traffic against alternative policies — always-frontier,
always-cheapest, stickiness on/off — with no model calls. Read the caveat in the
module docstring before acting on the numbers.

---

## Making it faster

**Measure before you change anything.** On this gateway every local stage
*together* costs about **1.3ms**, against an upstream call of 18–58 seconds. The
gateway is not what is slow, and switching parts of it off will not help.

The levers that do move wall-clock, in order:

| Lever | Effect | How |
|---|---|---|
| **Output budget** | 18s → 58s across 1,200 → 8,000 tokens | Omit `max_tokens` (auto-sized), or send a smaller one |
| **Effort** | `low` ≈ 25% faster than `medium`, comparable answer | `x_gateway.effort` |
| **Model tier** | small models finish sooner | `max_tier`, or a narrower intent |

The console's **Speed profile** sets the first two together: *Auto* (sized by
intent), *Fast* (low effort, 2,000), *Balanced*, *Thorough* (xhigh, 14,000).

### Switches

`GET /health` reports them; set them as env vars (`GATEWAY_` prefix):

| Switch | Default | What turning it off costs you |
|---|---|---|
| `auto_size_max_tokens` | on | Sizing an absent budget to the intent. **Leave this on** — it is the one that affects latency. |
| `latency_baselines_enabled` | on | The pipeline colours and `/admin/baselines`. |
| `hop_trace_enabled` | on | The per-transaction hop waterfall. |
| `quality_checks_enabled` | on | Every claimed saving becomes unfalsifiable. |
| `effort_tracking_enabled` | on | Effort-adjusted routing stops learning. |
| `quality_judge_enabled` | **off** | An extra billable model call per request. |
| `llm_classifier_enabled` | on | Layer 3 of intent classification. |

Only the last two cost real time — the judge is a whole extra model call, and the
LLM classifier is a small one. The middle four are the ~1.3ms. They exist for
running the serving path minimal, not as a latency fix.

---

## Operating it

- **Do not run multiple workers without Redis.** Budgets and rate limits are
  per-process; N workers means N times the intended limit. The gateway warns
  about this on start.
- **`GATEWAY_AUTH_MODE=dev` trusts request headers** for tenant identity. Local
  use only. Switch to `jwt` for anything reachable.
- **Provider keys live only in the gateway.** Agents authenticate to the gateway;
  the gateway authenticates to vendors. Key rotation is a one-place operation.
- **Budgets degrade before they fail.** Over the limit, a tenant is capped to the
  cheapest tier that can still do the job rather than being cut off.
- **The catalog is dated.** Prices carry `price_checked` and `price_verified`;
  `catalog_warnings()` names anything unverified and the gateway logs it on start.
  Re-check when the model line-up changes.

```bash
curl -s localhost:8000/admin/usage/demo | jq          # spend and limits
curl -s -X POST localhost:8000/admin/limits/demo \
  -H 'content-type: application/json' -d '{"daily_usd": 25, "rpm": 120}'
```

---

## Endpoint reference

| Endpoint | Purpose |
|---|---|
| `GET  /` | Demo console |
| `POST /v1/chat/completions` | OpenAI-compatible, streaming and unary |
| `GET  /v1/models` | OpenAI-shaped model list |
| `GET  /v1/catalog` | Full capability/pricing matrix the router scores against |
| `GET  /health` | Liveness plus active feature flags |
| **Routing** | |
| `POST /admin/route/preview` | Dry-run the router; classifies and scores, never calls a model |
| `GET  /admin/policy` | Current intent → tier table |
| **Observability** | |
| `GET  /admin/fleet` · `POST /admin/fleet/reset` | Where traffic goes, per server and per flow |
| `GET  /admin/baselines` | Latency baselines behind every colour on the pipeline |
| `GET  /admin/reputation` · `POST /admin/reputation/reset` | Per (model, intent) quality and effort |
| `POST /admin/effort` | Report caller-side effort for work already served |
| `GET  /admin/effort/signals` | The open effort table and what each signal needs |
| **Pool and switches** | |
| `GET  /admin/pool` | Health, breaker state, latency, success rate |
| `POST /admin/pool/probe` · `POST /admin/pool/reset` | Force a probe; close breakers (`?model=`) |
| `GET  /admin/switchboard` | Operator on/off state |
| `POST /admin/switchboard/model/{model}` · `/provider/{provider}` · `/reset` | Turn things off by hand |
| **Governance** | |
| `GET  /admin/usage/{tenant}` · `POST /admin/limits/{tenant}` | Spend, limits, utilisation |
| **Credentials (dev mode only)** | |
| `GET/POST /admin/credentials` · `DELETE /admin/credentials/{provider}` | Masked status; set and validate; remove |
| **Demo** | |
| `POST /demo/trace` | SSE: stream each pipeline stage with its baseline verdict |
| `POST /demo/fanout` | N sub-agents on one shared prefix; pilot on/off |
| `POST /demo/fanin` | Scatter/gather: N workers, then one synthesiser |
| `POST /demo/reset/{session}` | Forget a session's warm model |

Interactive API docs at **`/docs`**.

---

## Gateway extensions (`x_gateway`)

Sent in the request body; invisible to vendors.

| Field | Purpose |
|---|---|
| `session_id` | Groups turns of one workflow. Enables cache-aware, sticky routing. **Send this.** |
| `intent` | Caller-declared intent (layer 0). Accepted as given, but overridden if the request is plainly heavier than the label — see [CLASSIFICATION.md](CLASSIFICATION.md#3-l0--the-declared-hint-trusted-but-checked). |
| `effort` | `low`…`max`, mapped per vendor. Stepped down if `max_tokens` cannot support it. |
| `cache_hints` | Which regions get breakpoints: `system`, `tools`, `history`, `last_turn`. |
| `pin_model` | Bypass the router. Logged as a routing bypass. Needs the `model:pin` scope. |
| `max_tier` | Caller-imposed ceiling (`light`/`standard`/`heavy`). |
| `vendor_overrides` | Escape hatch, keyed by provider. |

Everything the gateway decided comes back on `x_gateway` in the response:
`chosen_model`, `routing_reason`, `tier`, `cache_state`, `cache_read_tokens`,
`estimated_cost_usd`, `actual_cost_usd`, `cache_savings_usd`, `fallback_chain`,
`degraded`, `quality`, `trace`, and the full `considered` list.

---

## Layout

```
src/aigateway/
  catalog.py         capability matrix, pricing, cache economics
  pipeline.py        request lifecycle, stage timing, baseline verdicts
  quality.py         deterministic response checks, reasoning-budget floors
  routing/
    intent.py        layered intent classification
    policy.py        intent → tier floor + effort
    router.py        cache-aware, quality-adjusted scoring (see ROUTING.md)
    reputation.py    per (model, intent) success rate and effort
    effort.py        the open effort table
    explain.py       the plain-language account of a decision
  cache/             breakpoint planning, prefix fingerprints, fan-out pilot
  providers/         Anthropic + OpenAI adapters, fallback chain, pool health
  governance/        budgets, rate limits, cost ledger
  observability/
    record.py        per-request JSONL record
    hops.py          per-transaction hop trace
    fleet.py         rolling fleet aggregate
    baselines.py     learned latency baselines and banding
  replay/            offline policy comparison
  static/index.html  the console
```

## Tests

```bash
./.venv/bin/python -m pytest -q      # 208 tests, no credentials needed
```

`tests/test_e2e.py` drives the real app through a stub provider that simulates a
model-scoped prompt cache — including the await between cache lookup and write,
without which the fan-out race cannot be reproduced.
