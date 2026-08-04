# Architecture

Everything about how this gateway is built and why. Decisions are recorded with
their alternatives and consequences so they can be argued with rather than
inherited. Nothing here is settled by authority — if a rationale no longer holds
on your traffic, the decision should change.

**Contents**

1. [Purpose and scope](#1-purpose-and-scope)
2. [System context](#2-system-context)
3. [Component map](#3-component-map)
4. [The core design problem](#4-the-core-design-problem-routing-fights-caching)
5. [Request lifecycle](#5-request-lifecycle)
6. [Decision records](#6-decision-records)
7. [Key data structures](#7-key-data-structures)
8. [Failure modes](#8-failure-modes)
9. [Concurrency and consistency](#9-concurrency-and-consistency)
10. [Extension points](#10-extension-points)
11. [Security posture](#11-security-posture)
12. [Performance characteristics](#12-performance-characteristics)
13. [Open questions](#13-open-questions)
14. [Known gaps](#14-known-gaps)

---

## 1. Purpose and scope

A central gateway for a multi-agent system that:

- routes each request to the cheapest model that can actually do the job,
- keeps prompt caching working *while* doing so,
- presents one vendor-neutral API over Anthropic and OpenAI, and
- enforces per-tenant budgets and rate limits with auditable cost attribution.

**In scope:** model selection, cross-vendor translation, cache orchestration,
governance, observability, offline policy evaluation.

**Explicitly out of scope:** agent orchestration (the gateway does not know what
a workflow *is* beyond a session id), prompt management, vector storage, model
hosting, and evaluation of answer quality. The gateway measures cost and
routing; it does not measure whether the answer was good. That boundary matters
— see [§13](#13-open-questions).

---

## 2. System context

```
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │  Agent A     │   │  Agent B     │   │  Sub-agents  │
   │ (planner)    │   │ (reviewer)   │   │  (fan-out)   │
   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
          │ OpenAI-shaped HTTP + x_gateway hints │
          └──────────────────┼───────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │    Moon Gateway     │   ← the only holder of provider keys
                  └──────┬───────┬──────┘
                         │       │
            ┌────────────┘       └────────────┐
            ▼                                 ▼
    ┌───────────────┐                 ┌───────────────┐
    │  Anthropic    │                 │    OpenAI     │
    └───────────────┘                 └───────────────┘

            Redis (shared state)      records.jsonl (telemetry)
```

Agents authenticate to the gateway; the gateway authenticates to vendors. No
agent ever holds a provider credential. That is most of the point of having a
gateway at all, and it is what makes key rotation a one-place operation.

---

## 3. Component map

| Module | Responsibility | Depends on |
|---|---|---|
| `api/chat.py` | OpenAI-compatible surface | `pipeline` |
| `api/admin.py` | Spend, limits, policy, dry-run router | `pipeline`, `governance` |
| `api/credentials.py` | Runtime key entry + validation (dev only) | `providers` |
| `api/demo.py` | Fan-out demo, session reset | `pipeline` |
| `api/models.py` | Model list and capability matrix | `catalog` |
| `pipeline.py` | Orchestrates the request lifecycle | everything below |
| `schemas.py` | Wire + canonical representations | — |
| `catalog.py` | Capability matrix, pricing, cache economics | — |
| `routing/intent.py` | Layered intent classification | `catalog`, providers |
| `routing/policy.py` | Intent → tier floor + effort | `catalog` |
| `routing/router.py` | Cache-aware model scoring | `catalog`, `cache`, `state` |
| `cache/hints.py` | Breakpoint planning, prefix fingerprinting | `catalog`, `tokens` |
| `cache/pilot.py` | Fan-out serialisation | `state` |
| `providers/anthropic_provider.py` | Anthropic adapter | `catalog`, `cache` |
| `providers/openai_provider.py` | OpenAI adapter | `catalog` |
| `providers/registry.py` | Provider lookup, fallback ordering | `catalog` |
| `providers/health.py` | Model pool health, circuit breaker, probes | `catalog` |
| `governance/budget.py` | Per-tenant budget, degradation | `ledger`, `state` |
| `governance/ledger.py` | Usage pricing, spend accounting | `catalog`, `state` |
| `governance/ratelimit.py` | Inbound RPM, upstream pool pressure | `state` |
| `observability/record.py` | Per-request record + JSONL sink | — |
| `replay/harness.py` | Offline policy comparison | `catalog`, `routing` |
| `state/` | KV, counters, locks (Redis or in-memory) | — |
| `auth.py` | Principal resolution (dev headers or JWT) | — |
| `static/index.html` | Demo console | the HTTP API only |

**Dependency direction is strictly downward.** `catalog`, `schemas`, `tokens`,
and `state` know nothing about routing; routing knows nothing about HTTP. The
one deliberate exception is `providers/*` importing `cache/hints`, because
compiling a cache plan into vendor syntax is exactly the adapter's job.

---

## 4. The core design problem: routing fights caching

This is the single most important section. Everything structural follows from it.

**Prompt caches are model-scoped.** A cache entry is keyed on the exact model
plus the exact prefix bytes. So intent-based routing — which by definition sends
different requests to different models — systematically destroys the cache that
prompt caching is supposed to build.

The arithmetic, using Anthropic's published multipliers:

| Action | Cost vs. fresh input |
|---|---|
| Cache read | ~0.1× |
| Cache write, 5-minute TTL | 1.25× |
| Cache write, 1-hour TTL | 2× |

A 20,000-token prefix on a repetitive workflow:

- **Pinned to one model:** 1 write + N reads → `1.25 + 0.1N`
- **Rerouted every turn:** N writes → `1.25N`

At N=10 that is 2.25 versus 12.5 — the "cost-optimising" router is **5.5× more
expensive**, while also being slower (no cache read) and less coherent.

This means a naive intent router is not merely suboptimal; on the workloads a
multi-agent system actually produces, it is *worse than no router at all*. Three
structural consequences:

1. **The routing unit is the session** ([D4](#d4-session-scoped-routing)).
2. **The cost function prices the cache transition** ([D5](#d5-cache-aware-cost-function)).
3. **Tier changes are escalation-only** ([D6](#d6-escalation-only-tier-changes)).

A fourth, subtler consequence is the fan-out problem, which the same arithmetic
produces in the parallel case — see [D9](#d9-the-cache-pilot).

---

## 5. Request lifecycle

```
 1. authenticate            → Principal(tenant, agent, scopes)
 2. rate limit              → cheapest rejection first, before any work
 3. canonicalise            → OpenAI shape → CanonicalRequest
 4. estimate tokens         → (prefix_tokens, volatile_tokens), local + free
 5. classify intent         → L0 declared → L1 rules → L2 embed → L3 small model
 6. route (dry)             → RoutingDecision + cost estimate
 7. budget check            → may return a tier ceiling
 8. route (final)           → re-run under the ceiling, if degraded
 9. plan cache              → breakpoints for the chosen model
10. cache pilot             → PILOT | FOLLOWER | WARM | TIMEOUT
11. invoke                  → with same-vendor-first fallback
12. mark warm               → release followers
13. remember session model  → stickiness for the TTL
14. price actual usage      → not the estimate
15. ledger + record         → spend accounting, JSONL telemetry
16. respond                 → answer + x_gateway routing metadata
```

**Why steps 6 and 8 both exist.** The budget needs a cost; the cost needs a
model; the model depends on the budget. Routing twice breaks the cycle. It is
free — arithmetic over a catalog of six entries, no I/O — whereas guessing a
cost to budget against is not.

**Why step 9 comes after step 8.** Cache planning is model-specific:
`min_cacheable_tokens` differs per model, and the mechanism differs per vendor.
Planning before the model is chosen would produce a plan for the wrong model.

**Why step 14 uses actual usage.** The estimate is for routing; the ledger must
be truth. Both are recorded, and their divergence (`estimate_error`) is a
first-class metric — an estimator nobody checks is how budget enforcement
quietly stops working.

---

## 6. Decision records

### D1 — Runtime: Python 3.11 + FastAPI
**Context.** The gateway does per-request decision-making, not high-volume byte
shuffling.
**Decision.** Python with FastAPI and async provider SDKs.
**Alternatives.** TypeScript/Fastify (better if agents are TS end-to-end);
Go/net-http (best throughput and memory per connection).
**Consequences.** First-class official SDKs for both vendors, and a short path to
the embedding/classifier work in routing layer 2. Python's per-request overhead
is immaterial next to a model call measured in seconds. Revisit if the gateway
ever becomes latency-critical rather than decision-critical.

### D2 — API surface: OpenAI-compatible `/v1/chat/completions`
**Context.** Every agent framework already speaks some LLM API.
**Decision.** Expose the OpenAI chat-completions shape. `model: "auto"`
delegates to the router; an explicit model id is treated as a pin.
**Alternatives.** Anthropic Messages shape (native for thinking/cache_control,
fewer clients); a custom envelope (maximum control, every agent needs a custom
client).
**Consequences.** Agents repoint `base_url` and change nothing else. The cost is
that the OpenAI shape has no vocabulary for thinking blocks, cache breakpoints,
or effort — resolved by [D3](#d3-neutral-core-with-typed-vendor-extensions), not
by degrading to the vendor intersection.

### D3 — Neutral core with typed vendor extensions
**Context.** "Vendor-agnostic" usually decays into lowest-common-denominator,
which discards exactly the features that make each vendor cheaper or better.
**Decision.** `CanonicalRequest` holds the neutral core. Gateway-specific fields
ride in an `x_gateway` object; per-vendor escapes live in `vendor_overrides`,
keyed by provider. Adapters are permitted to know vendor specifics; what they
may not do is leak a vendor error for something the caller did nothing wrong to
trigger.
**Consequences.** The neutral schema stays small and honest. Adapters carry the
complexity, which is where it belongs and where it is testable.

### D4 — Session-scoped routing
**Context.** [§4](#4-the-core-design-problem-routing-fights-caching).
**Decision.** Route once per session (`x_gateway.session_id`) rather than per
request. The chosen model is remembered in shared state for
`session_ttl_seconds`, defaulting to the 5-minute prompt-cache TTL.
**Alternatives.** Per-request routing (throws away the cache); per-tenant pinning
(too coarse — a tenant runs many different workloads).
**Consequences.** Callers that omit `session_id` get a fresh decision and a cold
cache every time — correct, but they forfeit most of the benefit. The TTL is
tied to the cache window because stickiness past cache expiry buys nothing.

### D5 — Cache-aware cost function
**Context.** Sticker price is the wrong comparison when one candidate is warm.
**Decision.** `Router._cost()` charges the read multiplier for the session's
warm model and the write multiplier for every other candidate, then adds
expected output. Expected output comes from the effort level, not `max_tokens` —
scoring against the ceiling makes every candidate look output-dominated and
flattens the comparison.
**Consequences.** A more expensive warm model can legitimately beat a cheaper
cold one, which is the correct answer and the one a naive router gets wrong.
Toggleable via `GATEWAY_CACHE_AWARE_ROUTING` so its value is measurable rather
than assumed.

### D6 — Escalation-only tier changes, **only when a cache exists**
**Context.** Escalating buys capability you established you needed. De-escalating
pays a cache write to save a fraction of one request.
**Decision.** Mid-session, the router keeps the warm model whenever it still
clears the intent's tier floor **and the prefix is actually cacheable on it**.
It will move up, never down.
**Consequences.** A session that escalates once stays escalated for the TTL.

> **The cacheability condition was missing in the first implementation, and it
> was a serious bug.** Stickiness held regardless of whether there was a cache
> to protect, so a session that opened with one hard question pinned *every*
> subsequent request — "classify this", "translate that" — to the heavyweight
> model it started on. The justification for stickiness is "switching discards a
> warm prefix"; with no cacheable prefix nothing is discarded, so the hold was
> pure cost. Fresh sessions routed correctly, which is exactly why spot-checks
> missed it. Regression coverage is in `test_router.py`.

### D7 — Layered intent classification
**Context.** An LLM classifier on every request adds a latency floor and a cost
floor to the traffic you were trying to make cheaper.
**Decision.** Four layers, cheapest first, stopping at the first confident
answer: declared hint → deterministic rules → embedding nearest-neighbour →
small-model call (memoised by prompt hash).
**Alternatives.** LLM-only (simple, expensive); rules-only (free, brittle).
**Consequences.** Most traffic should never reach layer 3. `intent_source` in the
records is the diagnostic: frequent L3 firing means the rules or exemplars need
work, not a bigger budget. A classifier failure never fails the request it was
labelling — it falls through to `unknown`, which floors at the middle tier.

### D8 — The gateway owns cache breakpoint placement
**Context.** Anthropic requires **explicit** `cache_control` markers (max 4,
prefix-match, per-model minimum prefix). OpenAI caches prefixes **automatically**
with none. A neutral request has nowhere natural to say "cache here".
**Decision.** Callers give intent-level hints (`system`, `tools`, `history`,
`last_turn`); the Anthropic adapter compiles them into markers; the OpenAI
adapter ignores them. A marker on the last system block already covers tools
(render order is tools → system → messages), so the planner drops a redundant
tools marker rather than spending one of only four.
**Consequences.** Callers never learn vendor cache syntax. The planner reports
*why* a plan is or is not cacheable, which matters because the failure is
otherwise silent: the API accepts a marker on a too-short prefix and simply
never caches.

### D9 — The cache pilot
**Context.** A cache entry is only readable once the first response *begins*. Fire
8 sub-agents in parallel on one system prompt and all 8 miss — 8 cold writes
instead of 1 write and 7 reads.
**Decision.** The first caller on an unseen prefix fingerprint becomes the
**pilot** and proceeds; others become **followers** and wait briefly for the
pilot to mark the prefix warm.
**Consequences.** Adds up to `cache_pilot_wait_ms` of latency to followers on a
cold prefix, in exchange for ~90% off their input cost. Followers never block
indefinitely: on timeout they proceed and pay exactly what they would have paid
without the module, so a stuck pilot degrades to the status quo rather than to an
outage. The fingerprint includes the model key, because a warm prefix on one
model is not warm on another.

### D10 — State behind one interface, Redis or in-memory
**Context.** Budgets, limits, stickiness, and the pilot lock all need state.
**Decision.** One `StateStore` protocol; `MemoryStore` for laptops, `RedisStore`
for anything real; automatic fallback with a loud warning.
**Alternatives.** In-memory only (breaks under scale-out); Postgres for
everything (right for the ledger, overkill for the hot path).
**Consequences.** Zero-infrastructure local runs. **Two workers without Redis do
not split a budget — they duplicate it**, warned about on every start. Postgres
is deferred, not rejected: `CostLedger` is a thin interface with one obvious
place to grow a durable writer.

### D11 — Budgets degrade before they fail
**Context.** Cutting an agent off mid-workflow wastes everything spent so far.
**Decision.** Over-budget tenants are capped to the cheapest capable tier and the
degradation is reported in `x_gateway.degraded`. Only when the budget is fully
exhausted does the request fail. `GATEWAY_BUDGET_MODE=hard` opts into immediate
rejection.
**Consequences.** Silent degradation would be worse than either alternative,
because the agent could not react; hence the flag on every response. The budget
ceiling deliberately outranks the intent floor — it is the one input allowed to
pull the required tier *down*.

### D12 — Tiered pre-flight estimation
**Context.** Exact token counting is a real round trip.
**Decision.** A local estimator (~3.6 chars/token, deliberately conservative)
for routing and normal budget checks; the provider's `count_tokens` endpoint
only when a tenant is within `preflight_exact_threshold` of their cap.
**Consequences.** Exactness is bought only where it can change the outcome. The
estimator's signed error is recorded per request so drift is visible. Note that
`tiktoken` is never used on the Anthropic path — it is OpenAI's tokenizer and
undercounts Claude tokens by 15–20% on prose and more on code.

### D13 — Fallback: same vendor first
**Context.** A cross-vendor hop mid-conversation discards the warm cache and any
provider-native state.
**Decision.** Fall back to an adjacent tier on the same vendor; cross vendors
only when the whole vendor is unreachable. The chain is recorded.
**Consequences.** Two classes deliberately do **not** fall back — policy refusals
(retrying is futile and launders a refusal) and 4xx caused by our own request
(another model will not fix a malformed request). Streaming skips the chain
entirely: once bytes are on the wire, a transparent retry would require
buffering the response, which defeats streaming.

### D14 — Refusals are a first-class response type
**Context.** Anthropic returns policy refusals as **HTTP 200** with
`stop_reason: "refusal"` and a possibly-empty `content` array. Code that reads
`content[0]` unconditionally crashes.
**Decision.** Adapters check `stop_reason` before touching `content`, and a
refusal becomes a typed `403` with the policy category.
**Consequences.** Agents get a stable, distinguishable signal. **This is a
contract your agents must honour** — one that treats every non-200 as retryable
will hammer a refusal. See [§13](#13-open-questions).

### D15 — Unsupported parameters are stripped, not forwarded
**Context.** Current Anthropic frontier models reject `temperature`/`top_p`/
`top_k` with a 400 rather than ignoring them. An ordinary OpenAI-shaped request
carrying `temperature: 0.7` would simply fail.
**Decision.** The catalog records `supports_sampling_params`; the adapter drops
what the model cannot accept and logs the drop. Same gating for `effort`, which
is unsupported on older small models.
**Consequences.** Callers are not punished for writing a normal request. The
trade-off is a silent behavioural difference — the drop is logged and recorded,
not raised, on the grounds that failing the request is worse.

### D16 — One record per request, and offline replay
**Context.** "Route by intent to save money" is a claim until measured.
**Decision.** A `RequestRecord` per request, JSONL, carrying the decision, its
reasoning, the alternatives considered, cache token counts, both costs, and the
outcome. `replay/harness.py` re-scores recorded traffic against alternative
policies with no model calls.
**Consequences.** The router becomes defensible rather than assumed.
`cache_read_tokens` is surfaced prominently because zero across repeated
identical prefixes is the clearest symptom of a broken caching design and is
invisible in a standard OpenAI response. Field names map onto OTel span
attributes for when JSONL is outgrown. **Replay's limitation is real and stated
in the module: it holds output length constant across models, so a weaker model
needing more turns looks cheaper than it is.**

### D17 — Auth: gateway holds all provider keys
**Context.** Distributing vendor keys to agents makes rotation impossible and
blast radius unbounded.
**Decision.** Agents authenticate to the gateway (JWT with `tenant_id` /
`agent_id` / `scopes`); the gateway authenticates to vendors. `dev` mode trusts
headers for local use and warns on every start.
**Consequences.** Key rotation is a one-place operation. Tenant identity is
trustworthy enough to bill against. Model pinning is scope-gated, so a
cost-ceiling policy cannot be bypassed by a caller simply naming an expensive
model.

### D20 — Model pool health, with a circuit breaker
**Context.** A model can be unreachable, overloaded, or rate-limited
independently of its provider being "up". Without a memory of that, every
request rediscovers the failure by paying a full timeout.
**Decision.** `HealthMonitor` tracks each model from two independent signals:
periodic **free** probes of the provider's models endpoint, and the outcome of
real traffic. Three consecutive upstream failures open a circuit breaker; after
a cooldown one trial request is admitted (HALF_OPEN); a success closes it.
**Passive evidence outranks probe evidence** — a model that just failed three
real requests is unhealthy however cheerfully the models endpoint answers, so a
failing probe alone never opens a breaker.
**Consequences.** Health is a **routing input, not an error path**: the router
drops open-breaker models from the candidate set, so an unhealthy model is never
selected rather than being selected and then failed over. Only genuine upstream
faults count — a 4xx we caused would otherwise open a breaker on a perfectly
healthy model. Probing is per *provider*, not per model, because the endpoint is
provider-scoped and per-model probing would be N identical calls.

### D22 — Per-transaction hop trace
**Context.** A routing decision says what the gateway *chose*. It does not say
what happened on the wire, which is the question you have when something is
slow, expensive, or wrong.
**Decision.** A `TraceContext` per request collects a hop for the origination
and every outbound call — including attempts that **failed** and triggered a
fallback, and any wait the cache pilot imposed. Each hop names the host actually
contacted, the endpoint, the model, the attempt number, status, latency, and
tokens. Exposed on `x_gateway.trace`, as a `hops` stage on the SSE stream, and
summarised into the JSONL record.
**Consequences.** `total_ms - upstream_ms` gives **gateway overhead** as a
first-class number, separating "we are slow" from "their servers are slow".
Failed hops are the point: a log that records only the attempt that succeeded
hides the two that timed out first along with the latency they cost. The context
is created per request and passed explicitly — never ambient — so nothing leaks
between concurrent requests.

### D23 — Prices are marked verified or not
**Context.** The router selects on price. A wrong price therefore does not
produce a rounding error in the ledger — it silently routes an entire tier to
the wrong vendor, and looks deliberate while doing it.
**Decision.** `ModelSpec.price_verified`. The OpenAI entries are `False`, and
`unverified_prices()` names them.
**Consequences.** Placeholder pricing is visible rather than load-bearing-by-
accident. Until those numbers are confirmed, any cross-vendor comparison in this
gateway should be read as provisional.

### D24 — Reasoning budgets are measured, and guarded before the call
**Context.** Reasoning tokens bill against the same output budget as the visible
answer. Under a tight cap the model spends the whole allowance thinking and
returns an empty reply with `finish_reason: length` — which reads as a gateway
fault, is not one, and is paid for regardless.

The first version of this guard used estimated floors (`low` 300, `medium` 700,
`high` 1800). They were 5-10× too low. `effort_that_fits()` was doing its job —
stepping `high` down to `medium` — and the answer still came back empty, because
the number it was fitting to was fiction.

**Decision.** Floors come from measurement. One heavy code-review prompt, run
against a budget large enough that nothing truncated, completion tokens actually
consumed before any visible character:

| effort | gpt-5 | claude-opus-5 |
|---|---|---|
| low | 1,937 | 2,828 |
| medium | 2,742 | 4,445 |
| high | 6,080 | 3,886 |

`REASONING_FLOOR_BY_EFFORT` takes the worse vendor at each level plus headroom.
`xhigh` and `max` are extrapolated and commented as such. Stepping effort down
rescues a tight budget only until the bottom rung, so `budget_starves_the_answer()`
reports a budget below the lowest floor at routing time, and the pipeline emits it
on the `routed` stage — before the tokens are spent, not after.

**Consequences.** The floors are honest about their provenance and their limits:
consumption is strongly task-dependent (the same models spent 74 tokens on a
one-line question), and this is one prompt on two models, so they are working
guards for demanding work rather than published constants. They need re-measuring
when the line-up changes — which is the same obligation D23 places on pricing.
Callers with tight budgets now get told up front instead of billed for silence.

### D25 — Latency banding: colour is a claim, so it has to be checkable
**Context.** A duration on its own tells you nothing. "1,847ms" is excellent for
a heavy model writing a cold cache and terrible for a rules classifier. Colouring
each stage turns the number into a judgement — and a judgement the dashboard makes
on your behalf is only worth having if it is right and if you can audit it.

**Decision.** `observability/baselines.py` learns a mean and standard deviation
per stage with Welford's algorithm, bands each observation at 1σ (warn) and 2σ
(critical), and publishes everything it judged against at `/admin/baselines`.
Four constraints do the real work, and each one exists because the naive version
fails:

- **Segmented, never global.** Keys carry what legitimately changes the expected
  duration: the upstream call is keyed per model *and* per cache state. Pooling
  a cold write with a warm read paints every cold request red.
- **Confidence-gated.** Under `MIN_SAMPLES` a segment reports `learning` and is
  drawn neutral. Mean and σ over three samples are noise with a decimal point.
- **Material as well as anomalous.** A band above `normal` also requires an
  absolute gap of `MIN_MATERIAL_DEVIATION_MS` (25ms). This was not a hypothetical:
  the first live run lit three stages red at 3.7σ over *30 microseconds* of
  jitter. Statistically correct, operationally worthless. When σ and the floor
  disagree the verdict says so — "1.1 sigma from the 0.04ms baseline, but only
  0.02ms in absolute terms" — so the number and the colour never look like they
  contradict each other.
- **Judged before it is recorded.** Scoring a sample against a baseline it has
  already moved is how an anomaly hides itself: the more extreme the outlier, the
  harder it drags the mean towards itself.

Verdicts ride on each SSE stage event, so the console is rendering a judgement
the server computed and published rather than a threshold invented in the browser
that nobody could check.

**Consequences.** Two reporting bugs fell out of building this and are worth
recording, because both were the report contradicting itself rather than the
maths being wrong. A zero-variance segment published `warn ≥ 40, critical ≥ 40`,
which reads as "anything at the mean is critical" and is not what `judge` does;
it now reports the rule instead of a number. And sub-millisecond segments printed
`σ = 0.0` while deriving thresholds from a non-zero σ — every published figure is
now derived from the *rounded* σ, so the report cannot disagree with itself.

Stage timing is sub-millisecond because gateway-local work is: whole-millisecond
truncation pinned canonicalise, classify and route at a flat 0 and threw away the
only signal those stages have. The upstream call is judged on its own measured
latency rather than wall-clock between stages, because the two differ by whatever
the cache pilot spent waiting and blaming the vendor for our own wait would be
both wrong and unfalsifiable. For the same reason the stage clock restarts after
the model call: `quality` was being charged the entire vendor round-trip and had
learned a 3.7-second baseline for work that takes 0.24ms.

**On SLAs.** The seeded priors are *ours*, not the vendors'. OpenAI and Anthropic
publish availability SLAs, not latency SLAs, so nothing here is labelled `sla`
on their behalf — a prior is marked `prior` and is replaced by observed data as
soon as a segment has enough of it. Treating a vendor's uptime commitment as a
latency promise would be the same class of error as D23's unverified pricing.

**Known limit.** Latency is right-skewed, so 2σ does not mean "the slowest 2.3%"
the way it would for a normal distribution — in practice it flags somewhat more.
The bands stay in σ because that is the agreed vocabulary; p50/p95 and a
percentile rank ship alongside so the skew is visible rather than implied. Moving
the bands to log-space or to fixed percentiles would be the principled fix and is
an open question, not a decision.

### D26 — Effort-adjusted routing: price the task, not the call
**Context.** D-era reputation adjusts a model's price by `1/success_rate`, on the
principle that a model succeeding a fraction `s` of the time needs `1/s` attempts
so its honest cost is `cost/s`. That is already an effort measure — it just counts
one kind of effort. A task that "succeeded" after four turns, two truncated
drafts and an escalation to a bigger model did not cost one call, and the sticker
price of the winning call is the least interesting number in that story.

**Decision.** `routing/effort.py` scores extra work in the same unit — **extra
ideal calls** — and the two factors multiply:

    multiplier = (1 + mean_extra_effort) / success_rate

With no effort evidence the left factor is exactly 1.0, so enabling this changes
nothing until there is something to change it with.

The signal table is **open**: each row is data (name, weight, attribution rule,
function), so adding "the user re-asked" later means appending a row, not editing
the scorer. Six signals are measured today from the request itself — retries,
wasted call, truncation, token overrun, invisible reasoning work, latency
overrun. Five need a human in the loop and are registered but inert, returning
"no opinion" until an orchestrator reports them via `POST /admin/effort`:
turns-to-goal, re-ask, rejection, manual escalation, human edit distance.

**Two hazards, both of which make routing worse if ignored.**

*Confounding.* Hard work legitimately takes more turns and more tokens. Scoring
raw effort punishes whichever model gets handed the hard problems — it would
route hard tasks to models that have never seen one. Every signal is normalised
against the same intent's observed norm, never an absolute threshold. Same lesson
as D25: only compare like with like.

*Attribution.* Session effort spans requests that different models served.
Blaming whoever answered last is arbitrary, so signals declare `call` or
`session` attribution, and anything that cannot be attributed honestly returns
`None`.

**Consequences.** "No data" and "no effort" are different claims and are kept
distinct throughout — a signal without data is reported as *silent*, with what it
is waiting for, rather than scoring zero. A model looking good because nobody
measured it is the failure this avoids. Every score is itemised: an effort
penalty that cannot be enumerated is indistinguishable from a grudge. Per-signal
and total caps stop one catastrophic task exiling a model, and D-era mandatory
exploration still applies, so a model that improves can climb back.

**Open questions.** The weights on the measured signals are reasoned, not
calibrated — truncation at half a call and escalation at 1.5 are arguments, not
measurements. The replay harness is the right place to fit them against
historical traffic, and until that happens they should be treated as a starting
position. Proportional session attribution is specified but not yet implemented:
a reported session signal is currently charged to the model named in the report.

### D21 — A streamed stage trace for the console
**Context.** A routing decision that appears all at once, after the fact, is
indistinguishable from a mock. Animating it on a timer would be a lie.
**Decision.** `GatewayPipeline.handle` takes an optional `emit(stage, payload)`
callback; `POST /demo/trace` bridges it to SSE through an `asyncio.Queue` and
the console renders each stage as it lands, with measured `elapsed_ms`.
**Consequences.** What you watch is the real sequence with real timings. The
bridge drains the queue after the pipeline task completes, so no stage is lost
to the race between the final `emit` and the return, and failures arrive as an
`error` stage rather than a dead stream. The serving path (`/v1/chat/completions`)
is unchanged — `emit` defaults to `None` and costs nothing.

### D19 — Runtime credential injection, dev-mode only
**Context.** A demo that cannot be run without editing a dotfile and restarting
is a demo nobody runs. But an HTTP endpoint that accepts secrets is a liability
if it is reachable.
**Decision.** `POST /admin/credentials` hot-swaps a provider client, gated on
`auth_mode == "dev"` (localhost-only by design). Keys are validated against each
vendor's **free** models endpoint, held in memory on the client object, never
logged beyond a mask, and never returned by any endpoint. Writing to `.env` is a
separate explicit opt-in. An *ambient* mode uses whatever the SDK can already
resolve — including an `ant auth login` OAuth profile — so a key need not be
pasted at all.
**Alternatives.** Env-only (safe, high friction); a persisted encrypted store
(more machinery than a prototype warrants, and the encryption key has the same
problem one level down).
**Consequences.** A rejected key is discarded rather than left in place, so a
typo cannot silently replace a working credential. This forced the router and
classifier to read the provider set **live** rather than snapshotting it at
startup — a snapshot would keep routing to a provider that had just been removed.
Under `jwt` mode the endpoints return `403` and credentials must come from the
environment, which is the correct posture for anything deployed.

### D18 — A demo console, served by the gateway itself
**Context.** The interesting behaviour here — a cold write becoming a warm read,
a pilot releasing followers — is invisible in a JSON response.
**Decision.** A single self-contained HTML page at `/`, no CDN and no build
step, driving the real public API rather than a special-cased path.
**Consequences.** What the console shows is what an agent would get, so a demo
cannot drift from reality. `/demo/fanout` and `/demo/reset` exist because
parallel fan-out and cold/warm comparison cannot be triggered through the normal
API alone.

---

## 7. Key data structures

**`CanonicalRequest`** — the vendor-neutral request. Notable: `system` is a
separate list rather than inline messages, because system content is the most
stable part of the prompt and therefore the natural home for the first cache
breakpoint. `sorted_tools()` exists because tool definitions render at position 0
and any ordering churn invalidates the entire prefix on every provider.

**`ModelSpec`** — capability, pricing, and cache economics per model. The two
fields that carry the most weight:

- `min_cacheable_tokens` — **not monotonic across generations**. A 3,000-token
  prefix caches on Opus 5 (512) and Sonnet 5 (1024) and silently does not on
  Haiku 4.5 (4096). A router assuming "cheaper model = cheaper request" gets
  this backwards.
- `rate_limit_pool` — model tiers sit in *separate* upstream pools, so shedding
  from a frontier model to a small one is a genuine capacity lever, not only a
  cost lever.

**`CachePlan`** — which regions get breakpoints, whether the prefix is cacheable
at all, and a human-readable `reason`. The `fingerprint` identifies the cacheable
prefix and is model-scoped and volatile-turn-excluding.

**`RoutingDecision`** — the chosen model plus `reason`, `cache_state`, the cost
estimate, and every `Candidate` considered with its own score. The alternatives
are carried all the way to the response so a decision can be argued with.

**`RequestRecord`** — see [D16](#d16--one-record-per-request-and-offline-replay).

---

## 8. Failure modes

| Failure | Detection | Behaviour |
|---|---|---|
| Tenant over RPM | Fixed-window counter | `429` + `retry-after`, before any work |
| Tenant over budget | Ledger vs. limit | Soft: cap to cheapest tier, flag `degraded`. Hard: `402` |
| No capable model | Empty candidate set | `422` listing why each model was rejected |
| Upstream 429/5xx | Adapter translation | Same-vendor fallback, then cross-vendor; chain recorded; counts against the model's health |
| Model failing repeatedly | 3 consecutive upstream faults | Circuit opens; router stops considering it; one trial after cooldown |
| Upstream 4xx (our fault) | Status classification | Propagates immediately — no model will fix it |
| Policy refusal | `stop_reason == "refusal"` | Typed `403` with category; **never** retried |
| Provider unreachable | Connection error | `503` after the chain is exhausted |
| Pilot dies mid-flight | Lock TTL | Followers time out and proceed cold; lock frees for a new pilot |
| Prefix below cache minimum | Planner check | Markers omitted, reason recorded; no silent no-op |
| Classifier fails | Exception in L3 | Falls through to `unknown` (middle tier). Never fails the request |
| Redis unavailable at boot | Connection attempt | Falls back to in-memory with a loud warning |
| Record write fails | `OSError` | Swallowed. Telemetry must never fail a request |

---

## 9. Concurrency and consistency

- **The pilot lock is advisory, not a correctness guarantee.** Losing it costs
  money, not correctness, so a TTL-bounded `SETNX` is the right weight of
  mechanism — no fencing tokens, no consensus.
- **Budget counters are eventually consistent.** `INCRBYFLOAT` is atomic, but a
  burst of concurrent requests can each pass a check that their combined spend
  would fail. Budgets are a cost-control mechanism, not a hard financial limit;
  treating them as the latter would require reserving budget before the call and
  refunding after, which doubles the state round trips for a bound nobody needs
  that precisely.
- **Rate limiting is a fixed window,** so a burst can straddle a boundary and
  briefly reach 2× the nominal rate. Adequate here; swap for a token bucket if
  burst shaping matters.
- **Session stickiness is last-write-wins.** Two concurrent requests opening the
  same session may pick different models and the last to finish wins. The pilot
  makes this rare, and the cost of being wrong is one extra cache write.
- **The ledger is written after the response,** so a crash between the upstream
  call and the ledger write under-counts spend. Durable pre-write would need a
  two-phase commit for a prototype-grade guarantee.

---

## 10. Extension points

**Add a provider.** Implement the `Provider` protocol (`invoke`, `stream`,
`classify`, `count_tokens`), add `ModelSpec` entries to `CATALOG`, and register
it in `ProviderRegistry.__init__`. The router, budget, and observability layers
need no changes — they operate on `ModelSpec`, not on vendors.

**Add an intent.** Add an `IntentPolicy` to `INTENT_POLICY` and, optionally, a
regex to `_KEYWORDS` in `routing/intent.py`. The classifier's L3 schema is
generated from the policy table's keys, so a new intent becomes available to the
small-model classifier automatically.

**Enable the embedding classifier (L2).** Implement the `Embedder` protocol and
fill in `IntentClassifier._classify_by_embedding` with a nearest-neighbour lookup
over labelled exemplars. It ships as a null implementation that abstains — a stub
returning confident garbage would be worse than one that declines to answer.

**Swap the state backend.** Implement `StateStore` and return it from
`build_store`. Nothing above it knows the difference.

**Replace the telemetry sink.** `RecordSink.write` is the seam. Field names were
chosen to map onto OTel span attributes.

**Add a routing policy variant.** Add a `PolicyVariant` to
`replay/harness.py::DEFAULT_VARIANTS` to score it against historical traffic
before it touches production.

---

## 11. Security posture

- **Provider credentials never leave the gateway.** Agents hold gateway
  credentials only.
- **Tenant isolation** is by `tenant_id` claim: budgets, rate limits, and the
  ledger are all keyed on it. There is no cross-tenant data path — the gateway
  holds no conversation state between requests.
- **Model pinning is scope-gated** (`model:pin`), so a caller cannot escape a
  cost ceiling by naming an expensive model directly.
- **`dev` auth mode is unsafe by construction** — it trusts request headers for
  identity. It warns on every start and must not be exposed beyond localhost.
- **Runtime credential entry is gated on dev mode** and returns `403` otherwise.
  Keys are memory-resident, mask-only in every response and log line, and
  discarded if validation fails. Persisting to `.env` is an explicit opt-in, and
  `.env` is gitignored.
- **Refusals are not laundered.** A policy refusal is never retried against
  another model, which would otherwise turn the gateway into a mechanism for
  shopping a declined request around vendors.
- **The record contains no prompt or completion text** — only metadata, token
  counts, and costs. Sending prompt bodies to telemetry would be a data-residency
  decision, not a logging one, so it is deliberately not the default.
- **Not implemented:** per-tenant data-residency routing constraints. If a tenant
  may not be served by a given vendor, that is a *hard* constraint that must sit
  above the cost optimiser. There is no hook for it yet — see
  [§13](#13-open-questions).

---

## 12. Performance characteristics

Gateway-added latency on the hot path, excluding the model call:

| Step | Cost |
|---|---|
| Auth (dev) / JWT decode | negligible / ~0.1 ms |
| Rate limit | 1 state round trip |
| Canonicalise + estimate | pure CPU, O(prompt length) |
| Intent L0/L1 | pure CPU, regex over the last user turn |
| Intent L3 (when it fires) | a full small-model call — the reason it is last |
| Route (×2) | pure CPU, O(catalog) — six entries |
| Budget check | 2 state round trips |
| Cache plan | one SHA-256 over the prefix |
| Pilot acquire | 1–2 state round trips, or a wait on a cold prefix |
| Ledger + record | 2 state round trips + one buffered append |

So roughly **6–8 state round trips** and a handful of milliseconds of CPU against
a model call measured in seconds — immaterial except for the pilot wait, which is
a deliberate latency-for-cost trade on cold-prefix followers only.

**The dominant cost lever is not gateway overhead; it is cache hit rate.** That
is why the record schema treats cache tokens as a headline metric rather than a
footnote.

---

## 13. Open questions

Not resolved. The code has defensible placeholders; placeholders are not answers.

1. **The real intent taxonomy.** `routing/policy.py` ships a generic 16-intent
   table. A routing table is only as good as its intent classes, and the right
   classes are the ones your agents actually emit. The first week of
   `records.jsonl` should tell you what they are.
2. **What "token availability for the enterprise" means concretely.** It is four
   mechanisms that get conflated: per-tenant dollar budget, upstream rate-limit
   headroom, quota fairness across teams, and cost attribution for chargeback.
   Three are built. **Fairness is not** — it needs a scheduler, not a counter,
   and it is the wrong thing to prototype first.
3. **Deployment shape.** Single process or scaled? This is the difference between
   the in-memory fallback being a convenience and being a correctness bug.
4. **Data residency / PII constraints per tenant.** A hard routing constraint
   with no hook today — see [§11](#11-security-posture).
5. **Do the agents have a refusal branch?** See [D14](#d14--refusals-are-a-first-class-response-type).
6. **How is answer quality measured?** The gateway optimises cost and can prove
   cost savings via replay. It cannot tell you whether the cheaper model was good
   enough. Without a quality signal, every routing "win" is unfalsifiable.

---

## 14. Known gaps

Named rather than hidden.

- **Embedding classifier (L2) abstains** — null implementation by design.
- **Semantic response caching is off and unimplemented** beyond the flag.
  Returning a near-miss answer to an agent mid-workflow is a correctness bug, not
  a saving. It is a different thing from prompt caching and the two are
  constantly conflated.
- **OpenAI model ids and prices are placeholders.** Verify before trusting the
  ledger for chargeback. Anthropic figures are current first-party rates.
- **Streaming skips the fallback chain** and writes the ledger only when the
  stream closes.
- **`mark_warm` fires on completion, not first token.** The provider cache is
  readable from the first token, so the streaming path is where this becomes
  worth tightening — the obvious next optimisation.
- **Upstream pool pressure is measured but not routed on.** Plumbing exists
  (`RateLimiter.pool_pressure`); the router does not read it. Model *health* is
  routed on — see D20 — but pool saturation is a separate, still-unused signal.
- **Rate limiting is a fixed window** — see [§9](#9-concurrency-and-consistency).
- **No idempotency keys.** Multi-agent retries will double-charge.
- **Replay assumes output length is model-independent.** It is not; treat its
  savings as an upper bound.
