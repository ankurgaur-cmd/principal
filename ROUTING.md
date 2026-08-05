# The router — what the decision is based on

Source: `src/aigateway/routing/router.py`. Tests: `tests/test_router.py`.
Upstream of this: [CLASSIFICATION.md](CLASSIFICATION.md). Rationale and
alternatives: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## The one-line answer

> **The cheapest *capable* model, where "cost" prices the cache transition and
> observed quality, and "capable" is a hard gate that price cannot buy past.**

Everything below is that sentence in detail. Each decision point is marked
**`R1`…`R12`** and is referenceable from code review or an incident.

---

## The decision order

Strictly sequential. An earlier step can end the decision; a later one never
undoes an earlier one.

| # | Step | Can it end the decision? |
|---|---|---|
| **R1** | Pin — caller named a model | **yes**, immediately |
| **R2** | Tier floor from intent | no, sets the bar |
| **R3** | Escalate on signals | no, raises the bar |
| **R4** | Caller ceiling (`max_tier`) | no, lowers the bar |
| **R5** | Budget ceiling | no, lowers the bar, marks `degraded` |
| **R6** | Exploration coin-flip | no, disables R10 for this request |
| **R7** | Hard gates — build the candidate set | **yes**, by exhaustion (error) |
| **R8** | Cache-transition pricing | no, produces the score |
| **R9** | Vendor weight | no, adjusts the score |
| **R10** | Quality multiplier | no, adjusts the score |
| **R11** | Session stickiness | **yes**, if a warm cache exists to protect |
| **R12** | Cheapest wins | **yes**, always |

---

## R1 — A pin bypasses everything

`x_gateway.pin_model` returns that model with `pinned: true` and no scoring.
An unknown key raises `NoCapableModel` rather than falling back silently.

⚠️ **A pin skips the hard gates too.** Pin a model without tool support for a
tool request and the vendor rejects it, not the router. That is deliberate — a
pin is an instruction, not a suggestion — but it means a pin can fail in ways
routed traffic cannot.

Quality penalties are **not** applied (`apply_quality=False`). Reputation exists
to change *choices*; there is no choice here, and applying it would only distort
the reported estimate.

---

## R2–R5 — Establishing the tier floor

`min_tier` is a **floor, not a target**. The router still picks the cheapest
model at or above it. Widening the catalog is how you get cheaper — not editing
the policy table downward.

**R2 — Policy floor.** From the intent (see
[CLASSIFICATION.md §8](CLASSIFICATION.md#8-what-the-label-is-used-for)).

**R3 — Escalation on signals.** If the intent policy sets `escalate_on_tools`
and the request carries tools, the floor rises one tier (capped at `heavy`).
Rationale: tool orchestration fails in ways that are expensive to detect.

**R4 — Caller ceiling.** `x_gateway.max_tier` lowers the floor if it is below it.

**R5 — Budget ceiling.** A tenant over its limit gets a ceiling from the budget
guard. This **wins over R4** and sets `degraded: true`.

📌 **Order matters: the budget ceiling is applied last so it can pull the floor
below what policy asked for.** That is the whole point of the degraded path —
serve something cheap rather than nothing. Every other lever can only be
overruled by this one.

---

## R6 — Exploration is decided once per request

```python
exploring = reputation.should_explore()      # default 5%
```

On an exploring request, R10 is skipped for **every** candidate.

📌 **Decided once for the whole request, never per candidate.** Applying the
penalty to some models and not others in the same comparison is not a comparison
at all. This is also what stops the reputation loop becoming a ratchet: a model
penalised out of contention gets no traffic, so it gets no new observations, so
it can never recover.

---

## R7 — The hard gates

Checked per model, in this order. **The first failure disqualifies — price is
never consulted.** A model that cannot do the job is not a cheap option.

| # | Gate | Exclusion kind |
|---|---|---|
| 1 | Tier below the floor (R2–R5) | `tier` |
| 2 | Operator switch off (model or vendor) | `switched_off` |
| 3 | Provider has no credentials | `no_credentials` |
| 4 | Circuit breaker open | `unhealthy` |
| 5 | Request has tools, model has no tool support | `capability` |
| 6 | Request has a schema, model has no structured outputs | `capability` |
| 7 | `prompt + max_tokens` exceeds the context window | `capacity` |
| 8 | `max_tokens` exceeds the model's output ceiling | `capacity` |

📌 **Operator switches are checked before health.** A human decision outranks an
observation. A switch never self-heals; a breaker does.

📌 **Switches are checked even on a dry run.** A preview that ignored your
switches would not be previewing the routing you actually configured.

📌 **A tier exclusion is a policy statement about the task, not a judgement on
the model.** The exclusion carries both the model's tier and the required tier so
the explanation can say *"light-tier model; this task was judged to need heavy"*
rather than implying the model is incompetent.

**Empty candidate set → `NoCapableModel`**, listing every rejection reason. The
router never silently downgrades below the floor to find something servable.

---

## R8 — The cost function: the cache transition is the point

This is the heart of the router. **Prompt caches are model-scoped**, so switching
models mid-session discards a warm prefix and pays to rebuild it. Scoring on the
sticker rate alone makes the "cost-optimising" choice routinely *more* expensive.

```
cost = input_cost + expected_output × price_out

input_cost, by cache state:
  uncached     (prefix + volatile) × price_in
  warm_read    prefix × price_in × 0.1     + volatile × price_in
  cold_write   prefix × price_in × 1.25    + volatile × price_in     (5m TTL)
                                     2.0                             (1h TTL)
```

📌 **Cacheability is per model and non-monotonic.** `min_cacheable_tokens` is 512
on Opus 5, 1,024 on most models, 4,096 on Haiku 4.5. The same prefix is a warm
read on one model and uncacheable on another — which is why this is computed per
candidate, not once.

📌 **Long-context rates are stepped, not flat.** `rates_for(total_tokens)` picks
the tier that applies at this request's size — OpenAI roughly doubles above 272K;
Anthropic has no such premium. Using the headline rate on a large request
under-prices it ~2×, which is exactly backwards for the requests where the bill
is biggest.

📌 **Expected output is not `max_tokens`.** `max_tokens` is a ceiling, not a
forecast. Scoring against the ceiling makes every model look output-dominated and
flattens the comparison. The router uses an effort-based expectation, capped by
`max_tokens`:

| effort | low | medium | high | xhigh | max |
|---|---|---|---|---|---|
| expected output | 600 | 1,200 | 2,500 | 5,000 | 8,000 |

---

## R9 — Vendor weight

```python
score = raw_cost × vendor_weights.get(provider, 1.0)
```

A deliberate thumb on the scale — contractual commitments, data residency, a
vendor you are trying to shift traffic toward.

📌 **Applied to the score, never to the ledger.** What you are *billed* stays the
real number. `Candidate.raw_cost_usd` keeps the unadjusted figure alongside, so
"what we pay" and "how we rank" are never confused.

---

## R10 — Observed quality, priced as expected cost

```python
score = ... × reputation.multiplier(model, intent)

multiplier = (1 + mean_extra_effort) / success_rate
```

📌 **This is a price, not a weight.** A model that succeeds a fraction `s` of the
time needs `1/s` attempts, so it genuinely costs `cost/s`. That makes the
adjustment comparable to money rather than a tunable fudge factor. Effort
composes on top — see [ARCHITECTURE.md D26](ARCHITECTURE.md).

📌 **Reputation is per `(model, intent)`, never per model.** A small model can be
excellent at classification and hopeless at code review; one global score is wrong
for both.

📌 **No evidence means no adjustment.** Below `quality_min_samples` (5) the
multiplier is exactly `1.0`. Penalising on one bad response would be superstition
and would make routing depend on arrival order.

Capped at `quality_max_penalty` (4.0) so a bad patch cannot exile a model forever.

---

## R11 — Session stickiness, escalation-only

If the session is warm on a model that still clears the floor, **keep it** — even
if something cheaper now exists.

The justification is arithmetic, on a 20,000-token prefix over N turns:

| | cost |
|---|---|
| Pinned to one model | `1.25 + 0.1N` |
| Rerouted every turn | `1.25N` |

At N=10 that is **2.25 vs 12.5** — the naive per-request optimum is 5.5× more
expensive, slower, and less coherent.

**All four conditions must hold:**

1. `escalate_only` is on
2. the session has a warm model
3. that model still clears the current floor — **escalation is allowed, de-escalation is not**
4. **the prefix is actually cacheable on that model**

⚠️ **Condition 4 was a real bug.** Stickiness held even when the prefix was too
short to cache, so there was no cache to protect — and every cheap follow-up
("classify this", "translate that") stayed pinned to whatever heavyweight the
session happened to open with. Reported as *"all requests are going to gpt-5
despite an easy query"*. If there is no cache, there is nothing to defend, and
the router falls through to price.

📌 **De-escalation is never sticky.** Coming *down* a tier mid-session almost
never recovers the write you just paid for.

Session state is `session:{id}:model`, TTL `session_ttl_seconds` (300s) — tracking
the prompt-cache window, because once the provider cache expires stickiness buys
nothing.

---

## R12 — Cheapest wins

```python
candidates.sort(key=lambda c: (c.cost_usd, c.model.tier))
```

Ties break toward the **lower tier** — same price, less capability held in
reserve, so the bigger model stays free for work that needs it.

When the winner differs from the warm model, the reason records both prices so
the switch is auditable:

```
switching off warm claude-opus-5 ($0.07149) to gpt-5 ($0.02681)
— cache write already priced in
```

---

## What the router never does

- ❌ Route below the tier floor to find something servable — it errors instead
- ❌ Let price overrule a hard gate
- ❌ Apply vendor or quality adjustments to what you are billed
- ❌ De-escalate a warm session to save money
- ❌ Snapshot the provider list — it is read per request, because credentials can
  arrive at runtime
- ❌ Make a decision it cannot explain — every path writes a `reason`, and every
  exclusion carries its kind

---

## Every output of a decision

```python
RoutingDecision(
    model, tier, effort,
    reason,                  # human-readable, every branch appends to it
    cache_state,             # uncached | warm_read | cold_write
    cache_plan,
    estimated_cost_usd,      # the adjusted score
    prefix_tokens, volatile_tokens,
    considered=[Candidate],  # everyone who qualified, with raw vs adjusted cost
    excluded=[{...}],        # everyone who did not, with kind and required_tier
    degraded,                # budget pulled the floor down
    escalated_from,          # the warm model we left, if any
    sticky, pinned,
    required_tier, intent,
)
```

`routing/explain.py` turns this into the plain-language account the console
shows. Inspect any decision without spending anything:

```bash
curl -s localhost:8000/admin/route/preview -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Review this diff"}]}' | jq
```

Add `"include_unavailable": true` to score models you hold no key for.

---

## Knobs

| Setting | Default | Effect |
|---|---|---|
| `GATEWAY_CACHE_AWARE_ROUTING` | `true` | R8 prices the cache transition |
| `GATEWAY_ESCALATE_ONLY` | `true` | R11 stickiness |
| `GATEWAY_SESSION_TTL_SECONDS` | `300` | how long a session stays warm |
| `GATEWAY_CACHE_TTL` | `5m` | write multiplier: 1.25× (5m) or 2× (1h) |
| `GATEWAY_VENDOR_WEIGHTS` | `{}` | R9, per provider |
| `GATEWAY_QUALITY_ROUTING_ENABLED` | `true` | R10 |
| `GATEWAY_QUALITY_MIN_SAMPLES` | `5` | evidence before R10 engages |
| `GATEWAY_QUALITY_MAX_PENALTY` | `4.0` | ceiling on R10 |
| `GATEWAY_QUALITY_EXPLORATION_RATE` | `0.05` | R6 |

---

## Known limits

**Cost is an estimate, priced before the call.** Actual usage is priced
afterwards and recorded separately (`estimated_cost_usd` vs `actual_cost_usd`).
They diverge when output length differs from the effort expectation. Watch the
gap — a systematic one means the `_EXPECTED_OUTPUT` table needs refitting.

**Warmth is assumed, not verified.** The router believes its own session record.
If the provider evicted the entry early, it priced a warm read and pays a cold
write. The quality check reports `cache_missed` when this happens; the router
does not currently learn from it.

**One warm model per session.** A session that legitimately alternates between
two models is not represented.

**Tier is a coarse proxy for capability.** Three buckets cannot express "good at
SQL, weak at long-context recall". Reputation (R10) patches this empirically per
intent, but only after enough traffic.

**Vendor weights are unvalidated.** A weight of `0.01` will route everything to
one vendor and the router will not warn you.

**The catalog is dated, not live.** Prices carry `price_checked` /
`price_verified`; `catalog_warnings()` names anything unverified. A stale price
does not produce a rounding error — it silently routes a whole tier to the wrong
vendor while looking deliberate.
