# Test playbook

Copy-paste scenarios for checking the gateway does what you think it does. Each
one states **what it proves**, the **exact input**, and **what you should see** —
plus what it means when you see something else, because "that looks wrong" and
"that is wrong" are different.

Every prompt below was run against this gateway and the expected results are the
observed ones, not guesses. Where a result is surprising it is flagged.

**Setup.** Console at <http://localhost:8000/>. Most scenarios are free —
`/admin/route/preview` classifies and scores without calling a model. Scenarios
that spend money are marked **💰**.

---

## Contents

| # | Scenario | Proves | Cost |
|---|---|---|---|
| [1](#1-intent-drives-the-tier) | Intent drives the tier | routing responds to what you asked | free |
| [2](#2-the-shape-of-a-request-outranks-its-words) | Shape beats prose | schema/tools override wording | free |
| [3](#3-your-declared-intent-is-checked) | Declared intent is checked | a wrong label cannot silently downgrade | free |
| [4](#4-ceilings-and-pins) | Ceilings and pins | you can overrule the router | free |
| [5](#5-caching-cold-then-warm) | Caching works | the saving is real | 💰 |
| [6](#6-the-cache-minimum-is-not-uniform) | Non-monotonic minimums | why a prefix caches on one model and not another | free |
| [7](#7-fan-out-and-the-cache-pilot) | Fan-out + pilot | N agents share one cache | 💰💰 |
| [8](#8-fan-in-the-tier-split) | Fan-in tier split | workers cheap, synthesis expensive | 💰💰 |
| [9](#9-switch-a-model-off) | Availability gates | off means off, everywhere | free |
| [10](#10-take-the-whole-fleet-down) | Alerts | red flag, beep, plain-language message | free |
| [11](#11-starve-the-answer) | Reasoning budget | the empty-answer trap and the warning | 💰 |
| [12](#12-quality-checks) | Quality checks | a cheap model being caught out | 💰 |
| [13](#13-latency-banding) | Latency baselines | colours mean something | 💰 |
| [14](#14-budget-degradation) | Budget ceiling | degrade rather than fail | free |
| [15](#15-effort-adjusted-routing) | Effort loop | reported effort changes routing | free |

---

## 1. Intent drives the tier

**Proves:** the router reads what you asked for and picks a floor from it. This
is the core claim of the whole gateway.

**Do:** paste each into the console's **Prompt** box (tab 1) and hit Send — or
run them free through preview:

```bash
curl -s localhost:8000/admin/route/preview -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"PROMPT HERE"}]}' \
  | jq '{intent:.intent.resolved, source:.intent.source, tier:.decision.tier, model:.decision.model}'
```

**Expect** — all observed, in order of increasing tier:

| Prompt | intent | tier | model |
|---|---|---|---|
| `Classify the sentiment of this support ticket as positive, neutral or negative.` | classify | light | gpt-5-nano |
| `Extract the invoice number, date and total from this email.` | extract | light | gpt-5-nano |
| `Translate this paragraph into French.` | translate | light | gpt-5-nano |
| `Summarise the key points of this thread.` | summarize | light | gpt-5-nano |
| `What does this error message mean?` | chat | standard | gpt-5-mini |
| `Break down the migration into steps.` | plan | standard | gpt-5-mini |
| `Refactor this function and write the tests.` | code_write | standard | gpt-5-mini |
| `Review this authentication middleware for security issues.` | code_review | **heavy** | gpt-5 |
| `There is a race condition and a deadlock in this stack trace.` | hard_debug | **heavy** | gpt-5 |
| `What are the trade-offs of event sourcing here? System design question.` | architecture | **heavy** | gpt-5 |

**If the model differs from the table:** that is fine — the tier is the claim,
not the model. The router picks the cheapest model *at or above* the floor, so
switching a model off or adding a cheaper one legitimately changes the answer.
**If the tier is wrong**, that is the interesting case: read
`.intent.rationale` to see what evidence it used.

**Known gap worth knowing about:** `Break this migration down into steps` (with
a noun in the middle) classifies as `chat`, not `plan` — the pattern is
`break (this )?down` and the noun defeats it. Keyword coverage is English and
literal by design; see [CLASSIFICATION.md §11](CLASSIFICATION.md#11-known-limits).

---

## 2. The shape of a request outranks its words

**Proves:** structure is stronger evidence than wording. Someone attaching a
JSON schema is doing extraction whatever the sentence says.

**Do & expect:**

| Input | Result | Why |
|---|---|---|
| `"Give me the fields."` + a `response_format` json_schema | `extract`, light | schema on a short prompt is near-definitional |
| `"Summarise this."` + **6 tools** | `tool_orchestration`, standard | many tools means orchestration, whatever the prose says |
| `"Implement this function."` + **1 tool** | `code_write`, **heavy** | ⚠️ tool escalation — see below |

```bash
curl -s localhost:8000/admin/route/preview -H 'content-type: application/json' -d '{
  "model":"auto","messages":[{"role":"user","content":"Implement this function."}],
  "tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}}]
}' | jq '{intent:.intent.resolved, tier:.decision.tier}'
```

⚠️ **The third row is the one to notice.** `code_write` has a *standard* floor,
but its policy sets `escalate_on_tools`, so one tool pushes it to **heavy**. That
is deliberate — tool orchestration fails in ways that are expensive to detect —
but it means adding a single tool definition can triple your cost per request.
If that surprises you, that is the point of testing it.

---

## 3. Your declared intent is checked

**Proves:** `x_gateway.intent` is trusted, but a label that plainly understates
the work does not silently get you a small model.

**Do & expect:**

| Input | Result | Meaning |
|---|---|---|
| `"Review this diff."` + `intent: classify` | `classify`, light, source **`declared`** | accepted — weak contrary evidence does not overrule you |
| `"There is a deadlock and a race condition in this stack trace."` + `intent: classify` | `hard_debug`, heavy, source **`declared-overridden`** | overridden — strong evidence, and only ever upward |
| `"Translate this."` + `intent: architecture` | `architecture`, heavy, source **`declared`** | left alone — over-declaring is you spending your own money |

```bash
curl -s localhost:8000/admin/route/preview -H 'content-type: application/json' -d '{
  "model":"auto",
  "messages":[{"role":"user","content":"There is a deadlock and a race condition in this stack trace."}],
  "x_gateway":{"intent":"classify"}
}' | jq '{intent:.intent.resolved, source:.intent.source, why:.intent.rationale}'
```

The override tells you what it did and how to insist:

> you declared 'classify' (light tier), but the request looks like hard_debug
> needing heavy — … Routed on the evidence; pin_model or max_tier if you meant it.

**The asymmetry is deliberate.** The check protects quality, not spend — see
[CLASSIFICATION.md §3](CLASSIFICATION.md#3-l0--the-declared-hint-trusted-but-checked).

---

## 4. Ceilings and pins

**Proves:** you can always overrule the router, and it tells you it was overruled.

```bash
curl -s localhost:8000/admin/route/preview -H 'content-type: application/json' -d '{
  "model":"auto","messages":[{"role":"user","content":"Review this diff for race conditions."}],
  "x_gateway":{"max_tier":"light"}
}' | jq '{intent:.intent.resolved, required:.decision.required_tier, got:.decision.tier, model:.decision.model}'
```

| Input | Result |
|---|---|
| `"Review this diff for race conditions."` + `max_tier: light` | `hard_debug` intent, but **light** tier, gpt-5-nano |
| any prompt + `pin_model: claude-opus-5` | that model, `pinned: true`, no scoring |

Note the first row: the intent is still `hard_debug`. The ceiling changes *which
models are eligible*, not what the gateway thinks you asked for — so the record
still shows you ran heavy work on a light model, which is what you want when
you go looking for a bad answer later.

---

## 5. Caching, cold then warm 💰

**Proves:** the cache saves real money, and the console tells you when it did not.

**Do:**
1. Tab 1 → click preset **Big cached prefix** (loads a ~1,400-token review standard)
2. Set **Session id** to something new, e.g. `cache-test-1`
3. **Send**
4. **Send again**, unchanged

**Expect:**

| | cache state | Cached tok | Note |
|---|---|---|---|
| Turn 1 | `↑ Building cache` | 0 | *"Building the cache. First request on this prefix…"* |
| Turn 2 | `✓ Reusing cache` | ~1,024+ | *"…read from cache — billed at about a tenth…, saving $X"* |

Observed on `gpt-5-mini`: 1,024 tokens read, `$0.00023` saved per turn.
On `claude-sonnet-5`: 1,650 tokens read, `$0.00297` saved.

**Reset session** puts it back to cold so you can repeat.

**If turn 2 still shows 0**, the console now tells you which of three things it
is rather than leaving you guessing:
- *"Nothing to cache"* — the prefix is below this model's minimum (see [6](#6-the-cache-minimum-is-not-uniform))
- *"Building the cache"* — you changed something, so this is a *new* prefix
- *"Expected a cache hit and got none"* — the only one that is actually a fault

---

## 6. The cache minimum is not uniform

**Proves:** the same prefix caches on one model and not another. This is the
single most common cause of "caching isn't working".

**Do:** watch the `prefix ≈ N tokens` hint under **Shared context** as you paste.

**Expect:**

| Prefix size | Caches on |
|---|---|
| < 512 | nothing |
| 512–1,023 | claude-opus-5 only |
| 1,024–4,095 | **11 of 12 models** — not claude-haiku-4-5 |
| ≥ 4,096 | everything |

The **Big cached prefix** preset is deliberately ~1,400 tokens so it lands in the
third band, where you can see two models treat identical input differently.

Verified: at a 1,204-token prefix, `claude-haiku-4-5` returns 0 cached tokens
and that is **correct behaviour**, not a bug — its floor is 4,096.

---

## 7. Fan-out and the cache pilot 💰💰

**Proves:** N parallel sub-agents share one cache write instead of paying N.

**Do:** tab 2 → **Fan-out**. It ships with a real capture diff and six
per-agent questions about it. Run with **Cache pilot: enabled**, then again with
**disabled**, and compare `total_cache_write_tokens`.

**Expect:** every agent's answer shown expanded, next to the question it
answered, with its model, cost, latency and quality verdict. Six visibly
*different* answers — they are asking different things about the same diff.

⚠️ **Two things that look like bugs and are not:**

- **More than one cold write with the pilot on** → the agents landed on
  different models. A prompt cache belongs to one model, so two models genuinely
  need two writes. The panel names the split when it happens; pin **Intent hint**
  to hold them all on one model.
- **`pilot_role: timeout` on followers** → known limitation. The pilot releases
  followers when its response *completes*, and generation takes 18–58s against a
  4s wait, so followers time out and proceed cold. On the unary path the pilot is
  currently costing 4s and buying nothing. Fix is designed, not shipped.

---

## 8. Fan-in — the tier split 💰💰

**Proves:** the router's value on a real multi-agent shape — narrow worker jobs
land on small models while the synthesis step, which must hold everything at
once, gets a big one.

**Do:** tab 2 → **Fan-in**. Run it.

**Expect:** `tier_split` showing workers at a lower tier than the synthesiser.
If they all come out the same tier, your subtasks are too similar in difficulty
to the synthesis — that is a property of your prompts, not a routing failure.

---

## 9. Switch a model off

**Proves:** an operator switch is honoured everywhere — routing, fallback, and
even over an explicit pin.

**Do:** tab 4 → click a model card to switch it off. Then re-run scenario 1's
`code_review` prompt.

**Expect** (observed, switching off each winner in turn):

```
all on           -> gpt-5
switch off gpt-5 -> gpt-5.4
switch off that  -> claude-opus-5
switch off that  -> gpt-5.6-sol
```

The switched-off model appears under **Not considered**, grouped as
*"Switched off by you"* — distinct from a tier exclusion or a missing key.

**Also test:** pin a model, then switch it off. You get a **503** telling you the
operator switch overrides the pin. That is deliberate: a pin is a caller
instruction, a switch is an operator decision about what this deployment may
talk to.

---

## 10. Take the whole fleet down

**Proves:** the gateway raises a system alert, flags for support, and tells the
end user something useful.

**Do:** tab 4 → switch **both vendors** off. Send any request.

**Expect:**
- A **red pulsing bulb and flag** at the top of the console
- A **beep** (once, on the new condition — not on every poll)
- Tab title becomes `⚑ Moon — service unavailable`
- **503** with `retry-after`, not a 422

Three lines, for three different readers:

```
TITLE : Paused by an operator
USER  : The service is paused. Please try again shortly.
FIX   : Turn a model back on in the console, or POST /admin/switchboard/reset.
```

**Note the amber dot, not red.** Switching everything off does not set
`needs_support` — it is a deliberate action, not an incident, and colouring it
like one is how a flag stops meaning anything. Remove your API keys instead and
the dot goes red with `needs_support: true`: that one nobody chose.

**Recovery:** turn a vendor back on; the flag clears immediately.

```bash
curl -s localhost:8000/admin/alerts | jq '{ok, needs_support, severity}'
```

---

## 11. Starve the answer 💰

**Proves:** the reasoning-budget trap, and that the gateway warns before spending.

**Do:** tab 1 → **Hard task** preset → set **Max tokens** to `1200` → Send.

**Expect:**
- A warning **before** the call: *"max_tokens=1200 is below the 3000 tokens a
  reasoning model typically needs…"*
- An **empty answer**, `finish_reason: length`
- Quality verdict **fail**, check `reasoning_starved`

**Then:** clear Max tokens (leave it blank) and Send again. The gateway sizes the
budget from the intent — 8,000 for `code_review` — and you get a real answer.

Measured across budgets on one heavy prompt:

| max_tokens | wall clock | result |
|---|---|---|
| 1,200 | 18.5s | **empty** |
| 4,000 | 26.9s | 1,423 tokens |
| 8,000 | 58.0s | 4,729 tokens |

This is also the latency lever — see [README § Making it faster](README.md#making-it-faster).

---

## 12. Quality checks 💰

**Proves:** the gateway can tell you the cheap model was *not* good enough,
which is what makes any claimed saving falsifiable.

**Do:** force a small model onto hard work — `pin_model: gpt-5-nano` with the
`code_review` prompt from scenario 1, `max_tokens: 8000`.

**Expect:** verdict `pass`, `warn` or `fail` in **Request & response**. Watch for:
- `thin_for_intent` — a very short answer for a demanding task (a warning, not a failure)
- `truncated` / `reasoning_starved` — budget problems
- `cache_missed` — the router priced a warm read and got none

A `fail` is recorded as evidence the routing was wrong and feeds back into
reputation, which makes that model more expensive for that intent next time.

---

## 13. Latency banding 💰

**Proves:** the pipeline colours mean something checkable.

**Do:** send the same request **8–10 times** on different session ids.

**Expect:** stages start grey (*learning*), then take colour once each segment has
8 samples. Hover any stage for its mean, σ, p50/p95 and sample count.

```bash
curl -s localhost:8000/admin/baselines | jq '.segments[] | {key, samples, mean_ms, stddev_ms, confident}'
```

**Two things that are deliberate, not bugs:**
- Gateway stages stay `normal` even at 3σ — a band also needs a **25ms absolute**
  gap. Without it, 30 microseconds of jitter lights the pipeline red.
- The upstream call is banded per **model *and* cache state**. A cold write is
  legitimately slower than a warm read; pooling them would paint every cold
  request red.

---

## 14. Budget degradation

**Proves:** a tenant over budget gets a cheaper model rather than an error.

**Do:** set a tiny limit, spend past it, then preview.

```bash
curl -s -X POST localhost:8000/admin/limits/demo \
  -H 'content-type: application/json' -d '{"daily_usd": 0.001, "rpm": 600}'

# spend something so the tenant is actually over
curl -s -o /dev/null localhost:8000/demo/trace -H 'content-type: application/json' \
  -H 'x-tenant-id: demo' -d '{"model":"auto","messages":[{"role":"user","content":"hi"}],"max_tokens":600}'

curl -s localhost:8000/admin/route/preview -H 'content-type: application/json' -H 'x-tenant-id: demo' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Review this authentication middleware for security issues."}]}' \
  | jq '{degraded:.decision.degraded, required:.decision.required_tier, got:.decision.tier, model:.decision.model}'
```

**Expect** (observed):

```json
{ "degraded": true, "required": "light", "got": "light", "model": "gpt-5-nano" }
```

and a reason that names it:

```
intent=code_review floor=heavy; DEGRADED to light by budget; cheapest capable: gpt-5-nano
```

**Why this one matters:** the budget ceiling is the *only* lever that can pull the
floor below what policy asked for — everything else can only raise it. It is what
makes the gateway serve something cheap rather than nothing.

**Restore your limit when done:**

```bash
curl -s -X POST localhost:8000/admin/limits/demo \
  -H 'content-type: application/json' -d '{"daily_usd": 50, "rpm": 600}'
```

---

## 15. Effort-adjusted routing

**Proves:** effort you report changes what the router picks — a model that
"works" but takes four turns is not cheap.

**Do:** report effort against a model five times, then look at its multiplier.

```bash
for i in $(seq 1 5); do
  curl -s -X POST localhost:8000/admin/effort -H 'content-type: application/json' \
    -d '{"model":"gpt-5-nano","intent":"code_review","turns_to_goal":5,
         "user_reasked":true,"manual_escalation":true}' > /dev/null
done
curl -s localhost:8000/admin/effort -H 'content-type: application/json' \
  -d '{"model":"gpt-5-nano","intent":"code_review","user_reasked":true}' \
  | jq '{effort:.scored.extra_effort, contributions:[.scored.contributions[].signal], multiplier_now}'
```

**Expect:** `multiplier_now` climbing above 1.0 — observed **3.5×** after five
reports. Every score is itemised; an effort penalty you cannot enumerate is
indistinguishable from a grudge.

**See what is being measured and what is still waiting for data:**

```bash
curl -s localhost:8000/admin/effort/signals | jq '.signals[] | {name, weight, status}'
```

Six signals are measured from the request itself; five need your orchestrator to
report them.

---

## Quick reference — everything free

```bash
curl -s localhost:8000/health                      | jq        # switches, providers
curl -s localhost:8000/admin/policy                | jq        # intent -> tier -> budget
curl -s localhost:8000/admin/pool                  | jq        # health, breakers, latency
curl -s localhost:8000/admin/switchboard           | jq        # what is switched off
curl -s localhost:8000/admin/alerts                | jq        # system alerts
curl -s localhost:8000/admin/baselines             | jq        # latency baselines
curl -s localhost:8000/admin/cache/effectiveness   | jq        # does each model deliver its cache
curl -s localhost:8000/admin/reputation            | jq        # quality + effort per model/intent
curl -s localhost:8000/admin/fleet                 | jq        # where traffic actually goes
curl -s localhost:8000/admin/usage/demo            | jq        # spend
```

## Resetting between runs

```bash
curl -s -X POST localhost:8000/admin/switchboard/reset      # all models back on
curl -s -X POST localhost:8000/admin/pool/reset             # close circuit breakers
curl -s -X POST localhost:8000/admin/alerts/clear           # clear the flag
curl -s -X POST localhost:8000/admin/reputation/reset       # forget quality history
curl -s -X POST localhost:8000/admin/fleet/reset            # forget traffic stats
curl -s -X POST localhost:8000/demo/reset/SESSION_ID        # forget a session's warm model
```
