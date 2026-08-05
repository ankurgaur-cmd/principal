# Intent classification — how the gateway decides what you asked for

Everything downstream starts here. The intent picks the **tier floor**, which
picks the **candidate models**, which picks the **price** and the **output
budget**. A wrong label is not a cosmetic problem: it is the wrong model, at the
wrong cost, for the whole request.

This document is the logic as implemented, in `src/aigateway/routing/intent.py`
and `src/aigateway/routing/policy.py`. Tests are in `tests/test_intent.py`.

---

## Contents

1. [The problem](#1-the-problem)
2. [The four layers](#2-the-four-layers)
3. [L0 — the declared hint](#3-l0--the-declared-hint-trusted-but-checked)
4. [L1 — deterministic rules](#4-l1--deterministic-rules)
5. [L2 — embeddings](#5-l2--embeddings-abstains-by-default)
6. [L3 — the small-model call](#6-l3--the-small-model-call)
7. [Confidence, and what it controls](#7-confidence-and-what-it-controls)
8. [What the label is used for](#8-what-the-label-is-used-for)
9. [Reading the output](#9-reading-the-output)
10. [Tuning it for your traffic](#10-tuning-it-for-your-traffic)
11. [Known limits](#11-known-limits)

---

## 1. The problem

Classifying with an LLM on every request adds a latency floor and a cost floor to
traffic you were trying to make *cheaper*. If the classifier costs a cent and
saves two, you have built an expensive way to break even.

So the layers run **cheapest-first and stop as soon as one is confident enough**.
The design target is that L0 and L1 absorb the large majority of traffic and L3
is a rarity. If `intent_source` shows `llm` firing often, that is a signal your
rules or exemplars need work — not that you need a bigger budget.

The second problem is subtler and is where most of the care went: **a cheap
classifier that is confidently wrong is worse than one that abstains.** Every
mechanism below exists to make abstention cheap and over-confidence hard.

---

## 2. The four layers

| Layer | Source | Cost | What it uses |
|---|---|---|---|
| **L0** | `declared` / `declared-overridden` | free | the caller's `x_gateway.intent`, checked against request shape |
| **L1** | `rules` | free | response schema, tool count, size, weighted keywords across recent turns |
| **L2** | `embedding` | ~free | nearest-neighbour over labelled exemplars (abstains unless wired) |
| **L3** | `llm` / `llm-cached` | cheap | a small model, memoised by request shape |
| — | `default` | free | nothing was confident; falls back to `unknown` |

```
                    ┌──────────────────────────────┐
  request ─────────▶│ L0  caller declared an intent │──── yes ──┐
                    └──────────────┬───────────────┘           │
                                   │ no                        ▼
                                   │                  ┌──────────────────┐
                                   │                  │ verify against   │
                                   │                  │ request shape    │
                                   │                  └────────┬─────────┘
                                   │                           │
                                   ▼                      accept │ override
                    ┌──────────────────────────────┐            │
                    │ L1  deterministic rules       │            │
                    └──────────────┬───────────────┘            │
                       confident?  │ no                          │
                                   ▼                             │
                    ┌──────────────────────────────┐             │
                    │ L2  embeddings (abstains)     │             │
                    └──────────────┬───────────────┘             │
                       confident?  │ no                          │
                                   ▼                             │
                    ┌──────────────────────────────┐             │
                    │ L3  small-model call, cached  │             │
                    └──────────────┬───────────────┘             │
                                   │ failed / unavailable        │
                                   ▼                             │
                    ┌──────────────────────────────┐             │
                    │ best rules guess, or `unknown`│◀────────────┘
                    └──────────────────────────────┘
```

Every layer returns the same shape:

```python
IntentResult(
    intent="code_review",       # always a key of INTENT_POLICY
    confidence=0.75,            # 0..1
    source="rules",             # which layer answered
    rationale="keyword evidence: code_review (1.6)",
    fallback=False,             # was this an absence of signal?
)
```

A test asserts every layer, on every path, produces a known intent **and** a
non-empty rationale. A label the router cannot price, or that nobody can explain,
is useless.

---

## 3. L0 — the declared hint, trusted but checked

Your agent usually knows what it is doing, so a declared intent is accepted at
confidence `1.0` and skips every other layer. That is the fast path and the
common case.

**But it is checked**, because the cheapest possible mistake is also the easiest
one to make: a template that labels everything `classify`, a stale constant, a
copy-pasted request. Before this check existed, such a request got a light-tier
model and a bad answer, and nothing anywhere said so — while the documented
contract said hints were "verified, and overridable, by the gateway".

The check is deliberately **one-directional**:

| Situation | What happens |
|---|---|
| Declared intent agrees with the evidence | accepted, `source: declared` |
| Declared intent is **heavier** than the evidence | accepted, `source: declared` |
| Declared intent is **lighter**, evidence is weak | accepted, `source: declared`, confidence `0.8` |
| Declared intent is **lighter**, evidence is strong | **overridden**, `source: declared-overridden` |

Declaring something *heavier* than the request looks is you spending your own
money on caution — that is your call, and the gateway leaves it alone. Only the
direction that costs **quality** is corrected.

Two guards before overruling a human:

1. **The evidence must be positive, not a structural default.** "Short and
   tool-free, so probably chat" is a sensible default and a terrible reason to
   overrule someone who declared `translate` — most short requests are short.
   Results carry `fallback=True` for exactly this, and a fallback never
   overrides.
2. **It must clear a higher bar than ordinary classification.** Overruling a
   declaration is a stronger action than merely being confident enough to stop
   classifying, so it needs `confidence >= 0.7`, not the usual `0.6`. Without the
   gap, one weak keyword was enough to overturn a label a human wrote on purpose.

An override always says what it did and how to insist:

```
you declared 'classify' (light tier), but the request looks like hard_debug
needing heavy — keyword evidence: hard_debug (2.4). Routed on the evidence;
pin_model or max_tier if you meant it.
```

An unknown intent name is **not** an error — it is logged and falls through to
L1, because a typo should not fail a request.

---

## 4. L1 — deterministic rules

Free, instant, and where most traffic should land. Rules run in order and the
first that fires wins, because they are ordered by how much they actually know.

### 4a. Shape rules — these outrank prose

Structure is stronger evidence than wording. Someone who attaches a JSON schema
is doing extraction whatever the sentence says.

| Rule | Result | Confidence | Why |
|---|---|---|---|
| Response schema **and** under 1,600 tokens | `extract` | 0.9 | near-definitional |
| 5 or more tools declared | `tool_orchestration` | 0.8 | whatever the prose says |
| Under 400 tokens, no tools | `chat` | 0.7 | **fallback** — absence of signal |
| Over 25,000 tokens **and** tools present | `long_horizon_agentic` | 0.7 | that shape is not a one-shot |

The schema rule has a ceiling on purpose: a schema-shaped request carrying 30,000
tokens is doing more than filling in fields, and calling it `extract` would floor
it at the light tier.

### 4b. Keyword evidence — scored, not first-match

The original version returned on the **first** matching pattern, which made list
order silently decide the answer:

> "review this stack trace and debug the race condition"

matched `code_review` because it was listed first, and `hard_debug` never got a
look. Order is a terrible tie-break because nobody reading a list of patterns
knows that its order is load-bearing.

Now **every** pattern is tested and the hits are scored. Weights reflect how much
one hit is actually worth:

| Evidence | Intent | Weight |
|---|---|---|
| `translate` | translate | 1.4 |
| `stack trace`, `race condition`, `deadlock`, `flaky`, `segfault` | hard_debug | 1.2 |
| `architect`, `design doc`, `system design` | architecture | 1.2 |
| `summarise`, `tl;dr`, `condense`, `key points` | summarize | 1.2 |
| `classif`, `categor`, `label this`, `sentiment` | classify | 1.2 |
| `extract`, `parse out`, `pull the fields` | extract | 1.2 |
| `write a function/class/test/script` | code_write | 1.2 |
| `audit`, `find bugs`, `vulnerab`, `security`, `injection`, `xss` | code_review | 1.0 |
| `break down`, `steps to`, `roadmap` | plan | 1.0 |
| `implement`, `refactor` | code_write | 0.9 |
| `debug` | hard_debug | 0.8 |
| `trade-off` | architecture | 0.8 |
| `plan` | plan | 0.7 |
| `review` | code_review | 0.6 |

Phrases that only ever appear in one kind of request score higher than words that
appear everywhere. **"Race condition" is decisive; "review" is not** — you review
a diff, a document, a plan, or a decision.

**Negation is handled.** "Do not summarise" matched `summarize` before the check
existed. A hit preceded within ~24 characters by `don't`, `no need to`, `without`,
`avoid`, `rather than`, or `instead of` is discarded.

### 4c. Evidence across turns, not just the last message

Classifying on the last user message alone is wrong in exactly the case that
matters most — a multi-turn session:

> **Turn 1:** "Review this authentication middleware for vulnerabilities."
> **Turn 2:** "yes, do that"

"yes, do that" carries no signal at all. A last-message classifier sends that
continuation to a light model.

So the last **3** user turns contribute, decaying by **0.5** per turn back:

```
score(intent) = Σ  keyword_score(turn_n) × 0.5ⁿ      n = 0, 1, 2
```

Older turns fade rather than counting equally — the conversation genuinely does
move on, so a review two turns back is weaker evidence than a translation right
now, but it is not *zero* evidence.

---

## 5. L2 — embeddings (abstains by default)

Ships as a null implementation on purpose. Plugging in an embedding provider is a
deployment decision, and a stub that silently returns garbage is worse than one
that abstains.

To enable it, implement the `Embedder` protocol and index your labelled
exemplars:

```python
class MyEmbedder:
    async def embed(self, text: str) -> list[float] | None: ...

IntentClassifier(store, registry, embedder=MyEmbedder())
```

Then fill in `_classify_by_embedding`. Until that exists it returns `None`, and
`None` means "no opinion" — never a guess.

---

## 6. L3 — the small-model call

Runs only when everything above abstained or was unsure, and only when a provider
is actually configured (checked live — credentials can arrive after startup).

It is a **structured** call against a schema constrained to the known intents, and
the system prompt biases towards cheapness:

> "Choose the cheapest intent that plausibly covers the request — over-labelling
> wastes money on an unnecessarily large model."

Three protections:

**A classifier failure never fails the request it was labelling.** Any exception
is logged and the layer returns `None`; the request falls back to the best rules
guess, or `unknown`.

**A label outside the taxonomy is rejected**, not passed through. The router has
no policy for `make_coffee` and therefore no tier floor for it.

**Results are cached for an hour, keyed on everything that could change the
answer** — the text, the classifier model, *and* a fingerprint of the label set:

```python
key = sha256(taxonomy_version + "\0" + model_key + "\0" + text)
```

Keyed on text alone, a stale label survived for an hour after the taxonomy or the
classifier model changed. A label that is quietly wrong for an hour is worse than
one recomputed for a fraction of a cent. A cache hit reports `source: llm-cached`
so it is never mistaken for a fresh judgement.

---

## 7. Confidence, and what it controls

Confidence is not decoration — it is the **escalation control**. A layer's answer
is accepted only if `confidence >= classifier_min_confidence` (default `0.6`);
otherwise the next layer gets a look.

| Confidence | Meaning |
|---|---|
| `1.0` | caller declared it and nothing contradicts it |
| `0.9` | a shape rule that is near-definitional |
| `0.8` | many tools, or a declaration kept over weak contrary evidence |
| `0.75` | strong, uncontested keyword evidence (the ceiling for keywords) |
| `0.7` | a structural default |
| `0.55` | keyword evidence on a large request |
| `0.5` | two intents in close contest |
| `0.3` | nothing worked; `unknown` |

Keyword confidence scales with the evidence — `min(0.75, 0.55 + 0.1 × score)` —
and is capped at **0.75**. No pile of keywords ever reaches the certainty of a
declared intent, because keywords are suggestive and never conclusive.

Two situations deliberately push confidence *below* the threshold so that L3 gets
a look:

- **Large requests** (≥ 25,000 tokens) drop to `0.55`. A big-context prompt saying
  "summarize" may be doing something much harder, and light-tier is an expensive
  place to be wrong.
- **Contested labels** — where the runner-up scores ≥ 75% of the winner — drop to
  `0.5`. Two intents neck and neck is precisely when a cheap label is most likely
  to be wrong, so it is worth paying for a better one.

---

## 8. What the label is used for

The intent indexes `INTENT_POLICY`, which supplies three things:

| Column | Effect |
|---|---|
| `min_tier` | the **floor** for model selection — the router still picks the cheapest model at or above it |
| `effort` | reasoning effort, mapped per vendor |
| `max_tokens` | default output budget when the caller gave none |

```
classify              light      low      600
extract               light      low    1,200
summarize             light      low    1,500
translate / format    light      low    2,000
qa / chat             standard   medium 5,000
plan / analysis       standard   high   8,000
tool_orchestration    standard   high   8,000
code_write            standard   high   8,000
code_review           heavy      high   8,000
hard_debug            heavy      xhigh 14,000
architecture          heavy      xhigh 14,000
long_horizon_agentic  heavy      xhigh 16,000
unknown               standard   medium 5,000
```

**`min_tier` is a floor, not a target.** Widening the catalog is how you get
cheaper, not editing this table downward. What the router does with the floor is
[ROUTING.md](ROUTING.md).

**`unknown` is deliberately `standard`, never `light`.** Abstention must not be a
discount: if the classifier does not know what the work is, guessing light is the
expensive kind of wrong. There is a test for this.

The classifier's taxonomy and the policy table are the same thing seen from two
sides, and a test asserts they cannot drift — a label with no policy has no tier
floor.

---

## 9. Reading the output

Every response carries the decision:

```python
resp.x_gateway.resolved_intent      # "code_review"
resp.x_gateway.intent_confidence    # 0.75
resp.x_gateway.intent_source        # "rules"
```

The console shows it live on the **Intent classifier** node, with four lamps for
which layer fired (L0/L1/L2/L3) and the rationale underneath. `/admin/route/preview`
gives you the same thing without spending anything:

```bash
curl -s localhost:8000/admin/route/preview -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Review this diff for race conditions"}]}' \
  | jq '.intent'
```

```json
{
  "resolved": "hard_debug",
  "confidence": 0.75,
  "source": "rules",
  "rationale": "keyword evidence: hard_debug (2.4), contested by code_review (0.6)"
}
```

The per-request JSONL record carries `resolved_intent`, `intent_confidence` and
`intent_source` on every line, which is what the replay harness re-scores against.

---

## 10. Tuning it for your traffic

**The taxonomy is placeholder and you should replace it.** A routing table is
only as good as its intent classes, and the right classes are the ones your agents
actually emit — not a generic list. Expect the first week of `records.jsonl` to
tell you what the real classes are.

Practical order of work:

1. **Watch `intent_source`.** If `llm` or `llm-cached` dominates, L1 is not
   pulling its weight — add keywords or shape rules for whatever L3 keeps
   labelling.
2. **Watch `declared-overridden`.** A steady trickle is the check earning its
   keep. A flood means one of your agents has a wrong constant, and the rationale
   names which intent it should be sending.
3. **Watch `unknown`.** Every `unknown` is standard-tier spend on work nobody has
   characterised.
4. **Add keywords with weights, not just patterns.** A word that appears in three
   kinds of request should score low; a phrase that only ever means one thing
   should score high.

Settings:

| Setting | Default | Effect |
|---|---|---|
| `GATEWAY_LLM_CLASSIFIER_ENABLED` | `true` | turn L3 off entirely |
| `GATEWAY_CLASSIFIER_MODEL` | `claude-haiku-4-5` | which small model labels |
| `GATEWAY_CLASSIFIER_MIN_CONFIDENCE` | `0.6` | how eagerly layers escalate |

Raising `min_confidence` sends more traffic to L3 — more accurate, more expensive.
Lowering it does the reverse. It is the single dial between classification cost
and classification accuracy.

---

## 11. Known limits

**The weights are reasoned, not calibrated.** `translate` at 1.4 and `review` at
0.6 are arguments about how specific a word is, not measurements against labelled
traffic. The replay harness is the right place to fit them once you have records;
until then treat them as a starting position.

**Keywords are English and literal.** No stemming, no synonyms, no other
languages. A request in French asking for a code review will not match, and will
fall through to L3 — which is the correct failure mode, but a costly one at
volume. This is the strongest argument for wiring up L2.

**The negation window is 24 characters and positional.** "Summarise it, but don't
be brief" has the negation *after* the keyword and will not be caught. Real
negation scope needs parsing, which is not free.

**Turn decay is fixed at 3 turns and 0.5.** Both are guesses. A long agentic
session with a stable goal probably wants a longer window; a chat that changes
subject constantly wants a shorter one.

**L2 is not implemented.** It is the layer best placed to fix the language and
synonym limits, and it is the obvious next piece of work.

**Overriding is one-directional by design.** The gateway will not route a
declared `architecture` down to `classify` even when the request is plainly
trivial, so a caller who over-declares pays for it. That is deliberate — see
[§3](#3-l0--the-declared-hint-trusted-but-checked) — but it does mean the check
protects quality and not spend.
