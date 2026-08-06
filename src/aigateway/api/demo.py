"""Endpoints backing the demo console.

Two things the ordinary API cannot show on its own:

* ``/demo/fanout`` fires N sub-agents in parallel against one shared prefix,
  which is what makes the cache-pilot behaviour visible. One pilot writes; the
  rest read.
* ``/demo/reset`` clears a session's stickiness so the same query can be run
  cold and then warm, back to back, without waiting out the TTL.
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..catalog import CATALOG
from ..errors import GatewayError
from ..pipeline import canonicalise
from ..schemas import ChatCompletionRequest, ChatMessage, GatewayExtensions
from ..tokens import estimate_request_tokens

router = APIRouter(prefix="/demo", tags=["demo"])


class FanoutRequest(BaseModel):
    prompt: str
    shared_context: str = ""
    agents: int = 4
    session_id: str = "fanout"
    intent: str | None = None
    pilot_enabled: bool = True
    # Optional per-agent tasks. Real fan-out gives each sub-agent a different
    # slice of the work against the same shared prefix; sending all N the
    # identical prompt produces N near-identical answers, which makes the
    # per-agent output impossible to judge even once you can see it. When this
    # is empty the agents fall back to the single `prompt`.
    subtasks: list[str] = Field(default_factory=list)
    # Sub-agents are reasoning models too, and hidden reasoning bills against
    # this same budget. At 800 every agent returned an empty answer and the
    # dashboard showed six rows of metrics with nothing in them. Measured floor
    # for demanding work is ~3,000 on both vendors; 4,000 leaves room to answer.
    max_tokens: int = 4000


@router.post("/fanout")
async def fanout(body: FanoutRequest, request: Request) -> dict:
    """Run N sub-agents concurrently on one shared prefix.

    With the pilot on, exactly one request writes the cache and the rest read
    it. With it off, all N race and every one pays a cold write — which is the
    default behaviour of every gateway that does not do this.
    """
    app = request.app
    principal = await app.state.auth.authenticate(request)
    pipeline = app.state.pipeline

    # Toggle the pilot for this run so the demo can show both sides.
    original = pipeline.pilot.set_enabled(body.pilot_enabled)
    await app.state.store.delete(f"session:{body.session_id}:model")

    def task_for(index: int) -> str:
        """What sub-agent `index` is actually asked to do.

        The shared prefix stays byte-identical across agents either way — that
        is what makes the cache shareable — so giving each agent its own task
        costs nothing and makes the parallel output worth reading.
        """
        if body.subtasks:
            return body.subtasks[index % len(body.subtasks)]
        return f"{body.prompt}\n\n(sub-agent #{index + 1})"

    def build(index: int) -> ChatCompletionRequest:
        messages = []
        if body.shared_context:
            messages.append(ChatMessage(role="system", content=body.shared_context))
        messages.append(ChatMessage(role="user", content=task_for(index)))
        return ChatCompletionRequest(
            model="auto",
            messages=messages,
            max_tokens=body.max_tokens,
            x_gateway=GatewayExtensions(session_id=body.session_id, intent=body.intent),
        )

    started = time.perf_counter()
    try:
        results = await asyncio.gather(
            *(pipeline.handle(build(i), principal) for i in range(body.agents)),
            return_exceptions=True,
        )
    finally:
        pipeline.pilot.set_enabled(original)

    agents = []
    total_cost = 0.0
    total_read = 0
    total_write = 0
    for index, result in enumerate(results):
        if isinstance(result, GatewayError):
            agents.append({"agent": index + 1, "task": task_for(index),
                           "error": str(result.detail)})
            continue
        if isinstance(result, BaseException):
            agents.append({"agent": index + 1, "task": task_for(index),
                           "error": repr(result)})
            continue
        meta = result.x_gateway
        total_cost += meta.actual_cost_usd
        total_read += meta.cache_read_tokens
        total_write += meta.cache_write_tokens
        agents.append(
            {
                "agent": index + 1,
                # An answer you cannot see the question for is not reviewable.
                "task": task_for(index),
                "model": meta.chosen_model,
                "provider": meta.provider,
                "tier": meta.tier,
                "pilot_role": meta.pilot_role,
                "cache_read_tokens": meta.cache_read_tokens,
                "cache_write_tokens": meta.cache_write_tokens,
                "cost_usd": meta.actual_cost_usd,
                "latency_ms": meta.latency_ms,
                "hosts": meta.trace.get("hosts_contacted", []),
                # Without the answer and its verdict this panel is a set of
                # numbers you cannot judge — you can see what it cost but not
                # whether it was worth anything.
                "answer": result.choices[0].message.content or "",
                "quality": meta.quality.get("verdict"),
                "quality_failures": [
                    c["title"]
                    for c in meta.quality.get("checks", [])
                    if c.get("level") == "fail"
                ],
            }
        )

    prefix_tokens, _ = estimate_request_tokens(canonicalise(build(0)))
    cacheable_on = [
        m.key for m in CATALOG.values() if prefix_tokens >= m.min_cacheable_tokens
    ]

    return {
        "agents": agents,
        "pilot_enabled": body.pilot_enabled,
        "wall_clock_ms": int((time.perf_counter() - started) * 1000),
        "total_cost_usd": round(total_cost, 6),
        "total_cache_read_tokens": total_read,
        "total_cache_write_tokens": total_write,
        # Why the run behaved as it did. Without this, "everything was cold"
        # looks like a broken cache when it usually means the shared prefix was
        # too small to cache anywhere.
        "prefix_tokens": prefix_tokens,
        "cacheable_on": cacheable_on,
        "cacheable": bool(cacheable_on),
    }


class FaninRequest(BaseModel):
    task: str
    subtasks: list[str]
    shared_context: str = ""
    session_id: str = "fanin"
    worker_intent: str | None = None
    synthesis_intent: str | None = "analysis"
    max_tokens: int = 4000  # same reasoning-budget floor as fan-out


@router.post("/fanin")
async def fanin(body: FaninRequest, request: Request) -> dict:
    """Scatter/gather: N workers in parallel, then one synthesiser over their output.

    This is the shape most multi-agent systems actually take, and it is where a
    router earns its keep — the workers are doing narrow, well-specified jobs
    that a small model handles fine, while the synthesis step has to hold all
    their output at once and reason across it. Routing them identically wastes
    money on the workers or under-powers the synthesis.

    Every leg is routed independently and traced, so you can see the tier split
    happen rather than being told it does.
    """
    app = request.app
    principal = await app.state.auth.authenticate(request)
    pipeline = app.state.pipeline
    started = time.perf_counter()

    def build(prompt: str, intent: str | None, session: str) -> ChatCompletionRequest:
        messages = []
        if body.shared_context:
            messages.append(ChatMessage(role="system", content=body.shared_context))
        messages.append(ChatMessage(role="user", content=prompt))
        return ChatCompletionRequest(
            model="auto",
            messages=messages,
            max_tokens=body.max_tokens,
            x_gateway=GatewayExtensions(session_id=session, intent=intent),
        )

    # --- scatter -----------------------------------------------------------
    scatter_started = time.perf_counter()
    results = await asyncio.gather(
        *(
            pipeline.handle(
                build(sub, body.worker_intent, f"{body.session_id}-w{i}"), principal
            )
            for i, sub in enumerate(body.subtasks)
        ),
        return_exceptions=True,
    )
    scatter_ms = int((time.perf_counter() - scatter_started) * 1000)

    workers = []
    findings: list[str] = []
    for i, (sub, result) in enumerate(zip(body.subtasks, results, strict=False)):
        if isinstance(result, BaseException):
            detail = str(result.detail) if isinstance(result, GatewayError) else repr(result)
            workers.append({"worker": i + 1, "subtask": sub, "error": detail})
            continue
        text = result.choices[0].message.content or ""
        meta = result.x_gateway
        findings.append(f"### {sub}\n{text}")
        workers.append(
            {
                "worker": i + 1,
                "subtask": sub,
                "model": meta.chosen_model,
                "provider": meta.provider,
                "tier": meta.tier,
                "intent": meta.resolved_intent,
                "cost_usd": meta.actual_cost_usd,
                "latency_ms": meta.latency_ms,
                "cached_tokens": meta.cache_read_tokens,
                "hosts": meta.trace.get("hosts_contacted", []),
                "quality": meta.quality.get("verdict"),
                "answer": text,
            }
        )

    if not findings:
        raise GatewayError(502, "every worker failed; nothing to synthesise", code="fanin_failed")

    # --- gather ------------------------------------------------------------
    gather_started = time.perf_counter()
    synthesis_prompt = (
        f"{body.task}\n\nYou have the following findings from parallel workers. "
        f"Synthesise them into one coherent answer, noting any disagreement.\n\n"
        + "\n\n".join(findings)
    )
    synthesis = await pipeline.handle(
        build(synthesis_prompt, body.synthesis_intent, f"{body.session_id}-synth"), principal
    )
    gather_ms = int((time.perf_counter() - gather_started) * 1000)
    smeta = synthesis.x_gateway

    worker_cost = sum(w.get("cost_usd", 0.0) for w in workers)
    tiers = {w.get("tier") for w in workers if w.get("tier")}

    return {
        "workers": workers,
        "synthesis": {
            "model": smeta.chosen_model,
            "provider": smeta.provider,
            "tier": smeta.tier,
            "intent": smeta.resolved_intent,
            "cost_usd": smeta.actual_cost_usd,
            "latency_ms": smeta.latency_ms,
            "prompt_tokens": synthesis.usage.prompt_tokens,
            "hosts": smeta.trace.get("hosts_contacted", []),
            "quality": smeta.quality.get("verdict"),
            "answer": synthesis.choices[0].message.content or "",
        },
        "totals": {
            "workers": len(workers),
            "worker_cost_usd": round(worker_cost, 6),
            "synthesis_cost_usd": round(smeta.actual_cost_usd, 6),
            "total_cost_usd": round(worker_cost + smeta.actual_cost_usd, 6),
            "scatter_ms": scatter_ms,
            "gather_ms": gather_ms,
            "wall_clock_ms": int((time.perf_counter() - started) * 1000),
            # The headline: did the router actually split the tiers, or send
            # everything to the same model?
            "worker_tiers": sorted(t for t in tiers if t),
            "synthesis_tier": smeta.tier,
            "tier_split": bool(tiers) and smeta.tier not in tiers,
        },
    }


@router.post("/trace")
async def trace(body: ChatCompletionRequest, request: Request):
    """Run a request and stream each pipeline stage as it completes.

    The console renders the route from these events, so what you watch is the
    real sequence with real timings — not an animation timed to look plausible.
    """
    app = request.app
    principal = await app.state.auth.authenticate(request)
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(stage_name: str, payload: dict) -> None:
        await queue.put({"stage": stage_name, **payload})

    async def generate():
        task = asyncio.create_task(app.state.pipeline.handle(body, principal, emit=emit))
        try:
            while True:
                drain = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {drain, task}, return_when=asyncio.FIRST_COMPLETED, timeout=180
                )
                if drain in done:
                    yield f"data: {json.dumps(drain.result())}\n\n"
                    continue

                drain.cancel()
                # The pipeline finished; flush anything still queued so no
                # stage is lost to the race between the last emit and the return.
                while not queue.empty():
                    yield f"data: {json.dumps(queue.get_nowait())}\n\n"

                if not done:  # the wait timed out
                    yield f"data: {json.dumps({'stage': 'error', 'error': 'timed out'})}\n\n"
                    task.cancel()
                    break

                try:
                    payload = {"stage": "done", "response": task.result().model_dump()}
                    yield f"data: {json.dumps(payload)}\n\n"
                except GatewayError as exc:
                    detail = exc.detail.get("error", {}) if isinstance(exc.detail, dict) else {}
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "stage": "error",
                                "error": detail.get("message", str(exc.detail)),
                                "code": detail.get("code", "error"),
                                "status": exc.status_code,
                                # Plain language for the person, the remedy for
                                # the operator. Both, or one audience is served
                                # the other's version.
                                "user_message": detail.get("user_message", ""),
                                "remedy": detail.get("remedy", ""),
                                "cause": detail.get("cause", ""),
                            }
                        )
                        + "\n\n"
                    )
                except Exception as exc:
                    yield f"data: {json.dumps({'stage': 'error', 'error': repr(exc)})}\n\n"
                break
        finally:
            if not task.done():
                task.cancel()
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


@router.post("/reset/{session_id}")
async def reset(session_id: str, request: Request) -> dict:
    """Forget a session's warm model so the next call routes cold."""
    await request.app.state.store.delete(f"session:{session_id}:model")
    return {"session_id": session_id, "reset": True}
