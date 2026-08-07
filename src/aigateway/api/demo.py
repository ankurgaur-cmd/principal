"""Endpoints backing the demo console.

Things the ordinary API cannot show on its own:

* ``/demo/fanout`` fires N sub-agents in parallel against one shared prefix,
  which is what makes the cache-pilot behaviour visible. One pilot writes; the
  rest read.
* ``/demo/fanout/live`` and ``/demo/fanin/live`` are the same runs as SSE:
  every worker's pipeline stages, multiplexed into one stream as they happen,
  then the same summary the JSON endpoints return. The console draws one lane
  per worker from these — real events with real timings, not an animation.
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


def _fan_task(body: FanoutRequest, index: int) -> str:
    """What sub-agent `index` is actually asked to do.

    The shared prefix stays byte-identical across agents either way — that
    is what makes the cache shareable — so giving each agent its own task
    costs nothing and makes the parallel output worth reading.
    """
    if body.subtasks:
        return body.subtasks[index % len(body.subtasks)]
    return f"{body.prompt}\n\n(sub-agent #{index + 1})"


def _fan_build(body: FanoutRequest, index: int) -> ChatCompletionRequest:
    messages = []
    if body.shared_context:
        messages.append(ChatMessage(role="system", content=body.shared_context))
    messages.append(ChatMessage(role="user", content=_fan_task(body, index)))
    return ChatCompletionRequest(
        model="auto",
        messages=messages,
        max_tokens=body.max_tokens,
        x_gateway=GatewayExtensions(session_id=body.session_id, intent=body.intent),
    )


def _fan_row(index: int, task: str, result) -> dict:
    """One agent's line in the summary — shared by the JSON and live shapes."""
    if isinstance(result, GatewayError):
        return {"agent": index + 1, "task": task, "error": str(result.detail)}
    if isinstance(result, BaseException):
        return {"agent": index + 1, "task": task, "error": repr(result)}
    meta = result.x_gateway
    return {
        "agent": index + 1,
        # An answer you cannot see the question for is not reviewable.
        "task": task,
        "model": meta.chosen_model,
        "provider": meta.provider,
        "tier": meta.tier,
        "pilot_role": meta.pilot_role,
        "cache_read_tokens": meta.cache_read_tokens,
        "cache_write_tokens": meta.cache_write_tokens,
        "cost_usd": meta.actual_cost_usd,
        "latency_ms": meta.latency_ms,
        # The latency split the whole panel is judged by: the gateway's own
        # compute, the deliberate pilot wait, and the model's time. The
        # gateway number is the one that recurs on *every* agent call.
        "gateway_ms": meta.trace.get("gateway_overhead_ms", 0),
        "upstream_ms": meta.trace.get("upstream_ms", 0),
        "pilot_wait_ms": meta.trace.get("pilot_wait_ms", 0),
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


def _time_split(rows: list[dict]) -> dict:
    """Cumulative gateway/LLM/wait time across a run's legs.

    The cumulative gateway number is the honest price of putting this gateway
    in front of *every* agent call: per-call overhead is microscopic next to a
    model call, but N agents pay it N times, and only the sum shows whether
    that stays true.
    """
    ok = [r for r in rows if "error" not in r]
    gateway = sum(r.get("gateway_ms", 0) - r.get("pilot_wait_ms", 0) for r in ok)
    return {
        "calls": len(ok),
        "gateway_ms_total": gateway,
        "gateway_ms_mean": round(gateway / len(ok), 1) if ok else 0,
        "pilot_wait_ms_total": sum(r.get("pilot_wait_ms", 0) for r in ok),
        "llm_ms_total": sum(r.get("upstream_ms", 0) for r in ok),
    }


def _fan_summary(body: FanoutRequest, agents: list[dict], started: float) -> dict:
    total_cost = sum(a.get("cost_usd", 0.0) for a in agents)
    total_read = sum(a.get("cache_read_tokens", 0) for a in agents)
    total_write = sum(a.get("cache_write_tokens", 0) for a in agents)

    prefix_tokens, _ = estimate_request_tokens(canonicalise(_fan_build(body, 0)))
    cacheable_on = [
        m.key for m in CATALOG.values() if prefix_tokens >= m.min_cacheable_tokens
    ]
    return {
        "agents": agents,
        "pilot_enabled": body.pilot_enabled,
        "time_split": _time_split(agents),
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

    started = time.perf_counter()
    try:
        results = await asyncio.gather(
            *(pipeline.handle(_fan_build(body, i), principal) for i in range(body.agents)),
            return_exceptions=True,
        )
    finally:
        pipeline.pilot.set_enabled(original)

    agents = [
        _fan_row(i, _fan_task(body, i), result) for i, result in enumerate(results)
    ]
    return _fan_summary(body, agents, started)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def _pump(queue: asyncio.Queue, task: asyncio.Task, timeout: float = 300):
    """Yield queued events until ``task`` completes, then flush the stragglers.

    The race this exists for: the last emit and the task's return are
    concurrent, so "task finished" must not mean "stop reading the queue" until
    the queue is provably empty — or the final stage of the slowest worker
    vanishes from the feed.
    """
    while True:
        drain = asyncio.create_task(queue.get())
        done, _ = await asyncio.wait(
            {drain, task}, return_when=asyncio.FIRST_COMPLETED, timeout=timeout
        )
        if drain in done:
            yield drain.result()
            continue
        drain.cancel()
        while not queue.empty():
            yield queue.get_nowait()
        if not done:  # the wait timed out
            yield {"event": "error", "error": "timed out"}
            task.cancel()
        return


def _live_emitter(queue: asyncio.Queue, worker, run_started: float):
    """An ``emit`` for one worker that tags every stage with who and when.

    ``at_ms`` is run-relative wall clock — the one clock every lane shares —
    so the unified feed can interleave workers in the order things actually
    happened. The per-request clocks (`elapsed_ms`, `stage_ms`) ride along
    untouched.
    """

    async def emit(stage_name: str, payload: dict) -> None:
        await queue.put(
            {
                "event": "stage",
                "worker": worker,
                "stage": stage_name,
                "at_ms": int((time.perf_counter() - run_started) * 1000),
                **payload,
            }
        )

    return emit


@router.post("/fanout/live")
async def fanout_live(body: FanoutRequest, request: Request):
    """The fan-out run as SSE: every worker's pipeline stages, live.

    Same run as ``/demo/fanout`` — same routing, same pilot, same summary at
    the end — but each worker's stage events stream out as they happen, tagged
    ``worker`` and ``at_ms``, so the console can draw N live pipelines and one
    merged feed instead of a spinner followed by a table.
    """
    app = request.app
    principal = await app.state.auth.authenticate(request)
    pipeline = app.state.pipeline
    queue: asyncio.Queue = asyncio.Queue()

    async def one(index: int, run_started: float) -> dict:
        try:
            result = await pipeline.handle(
                _fan_build(body, index), principal,
                emit=_live_emitter(queue, index + 1, run_started),
            )
        except BaseException as exc:  # noqa: BLE001 - row carries the error
            result = exc
        row = _fan_row(index, _fan_task(body, index), result)
        await queue.put({"event": "worker_done", "worker": index + 1, "agent": row})
        return row

    async def generate():
        # Toggled inside the generator: the response object is created before
        # the run starts, and the pilot must be restored even if the client
        # disconnects mid-stream.
        original = pipeline.pilot.set_enabled(body.pilot_enabled)
        await app.state.store.delete(f"session:{body.session_id}:model")
        started = time.perf_counter()
        run = asyncio.ensure_future(
            asyncio.gather(*(one(i, started) for i in range(body.agents)))
        )
        try:
            async for ev in _pump(queue, run):
                yield _sse(ev)
            if run.done() and not run.cancelled():
                yield _sse({"event": "summary", **_fan_summary(body, run.result(), started)})
        finally:
            pipeline.pilot.set_enabled(original)
            if not run.done():
                run.cancel()
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


class FaninRequest(BaseModel):
    task: str
    subtasks: list[str]
    shared_context: str = ""
    session_id: str = "fanin"
    worker_intent: str | None = None
    synthesis_intent: str | None = "analysis"
    max_tokens: int = 4000  # same reasoning-budget floor as fan-out


def _fanin_build(
    body: FaninRequest, prompt: str, intent: str | None, session: str
) -> ChatCompletionRequest:
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


def _fanin_worker_row(i: int, sub: str, result) -> dict:
    if isinstance(result, BaseException):
        detail = str(result.detail) if isinstance(result, GatewayError) else repr(result)
        return {"worker": i + 1, "subtask": sub, "error": detail}
    meta = result.x_gateway
    return {
        "worker": i + 1,
        "subtask": sub,
        "model": meta.chosen_model,
        "provider": meta.provider,
        "tier": meta.tier,
        "intent": meta.resolved_intent,
        "cost_usd": meta.actual_cost_usd,
        "latency_ms": meta.latency_ms,
        "gateway_ms": meta.trace.get("gateway_overhead_ms", 0),
        "upstream_ms": meta.trace.get("upstream_ms", 0),
        "pilot_wait_ms": meta.trace.get("pilot_wait_ms", 0),
        "cached_tokens": meta.cache_read_tokens,
        "hosts": meta.trace.get("hosts_contacted", []),
        "quality": meta.quality.get("verdict"),
        "answer": result.choices[0].message.content or "",
    }


def _fanin_synthesis_prompt(body: FaninRequest, findings: list[str]) -> str:
    return (
        f"{body.task}\n\nYou have the following findings from parallel workers. "
        f"Synthesise them into one coherent answer, noting any disagreement.\n\n"
        + "\n\n".join(findings)
    )


def _fanin_result(
    workers: list[dict], synthesis, scatter_ms: int, gather_ms: int, started: float
) -> dict:
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
            "gateway_ms": smeta.trace.get("gateway_overhead_ms", 0),
            "upstream_ms": smeta.trace.get("upstream_ms", 0),
            "pilot_wait_ms": smeta.trace.get("pilot_wait_ms", 0),
            "prompt_tokens": synthesis.usage.prompt_tokens,
            "hosts": smeta.trace.get("hosts_contacted", []),
            "quality": smeta.quality.get("verdict"),
            "answer": synthesis.choices[0].message.content or "",
        },
        "totals": {
            "workers": len(workers),
            # Workers and the synthesiser together: every leg pays the gateway
            # once, so this is the run's whole gateway bill in milliseconds.
            "time_split": _time_split(
                [*workers, {
                    "gateway_ms": smeta.trace.get("gateway_overhead_ms", 0),
                    "upstream_ms": smeta.trace.get("upstream_ms", 0),
                    "pilot_wait_ms": smeta.trace.get("pilot_wait_ms", 0),
                }]
            ),
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

    # --- scatter -----------------------------------------------------------
    scatter_started = time.perf_counter()
    results = await asyncio.gather(
        *(
            pipeline.handle(
                _fanin_build(body, sub, body.worker_intent, f"{body.session_id}-w{i}"),
                principal,
            )
            for i, sub in enumerate(body.subtasks)
        ),
        return_exceptions=True,
    )
    scatter_ms = int((time.perf_counter() - scatter_started) * 1000)

    workers = [
        _fanin_worker_row(i, sub, result)
        for i, (sub, result) in enumerate(zip(body.subtasks, results, strict=False))
    ]
    findings = [
        f"### {w['subtask']}\n{w['answer']}" for w in workers if "error" not in w
    ]
    if not findings:
        raise GatewayError(502, "every worker failed; nothing to synthesise", code="fanin_failed")

    # --- gather ------------------------------------------------------------
    gather_started = time.perf_counter()
    synthesis = await pipeline.handle(
        _fanin_build(
            body, _fanin_synthesis_prompt(body, findings),
            body.synthesis_intent, f"{body.session_id}-synth",
        ),
        principal,
    )
    gather_ms = int((time.perf_counter() - gather_started) * 1000)
    return _fanin_result(workers, synthesis, scatter_ms, gather_ms, started)


@router.post("/fanin/live")
async def fanin_live(body: FaninRequest, request: Request):
    """The scatter/gather run as SSE: every leg's pipeline stages, live.

    Workers stream tagged ``worker: 1..N``; the synthesiser streams tagged
    ``worker: "synth"`` — it is a routed request like any other, and watching
    it climb the tiers after the workers finish *is* the demo. Ends with the
    same summary ``/demo/fanin`` returns.
    """
    app = request.app
    principal = await app.state.auth.authenticate(request)
    pipeline = app.state.pipeline
    queue: asyncio.Queue = asyncio.Queue()

    async def one(i: int, sub: str, run_started: float) -> dict:
        try:
            result = await pipeline.handle(
                _fanin_build(body, sub, body.worker_intent, f"{body.session_id}-w{i}"),
                principal,
                emit=_live_emitter(queue, i + 1, run_started),
            )
        except BaseException as exc:  # noqa: BLE001 - row carries the error
            result = exc
        row = _fanin_worker_row(i, sub, result)
        await queue.put({"event": "worker_done", "worker": i + 1, "agent": row})
        return row

    tasks: list[asyncio.Task] = []

    async def events():
        started = time.perf_counter()
        scatter = asyncio.ensure_future(
            asyncio.gather(*(one(i, sub, started) for i, sub in enumerate(body.subtasks)))
        )
        tasks.append(scatter)
        async for ev in _pump(queue, scatter):
            yield ev
        if not scatter.done() or scatter.cancelled():
            return
        workers = scatter.result()
        scatter_ms = int((time.perf_counter() - started) * 1000)

        findings = [
            f"### {w['subtask']}\n{w['answer']}" for w in workers if "error" not in w
        ]
        if not findings:
            yield {"event": "error", "error": "every worker failed; nothing to synthesise"}
            return

        gather_started = time.perf_counter()
        synth = asyncio.ensure_future(
            pipeline.handle(
                _fanin_build(
                    body, _fanin_synthesis_prompt(body, findings),
                    body.synthesis_intent, f"{body.session_id}-synth",
                ),
                principal,
                emit=_live_emitter(queue, "synth", started),
            )
        )
        tasks.append(synth)
        async for ev in _pump(queue, synth):
            yield ev
        if not synth.done() or synth.cancelled():
            return
        try:
            synthesis = synth.result()
        except GatewayError as exc:
            yield {"event": "error", "error": str(exc.detail)}
            return
        gather_ms = int((time.perf_counter() - gather_started) * 1000)
        yield {
            "event": "summary",
            **_fanin_result(workers, synthesis, scatter_ms, gather_ms, started),
        }

    async def generate():
        try:
            async for ev in events():
                yield _sse(ev)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


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
