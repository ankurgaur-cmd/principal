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
from pydantic import BaseModel

from ..errors import GatewayError
from ..schemas import ChatCompletionRequest, ChatMessage, GatewayExtensions

router = APIRouter(prefix="/demo", tags=["demo"])


class FanoutRequest(BaseModel):
    prompt: str
    shared_context: str = ""
    agents: int = 4
    session_id: str = "fanout"
    intent: str | None = None
    pilot_enabled: bool = True


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

    def build(index: int) -> ChatCompletionRequest:
        messages = []
        if body.shared_context:
            messages.append(ChatMessage(role="system", content=body.shared_context))
        messages.append(
            ChatMessage(role="user", content=f"{body.prompt}\n\n(sub-agent #{index + 1})")
        )
        return ChatCompletionRequest(
            model="auto",
            messages=messages,
            max_tokens=300,
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
            agents.append({"agent": index + 1, "error": str(result.detail)})
            continue
        if isinstance(result, BaseException):
            agents.append({"agent": index + 1, "error": repr(result)})
            continue
        meta = result.x_gateway
        total_cost += meta.actual_cost_usd
        total_read += meta.cache_read_tokens
        total_write += meta.cache_write_tokens
        agents.append(
            {
                "agent": index + 1,
                "model": meta.chosen_model,
                "pilot_role": meta.pilot_role,
                "cache_read_tokens": meta.cache_read_tokens,
                "cache_write_tokens": meta.cache_write_tokens,
                "cost_usd": meta.actual_cost_usd,
                "latency_ms": meta.latency_ms,
            }
        )

    return {
        "agents": agents,
        "pilot_enabled": body.pilot_enabled,
        "wall_clock_ms": int((time.perf_counter() - started) * 1000),
        "total_cost_usd": round(total_cost, 6),
        "total_cache_read_tokens": total_read,
        "total_cache_write_tokens": total_write,
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
