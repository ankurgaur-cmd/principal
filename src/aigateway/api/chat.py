"""OpenAI-compatible chat completions.

Agents point ``base_url`` here and change nothing else. Gateway-specific hints
ride in an ``x_gateway`` object, which vendors never see and OpenAI clients
happily ignore.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..schemas import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest, request: Request
) -> ChatCompletionResponse | StreamingResponse:
    app = request.app
    principal = await app.state.auth.authenticate(request)

    if body.stream:
        generator = await app.state.pipeline.stream(body, principal)
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    return await app.state.pipeline.handle(body, principal)
