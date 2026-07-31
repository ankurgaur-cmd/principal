from __future__ import annotations

import pytest

from aigateway.config import Settings
from aigateway.routing import Router
from aigateway.schemas import CanonicalRequest, ChatMessage, ToolDef
from aigateway.state import MemoryStore


@pytest.fixture
def settings() -> Settings:
    return Settings(
        redis_url=None,
        cache_aware_routing=True,
        escalate_only=True,
        cache_ttl="5m",
        session_ttl_seconds=300,
    )


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def router(settings: Settings, store: MemoryStore) -> Router:
    # A plain set stands in for the live registry; Router accepts either.
    return Router(settings, store, {"anthropic", "openai"})


def make_request(
    *,
    system_tokens: int = 0,
    user_text: str = "hello",
    tools: int = 0,
    session_id: str | None = None,
    max_tokens: int = 4096,
    **kwargs,
) -> CanonicalRequest:
    """Build a canonical request with an approximately-sized prefix.

    ~3.6 chars/token is the estimator's ratio; tests that care about the
    per-model ``min_cacheable_tokens`` cliffs need to land on the right side
    of it deliberately.
    """
    system = ["x" * int(system_tokens * 3.6)] if system_tokens else []
    return CanonicalRequest(
        system=system,
        messages=[ChatMessage(role="user", content=user_text)],
        tools=[
            ToolDef(name=f"tool_{i}", description="d", parameters={"type": "object"})
            for i in range(tools)
        ],
        max_tokens=max_tokens,
        session_id=session_id,
        **kwargs,
    )
