"""OpenAI adapter.

Deliberately thinner than the Anthropic one, for a structural reason: the
inbound gateway schema is already OpenAI-shaped, and this provider caches
prefixes automatically. There is nothing to compile the cache plan into — the
adapter's whole caching responsibility is *not disturbing the prefix*, which
``sorted_tools()`` and the frozen-system-prompt rule already handle upstream.

Model ids and prices in the catalog are config-driven placeholders. Verify them
before trusting the cost ledger for chargeback.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..cache.hints import CachePlan
from ..catalog import get_model
from ..errors import UpstreamError
from ..schemas import CanonicalRequest, ProviderResponse, Usage

log = logging.getLogger(__name__)


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    @property
    def host(self) -> str:
        """The server this adapter actually talks to (honours base_url overrides)."""
        return str(getattr(self._client, "base_url", "https://api.openai.com")).rstrip("/")

    def _build(
        self, canonical: CanonicalRequest, model_key: str, effort: str
    ) -> dict[str, Any]:
        spec = get_model(model_key)
        assert spec is not None

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": s} for s in canonical.system if s
        ]
        for msg in canonical.messages:
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.name:
                entry["name"] = msg.name
            messages.append(entry)

        params: dict[str, Any] = {
            "model": spec.vendor_model_id,
            "messages": messages,
            "max_completion_tokens": min(canonical.max_tokens, spec.max_output_tokens),
        }

        if canonical.tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters or {"type": "object", "properties": {}},
                        **({"strict": True} if t.strict else {}),
                    },
                }
                for t in canonical.sorted_tools()  # stable order = stable prefix
            ]
            if canonical.tool_choice:
                params["tool_choice"] = canonical.tool_choice

        if canonical.response_schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": canonical.response_schema,
                    "strict": True,
                },
            }

        if spec.supports_sampling_params:
            if canonical.temperature is not None:
                params["temperature"] = canonical.temperature
            if canonical.top_p is not None:
                params["top_p"] = canonical.top_p

        # Neutral effort -> vendor reasoning effort. The neutral vocabulary is
        # wider than this provider's, so the top two levels collapse.
        params["reasoning_effort"] = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
            "max": "high",
        }.get(effort, "medium")

        params.update(canonical.vendor_overrides.get("openai", {}))
        return params

    async def invoke(
        self,
        canonical: CanonicalRequest,
        model_key: str,
        effort: str,
        cache_plan: CachePlan,
    ) -> ProviderResponse:
        params = self._build(canonical, model_key, effort)
        try:
            completion = await self._client.chat.completions.create(**params)
        except Exception as exc:
            raise _translate(exc) from exc

        choice = completion.choices[0]
        message = choice.message

        tool_calls = []
        for call in message.tool_calls or []:
            tool_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )

        u = completion.usage
        cached = 0
        if u and getattr(u, "prompt_tokens_details", None):
            cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0

        return ProviderResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            model=model_key,
            usage=Usage(
                prompt_tokens=u.prompt_tokens if u else 0,
                completion_tokens=u.completion_tokens if u else 0,
                total_tokens=u.total_tokens if u else 0,
                cache_read_tokens=cached,
                # Automatic caching has no separate write step to bill for.
                cache_write_tokens=0,
            ),
            raw_stop_reason=choice.finish_reason,
        )

    async def stream(
        self,
        canonical: CanonicalRequest,
        model_key: str,
        effort: str,
        cache_plan: CachePlan,
    ) -> AsyncIterator[dict[str, Any]]:
        params = self._build(canonical, model_key, effort)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        try:
            stream = await self._client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.usage:
                    cached = 0
                    if getattr(chunk.usage, "prompt_tokens_details", None):
                        cached = getattr(
                            chunk.usage.prompt_tokens_details, "cached_tokens", 0
                        ) or 0
                    yield {
                        "delta": {},
                        "finish_reason": "stop",
                        "usage": Usage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            cache_read_tokens=cached,
                        ),
                    }
                    continue
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield {"delta": {"content": delta.content}, "finish_reason": None}
        except Exception as exc:
            raise _translate(exc) from exc

    async def classify(
        self, model_key: str, system: str, text: str, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        spec = get_model(model_key)
        assert spec is not None
        completion = await self._client.chat.completions.create(
            model=spec.vendor_model_id,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
            max_completion_tokens=256,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "intent", "schema": schema, "strict": True},
            },
        )
        content = completion.choices[0].message.content
        try:
            return json.loads(content) if content else None
        except json.JSONDecodeError:
            return None

    async def validate(self) -> tuple[bool, str]:
        """Cheap credential check. Uses the free models endpoint — no tokens."""
        import openai

        try:
            page = await self._client.models.list()
            n = len(getattr(page, "data", []) or [])
            return True, f"authenticated ({n or 'some'} model(s) visible)"
        except openai.AuthenticationError:
            return False, "key rejected (401) — check it was copied in full"
        except openai.PermissionDeniedError:
            return False, "key is valid but lacks permission (403)"
        except openai.RateLimitError:
            return False, "key is valid but currently rate limited (429)"
        except openai.APIConnectionError:
            return False, "could not reach the OpenAI API — check network access"
        except Exception as exc:
            return False, type(exc).__name__

    async def count_tokens(
        self, canonical: CanonicalRequest, model_key: str
    ) -> int | None:
        # No pre-flight counting endpoint. The caller falls back to the local
        # estimate — and must not substitute tiktoken on the Anthropic path.
        return None


def _translate(exc: Exception) -> Exception:
    import openai

    if isinstance(exc, openai.RateLimitError):
        from ..errors import RateLimited

        return RateLimited("upstream rate limit")
    if isinstance(exc, openai.AuthenticationError):
        return UpstreamError("gateway is misconfigured: bad OpenAI credentials", 500)
    if isinstance(exc, openai.APIStatusError):
        return UpstreamError(f"openai {exc.status_code}", 502)
    if isinstance(exc, openai.APIConnectionError):
        return UpstreamError("could not reach OpenAI", 503)
    return exc
