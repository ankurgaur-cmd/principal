"""Anthropic adapter.

Vendor specifics this absorbs so callers never see them:

* **Sampling parameters are rejected** on the current frontier models —
  ``temperature``/``top_p``/``top_k`` return a 400 rather than being ignored.
  We drop them and note the drop, instead of surfacing a confusing vendor error
  to an agent that sent a perfectly ordinary OpenAI-shaped request.
* **``cache_control`` is explicit.** The cache plan is compiled into markers
  here; the OpenAI adapter ignores the same plan entirely.
* **Refusals arrive as HTTP 200** with ``stop_reason == "refusal"`` and a
  possibly-empty ``content`` array. Reading ``content[0]`` unconditionally
  crashes. We check ``stop_reason`` first, always.
* **``effort`` and ``format`` both live under ``output_config``**, and effort is
  unsupported on older small models — so it is gated on the catalog capability
  rather than sent blindly.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..cache.hints import CachePlan
from ..catalog import Capability, get_model
from ..errors import ProviderRefusal, UpstreamError
from ..schemas import CanonicalRequest, ProviderResponse, Usage

log = logging.getLogger(__name__)

# Above this, a non-streaming request risks an SDK HTTP timeout. We stream
# internally and hand the caller a unary response.
_STREAM_THRESHOLD = 16_000


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else (
            anthropic.AsyncAnthropic()
        )

    @property
    def host(self) -> str:
        """The server this adapter actually talks to (honours base_url overrides)."""
        return str(getattr(self._client, "base_url", "https://api.anthropic.com")).rstrip("/")

    # -- request compilation ------------------------------------------------
    def _system_blocks(self, canonical: CanonicalRequest, plan: CachePlan) -> list[dict]:
        blocks = [{"type": "text", "text": s} for s in canonical.system if s]
        if not blocks:
            return []
        if plan.enabled and "system" in plan.breakpoints:
            # The marker goes on the *last* system block. Render order is
            # tools -> system -> messages, so one marker here covers tools too.
            ttl = {"type": "ephemeral"}
            if plan.ttl == "1h":
                ttl["ttl"] = "1h"
            blocks[-1]["cache_control"] = ttl
        return blocks

    def _tools(self, canonical: CanonicalRequest, plan: CachePlan) -> list[dict]:
        tools = []
        for tool in canonical.sorted_tools():  # stable order = stable cache prefix
            spec: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters or {"type": "object", "properties": {}},
            }
            if tool.strict:
                spec["strict"] = True
            tools.append(spec)
        if tools and plan.enabled and "tools" in plan.breakpoints:
            tools[-1]["cache_control"] = {"type": "ephemeral"}
        return tools

    def _messages(self, canonical: CanonicalRequest, plan: CachePlan) -> list[dict]:
        out: list[dict] = []
        for msg in canonical.messages:
            if msg.role == "system":
                continue  # hoisted into the system field
            if msg.role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": str(msg.content or ""),
                            }
                        ],
                    }
                )
                continue

            content: Any = msg.content
            if msg.role == "assistant" and msg.tool_calls:
                blocks: list[dict] = []
                if isinstance(content, str) and content:
                    blocks.append({"type": "text", "text": content})
                for call in msg.tool_calls:
                    fn = call.get("function", {})
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": args,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
                continue

            out.append({"role": msg.role, "content": content if content is not None else ""})

        wants_message_marker = "history" in plan.breakpoints or "last_turn" in plan.breakpoints
        if out and plan.enabled and wants_message_marker:
            last = out[-1]
            if isinstance(last.get("content"), str):
                last["content"] = [{"type": "text", "text": last["content"]}]
            if isinstance(last.get("content"), list) and last["content"]:
                last["content"][-1]["cache_control"] = {"type": "ephemeral"}
        return out

    def _build(
        self, canonical: CanonicalRequest, model_key: str, effort: str, plan: CachePlan
    ) -> tuple[dict[str, Any], list[str]]:
        spec = get_model(model_key)
        assert spec is not None
        dropped: list[str] = []

        params: dict[str, Any] = {
            "model": spec.vendor_model_id,
            "max_tokens": min(canonical.max_tokens, spec.max_output_tokens),
            "messages": self._messages(canonical, plan),
        }

        if system := self._system_blocks(canonical, plan):
            params["system"] = system
        if tools := self._tools(canonical, plan):
            params["tools"] = tools
            if canonical.tool_choice and canonical.tool_choice != "auto":
                params["tool_choice"] = _tool_choice(canonical.tool_choice)

        # Sampling params: silently rejected upstream on current models.
        if spec.supports_sampling_params:
            if canonical.temperature is not None:
                params["temperature"] = canonical.temperature
            if canonical.top_p is not None:
                params["top_p"] = canonical.top_p
        else:
            for field in ("temperature", "top_p"):
                if getattr(canonical, field) is not None:
                    dropped.append(field)

        output_config: dict[str, Any] = {}
        if spec.supports(Capability.EXTENDED_THINKING):
            # Adaptive thinking is the only supported on-mode on these models;
            # the fixed-budget form is a 400. Effort is what controls depth.
            params["thinking"] = {"type": "adaptive"}
            output_config["effort"] = effort
        elif effort not in (None, "medium"):
            dropped.append(f"effort={effort} (unsupported on {model_key})")

        if canonical.response_schema:
            output_config["format"] = {
                "type": "json_schema",
                "schema": canonical.response_schema,
            }
        if output_config:
            params["output_config"] = output_config

        params.update(canonical.vendor_overrides.get("anthropic", {}))
        return params, dropped

    # -- invocation ---------------------------------------------------------
    async def invoke(
        self,
        canonical: CanonicalRequest,
        model_key: str,
        effort: str,
        cache_plan: CachePlan,
    ) -> ProviderResponse:
        params, dropped = self._build(canonical, model_key, effort, cache_plan)
        if dropped:
            log.info("anthropic: dropped unsupported params %s for %s", dropped, model_key)

        try:
            if params["max_tokens"] > _STREAM_THRESHOLD:
                async with self._client.messages.stream(**params) as stream:
                    message = await stream.get_final_message()
            else:
                message = await self._client.messages.create(**params)
        except Exception as exc:
            raise _translate(exc) from exc

        return self._to_response(message, model_key)

    def _to_response(self, message, model_key: str) -> ProviderResponse:
        # Check stop_reason BEFORE touching content: on a refusal, content may
        # be empty (pre-output) or a partial (mid-stream).
        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ProviderRefusal(
                "provider declined this request on policy grounds", category=category
            )

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in message.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )
            # thinking blocks carry no text under the default display setting

        u = message.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        usage = Usage(
            # input_tokens is the *uncached remainder* only — the full prompt is
            # the sum of all three. Reporting it alone understates prompt size.
            prompt_tokens=u.input_tokens + cache_read + cache_write,
            completion_tokens=u.output_tokens,
            total_tokens=u.input_tokens + cache_read + cache_write + u.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )

        finish = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
        }.get(message.stop_reason or "end_turn", "stop")

        return ProviderResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=finish,
            model=model_key,
            usage=usage,
            raw_stop_reason=message.stop_reason,
        )

    async def stream(
        self,
        canonical: CanonicalRequest,
        model_key: str,
        effort: str,
        cache_plan: CachePlan,
    ) -> AsyncIterator[dict[str, Any]]:
        params, _ = self._build(canonical, model_key, effort, cache_plan)
        try:
            async with self._client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield {"delta": {"content": text}, "finish_reason": None}
                final = await stream.get_final_message()
        except Exception as exc:
            raise _translate(exc) from exc

        if getattr(final, "stop_reason", None) == "refusal":
            yield {"delta": {}, "finish_reason": "content_filter"}
            return

        u = final.usage
        yield {
            "delta": {},
            "finish_reason": "stop",
            "usage": Usage(
                prompt_tokens=u.input_tokens
                + (getattr(u, "cache_read_input_tokens", 0) or 0)
                + (getattr(u, "cache_creation_input_tokens", 0) or 0),
                completion_tokens=u.output_tokens,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            ),
        }

    async def classify(
        self, model_key: str, system: str, text: str, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        spec = get_model(model_key)
        assert spec is not None
        params: dict[str, Any] = {
            "model": spec.vendor_model_id,
            "max_tokens": 256,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": text}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        message = await self._client.messages.create(**params)
        if getattr(message, "stop_reason", None) == "refusal":
            return None
        for block in message.content:
            if getattr(block, "type", None) == "text":
                try:
                    return json.loads(block.text)
                except json.JSONDecodeError:
                    return None
        return None

    async def validate(self) -> tuple[bool, str]:
        """Cheap credential check. Uses the free models endpoint — no tokens."""
        try:
            page = await self._client.models.list(limit=1)
            n = len(getattr(page, "data", []) or [])
            return True, f"authenticated ({n or 'some'} model(s) visible)"
        except Exception as exc:
            return False, _describe(exc)

    async def count_tokens(
        self, canonical: CanonicalRequest, model_key: str
    ) -> int | None:
        spec = get_model(model_key)
        assert spec is not None
        plan = CachePlan()
        params, _ = self._build(canonical, model_key, "medium", plan)
        try:
            result = await self._client.messages.count_tokens(
                model=params["model"],
                messages=params["messages"],
                system=params.get("system"),
                tools=params.get("tools"),
            )
            return result.input_tokens
        except Exception as exc:
            log.warning("count_tokens failed for %s: %s", model_key, exc)
            return None


def _tool_choice(choice: Any) -> dict[str, Any]:
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "tool", "name": choice["function"]["name"]}
    return {"type": "auto"}


def _describe(exc: Exception) -> str:
    """A human-readable reason a credential check failed.

    Deliberately does not include the key or the raw response body.
    """
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return "key rejected (401) — check it was copied in full"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "key is valid but lacks permission (403)"
    if isinstance(exc, anthropic.RateLimitError):
        return "key is valid but currently rate limited (429)"
    if isinstance(exc, anthropic.APIConnectionError):
        return "could not reach the Anthropic API — check network access"
    if isinstance(exc, anthropic.APIStatusError):
        return f"API returned {exc.status_code}"
    return type(exc).__name__


def _translate(exc: Exception) -> Exception:
    """Map SDK exceptions onto gateway errors, preserving retryability."""
    import anthropic

    if isinstance(exc, anthropic.RateLimitError):
        from ..errors import RateLimited

        retry_after = 60
        try:
            retry_after = int(exc.response.headers.get("retry-after", "60"))
        except Exception:
            pass
        return RateLimited("upstream rate limit", retry_after=retry_after)
    if isinstance(exc, anthropic.AuthenticationError):
        return UpstreamError("gateway is misconfigured: bad Anthropic credentials", 500)
    if isinstance(exc, anthropic.APIStatusError):
        return UpstreamError(f"anthropic {exc.status_code}: {exc.message}", 502)
    if isinstance(exc, anthropic.APIConnectionError):
        return UpstreamError("could not reach Anthropic", 503)
    return exc
