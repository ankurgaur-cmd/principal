"""Token estimation.

Two tiers deliberately:

* ``estimate_tokens`` — a free, local approximation used for routing and for
  the cheap path of budget pre-flight. It is an estimate and is labelled as one.
* ``count_tokens_exact`` — a real provider round trip, used only when a tenant
  is near their cap. An extra call on every request is not free.

Never reach for ``tiktoken`` on the Anthropic path: it is OpenAI's tokenizer and
undercounts Claude tokens by 15-20% on prose and considerably more on code.
"""

from __future__ import annotations

import json
from typing import Any

# Deliberately conservative: over-estimating input costs a slightly pessimistic
# routing decision; under-estimating blows a budget cap.
_CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_content_tokens(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
                elif part.get("type") in ("image_url", "image"):
                    # Coarse placeholder. High-resolution images can reach
                    # ~4.8k tokens each on current frontier models, so this is
                    # a floor, not a promise.
                    total += 1_500
                else:
                    total += estimate_tokens(json.dumps(part))
            else:
                total += estimate_tokens(str(part))
        return total
    return estimate_tokens(json.dumps(content, sort_keys=True))


def estimate_request_tokens(canonical) -> tuple[int, int]:
    """Return ``(prefix_tokens, volatile_tokens)``.

    ``prefix_tokens`` is the stable, cacheable head of the prompt: tools plus
    system plus all but the final turn. ``volatile_tokens`` is the tail that
    changes every request and can never be cached.
    """
    prefix = 0
    for tool in canonical.sorted_tools():
        prefix += estimate_tokens(tool.name + tool.description)
        prefix += estimate_tokens(json.dumps(tool.parameters, sort_keys=True))
    for block in canonical.system:
        prefix += estimate_tokens(block)

    messages = canonical.messages
    for msg in messages[:-1]:
        prefix += estimate_content_tokens(msg.content)
    volatile = estimate_content_tokens(messages[-1].content) if messages else 0
    return prefix, volatile
