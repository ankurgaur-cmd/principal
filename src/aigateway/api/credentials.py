"""Runtime credential management.

Lets you paste a provider key into the console and start making live calls
without restarting. Convenience for a local demo, so the safety rules are
strict and non-negotiable:

* **Keys are never returned.** ``GET`` reports a mask (``sk-…a1b2``) and nothing
  else. There is no endpoint that reads a key back out.
* **Keys are never logged.** Only the mask reaches the log line.
* **Keys stay in memory** by default — held on the provider client object, not
  written anywhere. Persisting to ``.env`` is a separate, explicit opt-in.
* **Dev mode only.** These endpoints are disabled unless
  ``GATEWAY_AUTH_MODE=dev``, which is itself localhost-only. A gateway
  reachable by anyone else must get its keys from the environment.
* **Validation costs nothing.** The check calls each vendor's free models
  endpoint rather than burning tokens on a hello-world completion.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..errors import GatewayError
from ..providers.registry import BUILDERS, CREDENTIAL_KINDS, mask

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/credentials", tags=["credentials"])


class CredentialUpdate(BaseModel):
    provider: str = Field(..., description="anthropic | openai")
    api_key: str = Field("", description="Omit when use_ambient is true.")
    kind: str = Field(
        "auto",
        description="auto | api_key | oauth_token. 'auto' detects a "
        "subscription OAuth token by its sk-ant-oat prefix; anything else is "
        "treated as an API key.",
    )
    persist: bool = Field(
        False,
        description="Also write to .env so the key survives a restart. Off by "
        "default: writing a secret to disk should be a deliberate act.",
    )
    use_ambient: bool = Field(
        False,
        description="Use whatever credentials the SDK can already resolve "
        "(env vars, or an OAuth profile from `ant auth login`) instead of a "
        "pasted key. Nothing is stored by the gateway.",
    )


def _guard(request: Request) -> None:
    """Refuse outside dev mode.

    Runtime credential injection is a local-development affordance. Anywhere
    else, keys belong in the environment or a secrets manager.
    """
    if request.app.state.settings.auth_mode != "dev":
        raise GatewayError(
            403,
            "runtime credential management is dev-mode only; set provider keys "
            "via environment variables instead",
            code="dev_mode_only",
        )


@router.get("")
async def status(request: Request) -> dict:
    """Masked credential status. Contains no key material."""
    registry = request.app.state.registry
    return {
        "auth_mode": request.app.state.settings.auth_mode,
        "editable": request.app.state.settings.auth_mode == "dev",
        "providers": registry.status(),
    }


@router.post("")
async def set_credentials(body: CredentialUpdate, request: Request) -> dict:
    _guard(request)
    registry = request.app.state.registry

    provider = body.provider.strip().lower()
    if provider not in BUILDERS:
        raise GatewayError(422, f"unknown provider '{provider}'", code="unknown_provider")

    api_key = body.api_key.strip()
    if not body.use_ambient and len(api_key) < 8:
        raise GatewayError(422, "api_key is required unless use_ambient is set", code="no_key")

    # Detect the credential shape rather than making the user know the
    # taxonomy: subscription OAuth tokens carry a distinctive prefix.
    kind = body.kind
    if kind == "auto":
        kind = "oauth_token" if api_key.startswith("sk-ant-oat") else "api_key"
    if kind not in CREDENTIAL_KINDS.get(provider, ()):
        raise GatewayError(
            422,
            f"{provider} does not accept {kind.replace('_', ' ')}s — a ChatGPT "
            f"subscription is not API access; create an API key at "
            f"platform.openai.com instead"
            if provider == "openai"
            else f"{provider} does not accept '{kind}' credentials",
            code="unsupported_credential_kind",
        )

    # Swap first, then validate through the real client — so what we test is
    # exactly what will serve traffic, not a separate throwaway client.
    previous_ok = provider in registry.enabled
    if body.use_ambient:
        registry.use_ambient(provider)
    else:
        registry.set_credentials(provider, api_key, source="console", kind=kind)

    ok, detail = await registry.get(provider).validate()
    if not ok:
        registry.clear_credentials(provider)
        if previous_ok:
            log.warning("credential check failed for %s; previous key removed", provider)
        raise GatewayError(400, f"{provider}: {detail}", code="invalid_credentials")

    persisted = False
    if body.persist and not body.use_ambient:
        persisted = _persist_to_env(provider, api_key, kind)

    return {
        "provider": provider,
        "configured": True,
        "kind": "ambient" if body.use_ambient else kind,
        "masked_key": "(ambient)" if body.use_ambient else mask(api_key),
        "validated": detail,
        "persisted_to_env": persisted,
    }


@router.delete("/{provider}")
async def clear(provider: str, request: Request) -> dict:
    _guard(request)
    request.app.state.registry.clear_credentials(provider.lower())
    return {"provider": provider.lower(), "configured": False}


ENV_VAR = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
# A subscription token is a different credential and gets its own variable —
# writing it into the API-key slot would make the SDK send it the wrong way.
ENV_VAR_OAUTH = {"anthropic": "ANTHROPIC_AUTH_TOKEN"}


def _persist_to_env(provider: str, api_key: str, kind: str = "api_key") -> bool:
    """Upsert the key into ./.env (which .gitignore already excludes).

    Opt-in only. Returns False rather than raising if the file cannot be
    written — failing to persist must not fail an otherwise-valid key.
    """
    var = (ENV_VAR_OAUTH if kind == "oauth_token" else ENV_VAR)[provider]
    path = Path(".env")
    try:
        lines = path.read_text().splitlines() if path.exists() else []
        pattern = re.compile(rf"^\s*{re.escape(var)}\s*=")
        replaced = False
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = f"{var}={api_key}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{var}={api_key}")
        path.write_text("\n".join(lines) + "\n")
        # Deliberately not logging the value, only that it happened.
        log.info("persisted %s to .env", var)
        return True
    except OSError as exc:
        log.warning("could not persist %s to .env: %s", var, exc)
        return False
