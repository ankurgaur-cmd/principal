"""Authentication and tenant resolution.

Provider keys live in the gateway and nowhere else — that is most of the point
of having a gateway. Agents authenticate to *us*, we authenticate to vendors.

``dev`` mode trusts request headers and exists so you can curl the thing on a
laptop. It is unsafe anywhere a caller you do not control can reach it, and it
logs a warning on every start so nobody ships it by accident.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import Request

from .config import Settings
from .errors import AuthError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    agent_id: str
    scopes: frozenset[str] = frozenset()

    def may_pin_model(self) -> bool:
        return "model:pin" in self.scopes or not self.scopes


class Authenticator:
    def __init__(self, settings: Settings):
        self._s = settings
        if settings.auth_mode == "dev":
            log.warning(
                "AUTH_MODE=dev — tenant identity is taken from request headers. "
                "Do not expose this gateway beyond localhost."
            )

    async def authenticate(self, request: Request) -> Principal:
        if self._s.auth_mode == "dev":
            return Principal(
                tenant_id=request.headers.get("x-tenant-id", "dev-tenant"),
                agent_id=request.headers.get("x-agent-id", "dev-agent"),
            )

        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise AuthError("expected 'Authorization: Bearer <jwt>'")

        import jwt

        try:
            claims = jwt.decode(
                header[7:], self._s.jwt_secret, algorithms=[self._s.jwt_algorithm]
            )
        except jwt.ExpiredSignatureError:
            raise AuthError("token expired") from None
        except jwt.InvalidTokenError as exc:
            raise AuthError(f"invalid token: {exc}") from None

        tenant = claims.get("tenant_id")
        if not tenant:
            raise AuthError("token is missing the tenant_id claim")

        return Principal(
            tenant_id=tenant,
            agent_id=claims.get("agent_id", "unknown"),
            scopes=frozenset(claims.get("scopes", [])),
        )
