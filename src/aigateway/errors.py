"""Gateway error types, rendered in the OpenAI error envelope."""

from __future__ import annotations

from fastapi import HTTPException


class GatewayError(HTTPException):
    error_type = "gateway_error"

    def __init__(self, status_code: int, message: str, code: str | None = None):
        super().__init__(
            status_code=status_code,
            detail={
                "error": {
                    "message": message,
                    "type": self.error_type,
                    "code": code or self.error_type,
                }
            },
        )


class AuthError(GatewayError):
    error_type = "authentication_error"

    def __init__(self, message: str = "missing or invalid credentials"):
        super().__init__(401, message)


class BudgetExceeded(GatewayError):
    error_type = "budget_exceeded"

    def __init__(self, message: str):
        super().__init__(402, message)


class RateLimited(GatewayError):
    error_type = "rate_limit_error"

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(429, message)
        self.headers = {"retry-after": str(retry_after)}


class NoCapableModel(GatewayError):
    error_type = "no_capable_model"

    def __init__(self, message: str):
        super().__init__(422, message)


class UpstreamError(GatewayError):
    error_type = "upstream_error"

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(status_code, message)


class ProviderRefusal(GatewayError):
    """A provider declined the request on policy grounds.

    Anthropic surfaces this as HTTP 200 with ``stop_reason == "refusal"``. We
    normalise it to a first-class response rather than letting callers trip over
    an empty ``content`` array.
    """

    error_type = "content_policy_refusal"

    def __init__(self, message: str, category: str | None = None):
        super().__init__(403, message, code=category or "refusal")
