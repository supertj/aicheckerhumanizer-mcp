"""Client for the AI Checker Humanizer REST API."""

from __future__ import annotations

from enum import Enum
from functools import partial
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import httpx
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, Field, SecretStr

from ._version import __version__

_API_KEY_ENV = "AICHECKERHUMANIZER_API_KEY"
_REQUEST_TIMEOUT_MESSAGE = "The request to AI Checker Humanizer timed out."


class HumanizeMode(str, Enum):
    """Writing style for a rewrite."""

    STANDARD = "standard"
    ACADEMIC = "academic"
    CASUAL = "casual"
    CONCISE = "concise"


class HumanizeStrength(str, Enum):
    """How extensively to rewrite the text."""

    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class AICheckerHumanizerError(Exception):
    """An API or transport error. Status is None if no response was received."""

    def __init__(
        self,
        *,
        status: int | None,
        code: str,
        message: str,
        retry_after: int | None = None,
        credits_remaining: int | None = None,
        credits_needed: int | None = None,
        max_words: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after
        self.credits_remaining = credits_remaining
        self.credits_needed = credits_needed
        self.max_words = max_words

    def __reduce__(
        self,
    ) -> tuple[partial[AICheckerHumanizerError], tuple[()], dict[str, Any]]:
        constructor = partial(
            type(self), status=self.status, code=self.code, message=self.message
        )
        return constructor, (), self.__dict__


def _seconds_header(response: httpx.Response) -> int | None:
    try:
        seconds = int(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _integer(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    return value if type(value) is int else None


def _submission_message(message: str, submission_key: str | None) -> str:
    if submission_key is None:
        return message
    return (
        f"{message} Retry with the same text and options. "
        f"Set idempotency_key to {submission_key!r}."
    )


def _parse_response(
    response: httpx.Response, *, is_task: bool, submission_key: str | None
) -> dict[str, Any]:
    has_json = True
    try:
        data = response.json()
    except ValueError:
        has_json = False
        data = None

    if not response.is_success:
        error = data if isinstance(data, dict) else {}
        code = error.get("error")
        message = error.get("message")
        if not isinstance(message, str) or not message:
            message = f"AI Checker Humanizer returned HTTP {response.status_code}."
        if response.status_code >= 500 or not has_json:
            message = _submission_message(message, submission_key)
        retry_after = _integer(error, "retryAfter")
        raise AICheckerHumanizerError(
            status=response.status_code,
            code=code if isinstance(code, str) else "http_error",
            message=message,
            retry_after=(
                retry_after if retry_after is not None else _seconds_header(response)
            ),
            credits_remaining=_integer(error, "creditsRemaining"),
            credits_needed=_integer(error, "creditsNeeded"),
            max_words=_integer(error, "maxWords"),
        )

    if not isinstance(data, dict) or (
        is_task
        and (
            not isinstance(data.get("status"), str)
            or not isinstance(data.get("taskId"), str)
        )
    ):
        raise AICheckerHumanizerError(
            status=response.status_code,
            code="invalid_response",
            message=_submission_message(
                "AI Checker Humanizer returned an invalid response.", submission_key
            ),
        )

    if is_task:
        retry_after = _seconds_header(response)
        if retry_after is not None and _integer(data, "retryAfter") is None:
            data["retryAfter"] = retry_after
    return data


def _request_error(
    exc: httpx.RequestError | httpx.InvalidURL, *, submission_key: str | None = None
) -> AICheckerHumanizerError:
    if isinstance(exc, httpx.TimeoutException):
        code = "timeout"
        message = _REQUEST_TIMEOUT_MESSAGE
    else:
        code = "request_error"
        message = "The request to AI Checker Humanizer could not be sent or completed."
    return AICheckerHumanizerError(
        status=None, code=code, message=_submission_message(message, submission_key)
    )


class AICheckerHumanizerAPIWrapper(BaseModel):
    """Call the API with an explicit key or AICHECKERHUMANIZER_API_KEY."""

    api_key: SecretStr = Field(
        default_factory=secret_from_env(
            _API_KEY_ENV,
            error_message=(
                "Set AICHECKERHUMANIZER_API_KEY or pass api_key to "
                "AICheckerHumanizerAPIWrapper."
            ),
        )
    )
    base_url: str = "https://aicheckerhumanizer.com/api/v1"
    timeout: float = Field(default=30, gt=0, allow_inf_nan=False)

    def detect(
        self, text: str, *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Check text for AI writing and return the detection response."""
        return self._request(
            "POST", "/detect", data={"text": text}, idempotency_key=idempotency_key
        )

    async def adetect(
        self, text: str, *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Check text for AI writing asynchronously."""
        return await self._arequest(
            "POST", "/detect", data={"text": text}, idempotency_key=idempotency_key
        )

    def humanize_start(
        self,
        text: str,
        *,
        mode: HumanizeMode | str = HumanizeMode.STANDARD,
        strength: HumanizeStrength | str = HumanizeStrength.MEDIUM,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Submit text for rewriting and return the task."""
        return self._request(
            "POST",
            "/humanize",
            data={"text": text, "mode": mode, "strength": strength},
            idempotency_key=(
                str(uuid4()) if idempotency_key is None else idempotency_key
            ),
        )

    async def ahumanize_start(
        self,
        text: str,
        *,
        mode: HumanizeMode | str = HumanizeMode.STANDARD,
        strength: HumanizeStrength | str = HumanizeStrength.MEDIUM,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Submit text for rewriting asynchronously."""
        return await self._arequest(
            "POST",
            "/humanize",
            data={"text": text, "mode": mode, "strength": strength},
            idempotency_key=(
                str(uuid4()) if idempotency_key is None else idempotency_key
            ),
        )

    def humanize_get(
        self, task_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Fetch a rewrite task and any available output."""
        return self._request(
            "GET", f"/humanize/{quote(task_id, safe='')}", timeout=timeout
        )

    async def ahumanize_get(
        self, task_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Fetch a rewrite task asynchronously."""
        return await self._arequest(
            "GET", f"/humanize/{quote(task_id, safe='')}", timeout=timeout
        )

    def balance(self) -> dict[str, Any]:
        """Fetch the remaining credits and expiration dates."""
        return self._request("GET", "/balance")

    async def abalance(self) -> dict[str, Any]:
        """Fetch the remaining credits asynchronously."""
        return await self._arequest("GET", "/balance")

    def _headers(self, idempotency_key: str | None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "User-Agent": f"langchain-aicheckerhumanizer/{__version__}",
        }
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        submission_key = idempotency_key if path == "/humanize" else None
        try:
            with httpx.Client(
                timeout=self.timeout if timeout is None else timeout
            ) as client:
                response = client.request(
                    method,
                    self.base_url.rstrip("/") + path,
                    headers=self._headers(idempotency_key),
                    json=data,
                )
        except (httpx.RequestError, httpx.InvalidURL) as exc:
            raise _request_error(exc, submission_key=submission_key) from exc
        return _parse_response(
            response,
            is_task=path.startswith("/humanize"),
            submission_key=submission_key,
        )

    async def _arequest(
        self,
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        submission_key = idempotency_key if path == "/humanize" else None
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout if timeout is None else timeout
            ) as client:
                response = await client.request(
                    method,
                    self.base_url.rstrip("/") + path,
                    headers=self._headers(idempotency_key),
                    json=data,
                )
        except (httpx.RequestError, httpx.InvalidURL) as exc:
            raise _request_error(exc, submission_key=submission_key) from exc
        return _parse_response(
            response,
            is_task=path.startswith("/humanize"),
            submission_key=submission_key,
        )
