"""Client for the AI Checker Humanizer REST API."""

from __future__ import annotations

import re
from contextlib import nullcontext
from enum import Enum
from functools import partial
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import httpx
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from ._version import __version__

_API_KEY_ENV = "AICHECKERHUMANIZER_API_KEY"
_REQUEST_TIMEOUT_MESSAGE = "The request to AI Checker Humanizer timed out."
_IDEMPOTENCY_KEY_PATTERN = r"^[\x21-\x7e]+$"
_IDEMPOTENCY_KEY_MAX_LENGTH = 128
_TERMINAL_STATUSES = frozenset(
    {"completed", "partial", "unchanged", "failed", "cancelled"}
)
_NEW_KEY_GUIDANCE = (
    "To retry, send the text with a new idempotency key, or without one."
)


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
    """An API or transport error.

    Status is None when the request fails before HTTPX returns a response.
    Resubmitting an expired rewrite reports status 410.
    """

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
        idempotency_key: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after
        self.credits_remaining = credits_remaining
        self.credits_needed = credits_needed
        self.max_words = max_words
        self.idempotency_key = idempotency_key
        self.body = body if body is not None else {}

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


def _submission_key(key: str | None) -> str:
    if key is None:
        return str(uuid4())
    if len(key) > _IDEMPOTENCY_KEY_MAX_LENGTH or not re.fullmatch(
        _IDEMPOTENCY_KEY_PATTERN, key
    ):
        raise ValueError("idempotency_key must be 1–128 visible ASCII characters.")
    return key


def _submission_message(message: str, submission_key: str | None) -> str:
    if submission_key is None:
        return message
    return (
        f"{message} Retry with the same input. "
        f"Set idempotency_key to {submission_key!r}."
    )


def _valid_response(data: dict[str, Any], path: str) -> bool:
    if path == "/detect":
        return "score" in data and (
            data["score"] is None or _integer(data, "score") is not None
        )
    if path.startswith("/humanize"):
        task_id = data.get("taskId")
        return (
            isinstance(task_id, str)
            and bool(task_id)
            and (path == "/humanize" or isinstance(data.get("status"), str))
        )
    return True


def _add_terminal_retry_guidance(task: dict[str, Any]) -> None:
    status = task.get("status")
    if not isinstance(status, str) or status not in _TERMINAL_STATUSES:
        return
    not_humanized = task.get("reason") == "not_humanized"
    if not not_humanized and status not in ("failed", "cancelled"):
        return

    task.pop("idempotencyKey", None)
    task.pop("idempotency_key", None)
    message = task.get("message")
    if not isinstance(message, str) or not message:
        message = "No credits were charged." if not_humanized else ""
    if not message.endswith(_NEW_KEY_GUIDANCE):
        task["message"] = f"{message} {_NEW_KEY_GUIDANCE}".strip()


def _parse_response(
    response: httpx.Response, *, path: str, submission_key: str | None
) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        data = None

    status = response.status_code
    if (
        path == "/humanize"
        and status == 202
        and isinstance(data, dict)
        and data.get("error") == "result_expired"
    ):
        status = 410

    if not 200 <= status < 300:
        error = data if isinstance(data, dict) else {}
        if path.startswith("/humanize") and status == 410:
            _add_terminal_retry_guidance(error)
        code = error.get("error")
        message = error.get("message")
        if not isinstance(message, str) or not message:
            message = f"AI Checker Humanizer returned HTTP {status}."
        retry_key = (
            submission_key if 500 <= status < 600 or code == "in_progress" else None
        )
        retry_after = _integer(error, "retryAfter")
        raise AICheckerHumanizerError(
            status=status,
            code=code if isinstance(code, str) else "http_error",
            message=_submission_message(message, retry_key),
            retry_after=(
                retry_after if retry_after is not None else _seconds_header(response)
            ),
            credits_remaining=_integer(error, "creditsRemaining"),
            credits_needed=_integer(error, "creditsNeeded"),
            max_words=_integer(error, "maxWords"),
            idempotency_key=retry_key,
            body=error,
        )

    if not isinstance(data, dict) or not _valid_response(data, path):
        raise AICheckerHumanizerError(
            status=status,
            code="invalid_response",
            message=_submission_message(
                "AI Checker Humanizer returned an invalid response.", submission_key
            ),
            idempotency_key=submission_key,
        )

    if path.startswith("/humanize"):
        # A replay can return its task ID even if the task view is unavailable.
        if not isinstance(data.get("status"), str):
            data["status"] = "pending"
        _add_terminal_retry_guidance(data)
        retry_after = _seconds_header(response)
        if retry_after is not None and _integer(data, "retryAfter") is None:
            data["retryAfter"] = retry_after
    return data


def _request_error(
    exc: httpx.RequestError | httpx.InvalidURL, *, submission_key: str | None = None
) -> AICheckerHumanizerError:
    retry_key = (
        submission_key
        if isinstance(exc, (httpx.TransportError, httpx.DecodingError))
        and not isinstance(exc, httpx.UnsupportedProtocol)
        else None
    )
    if isinstance(exc, httpx.TimeoutException):
        code = "timeout"
        message = _REQUEST_TIMEOUT_MESSAGE
    else:
        code = "request_error"
        message = "The request to AI Checker Humanizer could not be sent or completed."
    return AICheckerHumanizerError(
        status=None,
        code=code,
        message=_submission_message(message, retry_key),
        idempotency_key=retry_key,
    )


class AICheckerHumanizerAPIWrapper(BaseModel):
    """Call the API with an explicit key or AICHECKERHUMANIZER_API_KEY."""

    model_config = ConfigDict(validate_assignment=True)

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
    # Allow time for the synchronous detection route to reach its 60-second limit.
    timeout: float = Field(default=65, gt=0, allow_inf_nan=False)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        try:
            url = httpx.URL(value)
        except httpx.InvalidURL as exc:
            raise ValueError("base_url must be an absolute HTTPS URL.") from exc
        if url.scheme != "https" or not url.host:
            raise ValueError("base_url must be an absolute HTTPS URL.")
        return value

    def detect(
        self, text: str, *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Check text for AI writing and return the detection response."""
        return self._request(
            "POST",
            "/detect",
            data={"text": text},
            idempotency_key=_submission_key(idempotency_key),
        )

    async def adetect(
        self, text: str, *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Check text for AI writing asynchronously."""
        return await self._arequest(
            "POST",
            "/detect",
            data={"text": text},
            idempotency_key=_submission_key(idempotency_key),
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
            idempotency_key=_submission_key(idempotency_key),
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
            idempotency_key=_submission_key(idempotency_key),
        )

    def humanize_get(
        self,
        task_id: str,
        *,
        timeout: float | None = None,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        """Fetch a rewrite task and any available output."""
        return self._request(
            "GET",
            f"/humanize/{quote(task_id, safe='')}",
            timeout=timeout,
            client=client,
        )

    async def ahumanize_get(
        self,
        task_id: str,
        *,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        """Fetch a rewrite task asynchronously."""
        return await self._arequest(
            "GET",
            f"/humanize/{quote(task_id, safe='')}",
            timeout=timeout,
            client=client,
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
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        try:
            context = httpx.Client() if client is None else nullcontext(client)
            with context as session:
                response = session.request(
                    method,
                    self.base_url.rstrip("/") + path,
                    headers=self._headers(idempotency_key),
                    json=data,
                    timeout=self.timeout if timeout is None else timeout,
                )
        except (httpx.RequestError, httpx.InvalidURL) as exc:
            raise _request_error(exc, submission_key=idempotency_key) from exc
        return _parse_response(response, path=path, submission_key=idempotency_key)

    async def _arequest(
        self,
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        try:
            context = httpx.AsyncClient() if client is None else nullcontext(client)
            async with context as session:
                response = await session.request(
                    method,
                    self.base_url.rstrip("/") + path,
                    headers=self._headers(idempotency_key),
                    json=data,
                    timeout=self.timeout if timeout is None else timeout,
                )
        except (httpx.RequestError, httpx.InvalidURL) as exc:
            raise _request_error(exc, submission_key=idempotency_key) from exc
        return _parse_response(response, path=path, submission_key=idempotency_key)
