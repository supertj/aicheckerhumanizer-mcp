"""Tool handlers for the AI Checker Humanizer REST API."""

import json
import math
import time
from contextlib import closing
from datetime import UTC
from email.utils import parsedate_to_datetime
from http.client import HTTPException, HTTPSConnection
from sys import get_int_max_str_digits
from uuid import UUID, uuid4

API_KEY_NAME = "AICHECKERHUMANIZER_API_KEY"
MODES = ("standard", "academic", "casual", "concise")
STRENGTHS = ("light", "medium", "heavy")

_API_HOST = "aicheckerhumanizer.com"
_API_PATH = "/api/v1"
_USER_AGENT = "aicheckerhumanizer-hermes/0.1.0"
_OTHER_TIMEOUT = 30
_HUMANIZE_WAIT = 50
_POLL_INTERVAL = 2.5
_PENDING = ("queued", "running")


def _json(result):
    return json.dumps(result, separators=(",", ":"))


def _error(code, message):
    return {"error": code, "message": message}


def _retry_after(header):
    if not header:
        return None
    header = header.strip()
    if header.isascii() and header.isdecimal():
        limit = get_int_max_str_digits()
        if limit and len(header) > limit:
            return None
        return int(header)
    try:
        when = parsedate_to_datetime(header)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return max(0, math.ceil(when.timestamp() - time.time()))
    except (ValueError, OverflowError, OSError):
        return None


def _http_error(status, payload, retry_after):
    result = {
        "error": payload.get("error", "http_error"),
        "message": payload.get("message", f"AI Checker Humanizer returned HTTP {status}."),
        "status": status,
    }
    for field in (
        "retryAfter", "creditsRemaining", "creditsNeeded", "maxWords", "maxUnits", "units",
        "taskId", "taskStatus", "creditsCharged",
    ):
        if field in payload:
            result[field] = payload[field]
    if status == 410 and isinstance(payload.get("status"), str):
        result.setdefault("taskStatus", payload["status"])
    if retry_after is not None:
        result.setdefault("retryAfter", retry_after)
    return result


def _request(key, method, path, *, timeout, body=None, idempotency_key=None):
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
        body = _json(body).encode("utf-8")
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key

    # A direct HTTPS connection never follows redirects with the key or text.
    try:
        with closing(HTTPSConnection(_API_HOST, timeout=timeout)) as connection:
            connection.request(method, f"{_API_PATH}{path}", body=body, headers=headers)
            with connection.getresponse() as response:
                status = response.status
                retry_after = _retry_after(response.getheader("Retry-After"))
                try:
                    payload = json.loads(response.read())
                except (ValueError, RecursionError):
                    payload = None
    except (OSError, HTTPException):
        return _error("network_error", "Could not reach AI Checker Humanizer."), None

    if not 200 <= status < 300:
        return _http_error(status, payload if isinstance(payload, dict) else {}, retry_after), None
    if not isinstance(payload, dict):
        return _error("invalid_response", "AI Checker Humanizer returned an invalid response."), None
    return payload, retry_after


def _task_path(task_id):
    if isinstance(task_id, str):
        try:
            return f"/humanize/{UUID(task_id)}"
        except ValueError:
            pass
    return None


class Tools:
    def __init__(self, get_secret):
        self.get_secret = get_secret

    def _api_key(self):
        key = (self.get_secret(API_KEY_NAME) or "").strip()
        if not key:
            return None, _error(
                "missing_api_key",
                "Set AICHECKERHUMANIZER_API_KEY in the active Hermes profile. "
                "Create a key at https://aicheckerhumanizer.com/account#api",
            )
        if not key.isascii() or not key.isprintable():
            return None, _error("invalid_api_key", "The API key contains invalid characters.")
        return key, None

    def _submit(self, args, path, *, timeout, options=None):
        key, error = self._api_key()
        if error:
            return key, error, None

        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return key, _error("invalid_request", "Provide text to send."), None
        idempotency_key = args.get("idempotency_key")
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not idempotency_key
            or not idempotency_key.isascii()
            or not idempotency_key.isprintable()
            or len(idempotency_key) > 128
        ):
            return key, _error(
                "invalid_request", "Use 1 to 128 printable ASCII characters for idempotency_key."
            ), None

        result, retry_after = _request(
            key, "POST", path, timeout=timeout, body={"text": text, **(options or {})},
            idempotency_key=idempotency_key,
        )
        return key, result, retry_after

    def detect_ai_text(self, args, **kwargs):
        _, result, _ = self._submit(args, "/detect", timeout=60)
        return _json(result)

    def humanize_text(self, args, **kwargs):
        deadline = time.monotonic() + _HUMANIZE_WAIT
        mode = args.get("mode", "standard")
        strength = args.get("strength", "medium")
        if mode not in MODES or strength not in STRENGTHS:
            return _json(_error(
                "invalid_request",
                "Use mode standard, academic, casual or concise, and strength light, medium or heavy.",
            ))

        idempotency_key = args.get("idempotency_key")
        if idempotency_key is None:
            idempotency_key = str(uuid4())
        key, result, retry_after = self._submit(
            {**args, "idempotency_key": idempotency_key}, "/humanize", timeout=_OTHER_TIMEOUT,
            options={"mode": mode, "strength": strength},
        )
        if "error" in result:
            if result["error"] == "network_error":
                result["idempotency_key"] = idempotency_key
                result["message"] = (
                    "Could not confirm the rewrite submission. Retry humanize_text with "
                    "the same text, options, and returned idempotency_key."
                )
            return _json(result)
        task_id = result.get("taskId")
        path = _task_path(task_id)
        if path is None:
            return _json(_error("invalid_response", "The rewrite response did not include a valid task ID."))

        while result.get("status") in _PENDING:
            pending = {
                "taskId": task_id,
                "status": result["status"],
                "idempotency_key": idempotency_key,
            }
            # Retry-After is a minimum. Keep the normal interval to avoid rapid polling.
            delay = max(_POLL_INTERVAL, retry_after or 0)
            if delay >= deadline - time.monotonic():
                return _json(pending)
            time.sleep(delay)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _json(pending)
            result, retry_after = _request(
                key, "GET", path, timeout=min(_OTHER_TIMEOUT, remaining),
            )
            if result.get("error") == "network_error":
                if time.monotonic() < deadline:
                    pending["error"] = result["error"]
                    pending["message"] = (
                        "Could not check the rewrite status. Use taskId with get_humanize_result."
                    )
                return _json(pending)

        result["taskId"] = task_id
        result["idempotency_key"] = idempotency_key
        return _json(result)

    def get_humanize_result(self, args, **kwargs):
        key, result = self._api_key()
        task_id = args.get("task_id")
        path = _task_path(task_id)
        if path is None:
            return _json(result or _error(
                "invalid_request", "Provide the task ID returned by humanize_text.",
            ))
        if result is None:
            result, _ = _request(key, "GET", path, timeout=_OTHER_TIMEOUT)
        result.setdefault("taskId", task_id)
        return _json(result)

    def get_balance(self, args, **kwargs):
        key, error = self._api_key()
        if error:
            return _json(error)
        result, _ = _request(key, "GET", "/balance", timeout=_OTHER_TIMEOUT)
        return _json(result)
