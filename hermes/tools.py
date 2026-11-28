"""Tool handlers for the AI Checker Humanizer REST API."""

import json
import math
import re
from contextlib import closing
from datetime import UTC
from email.utils import parsedate_to_datetime
from http.client import HTTPException, HTTPSConnection
from sys import get_int_max_str_digits
from time import monotonic, sleep, time
from uuid import UUID, uuid4

API_KEY_NAME = "AICHECKERHUMANIZER_API_KEY"
MODES = ("standard", "academic", "casual", "concise")
STRENGTHS = ("light", "medium", "heavy")
IDEMPOTENCY_KEY_PATTERN = r"^[\x21-\x7e]+$"

_API_HOST = "aicheckerhumanizer.com"
_API_PATH = "/api/v1"
_USER_AGENT = "aicheckerhumanizer-hermes/0.1.0"
_DETECT_TIMEOUT = 65  # The synchronous endpoint can run for 60 seconds.
_OTHER_TIMEOUT = 30
_HUMANIZE_WAIT = 50
_POLL_INTERVAL = 2.5
_TERMINAL_STATUSES = frozenset({"completed", "partial", "unchanged", "failed", "cancelled"})


class SecretScopeUnavailableError(Exception):
    pass


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
        return max(0, math.ceil(when.timestamp() - time()))
    except (ValueError, OverflowError, OSError):
        return None


def _http_error(status, payload, retry_after):
    result = {
        "error": "http_error",
        "message": f"AI Checker Humanizer returned HTTP {status}.",
        **payload,
        "httpStatus": status,
    }
    if status == 410:
        if isinstance(result.get("status"), str):
            result.setdefault("taskStatus", result["status"])
        task_id = _normalize_task_id(result.get("taskId"))
        if task_id is not None:
            result["taskId"] = task_id
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
    body_decode_failed = False
    try:
        with closing(HTTPSConnection(_API_HOST, timeout=timeout)) as connection:
            connection.request(method, f"{_API_PATH}{path}", body=body, headers=headers)
            with connection.getresponse() as response:
                status = response.status
                retry_after = _retry_after(response.getheader("Retry-After"))
                try:
                    payload = json.loads(response.read())
                except UnicodeDecodeError:
                    body_decode_failed = True
                    payload = None
                except (ValueError, RecursionError):
                    payload = None
    except (OSError, HTTPException):
        return _error("network_error", "Could not reach AI Checker Humanizer."), None

    # A replay can wrap the expired task view in HTTP 202.
    if status == 202 and isinstance(payload, dict) and payload.get("error") == "result_expired":
        status = 410
    if not 200 <= status < 300:
        result = _http_error(status, payload if isinstance(payload, dict) else {}, retry_after)
        if body_decode_failed and idempotency_key is not None:
            result["idempotency_key"] = idempotency_key
        return result, None
    if not isinstance(payload, dict) or (method == "GET" and "error" in payload):
        return _error("invalid_response", "AI Checker Humanizer returned an invalid response."), None
    if "error" in payload and idempotency_key is not None:
        payload["idempotency_key"] = idempotency_key
    return payload, retry_after


def _get_task(key, task_id, *, timeout):
    result, retry_after = _request(key, "GET", f"/humanize/{task_id}", timeout=timeout)
    if "error" not in result and not isinstance(result.get("status"), str):
        return _error("invalid_response", "The rewrite response did not include a valid status."), None
    return result, retry_after


def _normalize_task_id(task_id):
    if isinstance(task_id, str):
        try:
            return str(UUID(task_id))
        except ValueError:
            pass
    return None


def _keep_retry_key(result, idempotency_key):
    status = result.get("httpStatus")
    code = result.get("error")
    if idempotency_key is not None and (
        code in ("network_error", "in_progress")
        or (code == "invalid_response" and status is None)
        or (isinstance(status, int) and 500 <= status < 600)
    ):
        result["idempotency_key"] = idempotency_key


def _rewrite_result(result):
    status = result.get("status")
    not_humanized = result.get("reason") == "not_humanized"
    if isinstance(status, str) and status in _TERMINAL_STATUSES and (
        not_humanized or status in ("failed", "cancelled")
    ):
        result.pop("idempotency_key", None)
        message = result.get("message")
        if not (not_humanized and isinstance(message, str) and message):
            advice = "To retry, call humanize_text with a new idempotency_key or omit it."
            result["message"] = f"{message} {advice}" if isinstance(message, str) and message else advice
    return _json(result)


class Tools:
    def __init__(self, get_secret):
        self.get_secret = get_secret

    def _api_key(self):
        try:
            key = (self.get_secret(API_KEY_NAME) or "").strip()
        except SecretScopeUnavailableError:
            return None, _error(
                "secret_scope_unavailable",
                "Hermes could not read the profile's secrets for this call. Check that Hermes runs the tool "
                "in the profile where AICHECKERHUMANIZER_API_KEY is set.",
            )
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
            return key, error, None, None

        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return key, _error("invalid_request", "Provide text to send."), None, None
        idempotency_key = args.get("idempotency_key")
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or len(idempotency_key) > 128
            or not re.fullmatch(IDEMPOTENCY_KEY_PATTERN, idempotency_key)
        ):
            return key, _error(
                "invalid_request", "Use 1 to 128 visible ASCII characters (! through ~) for idempotency_key."
            ), None, None
        if idempotency_key is None:
            idempotency_key = str(uuid4())

        result, retry_after = _request(
            key, "POST", path, timeout=timeout, body={"text": text, **(options or {})},
            idempotency_key=idempotency_key,
        )
        return key, result, retry_after, idempotency_key

    def detect_ai_text(self, args, **kwargs):
        _, result, _, idempotency_key = self._submit(args, "/detect", timeout=_DETECT_TIMEOUT)
        if "error" not in result and not {"score", "verdict"} <= result.keys():
            result = _error("invalid_response", "AI Checker Humanizer returned an invalid response.")
        if result.get("error") == "network_error":
            result["message"] = (
                "The detection result is unknown and the request may have used credits. "
                "To avoid a second charge, retry detect_ai_text within 24 hours with the returned "
                "idempotency_key and the same text."
            )
        if "error" in result:
            _keep_retry_key(result, idempotency_key)
        return _json(result)

    def humanize_text(self, args, **kwargs):
        deadline = monotonic() + _HUMANIZE_WAIT
        mode = args.get("mode")
        strength = args.get("strength")
        mode = "standard" if mode is None else mode
        strength = "medium" if strength is None else strength
        if mode not in MODES or strength not in STRENGTHS:
            return _json(_error(
                "invalid_request",
                "Use mode standard, academic, casual or concise, and strength light, medium or heavy.",
            ))

        key, result, retry_after, idempotency_key = self._submit(
            args, "/humanize", timeout=_OTHER_TIMEOUT,
            options={"mode": mode, "strength": strength},
        )
        if "error" in result:
            _keep_retry_key(result, idempotency_key)
            if result["error"] == "network_error":
                result["message"] = (
                    "Could not confirm the rewrite submission. To avoid a second charge, retry "
                    "humanize_text within 24 hours with the returned idempotency_key and the same text and options."
                )
            return _rewrite_result(result)
        task_id = _normalize_task_id(result.get("taskId"))
        if task_id is None:
            result = _error("invalid_response", "The rewrite response did not include a valid task ID.")
            _keep_retry_key(result, idempotency_key)
            return _json(result)

        task_info = {"taskId": task_id, "idempotency_key": idempotency_key}
        # Keep submission metadata available if the task lookup omits it.
        for field in ("skippedWords", "replayed"):
            if field in result:
                task_info[field] = result[field]
        while "error" not in result:
            status = result.get("status")
            pending = dict(task_info)
            if isinstance(status, str):
                if status in _TERMINAL_STATUSES:
                    break
                pending["status"] = status
            # A replay without a status can be checked immediately unless Retry-After says to wait.
            delay = max(_POLL_INTERVAL if "status" in pending else 0, retry_after or 0)
            if delay >= deadline - monotonic():
                return _json(pending)
            if delay:
                sleep(delay)
            remaining = deadline - monotonic()
            if remaining <= 0:
                return _json(pending)
            result, retry_after = _get_task(key, task_id, timeout=min(_OTHER_TIMEOUT, remaining))
            if result.get("error") in ("network_error", "invalid_response") and "httpStatus" not in result:
                if monotonic() < deadline:
                    pending["error"] = result["error"]
                    pending["message"] = (
                        "Could not check the rewrite status. Use taskId with get_humanize_result."
                    )
                return _json(pending)

        if "skippedWords" in result:
            task_info.pop("skippedWords", None)
        return _rewrite_result({**result, **task_info})

    def get_humanize_result(self, args, **kwargs):
        key, result = self._api_key()
        task_id = _normalize_task_id(args.get("task_id"))
        if task_id is None:
            return _json(result or _error(
                "invalid_request", "Provide the task ID returned by humanize_text.",
            ))
        if result is None:
            result, _ = _get_task(key, task_id, timeout=_OTHER_TIMEOUT)
        result["taskId"] = task_id
        return _rewrite_result(result)

    def get_balance(self, args, **kwargs):
        key, error = self._api_key()
        if error:
            return _json(error)
        result, _ = _request(key, "GET", "/balance", timeout=_OTHER_TIMEOUT)
        return _json(result)
