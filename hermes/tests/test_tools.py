import json
from datetime import UTC, datetime
from email.utils import format_datetime
from unittest.mock import MagicMock, Mock

import pytest

from hermes import tools

_KEY = "ach_live_" + "x" * 32
_TASK_ID = "74b33b79-d17b-4f1a-b765-9a952f01d270"


def response(status, payload, retry_after=None):
    reply = MagicMock()
    reply.__enter__.return_value = reply
    reply.status = status
    reply.getheader.return_value = retry_after
    reply.read.return_value = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return reply


@pytest.fixture(autouse=True)
def http(monkeypatch):
    connection = Mock()
    connection.getresponse.side_effect = []
    fake = Mock(return_value=connection)
    monkeypatch.setattr(tools, "HTTPSConnection", fake)
    return fake, connection


@pytest.fixture
def handlers():
    return tools.Tools(Mock(return_value=_KEY))


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        def __init__(self):
            self.seconds = 0
            self.sleeps = []

        def sleep(self, delay):
            self.sleeps.append(delay)
            self.seconds += delay

    fake = Clock()
    monkeypatch.setattr(tools.time, "monotonic", lambda: fake.seconds)
    monkeypatch.setattr(tools.time, "sleep", fake.sleep)
    return fake


def test_detect_returns_api_result_and_passes_idempotency_key(handlers, http):
    connect, connection = http
    payload = {"score": 92, "verdict": {"label": "likely_ai"}, "creditsCharged": 158}
    connection.getresponse.side_effect = [response(200, payload)]

    result = handlers.detect_ai_text({"text": "Check this.", "idempotency_key": "check-123"})

    assert result == json.dumps(payload, separators=(",", ":"))
    handlers.get_secret.assert_called_once_with("AICHECKERHUMANIZER_API_KEY")
    connect.assert_called_once_with("aicheckerhumanizer.com", timeout=60)
    connection.request.assert_called_once_with(
        "POST", "/api/v1/detect", body=b'{"text":"Check this."}', headers={
            "Authorization": f"Bearer {_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "aicheckerhumanizer-hermes/0.1.0",
            "Idempotency-Key": "check-123",
        },
    )
    connection.close.assert_called_once()


def test_detect_without_verdict(handlers, http):
    payload = {"score": None, "verdict": None, "reason": "too_short", "creditsCharged": 0}
    http[1].getresponse.side_effect = [response(200, payload)]

    assert json.loads(handlers.detect_ai_text({"text": "Short text."})) == payload


def test_humanize_polls_using_server_delay_and_default_interval(handlers, http, clock):
    connect, connection = http
    done = {"taskId": _TASK_ID, "status": "completed", "output": "Rewritten text."}
    connection.getresponse.side_effect = [
        response(202, {"taskId": _TASK_ID, "status": "queued"}, "4"),
        response(200, {"taskId": _TASK_ID, "status": "running"}),
        response(200, done),
    ]

    result = handlers.humanize_text({"text": "Rewrite this.", "idempotency_key": "rewrite-123"})

    assert json.loads(result) == done
    assert clock.sleeps == [4, 2.5]
    assert [call.kwargs["timeout"] for call in connect.call_args_list] == [30, 30, 30]
    post, *polls = connection.request.call_args_list
    assert post.args == ("POST", "/api/v1/humanize")
    assert json.loads(post.kwargs["body"]) == {
        "text": "Rewrite this.", "mode": "standard", "strength": "medium",
    }
    assert post.kwargs["headers"]["Idempotency-Key"] == "rewrite-123"
    for poll in polls:
        assert poll.args == ("GET", f"/api/v1/humanize/{_TASK_ID}")
        assert poll.kwargs["body"] is None
        assert "Idempotency-Key" not in poll.kwargs["headers"]


@pytest.mark.parametrize("status", ["completed", "partial", "unchanged", "failed", "cancelled"])
def test_humanize_returns_terminal_post_without_polling(handlers, http, clock, status):
    payload = {"taskId": _TASK_ID, "status": status, "output": None}
    http[1].getresponse.side_effect = [response(202, payload)]

    result = handlers.humanize_text({"text": "Rewrite this.", "mode": "academic", "strength": "heavy"})

    assert json.loads(result) == payload
    assert clock.sleeps == []
    assert http[1].request.call_count == 1
    assert json.loads(http[1].request.call_args.kwargs["body"]) == {
        "text": "Rewrite this.", "mode": "academic", "strength": "heavy",
    }


def test_humanize_timeout_caps_poll_from_post_start(handlers, http, clock):
    connect, connection = http

    def exchange():
        clock.seconds += 29
        yield response(202, {"taskId": _TASK_ID, "status": "queued"})
        yield response(200, {"taskId": _TASK_ID, "status": "running"})
        clock.seconds += connect.call_args.kwargs["timeout"]
        raise TimeoutError

    connection.getresponse.side_effect = exchange()

    result = handlers.humanize_text({"text": "Rewrite this."})

    assert json.loads(result) == {"taskId": _TASK_ID, "status": "running"}
    assert [call.kwargs["timeout"] for call in connect.call_args_list] == [30, 18.5, 16]
    assert clock.seconds == 50


def test_humanize_returns_pending_when_retry_after_exceeds_budget(handlers, http, clock):
    payload = {"taskId": _TASK_ID, "status": "queued"}
    http[1].getresponse.side_effect = [response(202, payload, "60")]

    assert json.loads(handlers.humanize_text({"text": "Rewrite this."})) == payload
    assert clock.sleeps == []
    assert http[1].request.call_count == 1


def test_get_task(handlers, http):
    connect, connection = http
    payload = {"taskId": _TASK_ID, "status": "partial", "output": "The full text."}
    connection.getresponse.side_effect = [response(200, payload)]

    assert json.loads(handlers.get_humanize_result({"task_id": _TASK_ID})) == payload
    assert connection.request.call_args.args == ("GET", f"/api/v1/humanize/{_TASK_ID}")
    connect.assert_called_once_with("aicheckerhumanizer.com", timeout=30)


def test_get_balance(handlers, http):
    connect, connection = http
    payload = {"creditsRemaining": 300, "batches": [{"expiresAt": "2027-01-23T00:00:00Z"}]}
    connection.getresponse.side_effect = [response(200, payload)]

    assert json.loads(handlers.get_balance({})) == payload
    assert connection.request.call_args.args == ("GET", "/api/v1/balance")
    connect.assert_called_once_with("aicheckerhumanizer.com", timeout=30)


@pytest.mark.parametrize(("status", "code", "fields"), [
    (400, "bad_request", {}),
    (401, "unauthorized", {}),
    (402, "insufficient_credits", {"creditsRemaining": 0, "creditsNeeded": 200}),
    (402, "trial_paused", {}),
    (404, "not_found", {}),
    (409, "in_progress", {"retryAfter": 3}),
    (409, "busy", {"retryAfter": 4}),
    (410, "result_expired", {"taskStatus": "completed", "creditsCharged": 5}),
    (413, "too_long", {"maxWords": 10_000}),
    (413, "too_many_units", {"maxUnits": 200, "units": 201}),
    (422, "language_unsupported", {}),
    (422, "mode_unavailable", {}),
    (422, "nothing_to_rewrite", {}),
    (422, "idempotency_conflict", {}),
    (429, "rate_limited", {"retryAfter": 10}),
    (429, "api_paused", {"retryAfter": 10}),
    (503, "capacity", {"retryAfter": 20}),
    (503, "detection_unavailable", {"retryAfter": 20}),
    (503, "unavailable", {"retryAfter": 20}),
])
def test_http_error_keeps_code_message_and_fields(handlers, http, status, code, fields):
    payload = {"error": code, "message": "Service detail.", **fields}
    if "taskStatus" in fields:
        payload["status"] = payload.pop("taskStatus")
    http[1].getresponse.side_effect = [response(status, payload)]

    result = handlers.detect_ai_text({"text": "Check this."})

    assert json.loads(result) == {"error": code, "message": "Service detail.", "status": status, **fields}
    assert http[1].request.call_count == 1


def test_http_error_falls_back_to_retry_after_date(handlers, http, monkeypatch):
    monkeypatch.setattr(tools.time, "time", lambda: 1000)
    date = format_datetime(datetime.fromtimestamp(1020, UTC), usegmt=True)
    http[1].getresponse.side_effect = [response(503, b"<html>", date)]

    assert json.loads(handlers.get_balance({})) == {
        "error": "http_error",
        "message": "AI Checker Humanizer returned HTTP 503.",
        "status": 503,
        "retryAfter": 20,
    }


def test_redirect_is_an_http_error(handlers, http):
    http[1].getresponse.side_effect = [response(307, b"")]

    assert json.loads(handlers.detect_ai_text({"text": "Check this."}))["status"] == 307
    assert http[1].request.call_count == 1


def test_poll_error_is_returned_without_retry(handlers, http, clock):
    http[1].getresponse.side_effect = [
        response(202, {"taskId": _TASK_ID, "status": "running"}),
        response(429, {"error": "rate_limited", "message": "Slow down."}, "6"),
    ]

    assert json.loads(handlers.humanize_text({"text": "Rewrite this."})) == {
        "error": "rate_limited", "message": "Slow down.", "status": 429, "retryAfter": 6,
    }
    assert clock.sleeps == [2.5]
    assert http[1].request.call_count == 2


@pytest.mark.parametrize("failure", [TimeoutError, OSError, tools.HTTPException])
def test_network_error_does_not_retry(handlers, http, failure):
    http[1].getresponse.side_effect = failure("connection details")

    assert json.loads(handlers.get_balance({})) == {
        "error": "network_error", "message": "Could not reach AI Checker Humanizer.",
    }
    assert http[1].request.call_count == 1
    http[1].close.assert_called_once()


def test_early_transport_failure(handlers, http):
    http[0].side_effect = OSError("TLS setup failed")

    assert json.loads(handlers.get_balance({}))["error"] == "network_error"
    http[0].assert_called_once()
    http[1].request.assert_not_called()


@pytest.mark.parametrize("body", [b"{", [], None])
def test_invalid_success_response(handlers, http, body):
    http[1].getresponse.side_effect = [response(200, body)]

    assert json.loads(handlers.get_balance({}))["error"] == "invalid_response"


@pytest.mark.parametrize(("handler", "arguments"), [
    ("detect_ai_text", {"text": "Check this."}),
    ("humanize_text", {"text": "Rewrite this."}),
    ("get_humanize_result", {"task_id": _TASK_ID}),
    ("get_balance", {}),
])
def test_missing_api_key(handler, arguments, http):
    handlers = tools.Tools(lambda _name: None)

    result = json.loads(getattr(handlers, handler)(arguments))

    assert result["error"] == "missing_api_key"
    assert "AICHECKERHUMANIZER_API_KEY" in result["message"]
    assert "https://aicheckerhumanizer.com/account#api" in result["message"]
    http[0].assert_not_called()


@pytest.mark.parametrize("key", ["bad\nheader", "é", "x" * 129, ""])
def test_invalid_idempotency_key_cannot_reach_http(handlers, http, key):
    result = handlers.detect_ai_text({"text": "Check this.", "idempotency_key": key})

    assert json.loads(result)["error"] == "invalid_request"
    http[0].assert_not_called()


def test_invalid_task_id_cannot_change_request_path(handlers, http):
    result = handlers.get_humanize_result({"task_id": "not a UUID"})

    assert json.loads(result)["error"] == "invalid_request"
    http[0].assert_not_called()
