import json
import sys
from datetime import UTC, datetime
from email.utils import format_datetime
from types import ModuleType
from unittest.mock import MagicMock, Mock
from uuid import UUID

import pytest

from hermes import register, tools

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
    connect.assert_called_once_with("aicheckerhumanizer.com", timeout=65)
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
    assert UUID(http[1].request.call_args.kwargs["headers"]["Idempotency-Key"]).version == 4


@pytest.mark.parametrize("key", [None, "detect-123"])
@pytest.mark.parametrize(("failure", "code", "status"), [
    (TimeoutError, "network_error", None),
    (OSError, "network_error", None),
    (response(502, b"<html>Bad Gateway</html>"), "http_error", 502),
    (response(503, {"error": "detection_unavailable"}), "detection_unavailable", 503),
    (response(200, b"<html>Unexpected response</html>"), "invalid_response", None),
    (response(200, {}), "invalid_response", None),
])
def test_detect_uncertain_result_can_be_retried_with_same_key(handlers, http, key, failure, code, status):
    _, connection = http
    detected = {"score": 92, "verdict": {"label": "likely_ai"}}
    connection.getresponse.side_effect = [failure, response(200, detected)]
    args = {"text": "Check this."}
    if key is not None:
        args["idempotency_key"] = key

    result = json.loads(handlers.detect_ai_text(args))

    submitted_key = connection.request.call_args.kwargs["headers"]["Idempotency-Key"]
    if key is None:
        assert UUID(submitted_key).version == 4
    else:
        assert submitted_key == key
    assert result["error"] == code
    assert result.get("httpStatus") == status
    assert result["idempotency_key"] == submitted_key
    if code in ("network_error", "invalid_response"):
        assert "request may have used credits" in result["message"]
        assert "within 24 hours" in result["message"]
    assert connection.request.call_count == 1

    retry = handlers.detect_ai_text({**args, "idempotency_key": result["idempotency_key"]})

    assert json.loads(retry) == detected
    first_post, second_post = connection.request.call_args_list
    assert first_post == second_post


def test_humanize_polls_using_server_delay_and_default_interval(handlers, http, clock):
    connect, connection = http
    done = {"status": "completed", "output": "Rewritten text."}
    connection.getresponse.side_effect = [
        response(202, {"taskId": _TASK_ID, "status": "queued"}, "4"),
        response(200, {"taskId": _TASK_ID, "status": "running"}),
        response(200, done),
    ]

    result = handlers.humanize_text({"text": "Rewrite this.", "idempotency_key": "rewrite-123"})

    assert json.loads(result) == {**done, "taskId": _TASK_ID, "idempotency_key": "rewrite-123"}
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

    post = http[1].request.call_args
    idempotency_key = post.kwargs["headers"]["Idempotency-Key"]
    assert UUID(idempotency_key).version == 4
    assert json.loads(result) == {**payload, "idempotency_key": idempotency_key}
    assert clock.sleeps == []
    assert http[1].request.call_count == 1
    assert json.loads(post.kwargs["body"]) == {
        "text": "Rewrite this.", "mode": "academic", "strength": "heavy",
    }


def test_humanize_accepts_a_replayed_task_id_without_status(handlers, http, clock):
    accepted = {"taskId": _TASK_ID, "replayed": True}
    complete = {"taskId": _TASK_ID, "status": "completed", "output": "Rewritten text."}
    http[1].getresponse.side_effect = [response(202, accepted), response(200, complete)]

    result = handlers.humanize_text({"text": "Rewrite this.", "idempotency_key": "rewrite-123"})

    assert json.loads(result) == {**accepted, "status": "pending", "idempotency_key": "rewrite-123"}
    assert clock.sleeps == []
    assert http[1].request.call_count == 1
    assert json.loads(handlers.get_humanize_result({"task_id": _TASK_ID})) == complete


@pytest.mark.parametrize("key", [None, "rewrite-123"])
def test_humanize_submission_network_error_can_be_retried_with_same_key(handlers, http, clock, key):
    _, connection = http
    accepted = {"taskId": _TASK_ID, "status": "queued"}
    connection.getresponse.side_effect = [TimeoutError, response(202, accepted, "60")]
    args = {"text": "Rewrite this.", "mode": "casual", "strength": "light"}
    if key is not None:
        args["idempotency_key"] = key

    result = json.loads(handlers.humanize_text(args))

    submitted_key = connection.request.call_args.kwargs["headers"]["Idempotency-Key"]
    if key is None:
        assert UUID(submitted_key).version == 4
    else:
        assert submitted_key == key
    assert result == {
        "error": "network_error",
        "message": (
            "Could not confirm the rewrite submission. Retry humanize_text within 24 hours. "
            "Use the returned idempotency_key and the same text and options."
        ),
        "idempotency_key": submitted_key,
    }
    assert connection.request.call_count == 1

    retry = handlers.humanize_text({**args, "idempotency_key": result["idempotency_key"]})

    assert json.loads(retry) == {**accepted, "idempotency_key": submitted_key}
    first_post, second_post = connection.request.call_args_list
    assert first_post == second_post


def test_humanize_submission_html_gateway_error_can_be_retried_with_same_key(handlers, http):
    _, connection = http
    accepted = {"taskId": _TASK_ID, "status": "completed", "output": "Rewritten text."}
    connection.getresponse.side_effect = [response(502, b"<html>Bad Gateway</html>"), response(202, accepted)]
    args = {"text": "Rewrite this.", "mode": "academic", "strength": "heavy"}

    result = json.loads(handlers.humanize_text(args))

    submitted_key = connection.request.call_args.kwargs["headers"]["Idempotency-Key"]
    assert UUID(submitted_key).version == 4
    assert result == {
        "error": "http_error",
        "message": "AI Checker Humanizer returned HTTP 502.",
        "httpStatus": 502,
        "idempotency_key": submitted_key,
    }
    assert connection.request.call_count == 1

    retry = handlers.humanize_text({**args, "idempotency_key": result["idempotency_key"]})

    assert json.loads(retry) == {**accepted, "idempotency_key": submitted_key}
    first_post, second_post = connection.request.call_args_list
    assert first_post == second_post


@pytest.mark.parametrize(("status", "body", "error"), [
    (503, {"error": "capacity"}, "capacity"),
    (503, {"error": "unavailable"}, "unavailable"),
    (202, b"<html>Unexpected response</html>", "invalid_response"),
    (202, {"status": "queued"}, "invalid_response"),
])
def test_humanize_submission_errors_keep_idempotency_key(handlers, http, status, body, error):
    http[1].getresponse.side_effect = [response(status, body)]

    result = json.loads(handlers.humanize_text({"text": "Rewrite this.", "idempotency_key": "rewrite-123"}))

    assert result["error"] == error
    assert result["idempotency_key"] == "rewrite-123"
    assert http[1].request.call_count == 1


def test_humanize_null_options_use_defaults(handlers, http):
    http[1].getresponse.side_effect = [response(202, {"taskId": _TASK_ID, "status": "completed"})]

    result = handlers.humanize_text({"text": "Rewrite this.", "mode": None, "strength": None})

    assert json.loads(result)["status"] == "completed"
    assert json.loads(http[1].request.call_args.kwargs["body"]) == {
        "text": "Rewrite this.", "mode": "standard", "strength": "medium",
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

    assert json.loads(result) == {
        "taskId": _TASK_ID,
        "status": "running",
        "idempotency_key": connection.request.call_args_list[0].kwargs["headers"]["Idempotency-Key"],
    }
    assert [call.kwargs["timeout"] for call in connect.call_args_list] == [30, 18.5, 16]
    assert clock.seconds == 50


def test_humanize_returns_pending_when_retry_after_exceeds_budget(handlers, http, clock):
    payload = {"taskId": _TASK_ID, "status": "queued"}
    http[1].getresponse.side_effect = [response(202, payload, "60")]

    result = handlers.humanize_text({"text": "Rewrite this.", "idempotency_key": "rewrite-123"})

    assert json.loads(result) == {**payload, "idempotency_key": "rewrite-123"}
    assert clock.sleeps == []
    assert http[1].request.call_count == 1


def test_get_task(handlers, http):
    connect, connection = http
    payload = {"taskId": _TASK_ID, "status": "partial", "output": "The full text."}
    connection.getresponse.side_effect = [response(200, payload)]

    assert json.loads(handlers.get_humanize_result({"task_id": _TASK_ID})) == payload
    assert connection.request.call_args.args == ("GET", f"/api/v1/humanize/{_TASK_ID}")
    connect.assert_called_once_with("aicheckerhumanizer.com", timeout=30)


@pytest.mark.parametrize(("poll", "status"), [(False, "completed"), (True, "completed"), (True, "running")])
def test_expired_task_keeps_status_charge_and_expiration(handlers, http, clock, poll, status):
    payload = {
        "error": "result_expired",
        "message": "The result has expired.",
        "taskId": _TASK_ID,
        "status": status,
        "creditsCharged": 50,
        "expiredAt": "2027-01-23T00:00:00Z",
    }
    replies = [response(410, payload)]
    expected = {**payload, "httpStatus": 410, "taskStatus": status}
    if poll:
        replies.insert(0, response(202, {"taskId": _TASK_ID, "status": "queued"}))
        expected["idempotency_key"] = "rewrite-123"
    http[1].getresponse.side_effect = replies

    if poll:
        result = handlers.humanize_text({"text": "Rewrite this.", "idempotency_key": "rewrite-123"})
    else:
        result = handlers.get_humanize_result({"task_id": _TASK_ID})

    assert json.loads(result) == expected


@pytest.mark.parametrize("task_id", [_TASK_ID, f"{{{_TASK_ID}}}", f"urn:uuid:{_TASK_ID}", UUID(_TASK_ID).hex])
def test_get_task_network_error_returns_normalized_task_id(handlers, http, task_id):
    http[1].getresponse.side_effect = TimeoutError

    assert json.loads(handlers.get_humanize_result({"task_id": task_id})) == {
        "taskId": _TASK_ID,
        "error": "network_error",
        "message": "Could not reach AI Checker Humanizer.",
    }
    assert http[1].request.call_count == 1
    assert http[1].request.call_args.args == ("GET", f"/api/v1/humanize/{_TASK_ID}")


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
    http[1].getresponse.side_effect = [response(status, payload)]

    result = handlers.detect_ai_text({"text": "Check this.", "idempotency_key": "detect-123"})

    assert json.loads(result) == {**payload, "httpStatus": status, "idempotency_key": "detect-123"}
    assert http[1].request.call_count == 1


def test_http_error_falls_back_to_retry_after_date(handlers, http, monkeypatch):
    monkeypatch.setattr(tools.time, "time", lambda: 1000)
    date = format_datetime(datetime.fromtimestamp(1020, UTC), usegmt=True)
    http[1].getresponse.side_effect = [response(503, b"<html>", date)]

    assert json.loads(handlers.get_balance({})) == {
        "error": "http_error",
        "message": "AI Checker Humanizer returned HTTP 503.",
        "httpStatus": 503,
        "retryAfter": 20,
    }


def test_retry_after_ignores_number_above_python_digit_limit(monkeypatch):
    monkeypatch.setattr(tools, "get_int_max_str_digits", lambda: 4300)

    assert tools._retry_after("9" * 4301) is None


def test_redirect_is_an_http_error(handlers, http):
    http[1].getresponse.side_effect = [response(307, b"")]

    assert json.loads(handlers.detect_ai_text({"text": "Check this."}))["httpStatus"] == 307
    assert http[1].request.call_count == 1


def test_poll_error_is_returned_without_retry(handlers, http, clock):
    http[1].getresponse.side_effect = [
        response(202, {"taskId": _TASK_ID, "status": "running"}),
        response(429, {"error": "rate_limited", "message": "Slow down."}, "6"),
    ]

    result = handlers.humanize_text({"text": "Rewrite this.", "idempotency_key": "rewrite-123"})

    assert json.loads(result) == {
        "error": "rate_limited", "message": "Slow down.", "httpStatus": 429, "retryAfter": 6,
        "taskId": _TASK_ID, "idempotency_key": "rewrite-123",
    }
    assert clock.sleeps == [2.5]
    assert http[1].request.call_count == 2


@pytest.mark.parametrize("body", [{}, {"status": 1}, {"status": None}])
def test_poll_without_string_status_returns_invalid_response(handlers, http, clock, body):
    http[1].getresponse.side_effect = [
        response(202, {"taskId": _TASK_ID, "status": "running"}),
        response(200, body),
    ]

    result = json.loads(handlers.humanize_text({"text": "Rewrite this.", "idempotency_key": "rewrite-123"}))

    assert result["error"] == "invalid_response"
    assert result["taskId"] == _TASK_ID
    assert result["idempotency_key"] == "rewrite-123"
    assert http[1].request.call_count == 2


def test_get_task_without_status_returns_invalid_response(handlers, http):
    http[1].getresponse.side_effect = [response(200, {"taskId": _TASK_ID})]

    result = json.loads(handlers.get_humanize_result({"task_id": _TASK_ID}))

    assert result["error"] == "invalid_response"
    assert result["taskId"] == _TASK_ID


def test_poll_network_error_before_deadline_returns_pending_task(handlers, http, clock):
    http[1].getresponse.side_effect = [
        response(202, {"taskId": _TASK_ID, "status": "queued"}),
        response(200, {"status": "running"}),
        TimeoutError,
    ]

    result = handlers.humanize_text({"text": "Rewrite this.", "idempotency_key": "rewrite-123"})

    assert json.loads(result) == {
        "taskId": _TASK_ID,
        "status": "running",
        "error": "network_error",
        "message": "Could not check the rewrite status. Use taskId with get_humanize_result.",
        "idempotency_key": "rewrite-123",
    }
    assert clock.sleeps == [2.5, 2.5]
    assert http[1].request.call_count == 3


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
    if "task_id" in arguments:
        assert result["taskId"] == arguments["task_id"]
    http[0].assert_not_called()


def test_registered_handlers_report_unscoped_hermes_secrets(monkeypatch, http):
    class UnscopedSecretError(RuntimeError):
        pass

    secret_scope = ModuleType("agent.secret_scope")
    secret_scope.UnscopedSecretError = UnscopedSecretError
    secret_scope.get_secret = Mock(side_effect=UnscopedSecretError("internal gateway detail"))
    monkeypatch.setitem(sys.modules, "agent.secret_scope", secret_scope)
    ctx = Mock()
    register(ctx)
    handlers = {call.kwargs["name"]: call.kwargs["handler"] for call in ctx.register_tool.call_args_list}

    for name, args in {
        "detect_ai_text": {"text": "Check this."},
        "humanize_text": {"text": "Rewrite this."},
        "get_humanize_result": {"task_id": _TASK_ID},
        "get_balance": {},
    }.items():
        result = json.loads(handlers[name](args))
        assert result["error"] == "secret_scope_unavailable"
        assert "Hermes" in result["message"]
        assert "internal gateway detail" not in result["message"]
    http[0].assert_not_called()


@pytest.mark.parametrize("options", [{"mode": "formal"}, {"strength": "high"}])
def test_invalid_humanize_options_cannot_reach_http(handlers, http, options):
    result = handlers.humanize_text({"text": "Rewrite this.", **options})

    assert json.loads(result)["error"] == "invalid_request"
    http[0].assert_not_called()


@pytest.mark.parametrize("handler", ["detect_ai_text", "humanize_text"])
@pytest.mark.parametrize("key", ["bad\nheader", "trailing\n", "check 1", " ", "é", "\x7f", "x" * 129, ""])
def test_invalid_idempotency_key_cannot_reach_http(handlers, http, handler, key):
    result = getattr(handlers, handler)({"text": "Check this.", "idempotency_key": key})

    assert json.loads(result)["error"] == "invalid_request"
    http[0].assert_not_called()


@pytest.mark.parametrize("key", ["!", "~" * 128])
def test_idempotency_key_accepts_visible_ascii_and_length_boundaries(handlers, http, key):
    http[1].getresponse.side_effect = [response(200, {"score": None, "verdict": None})]

    assert "error" not in json.loads(handlers.detect_ai_text({"text": "Check this.", "idempotency_key": key}))
    assert http[1].request.call_args.kwargs["headers"]["Idempotency-Key"] == key


def test_invalid_task_id_cannot_change_request_path(handlers, http):
    result = handlers.get_humanize_result({"task_id": "not a UUID"})

    assert json.loads(result)["error"] == "invalid_request"
    http[0].assert_not_called()
