import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import httpx
import pytest
import respx
from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool, ToolException
from pydantic import ValidationError

from langchain_aicheckerhumanizer import (
    AICheckerHumanizerAPIWrapper,
    AICheckerHumanizerDetectTool,
    AICheckerHumanizerError,
    AICheckerHumanizerGetResultTool,
    AICheckerHumanizerHumanizeTool,
    AICheckerHumanizerToolkit,
)
from langchain_aicheckerhumanizer import tools as tool_module

BASE = "https://aicheckerhumanizer.com/api/v1"
TASK_ID = "0f92ace1-57c9-4dd1-a675-5c3967295244"
EXPIRED_ERROR = {
    "error": "result_expired",
    "message": "The result expired.",
    "status": "completed",
    "reason": "retention_expired",
    "creditsReserved": 60,
    "creditsCharged": 50,
    "expiresAt": None,
    "expiredAt": None,
}
EXPIRED_RESULT = {**EXPIRED_ERROR, "httpStatus": 410, "taskStatus": "completed"}
INVALID_RESPONSE_ERROR = {
    "error": "invalid_response",
    "message": "AI Checker Humanizer returned an invalid response.",
    "httpStatus": 200,
}


class PollClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleep = Mock(side_effect=self.advance)
        self.async_sleep = AsyncMock(side_effect=self.advance)

    def monotonic(self) -> float:
        return self.now

    def advance(self, delay: float) -> None:
        self.now += delay


@pytest.fixture
def poll_clock(monkeypatch: pytest.MonkeyPatch) -> PollClock:
    clock = PollClock()
    monkeypatch.setattr(tool_module, "monotonic", clock.monotonic)
    monkeypatch.setattr(tool_module, "sleep", clock.sleep)
    monkeypatch.setattr(tool_module, "async_sleep", clock.async_sleep)
    return clock


@pytest.mark.parametrize("async_call", [False, True])
async def test_detect_retains_api_fields(
    api: AICheckerHumanizerAPIWrapper, respx_mock: respx.MockRouter, async_call: bool
) -> None:
    expected = {
        "score": 88,
        "verdict": {"label": "likely_ai"},
        "reason": None,
        "scoredWords": 200,
        "sentences": [{"score": 88, "text": "Draft."}],
    }
    respx_mock.post(f"{BASE}/detect").respond(json=expected)
    tool = AICheckerHumanizerDetectTool(api_wrapper=api)
    inputs = {"text": "Draft", "idempotency_key": "check-1"}
    result = await tool.ainvoke(inputs) if async_call else tool.invoke(inputs)
    assert json.loads(result) == expected


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (httpx.ReadTimeout("Timed out"), "timeout"),
        (httpx.ReadError("Connection lost"), "request_error"),
        (httpx.Response(502, html="<h1>Bad Gateway</h1>"), "http_error"),
        (httpx.Response(503, json={"error": "unavailable"}), "unavailable"),
        (httpx.Response(409, json={"error": "in_progress"}), "in_progress"),
    ],
)
async def test_detect_retries_with_the_key_from_the_error(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    async_call: bool,
    failure: httpx.Response | httpx.RequestError,
    code: str,
) -> None:
    detection = {"score": 42, "replayed": True}
    route = respx_mock.post(f"{BASE}/detect").mock(
        side_effect=[failure, httpx.Response(200, json=detection)]
    )
    tool = AICheckerHumanizerDetectTool(api_wrapper=api, handle_tool_error=False)
    inputs = {"text": "Draft"}
    with pytest.raises(ToolException) as caught:
        if async_call:
            await tool.ainvoke(inputs)
        else:
            tool.invoke(inputs)
    first = route.calls.last.request
    key = first.headers["Idempotency-Key"]
    assert UUID(key).version == 4
    assert first.extensions["timeout"]["read"] >= 65
    error = caught.value.__cause__
    assert isinstance(error, AICheckerHumanizerError)
    assert error.code == code
    assert error.idempotency_key == key
    assert f"Set idempotency_key to {key!r}" in str(caught.value)
    assert route.call_count == 1

    inputs["idempotency_key"] = error.idempotency_key
    result = await tool.ainvoke(inputs) if async_call else tool.invoke(inputs)
    assert json.loads(result) == detection
    retry = route.calls.last.request
    assert retry.headers["Idempotency-Key"] == key
    assert retry.content == first.content


@pytest.mark.parametrize("async_call", [False, True])
async def test_humanize_polls_for_result(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    poll_clock: PollClock,
    async_call: bool,
) -> None:
    start = respx_mock.post(f"{BASE}/humanize").respond(
        202,
        json={
            "taskId": TASK_ID,
            "status": "queued",
            "retryAfter": 2,
            "skippedWords": 4,
        },
    )
    expected = {
        "taskId": TASK_ID,
        "status": "partial",
        "output": "Rewritten text.",
        "originalScore": 92,
        "humanizedScore": 13,
        "creditsCharged": 50,
        "units": [{"text": "Original text.", "score": 13}],
    }
    poll = respx_mock.get(f"{BASE}/humanize/{TASK_ID}").mock(
        side_effect=[
            respx.MockResponse(json={"taskId": TASK_ID, "status": "running"}),
            respx.MockResponse(json=expected),
        ]
    )
    tool = AICheckerHumanizerHumanizeTool(api_wrapper=api)
    inputs = {
        "text": "Original text.",
        "mode": "academic",
        "strength": "heavy",
        "idempotency_key": "rewrite-1",
    }
    client_class = httpx.AsyncClient if async_call else httpx.Client
    with patch.object(
        client_class, "request", autospec=True, side_effect=client_class.request
    ) as request:
        result = await tool.ainvoke(inputs) if async_call else tool.invoke(inputs)
    submit_client, first_poll_client, second_poll_client = (
        call.args[0] for call in request.call_args_list
    )
    assert submit_client is not first_poll_client
    assert first_poll_client is second_poll_client
    assert first_poll_client.is_closed
    assert json.loads(result) == {**expected, "skippedWords": 4}
    assert poll.call_count == 2
    assert json.loads(start.calls.last.request.content) == {
        "text": "Original text.",
        "mode": "academic",
        "strength": "heavy",
    }
    assert start.calls.last.request.headers["Idempotency-Key"] == "rewrite-1"
    if async_call:
        assert [call.args[0] for call in poll_clock.async_sleep.await_args_list] == [
            2,
            3,
        ]
        poll_clock.sleep.assert_not_called()
    else:
        assert [call.args[0] for call in poll_clock.sleep.call_args_list] == [2, 3]
        poll_clock.async_sleep.assert_not_called()


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("wait_seconds", [0, 10])
@pytest.mark.usefixtures("poll_clock")
async def test_replayed_humanize_without_status(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    async_call: bool,
    wait_seconds: int,
) -> None:
    respx_mock.post(f"{BASE}/humanize").respond(
        202, json={"taskId": TASK_ID, "replayed": True}
    )
    completed = {"taskId": TASK_ID, "status": "completed", "output": "Edited draft."}
    if wait_seconds:
        respx_mock.get(f"{BASE}/humanize/{TASK_ID}").respond(json=completed)
    tool = AICheckerHumanizerHumanizeTool(
        api_wrapper=api, max_wait_seconds=wait_seconds
    )
    inputs = {"text": "Draft", "idempotency_key": "rewrite-1"}
    result = await tool.ainvoke(inputs) if async_call else tool.invoke(inputs)
    task = json.loads(result)
    if wait_seconds:
        assert task == completed
        assert len(respx_mock.calls) == 2
    else:
        assert task["taskId"] == TASK_ID
        assert task["status"] == "pending"
        assert "ai_checker_humanizer_get_result" in task["message"]
        assert len(respx_mock.calls) == 1


@pytest.mark.parametrize("async_call", [False, True])
async def test_humanize_returns_pending_when_next_poll_exceeds_budget(
    api: AICheckerHumanizerAPIWrapper, respx_mock: respx.MockRouter, async_call: bool
) -> None:
    respx_mock.post(f"{BASE}/humanize").respond(
        202,
        json={"taskId": TASK_ID, "status": "queued", "skippedWords": 0},
        headers={"Retry-After": "5"},
    )
    tool = AICheckerHumanizerHumanizeTool(api_wrapper=api, max_wait_seconds=3)
    inputs = {"text": "Draft"}
    result = await tool.ainvoke(inputs) if async_call else tool.invoke(inputs)
    pending = json.loads(result)
    assert pending["taskId"] == TASK_ID
    assert pending["status"] == "queued"
    assert pending["retryAfter"] == 5
    assert pending["skippedWords"] == 0
    assert "ai_checker_humanizer_get_result" in pending["message"]
    assert "task_id" in pending["message"]
    assert len(respx_mock.calls) == 1


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    ("failure", "code", "message", "retry_after"),
    [
        (
            httpx.Response(
                429,
                json={
                    "error": "rate_limited",
                    "message": "Slow down.",
                    "reason": "too_many_concurrent",
                },
                headers={"Retry-After": "10"},
            ),
            "rate_limited",
            "Slow down.",
            10,
        ),
        (
            httpx.Response(503, json={"error": "capacity", "message": "At capacity."}),
            "capacity",
            "At capacity.",
            1,
        ),
        (
            httpx.ReadError("Connection lost"),
            "request_error",
            "could not be sent or completed",
            1,
        ),
        (httpx.ReadTimeout("Read timed out"), "timeout", "timed out", 1),
    ],
)
@pytest.mark.usefixtures("poll_clock")
async def test_humanize_returns_pending_on_poll_error(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    async_call: bool,
    failure: httpx.Response | httpx.RequestError,
    code: str,
    message: str,
    retry_after: int,
) -> None:
    start = respx_mock.post(f"{BASE}/humanize").respond(
        202, json={"taskId": TASK_ID, "status": "queued", "skippedWords": 4}
    )
    poll = respx_mock.get(f"{BASE}/humanize/{TASK_ID}").mock(
        side_effect=[
            httpx.Response(
                200, json={"taskId": TASK_ID, "status": "running", "retryAfter": 1}
            ),
            failure,
        ]
    )
    tool = AICheckerHumanizerHumanizeTool(api_wrapper=api, handle_tool_error=False)
    inputs = {"text": "Draft"}
    result = await tool.ainvoke(inputs) if async_call else tool.invoke(inputs)
    pending = json.loads(result)
    assert pending["taskId"] == TASK_ID
    assert pending["status"] == "running"
    assert pending["skippedWords"] == 4
    assert pending["error"] == code
    assert message in pending["message"]
    assert "ai_checker_humanizer_get_result" in pending["message"]
    assert pending.get("retryAfter") == retry_after
    if isinstance(failure, httpx.Response):
        assert pending["httpStatus"] == failure.status_code
    if code == "rate_limited":
        assert pending["reason"] == "too_many_concurrent"
    assert start.call_count == 1
    assert poll.call_count == 2


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (
            401,
            {"error": "unauthorized", "message": "Check your API key."},
            {
                "error": "unauthorized",
                "message": "Check your API key.",
                "httpStatus": 401,
            },
        ),
        (
            404,
            {"error": "not_found", "message": "Task not found."},
            {"error": "not_found", "message": "Task not found.", "httpStatus": 404},
        ),
        (410, EXPIRED_ERROR, EXPIRED_RESULT),
        (200, {"taskId": TASK_ID}, INVALID_RESPONSE_ERROR),
    ],
)
@pytest.mark.usefixtures("poll_clock")
async def test_humanize_raises_on_permanent_poll_error(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    async_call: bool,
    status: int,
    body: dict[str, object],
    expected: dict[str, object],
) -> None:
    respx_mock.post(f"{BASE}/humanize").respond(
        202, json={"taskId": TASK_ID, "status": "running"}
    )
    route = respx_mock.get(f"{BASE}/humanize/{TASK_ID}").respond(status, json=body)
    tool = AICheckerHumanizerHumanizeTool(api_wrapper=api, handle_tool_error=False)
    with pytest.raises(ToolException) as caught:
        if async_call:
            await tool.ainvoke({"text": "Draft"})
        else:
            tool.invoke({"text": "Draft"})
    assert json.loads(str(caught.value)) == {"taskId": TASK_ID, **expected}
    assert route.call_count == 1


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.usefixtures("poll_clock")
async def test_poll_timeout_at_deadline_returns_pending(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    async_call: bool,
) -> None:
    max_wait = 3.01 if async_call else 10
    respx_mock.post(f"{BASE}/humanize").respond(
        202, json={"taskId": TASK_ID, "status": "running"}
    )
    tool = AICheckerHumanizerHumanizeTool(api_wrapper=api, max_wait_seconds=max_wait)
    inputs = {"text": "Draft"}
    if async_call:

        async def slow_poll(
            task_id: str, *, client: httpx.AsyncClient
        ) -> dict[str, str]:
            assert not client.is_closed
            await asyncio.sleep(0.1)
            return {"taskId": task_id, "status": "completed"}

        poll = AsyncMock(side_effect=slow_poll)
        monkeypatch.setattr(AICheckerHumanizerAPIWrapper, "ahumanize_get", poll)
        result = await tool.ainvoke(inputs)
        poll.assert_awaited_once()
        assert poll.await_args is not None
        assert poll.await_args.args == (TASK_ID,)
        assert poll.await_args.kwargs["client"].is_closed
    else:
        route = respx_mock.get(f"{BASE}/humanize/{TASK_ID}").mock(
            side_effect=httpx.ReadTimeout("Read timed out")
        )
        result = tool.invoke(inputs)
        assert route.calls.last.request.extensions["timeout"]["read"] == 7
    assert json.loads(result) == {
        "taskId": TASK_ID,
        "status": "running",
        "error": "timeout",
        "message": (
            "The request to AI Checker Humanizer timed out. "
            "Check again later with ai_checker_humanizer_get_result. "
            "Set task_id to the returned taskId. "
            "Results are available for about 70 minutes after submission."
        ),
    }


def test_immediate_terminal_task(
    api: AICheckerHumanizerAPIWrapper, respx_mock: respx.MockRouter
) -> None:
    expected = {
        "taskId": TASK_ID,
        "status": "failed",
        "reason": "not_humanized",
        "message": "No credits were charged.",
    }
    respx_mock.post(f"{BASE}/humanize").respond(202, json=expected)
    tool = AICheckerHumanizerHumanizeTool(api_wrapper=api)
    assert json.loads(tool.invoke({"text": "Draft"})) == expected
    assert len(respx_mock.calls) == 1


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    ("tool_class", "endpoint", "inputs", "status", "body"),
    [
        (
            AICheckerHumanizerDetectTool,
            "/detect",
            {"text": "Draft"},
            413,
            {
                "error": "too_long",
                "message": "Text is too long.",
                "words": 10_001,
                "chars": 65_000,
                "maxWords": 10_000,
                "maxChars": 60_000,
            },
        ),
        (
            AICheckerHumanizerHumanizeTool,
            "/humanize",
            {"text": "Draft"},
            400,
            {"error": "bad_request", "message": "Invalid input.", "field": "text"},
        ),
        (
            AICheckerHumanizerGetResultTool,
            f"/humanize/{TASK_ID}",
            {"task_id": TASK_ID},
            401,
            {"error": "unauthorized", "message": "Check your API key."},
        ),
    ],
)
async def test_tool_error_fields(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    async_call: bool,
    tool_class: type[BaseTool],
    endpoint: str,
    inputs: dict[str, str],
    status: int,
    body: dict[str, object],
) -> None:
    method = "GET" if "task_id" in inputs else "POST"
    route = respx_mock.request(method, f"{BASE}{endpoint}")
    tool = tool_class(api_wrapper=api)
    route.respond(status, json=body)
    call = ToolCall(name=tool.name, args=inputs, id="call-1", type="tool_call")
    message = await tool.ainvoke(call) if async_call else tool.invoke(call)
    assert message.status == "error"
    expected = {**body, "httpStatus": status}
    if "task_id" in inputs:
        expected["taskId"] = TASK_ID
    assert json.loads(message.content) == expected


@pytest.mark.parametrize("async_call", [False, True])
async def test_expired_submission_replay_is_a_410_error(
    api: AICheckerHumanizerAPIWrapper, respx_mock: respx.MockRouter, async_call: bool
) -> None:
    respx_mock.post(f"{BASE}/humanize").respond(
        202, json={**EXPIRED_ERROR, "taskId": TASK_ID, "replayed": True}
    )
    tool = AICheckerHumanizerHumanizeTool(api_wrapper=api, handle_tool_error=False)
    inputs = {"text": "Draft", "idempotency_key": "rewrite-1"}
    with pytest.raises(ToolException) as caught:
        if async_call:
            await tool.ainvoke(inputs)
        else:
            tool.invoke(inputs)
    error = caught.value.__cause__
    assert isinstance(error, AICheckerHumanizerError)
    assert error.status == 410
    assert error.idempotency_key is None
    assert json.loads(str(caught.value)) == {
        **EXPIRED_RESULT,
        "taskId": TASK_ID,
        "replayed": True,
    }
    assert len(respx_mock.calls) == 1


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("key", [None, "rewrite-1"])
@pytest.mark.parametrize(
    "failure",
    [
        httpx.ReadError("Connection lost"),
        httpx.Response(502, text="Bad gateway"),
        httpx.Response(503, json={"error": "unavailable", "message": "Unavailable."}),
    ],
)
async def test_ambiguous_submission_error_exposes_key_for_retry(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    async_call: bool,
    key: str | None,
    failure: httpx.Response | httpx.RequestError,
) -> None:
    task = {"taskId": TASK_ID, "status": "queued"}
    route = respx_mock.post(f"{BASE}/humanize").mock(
        side_effect=[failure, httpx.Response(202, json=task)]
    )
    tool = AICheckerHumanizerHumanizeTool(
        api_wrapper=api, handle_tool_error=False, max_wait_seconds=0
    )
    inputs = {"text": "Draft", "mode": "academic", "strength": "heavy"}
    if key is not None:
        inputs["idempotency_key"] = key
    with pytest.raises(ToolException, match="Retry with the same input") as exc:
        if async_call:
            await tool.ainvoke(inputs)
        else:
            tool.invoke(inputs)

    first_request = route.calls.last.request
    sent_key = first_request.headers["Idempotency-Key"]
    if key is None:
        assert UUID(sent_key).version == 4
    else:
        assert sent_key == key
    assert f"Set idempotency_key to {sent_key!r}" in str(exc.value)
    assert route.call_count == 1

    inputs["idempotency_key"] = sent_key
    result = await tool.ainvoke(inputs) if async_call else tool.invoke(inputs)
    assert json.loads(result)["taskId"] == TASK_ID
    retry = route.calls.last.request
    assert retry.headers["Idempotency-Key"] == sent_key
    assert retry.content == first_request.content


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    ("tool_class", "endpoint", "failure", "code"),
    [
        (
            AICheckerHumanizerDetectTool,
            "/detect",
            httpx.Response(200, json={"message": "No score"}),
            "invalid_response",
        ),
        (
            AICheckerHumanizerHumanizeTool,
            "/humanize",
            httpx.Response(202, text="Invalid submission response"),
            "invalid_response",
        ),
        (
            AICheckerHumanizerDetectTool,
            "/detect",
            httpx.Response(400, text="Bad request"),
            "http_error",
        ),
        (
            AICheckerHumanizerHumanizeTool,
            "/humanize",
            httpx.Response(413, html="<h1>Content Too Large</h1>"),
            "http_error",
        ),
        (
            AICheckerHumanizerDetectTool,
            "/detect",
            httpx.DecodingError("Invalid encoding"),
            "request_error",
        ),
    ],
)
async def test_submission_errors_without_a_retry_hint(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    async_call: bool,
    tool_class: type[BaseTool],
    endpoint: str,
    failure: httpx.Response | httpx.RequestError,
    code: str,
) -> None:
    route = respx_mock.post(f"{BASE}{endpoint}").mock(side_effect=[failure])
    tool = tool_class(api_wrapper=api, handle_tool_error=False)
    inputs = {"text": "Draft", "idempotency_key": "submission-1"}
    with pytest.raises(ToolException) as caught:
        if async_call:
            await tool.ainvoke(inputs)
        else:
            tool.invoke(inputs)
    error = caught.value.__cause__
    assert isinstance(error, AICheckerHumanizerError)
    assert error.code == code
    assert error.idempotency_key is None
    assert "Retry with the same input" not in str(caught.value)
    assert "submission-1" not in str(caught.value)
    assert route.call_count == 1


@pytest.mark.parametrize(
    ("tool_class", "inputs", "field"),
    [
        (AICheckerHumanizerDetectTool, {"text": "Draft"}, "idempotency_key"),
        (AICheckerHumanizerHumanizeTool, {"text": "Draft"}, "idempotency_key"),
        (AICheckerHumanizerGetResultTool, {}, "task_id"),
    ],
)
def test_tool_rejects_empty_key_or_task_id(
    api: AICheckerHumanizerAPIWrapper,
    tool_class: type[BaseTool],
    inputs: dict[str, str],
    field: str,
) -> None:
    tool = tool_class(api_wrapper=api)
    with pytest.raises(ValidationError, match=field):
        tool.invoke({**inputs, field: ""})


@pytest.mark.parametrize(
    "tool_class", [AICheckerHumanizerDetectTool, AICheckerHumanizerHumanizeTool]
)
@pytest.mark.parametrize("key", ["check 1", "key\n", "é", "x" * 129])
def test_tool_rejects_invalid_keys(
    api: AICheckerHumanizerAPIWrapper, tool_class: type[BaseTool], key: str
) -> None:
    tool = tool_class(api_wrapper=api)
    with pytest.raises(ValidationError, match="idempotency_key"):
        tool.invoke({"text": "Draft", "idempotency_key": key})


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "task",
    [
        {"taskId": TASK_ID, "status": "running", "retryAfter": 1},
        {"taskId": TASK_ID, "status": "completed", "output": "Edited draft."},
    ],
)
async def test_get_result_returns_task_json(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    async_call: bool,
    task: dict[str, object],
) -> None:
    route = respx_mock.get(f"{BASE}/humanize/{TASK_ID}").respond(json=task)
    tool = AICheckerHumanizerGetResultTool(api_wrapper=api)
    inputs = {"task_id": TASK_ID}
    result = await tool.ainvoke(inputs) if async_call else tool.invoke(inputs)
    assert json.loads(result) == task
    assert route.call_count == 1
    assert "Idempotency-Key" not in route.calls.last.request.headers


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (410, EXPIRED_ERROR, EXPIRED_RESULT),
        (200, {"taskId": TASK_ID, "status": None}, INVALID_RESPONSE_ERROR),
    ],
)
async def test_get_result_reports_task_details_on_error(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    async_call: bool,
    status: int,
    body: dict[str, object],
    expected: dict[str, object],
) -> None:
    respx_mock.get(f"{BASE}/humanize/{TASK_ID}").respond(status, json=body)
    tool = AICheckerHumanizerGetResultTool(api_wrapper=api)
    call = ToolCall(
        name=tool.name, args={"task_id": TASK_ID}, id="call-1", type="tool_call"
    )
    message = await tool.ainvoke(call) if async_call else tool.invoke(call)
    assert message.status == "error"
    assert isinstance(message.content, str)
    assert json.loads(message.content) == {"taskId": TASK_ID, **expected}


def test_toolkit_shares_wrapper(api: AICheckerHumanizerAPIWrapper) -> None:
    detect, humanize, get_result = AICheckerHumanizerToolkit(
        api_wrapper=api
    ).get_tools()
    assert isinstance(detect, AICheckerHumanizerDetectTool)
    assert isinstance(humanize, AICheckerHumanizerHumanizeTool)
    assert isinstance(get_result, AICheckerHumanizerGetResultTool)
    assert get_result.name == "ai_checker_humanizer_get_result"
    assert detect.api_wrapper is humanize.api_wrapper is get_result.api_wrapper is api
