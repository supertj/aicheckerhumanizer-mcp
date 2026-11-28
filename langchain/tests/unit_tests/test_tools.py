import asyncio
import json
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import httpx
import pytest
import respx
from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool, ToolException

from langchain_aicheckerhumanizer import (
    AICheckerHumanizerAPIWrapper,
    AICheckerHumanizerDetectTool,
    AICheckerHumanizerGetResultTool,
    AICheckerHumanizerHumanizeTool,
    AICheckerHumanizerToolkit,
)
from langchain_aicheckerhumanizer import tools as tool_module

BASE = "https://aicheckerhumanizer.com/api/v1"
TASK_ID = "0f92ace1-57c9-4dd1-a675-5c3967295244"


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
async def test_humanize_polls_for_result(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    async_call: bool,
) -> None:
    sync_sleep = Mock()
    async_sleep = AsyncMock()
    monkeypatch.setattr(tool_module, "sleep", sync_sleep)
    monkeypatch.setattr(asyncio, "sleep", async_sleep)
    start = respx_mock.post(f"{BASE}/humanize").respond(
        202, json={"taskId": TASK_ID, "status": "queued", "retryAfter": 2}
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
    result = await tool.ainvoke(inputs) if async_call else tool.invoke(inputs)
    assert json.loads(result) == expected
    assert poll.call_count == 2
    assert json.loads(start.calls.last.request.content) == {
        "text": "Original text.",
        "mode": "academic",
        "strength": "heavy",
    }
    assert start.calls.last.request.headers["Idempotency-Key"] == "rewrite-1"
    if async_call:
        assert [call.args[0] for call in async_sleep.await_args_list] == [2, 3]
        sync_sleep.assert_not_called()
    else:
        assert [call.args[0] for call in sync_sleep.call_args_list] == [2, 3]
        async_sleep.assert_not_called()


@pytest.mark.parametrize("async_call", [False, True])
async def test_humanize_returns_pending_when_next_poll_exceeds_budget(
    api: AICheckerHumanizerAPIWrapper, respx_mock: respx.MockRouter, async_call: bool
) -> None:
    respx_mock.post(f"{BASE}/humanize").respond(
        202, json={"taskId": TASK_ID, "status": "queued"}, headers={"Retry-After": "5"}
    )
    tool = AICheckerHumanizerHumanizeTool(api_wrapper=api, max_wait_seconds=3)
    inputs = {"text": "Draft"}
    result = await tool.ainvoke(inputs) if async_call else tool.invoke(inputs)
    pending = json.loads(result)
    assert pending["taskId"] == TASK_ID
    assert pending["status"] == "queued"
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
                json={"error": "rate_limited", "message": "Slow down."},
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
            None,
        ),
        (
            httpx.ReadError("Connection lost"),
            "request_error",
            "could not be sent or completed",
            None,
        ),
        (httpx.ReadTimeout("Read timed out"), "timeout", "timed out", None),
    ],
)
async def test_humanize_returns_pending_on_poll_error(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    async_call: bool,
    failure: httpx.Response | httpx.RequestError,
    code: str,
    message: str,
    retry_after: int | None,
) -> None:
    monkeypatch.setattr(tool_module, "_DEFAULT_POLL_SECONDS", 0)
    start = respx_mock.post(f"{BASE}/humanize").respond(
        202, json={"taskId": TASK_ID, "status": "queued"}
    )
    poll = respx_mock.get(f"{BASE}/humanize/{TASK_ID}").mock(
        side_effect=[
            httpx.Response(200, json={"taskId": TASK_ID, "status": "running"}),
            failure,
        ]
    )
    tool = AICheckerHumanizerHumanizeTool(api_wrapper=api, handle_tool_error=False)
    inputs = {"text": "Draft"}
    result = await tool.ainvoke(inputs) if async_call else tool.invoke(inputs)
    pending = json.loads(result)
    assert pending["taskId"] == TASK_ID
    assert pending["status"] == "running"
    assert pending["error"] == code
    assert message in pending["message"]
    assert "ai_checker_humanizer_get_result" in pending["message"]
    assert pending.get("retryAfter") == retry_after
    assert start.call_count == 1
    assert poll.call_count == 2


async def test_async_poll_stops_at_deadline(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_module, "_DEFAULT_POLL_SECONDS", 0.001)

    async def slow_poll(task_id: str) -> dict[str, str]:
        await asyncio.sleep(30)
        return {"taskId": task_id, "status": "completed"}

    monkeypatch.setattr(
        AICheckerHumanizerAPIWrapper, "ahumanize_get", AsyncMock(side_effect=slow_poll)
    )
    respx_mock.post(f"{BASE}/humanize").respond(
        202, json={"taskId": TASK_ID, "status": "running"}
    )
    tool = AICheckerHumanizerHumanizeTool(api_wrapper=api, max_wait_seconds=0.05)
    result = await tool.ainvoke({"text": "Draft"})
    assert json.loads(result)["status"] == "running"


def test_sync_poll_timeout_at_deadline_returns_pending(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_module, "monotonic", Mock(side_effect=[0, 0, 3]))
    monkeypatch.setattr(tool_module, "sleep", Mock())
    respx_mock.post(f"{BASE}/humanize").respond(
        202, json={"taskId": TASK_ID, "status": "running"}
    )
    route = respx_mock.get(f"{BASE}/humanize/{TASK_ID}").mock(
        side_effect=httpx.ReadTimeout("Read timed out")
    )
    tool = AICheckerHumanizerHumanizeTool(api_wrapper=api, max_wait_seconds=10)
    result = json.loads(tool.invoke({"text": "Draft"}))
    assert result["taskId"] == TASK_ID
    assert result["status"] == "running"
    assert result["error"] == "timeout"
    assert route.calls.last.request.extensions["timeout"]["read"] == 7


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
    ("tool_class", "endpoint", "inputs"),
    [
        (AICheckerHumanizerDetectTool, "/detect", {"text": "Draft"}),
        (AICheckerHumanizerHumanizeTool, "/humanize", {"text": "Draft"}),
        (
            AICheckerHumanizerGetResultTool,
            f"/humanize/{TASK_ID}",
            {"task_id": TASK_ID},
        ),
    ],
)
async def test_tool_error_message(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    async_call: bool,
    tool_class: type[BaseTool],
    endpoint: str,
    inputs: dict[str, str],
) -> None:
    method = "GET" if "task_id" in inputs else "POST"
    route = respx_mock.request(method, f"{BASE}{endpoint}")
    tool = tool_class(api_wrapper=api)
    route.respond(401, json={"error": "unauthorized", "message": "Check your API key."})
    call = ToolCall(name=tool.name, args=inputs, id="call-1", type="tool_call")
    message = await tool.ainvoke(call) if async_call else tool.invoke(call)
    assert message.status == "error"
    assert message.content == "Check your API key."


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("key", [None, "rewrite-1"])
async def test_submission_network_error_exposes_key_for_retry(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    async_call: bool,
    key: str | None,
) -> None:
    task = {"taskId": TASK_ID, "status": "queued"}
    route = respx_mock.post(f"{BASE}/humanize").mock(
        side_effect=[httpx.ReadError("Connection lost"), httpx.Response(202, json=task)]
    )
    tool = AICheckerHumanizerHumanizeTool(
        api_wrapper=api, handle_tool_error=False, max_wait_seconds=0
    )
    inputs = {"text": "Draft", "mode": "academic", "strength": "heavy"}
    if key is not None:
        inputs["idempotency_key"] = key
    with pytest.raises(
        ToolException, match="Retry with the same text and options"
    ) as exc:
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


def test_toolkit_shares_wrapper(api: AICheckerHumanizerAPIWrapper) -> None:
    detect, humanize, get_result = AICheckerHumanizerToolkit(
        api_wrapper=api
    ).get_tools()
    assert isinstance(detect, AICheckerHumanizerDetectTool)
    assert isinstance(humanize, AICheckerHumanizerHumanizeTool)
    assert isinstance(get_result, AICheckerHumanizerGetResultTool)
    assert get_result.name == "ai_checker_humanizer_get_result"
    assert detect.api_wrapper is humanize.api_wrapper is get_result.api_wrapper is api
