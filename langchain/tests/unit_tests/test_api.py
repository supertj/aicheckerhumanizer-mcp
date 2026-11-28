import json
import pickle
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from langchain_aicheckerhumanizer import (
    AICheckerHumanizerAPIWrapper,
    AICheckerHumanizerError,
    __version__,
)

BASE = "https://aicheckerhumanizer.com/api/v1"
TASK_ID = "0f92ace1-57c9-4dd1-a675-5c3967295244"


def assert_request(
    route: respx.Route,
    api: AICheckerHumanizerAPIWrapper,
    *,
    body: dict[str, str] | None = None,
    key: str | None = None,
) -> None:
    assert route.call_count == 1
    request = route.calls.last.request
    assert (
        request.headers["Authorization"] == f"Bearer {api.api_key.get_secret_value()}"
    )
    assert (
        request.headers["User-Agent"] == f"langchain-aicheckerhumanizer/{__version__}"
    )
    assert request.headers.get("Idempotency-Key") == key
    assert request.extensions["timeout"] == dict.fromkeys(
        ("connect", "read", "write", "pool"), api.timeout
    )
    if body is None:
        assert request.content == b""
    else:
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(request.content) == body


@pytest.mark.parametrize("async_call", [False, True])
async def test_requests(
    api: AICheckerHumanizerAPIWrapper, respx_mock: respx.MockRouter, async_call: bool
) -> None:
    detection = {"score": 41, "verdict": {"label": "likely_human"}}
    queued = {"taskId": TASK_ID, "status": "queued", "retryAfter": 4}
    completed = {"taskId": TASK_ID, "status": "completed", "output": "Rewritten."}
    balance = {"creditsRemaining": 100, "batches": []}
    detect = respx_mock.post(f"{BASE}/detect").respond(json=detection)
    start = respx_mock.post(f"{BASE}/humanize").respond(
        202,
        json={k: v for k, v in queued.items() if k != "retryAfter"},
        headers={"Retry-After": "4"},
    )
    get = respx_mock.get(f"{BASE}/humanize/{TASK_ID}").respond(json=completed)
    balance_route = respx_mock.get(f"{BASE}/balance").respond(json=balance)

    if async_call:
        assert await api.adetect("Draft", idempotency_key="check-1") == detection
        assert await api.ahumanize_start("Draft", idempotency_key="rewrite-1") == queued
        assert await api.ahumanize_get(TASK_ID) == completed
        assert await api.abalance() == balance
    else:
        assert api.detect("Draft", idempotency_key="check-1") == detection
        assert api.humanize_start("Draft", idempotency_key="rewrite-1") == queued
        assert api.humanize_get(TASK_ID) == completed
        assert api.balance() == balance

    assert_request(detect, api, body={"text": "Draft"}, key="check-1")
    assert_request(
        start,
        api,
        body={"text": "Draft", "mode": "standard", "strength": "medium"},
        key="rewrite-1",
    )
    assert_request(get, api)
    assert_request(balance_route, api)


def test_explicit_key_and_custom_settings(
    api: AICheckerHumanizerAPIWrapper, respx_mock: respx.MockRouter
) -> None:
    custom = AICheckerHumanizerAPIWrapper(
        api_key=SecretStr("ach_live_" + "y" * 32),
        base_url="https://example.test/api/v1/",
        timeout=4.5,
    )
    assert custom.api_key != api.api_key
    assert custom.api_key.get_secret_value() not in repr(custom)
    route = respx_mock.post("https://example.test/api/v1/detect").respond(json={})
    assert custom.detect("Draft") == {}
    assert_request(route, custom, body={"text": "Draft"})


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AICHECKERHUMANIZER_API_KEY", raising=False)
    with pytest.raises(
        ValueError, match="Set AICHECKERHUMANIZER_API_KEY or pass api_key"
    ):
        AICheckerHumanizerAPIWrapper()


@pytest.mark.parametrize(
    ("status", "code", "extra", "headers", "retry_after"),
    [
        (400, "bad_request", {}, {}, None),
        (401, "unauthorized", {}, {}, None),
        (
            402,
            "insufficient_credits",
            {"creditsRemaining": 2, "creditsNeeded": 10},
            {},
            None,
        ),
        (402, "trial_paused", {}, {}, None),
        (404, "not_found", {}, {}, None),
        (409, "in_progress", {"retryAfter": 2}, {"Retry-After": "4"}, 2),
        (409, "busy", {}, {"Retry-After": "6"}, 6),
        (410, "result_expired", {}, {}, None),
        (413, "too_long", {"maxWords": 10_000}, {}, None),
        (413, "too_many_units", {"maxUnits": 200}, {}, None),
        (422, "language_unsupported", {}, {}, None),
        (422, "mode_unavailable", {}, {}, None),
        (422, "nothing_to_rewrite", {}, {}, None),
        (422, "idempotency_conflict", {}, {}, None),
        (429, "rate_limited", {"retryAfter": 10}, {}, 10),
        (429, "api_paused", {}, {"Retry-After": "60"}, 60),
        (503, "capacity", {}, {"Retry-After": "8"}, 8),
        (503, "detection_unavailable", {"retryAfter": 5}, {}, 5),
        (503, "unavailable", {}, {}, None),
    ],
)
def test_api_error_mapping(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    status: int,
    code: str,
    extra: dict[str, Any],
    headers: dict[str, str],
    retry_after: int | None,
) -> None:
    respx_mock.post(f"{BASE}/detect").respond(
        status,
        json={"error": code, "message": "Request failed.", **extra},
        headers=headers,
    )
    with pytest.raises(AICheckerHumanizerError, match=r"Request failed\.") as caught:
        api.detect("Draft")
    error = caught.value
    assert error.status == status
    assert error.code == code
    assert error.message == "Request failed."
    assert error.retry_after == retry_after
    assert error.credits_remaining == extra.get("creditsRemaining")
    assert error.credits_needed == extra.get("creditsNeeded")
    assert error.max_words == extra.get("maxWords")


def test_api_error_pickle_round_trip() -> None:
    error = AICheckerHumanizerError(
        status=402,
        code="insufficient_credits",
        message="Request failed.",
        credits_remaining=2,
        credits_needed=10,
    )
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is type(error)
    assert restored.args == error.args
    assert vars(restored) == vars(error)


async def test_non_json_error(
    api: AICheckerHumanizerAPIWrapper, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(f"{BASE}/balance").respond(
        502, text="Proxy error", headers={"Retry-After": "unknown"}
    )
    with pytest.raises(AICheckerHumanizerError, match="HTTP 502") as caught:
        await api.abalance()
    assert caught.value.status == 502
    assert caught.value.code == "http_error"
    assert caught.value.retry_after is None


def test_invalid_success_response(
    api: AICheckerHumanizerAPIWrapper, respx_mock: respx.MockRouter
) -> None:
    respx_mock.post(f"{BASE}/humanize").respond(202, json={"status": "queued"})
    with pytest.raises(AICheckerHumanizerError, match="invalid response") as caught:
        api.humanize_start("Draft")
    assert caught.value.code == "invalid_response"
    assert caught.value.status == 202


@pytest.mark.parametrize(
    ("http_error", "code"),
    [(httpx.ReadTimeout, "timeout"), (httpx.ConnectError, "request_error")],
)
async def test_transport_errors(
    api: AICheckerHumanizerAPIWrapper,
    respx_mock: respx.MockRouter,
    http_error: type[httpx.RequestError],
    code: str,
) -> None:
    respx_mock.post(f"{BASE}/detect").mock(side_effect=http_error("Network error"))
    with pytest.raises(AICheckerHumanizerError) as caught:
        await api.adetect("Draft")
    assert caught.value.code == code
    assert caught.value.status is None
