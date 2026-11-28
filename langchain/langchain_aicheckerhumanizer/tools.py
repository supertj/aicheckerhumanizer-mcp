"""LangChain tools for AI detection and rewriting."""

import asyncio
import json
from asyncio import sleep as async_sleep
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from contextvars import copy_context
from time import monotonic
from typing import TYPE_CHECKING, Any

import httpx
from langchain_core.tools import BaseTool, BaseToolkit, ToolException
from langchain_core.tools import base as tool_base
from pydantic import BaseModel, Field

from .api import (
    _IDEMPOTENCY_KEY_MAX_LENGTH,
    _IDEMPOTENCY_KEY_PATTERN,
    _REQUEST_TIMEOUT_MESSAGE,
    AICheckerHumanizerAPIWrapper,
    AICheckerHumanizerError,
    HumanizeMode,
    HumanizeStrength,
)

if TYPE_CHECKING:
    from langchain_core.tools.base import ToolExceptionHandlerOutput as _ToolErrorOutput
else:
    # Older LangChain versions only support string output from an error handler.
    _ToolErrorOutput = getattr(tool_base, "ToolExceptionHandlerOutput", str)

_PENDING_STATUSES = ("queued", "running", "pending")
_DEFAULT_POLL_SECONDS = 3
_GET_RESULT_NAME = "ai_checker_humanizer_get_result"


class _DetectInput(BaseModel):
    text: str = Field(description="Text to check, up to 10,000 words.")
    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=_IDEMPOTENCY_KEY_MAX_LENGTH,
        pattern=_IDEMPOTENCY_KEY_PATTERN,
        description=(
            "Reuse with the same text within 24 hours to avoid a second charge."
        ),
    )


class _HumanizeInput(BaseModel):
    text: str = Field(description="Text to rewrite, up to 3,000 words.")
    mode: HumanizeMode = Field(
        default=HumanizeMode.STANDARD, description="Writing style."
    )
    strength: HumanizeStrength = Field(
        default=HumanizeStrength.MEDIUM, description="How extensively to rewrite."
    )
    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=_IDEMPOTENCY_KEY_MAX_LENGTH,
        pattern=_IDEMPOTENCY_KEY_PATTERN,
        description=(
            "Reuse with the same text and options within 24 hours "
            "to get the same task without another charge."
        ),
    )


class _GetResultInput(BaseModel):
    task_id: str = Field(
        min_length=1, description="The taskId returned by the rewrite tool."
    )


class AICheckerHumanizerDetectTool(BaseTool):
    """Check text for AI writing."""

    name: str = "ai_checker_humanizer_detect"
    description: str = (
        "Check up to 10,000 words for AI writing. Returns JSON with a 0–100 score, "
        "verdict and scored word count. English and Portuguese can include sentence "
        "scores. If there is no verdict, the response gives a reason. "
        "The API charges 1 credit per scored word only when it returns a verdict. "
        "If the error JSON includes idempotencyKey, pass it as idempotency_key "
        "with the same text when retrying within 24 hours."
    )
    args_schema: type[BaseModel] = _DetectInput
    api_wrapper: AICheckerHumanizerAPIWrapper = Field(
        default_factory=AICheckerHumanizerAPIWrapper
    )
    handle_tool_error: (
        bool | str | Callable[[ToolException], _ToolErrorOutput] | None
    ) = True

    def _run(self, text: str, idempotency_key: str | None = None) -> str:
        try:
            result = self.api_wrapper.detect(text, idempotency_key=idempotency_key)
        except AICheckerHumanizerError as exc:
            raise _tool_error(exc) from exc
        return json.dumps(result, ensure_ascii=False)

    async def _arun(self, text: str, idempotency_key: str | None = None) -> str:
        try:
            result = await self.api_wrapper.adetect(
                text, idempotency_key=idempotency_key
            )
        except AICheckerHumanizerError as exc:
            raise _tool_error(exc) from exc
        return json.dumps(result, ensure_ascii=False)


def _poll_delay(task: dict[str, Any]) -> int:
    hint = task.get("retryAfter")
    if type(hint) is int and hint > 0:
        return hint
    return _DEFAULT_POLL_SECONDS


def _humanize_result(
    task: dict[str, Any],
    *,
    skipped_words: int | None = None,
    error: AICheckerHumanizerError | None = None,
) -> str:
    if task.get("status") in _PENDING_STATUSES:
        message = (
            f"Check again later with {_GET_RESULT_NAME}. "
            "Set task_id to the returned taskId. "
            "Results are available for about 70 minutes after submission."
        )
        fields = ("taskId", "status", "retryAfter", "skippedWords")
        task = {field: task[field] for field in fields if field in task}
        task["message"] = message
        if error is not None:
            task = {
                **task,
                **_error_fields(error),
                "taskId": task["taskId"],
                "status": task["status"],
                "message": f"{error.message} {message}",
            }
    if skipped_words is not None:
        task.setdefault("skippedWords", skipped_words)
    return json.dumps(task, ensure_ascii=False)


def _poll_error_result(
    task: dict[str, Any], error: AICheckerHumanizerError, *, skipped_words: int | None
) -> str:
    status = error.status
    if status is None or status == 429 or 500 <= status < 600:
        return _humanize_result(task, skipped_words=skipped_words, error=error)
    raise _tool_error(error, task_id=task["taskId"]) from error


def _error_fields(error: AICheckerHumanizerError) -> dict[str, Any]:
    fields = {**error.body, "error": error.code, "message": error.message}
    if error.status is not None:
        fields["httpStatus"] = error.status
    if error.retry_after is not None:
        fields["retryAfter"] = error.retry_after
    if error.idempotency_key is not None:
        fields["idempotencyKey"] = error.idempotency_key
    if error.status == 410 and "status" in fields:
        fields["taskStatus"] = fields["status"]
    return fields


def _tool_error(
    error: AICheckerHumanizerError, *, task_id: str | None = None
) -> ToolException:
    fields = _error_fields(error)
    if task_id is not None:
        fields["taskId"] = task_id
    return ToolException(json.dumps(fields, ensure_ascii=False))


class AICheckerHumanizerHumanizeTool(BaseTool):
    """Rewrite text with AI Checker Humanizer."""

    name: str = "ai_checker_humanizer_humanize"
    description: str = (
        "Rewrite up to 3,000 words. Choose standard, academic, casual or concise "
        "mode and light, medium or heavy strength. Defaults are standard and medium. "
        "The API charges 5 credits per original word for each chunk it rewrites "
        "successfully. Chunks under 10 words are billed as 10. "
        "Reusing an idempotency_key with the same text and options within 24 hours "
        "returns the same task without another charge. If the error JSON includes "
        "idempotencyKey, retry with it as idempotency_key. Returns JSON "
        "with the rewritten text and any available scores, or a task ID and status "
        "if the rewrite is still pending."
    )
    args_schema: type[BaseModel] = _HumanizeInput
    api_wrapper: AICheckerHumanizerAPIWrapper = Field(
        default_factory=AICheckerHumanizerAPIWrapper
    )
    max_wait_seconds: float = Field(default=120, ge=0, allow_inf_nan=False)
    handle_tool_error: (
        bool | str | Callable[[ToolException], _ToolErrorOutput] | None
    ) = True

    def _run(
        self,
        text: str,
        mode: HumanizeMode = HumanizeMode.STANDARD,
        strength: HumanizeStrength = HumanizeStrength.MEDIUM,
        idempotency_key: str | None = None,
    ) -> str:
        try:
            task = self.api_wrapper.humanize_start(
                text, mode=mode, strength=strength, idempotency_key=idempotency_key
            )
        except AICheckerHumanizerError as exc:
            raise _tool_error(exc) from exc

        if task.get("status") not in _PENDING_STATUSES or not self.max_wait_seconds:
            return _humanize_result(task)

        deadline = monotonic() + self.max_wait_seconds
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            # Use a separate loop in this case and join its thread before returning.
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(
                    copy_context().run, self._poll_from_sync, task, deadline=deadline
                ).result()

        return self._poll_from_sync(task, deadline=deadline)

    def _poll_from_sync(self, task: dict[str, Any], *, deadline: float) -> str:
        # Closing the loop does not wait for an OS DNS lookup after cancellation.
        with closing(asyncio.new_event_loop()) as loop:
            return loop.run_until_complete(self._poll(task, deadline=deadline))

    async def _arun(
        self,
        text: str,
        mode: HumanizeMode = HumanizeMode.STANDARD,
        strength: HumanizeStrength = HumanizeStrength.MEDIUM,
        idempotency_key: str | None = None,
    ) -> str:
        try:
            task = await self.api_wrapper.ahumanize_start(
                text, mode=mode, strength=strength, idempotency_key=idempotency_key
            )
        except AICheckerHumanizerError as exc:
            raise _tool_error(exc) from exc

        if task.get("status") not in _PENDING_STATUSES or not self.max_wait_seconds:
            return _humanize_result(task)

        return await self._poll(task, deadline=monotonic() + self.max_wait_seconds)

    async def _poll(self, task: dict[str, Any], *, deadline: float) -> str:
        skipped_words = task.get("skippedWords")
        async with httpx.AsyncClient() as client:
            while task.get("status") in _PENDING_STATUSES:
                delay = _poll_delay(task)
                if delay >= deadline - monotonic():
                    break
                await async_sleep(delay)
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                try:
                    task = await asyncio.wait_for(
                        self.api_wrapper.ahumanize_get(task["taskId"], client=client),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    return _humanize_result(
                        task,
                        skipped_words=skipped_words,
                        error=AICheckerHumanizerError(
                            status=None,
                            code="timeout",
                            message=_REQUEST_TIMEOUT_MESSAGE,
                        ),
                    )
                except AICheckerHumanizerError as exc:
                    return _poll_error_result(task, exc, skipped_words=skipped_words)
        return _humanize_result(task, skipped_words=skipped_words)


class AICheckerHumanizerGetResultTool(BaseTool):
    """Fetch an existing rewrite task."""

    name: str = _GET_RESULT_NAME
    description: str = (
        "Fetch a rewrite task using its taskId as task_id. Returns task JSON with "
        "the current status and rewritten text when available. Checking status or "
        "fetching the result costs no credits. Results are available for about "
        "70 minutes after submission."
    )
    args_schema: type[BaseModel] = _GetResultInput
    api_wrapper: AICheckerHumanizerAPIWrapper = Field(
        default_factory=AICheckerHumanizerAPIWrapper
    )
    handle_tool_error: (
        bool | str | Callable[[ToolException], _ToolErrorOutput] | None
    ) = True

    def _run(self, task_id: str) -> str:
        try:
            task = self.api_wrapper.humanize_get(task_id)
        except AICheckerHumanizerError as exc:
            raise _tool_error(exc, task_id=task_id) from exc
        return json.dumps(task, ensure_ascii=False)

    async def _arun(self, task_id: str) -> str:
        try:
            task = await self.api_wrapper.ahumanize_get(task_id)
        except AICheckerHumanizerError as exc:
            raise _tool_error(exc, task_id=task_id) from exc
        return json.dumps(task, ensure_ascii=False)


class AICheckerHumanizerToolkit(BaseToolkit):
    """Share one API wrapper across the tools."""

    api_wrapper: AICheckerHumanizerAPIWrapper = Field(
        default_factory=AICheckerHumanizerAPIWrapper
    )

    def get_tools(self) -> list[BaseTool]:
        """Return the detection, rewriting and result tools."""
        return [
            AICheckerHumanizerDetectTool(api_wrapper=self.api_wrapper),
            AICheckerHumanizerHumanizeTool(api_wrapper=self.api_wrapper),
            AICheckerHumanizerGetResultTool(api_wrapper=self.api_wrapper),
        ]
