"""LangChain tools for AI detection and rewriting."""

import asyncio
import json
from time import monotonic, sleep
from typing import Any

from langchain_core.tools import BaseTool, BaseToolkit, ToolException
from pydantic import BaseModel, Field

from .api import (
    AICheckerHumanizerAPIWrapper,
    AICheckerHumanizerError,
    HumanizeMode,
    HumanizeStrength,
)

_PENDING_STATUSES = ("queued", "running")
_DEFAULT_POLL_SECONDS = 3
_GET_RESULT_NAME = "ai_checker_humanizer_get_result"


class _DetectInput(BaseModel):
    text: str = Field(description="Text to check, up to 10,000 words.")
    idempotency_key: str | None = Field(
        default=None,
        max_length=128,
        pattern=r"^[ -~]*$",
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
        max_length=128,
        pattern=r"^[ -~]*$",
        description=(
            "Reuse with the same text and options within 24 hours "
            "to get the same task without another charge."
        ),
    )


class _GetResultInput(BaseModel):
    task_id: str = Field(description="The taskId returned by the rewrite tool.")


class AICheckerHumanizerDetectTool(BaseTool):
    """Check text for AI writing."""

    name: str = "ai_checker_humanizer_detect"
    description: str = (
        "Check up to 10,000 words for AI writing. Returns JSON with a 0–100 score, "
        "verdict and scored word count. English and Portuguese can include sentence "
        "scores. If there is no verdict, the response gives a reason. "
        "The API charges 1 credit per scored "
        "word only when it returns a verdict. Repeat the same text with the same "
        "idempotency_key within 24 hours to avoid a second charge."
    )
    args_schema: type[BaseModel] = _DetectInput
    api_wrapper: AICheckerHumanizerAPIWrapper = Field(
        default_factory=AICheckerHumanizerAPIWrapper
    )
    handle_tool_error: bool = True

    def _run(self, text: str, idempotency_key: str | None = None) -> str:
        try:
            result = self.api_wrapper.detect(text, idempotency_key=idempotency_key)
        except AICheckerHumanizerError as exc:
            raise ToolException(exc.message) from exc
        return json.dumps(result, ensure_ascii=False)

    async def _arun(self, text: str, idempotency_key: str | None = None) -> str:
        try:
            result = await self.api_wrapper.adetect(
                text, idempotency_key=idempotency_key
            )
        except AICheckerHumanizerError as exc:
            raise ToolException(exc.message) from exc
        return json.dumps(result, ensure_ascii=False)


def _poll_delay(task: dict[str, Any]) -> int:
    hint = task.get("retryAfter")
    if type(hint) is int and hint > 0:
        return hint
    return _DEFAULT_POLL_SECONDS


def _humanize_result(
    task: dict[str, Any], error: AICheckerHumanizerError | None = None
) -> str:
    if task.get("status") in _PENDING_STATUSES:
        message = (
            f"Check again later with {_GET_RESULT_NAME}. "
            "Set task_id to the returned taskId. "
            "Results are available for about 70 minutes after submission."
        )
        task = {
            "taskId": task["taskId"],
            "status": task["status"],
            "message": message,
        }
        if error is not None:
            task["error"] = error.code
            task["message"] = f"{error.message} {message}"
            if error.retry_after is not None:
                task["retryAfter"] = error.retry_after
    return json.dumps(task, ensure_ascii=False)


class AICheckerHumanizerHumanizeTool(BaseTool):
    """Rewrite text with AI Checker Humanizer."""

    name: str = "ai_checker_humanizer_humanize"
    description: str = (
        "Rewrite up to 3,000 words. Choose standard, academic, casual or concise "
        "mode and light, medium or heavy strength. Defaults are standard and medium. "
        "The API charges 5 credits per original word in each rewritten chunk "
        "(minimum 10 words per chunk). Repeat the same text and options "
        "with the same idempotency_key within 24 hours to get the same task without "
        "another charge. Returns JSON with the rewritten text and any available "
        "scores, or a task ID and status if the rewrite is still pending."
    )
    args_schema: type[BaseModel] = _HumanizeInput
    api_wrapper: AICheckerHumanizerAPIWrapper = Field(
        default_factory=AICheckerHumanizerAPIWrapper
    )
    max_wait_seconds: float = Field(default=120, ge=0, allow_inf_nan=False)
    handle_tool_error: bool = True

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
            raise ToolException(exc.message) from exc

        deadline = monotonic() + self.max_wait_seconds
        while task.get("status") in _PENDING_STATUSES:
            delay = _poll_delay(task)
            if delay >= deadline - monotonic():
                break
            sleep(delay)
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            try:
                task = self.api_wrapper.humanize_get(
                    task["taskId"], timeout=min(self.api_wrapper.timeout, remaining)
                )
            except AICheckerHumanizerError as exc:
                return _humanize_result(task, exc)
        return _humanize_result(task)

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
            raise ToolException(exc.message) from exc

        deadline = monotonic() + self.max_wait_seconds
        while task.get("status") in _PENDING_STATUSES:
            delay = _poll_delay(task)
            if delay >= deadline - monotonic():
                break
            await asyncio.sleep(delay)
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            try:
                task = await asyncio.wait_for(
                    self.api_wrapper.ahumanize_get(task["taskId"]),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                break
            except AICheckerHumanizerError as exc:
                return _humanize_result(task, exc)
        return _humanize_result(task)


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
    handle_tool_error: bool = True

    def _run(self, task_id: str) -> str:
        try:
            task = self.api_wrapper.humanize_get(task_id)
        except AICheckerHumanizerError as exc:
            raise ToolException(exc.message) from exc
        return json.dumps(task, ensure_ascii=False)

    async def _arun(self, task_id: str) -> str:
        try:
            task = await self.api_wrapper.ahumanize_get(task_id)
        except AICheckerHumanizerError as exc:
            raise ToolException(exc.message) from exc
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
