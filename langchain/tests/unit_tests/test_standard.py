from collections.abc import Iterator
from typing import Any

import pytest
import respx
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests
from pydantic import SecretStr

from langchain_aicheckerhumanizer import (
    AICheckerHumanizerAPIWrapper,
    AICheckerHumanizerDetectTool,
    AICheckerHumanizerHumanizeTool,
)

_KEY = "ach_live_" + "x" * 32


class _ToolsStandard(ToolsUnitTests):
    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_wrapper": AICheckerHumanizerAPIWrapper(api_key=SecretStr(_KEY))}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "A draft paragraph."}

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"AICHECKERHUMANIZER_API_KEY": _KEY},
            {},
            self.tool_constructor_params,
        )

    @pytest.fixture(autouse=True)
    def mock_http(self) -> Iterator[None]:
        base = "https://aicheckerhumanizer.com/api/v1"
        with respx.mock(assert_all_called=False) as router:
            router.post(f"{base}/detect").respond(json={"score": 30})
            router.post(f"{base}/humanize").respond(
                202,
                json={
                    "taskId": "0f92ace1-57c9-4dd1-a675-5c3967295244",
                    "status": "completed",
                    "output": "Edited draft.",
                },
            )
            yield


class TestDetectStandard(_ToolsStandard):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return AICheckerHumanizerDetectTool


class TestHumanizeStandard(_ToolsStandard):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return AICheckerHumanizerHumanizeTool
