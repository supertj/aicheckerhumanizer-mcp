import os
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_aicheckerhumanizer import (
    AICheckerHumanizerDetectTool,
    AICheckerHumanizerHumanizeTool,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("AICHECKERHUMANIZER_API_KEY"),
    reason="Set AICHECKERHUMANIZER_API_KEY to run live integration tests.",
)

# Reuse the same submissions across the standard sync and async checks.
_RUN_ID = uuid4().hex
_DRAFT = (
    "The library moved the reference desk last month. It now sits just inside "
    "the south entrance, across from the cart where the staff leave books for "
    "people to browse. I stopped by Tuesday afternoon to ask about an old "
    "street map and was surprised to find two versions in the local history "
    "room. One showed a footbridge behind the school. The other had the same "
    "bridge drawn as a dotted line with no name. A librarian suggested checking "
    "the council minutes to see when the bridge was removed. The minutes were "
    "filed by year, so I started with the date on the newer map. It turned out "
    "that the bridge stayed in use for several years after the council approved "
    "its replacement. The new bridge went in farther upstream, where the banks "
    "are flatter and the path can reach the school without crossing the "
    "playing field. I made a note to walk that route on my way home."
)


class _ToolsLive(ToolsIntegrationTests):
    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"handle_tool_error": False}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "text": _DRAFT,
            "idempotency_key": f"{type(self).__name__}-{_RUN_ID}",
        }


class TestDetectLive(_ToolsLive):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return AICheckerHumanizerDetectTool


class TestHumanizeLive(_ToolsLive):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return AICheckerHumanizerHumanizeTool
