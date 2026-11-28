from collections.abc import Iterator

import pytest
from pytest_socket import disable_socket, enable_socket

from langchain_aicheckerhumanizer import AICheckerHumanizerAPIWrapper


@pytest.fixture(autouse=True)
def block_network() -> Iterator[None]:
    disable_socket(allow_unix_socket=True)
    try:
        yield
    finally:
        enable_socket()


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> AICheckerHumanizerAPIWrapper:
    monkeypatch.setenv("AICHECKERHUMANIZER_API_KEY", "ach_live_" + "x" * 32)
    return AICheckerHumanizerAPIWrapper()
