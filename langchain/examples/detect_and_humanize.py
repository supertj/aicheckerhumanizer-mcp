"""Check a sample, rewrite it, and check the rewrite."""

import json
from typing import Any

from langchain_core.tools import BaseTool

from langchain_aicheckerhumanizer import (
    AICheckerHumanizerAPIWrapper,
    AICheckerHumanizerDetectTool,
    AICheckerHumanizerHumanizeTool,
)

SAMPLE_TEXT = (
    "Remote work has become an increasingly common operating model for organizations "
    "that want to recruit beyond a single city. By giving employees more control over "
    "where they work, companies can reduce commuting time and make it easier for staff "
    "to manage commitments outside the office. However, the success of a distributed "
    "team depends on clear communication and deliberate coordination. Managers should "
    "document decisions where everyone can find them and specify when colleagues can "
    "expect a response. Meetings should have a clear purpose, especially when team "
    "members are spread across time zones. It is also important to give new hires a "
    "regular point of contact so questions do not pile up during their first few "
    "weeks. "
    "Teams can use recurring check-ins to identify obstacles early and adjust their "
    "working agreements as needed. With these practices in place, employers can "
    "support flexible schedules while keeping projects on track and making "
    "responsibilities clear to everyone."
)


def _invoke_json(tool: BaseTool, **inputs: str) -> dict[str, Any]:
    result: str = tool.invoke(inputs)
    try:
        decoded = json.loads(result)
    except json.JSONDecodeError:
        raise SystemExit(result) from None
    if not isinstance(decoded, dict):
        raise SystemExit("Expected a JSON object from the tool.")
    return decoded


def main() -> None:
    api = AICheckerHumanizerAPIWrapper()
    detect = AICheckerHumanizerDetectTool(api_wrapper=api)
    humanize = AICheckerHumanizerHumanizeTool(api_wrapper=api)

    before = _invoke_json(detect, text=SAMPLE_TEXT)
    print("Detection result")
    print(json.dumps(before, indent=2, ensure_ascii=False))

    rewrite = _invoke_json(
        humanize, text=SAMPLE_TEXT, mode="standard", strength="medium"
    )
    rewritten_text = rewrite.get("output")
    if not rewritten_text:
        print(json.dumps(rewrite, indent=2, ensure_ascii=False))
        return

    print("\nRewritten text")
    print(rewritten_text)

    after = _invoke_json(detect, text=rewritten_text)
    print(f"\nOriginal score {before.get('score')}")
    print(f"Rewrite score {after.get('score')}")


if __name__ == "__main__":
    main()
