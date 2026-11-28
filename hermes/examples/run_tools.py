"""Run the four tool handlers against the live API with a test-account key."""

import json
from os import getenv

from hermes.tools import API_KEY_NAME, Tools

SAMPLE = (
    "In today's rapidly evolving professional landscape, remote work has become an increasingly "
    "important component of organizational strategy. By giving employees greater flexibility "
    "in where they work, companies can support productivity while expanding access to talent "
    "across different regions. However, the successful implementation of remote work requires "
    "a proactive approach to communication and coordination. Managers should establish clear "
    "expectations about availability, create consistent opportunities for feedback, and provide "
    "employees with the resources they need to perform their responsibilities effectively. "
    "At the same time, team members can maintain strong working relationships by sharing "
    "updates regularly and making space for informal conversations. It is also important to "
    "recognize that remote work can make it harder to maintain boundaries between personal "
    "and professional life. Encouraging regular breaks and respecting agreed working hours "
    "can help address this challenge. Ultimately, organizations that adapt their practices to "
    "the needs of distributed teams are better positioned to sustain engagement over time."
)


def main():
    key = getenv(API_KEY_NAME)
    if not key:
        raise SystemExit(f"Set {API_KEY_NAME} before running this example.")

    handlers = Tools(lambda _name: key)
    print("detect_ai_text", handlers.detect_ai_text({"text": SAMPLE}), flush=True)
    result = handlers.humanize_text({"text": SAMPLE})
    print("humanize_text", result, flush=True)
    task_id = json.loads(result).get("taskId")
    if task_id:
        print("get_humanize_result", handlers.get_humanize_result({"task_id": task_id}), flush=True)
    else:
        print("get_humanize_result skipped because no task ID was returned.", flush=True)
    print("get_balance", handlers.get_balance({}), flush=True)


if __name__ == "__main__":
    main()
