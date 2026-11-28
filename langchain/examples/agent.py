"""Give a LangChain agent both tools."""

import os

from detect_and_humanize import SAMPLE_TEXT
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from langchain_aicheckerhumanizer import AICheckerHumanizerToolkit


def main() -> None:
    model = init_chat_model(os.environ["MODEL"])
    agent = create_agent(model, tools=AICheckerHumanizerToolkit().get_tools())
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Check the paragraph below for AI writing. Report its score. "
                        "If the score is above 50, rewrite the paragraph and show me "
                        f"the rewritten text.\n\n{SAMPLE_TEXT}"
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
