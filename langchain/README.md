# AI Checker Humanizer for LangChain

Use the [AI detection API](https://aicheckerhumanizer.com/ai-detector-api) to score a draft and the [AI humanizer API](https://aicheckerhumanizer.com/ai-humanizer-api) to rewrite it. Both tools work with LangChain v1.

## Install

```bash
pip install langchain-aicheckerhumanizer
```

Create a key on your [account page](https://aicheckerhumanizer.com/account#api) and set `AICHECKERHUMANIZER_API_KEY` in your environment.

## Call the tools directly

```python
from pathlib import Path

from langchain_aicheckerhumanizer import AICheckerHumanizerToolkit

toolkit = AICheckerHumanizerToolkit()
detect, humanize = toolkit.get_tools()
text = Path("draft.txt").read_text(encoding="utf-8")

print(detect.invoke({"text": text}))
print(humanize.invoke({"text": text, "mode": "standard", "strength": "medium"}))
```

Both tools return JSON strings and support `ainvoke`. Detection accepts up to 10,000 words and can return a null score with a reason. Rewriting accepts up to 3,000 words.

- Mode can be `standard` (default), `academic`, `casual` or `concise`.
- Strength can be `light`, `medium` (default) or `heavy`.

The rewrite tool waits up to 120 seconds for the task. If it returns a queued or running task, call `toolkit.api_wrapper.humanize_get(task_id)` later with the returned `taskId`. The result is available for about 70 minutes after submission.

## Use the toolkit in an agent

Run `pip install langchain` and install the provider package for your chat model. Choose a model that supports tool calls, set `MODEL` to its `provider:model` identifier and configure the provider's credentials.

```python
import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from langchain_aicheckerhumanizer import AICheckerHumanizerToolkit

toolkit = AICheckerHumanizerToolkit()
agent = create_agent(init_chat_model(os.environ["MODEL"]), tools=toolkit.get_tools())
```

See `examples/agent.py` for a complete invocation.

## Credits

Detection costs 1 credit per scored word when there is a verdict. A chunk is a paragraph or part of a long paragraph. The API charges 5 credits per original word in each chunk it successfully rewrites. Chunks under 10 words count as 10 for billing.

Reuse the same `idempotency_key`, text and options within 24 hours to avoid another charge for a completed check or accepted rewrite.

See the [API documentation](https://aicheckerhumanizer.com/docs/api) for response fields, errors and service limits.
