# AI Checker Humanizer for LangChain

Use the [AI detection API](https://aicheckerhumanizer.com/ai-detector-api) to score a draft and the [AI humanizer API](https://aicheckerhumanizer.com/ai-humanizer-api) to rewrite it. The toolkit works with LangChain v1.

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
detect, humanize, get_result = toolkit.get_tools()
text = Path("draft.txt").read_text(encoding="utf-8")

print(detect.invoke({"text": text}))
rewrite = humanize.invoke({"text": text, "mode": "standard", "strength": "medium"})
print(rewrite)
```

Each tool returns a JSON string on success and supports `ainvoke`. Detection accepts up to 10,000 words and can return a null score with a reason. Rewriting accepts up to 3,000 words.

- Mode can be `standard` (default), `academic`, `casual` or `concise`.
- Strength can be `light`, `medium` (default) or `heavy`.

By default, the rewrite tool waits up to 120 seconds after submission, including the status checks and the pauses between them. Set `humanize.max_wait_seconds` to change the limit.

If the rewrite is still pending when polling stops, you get the `taskId` and last known status. If a status check fails with a temporary error, the response also includes an `error` code. Check again with `get_result.invoke({"task_id": task_id})`. The result is available for about 70 minutes after submission.

When the API reports an expired result, `get_result` returns `result_expired` with `httpStatus` 410. Resubmitting with the same `idempotency_key` within 24 hours reports the same error. The response also includes `taskStatus`, `creditsCharged` and `expiredAt` when available.

`skippedWords` counts words left unchanged because their language is unsupported or the chosen mode is unavailable for that language. If a status check leaves out `skippedWords`, the rewrite tool uses the count from submission when available.

With the default error handling, a failed tool call returns a JSON string with the API's error fields and `httpStatus` if the server responded.

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

Detection costs 1 credit per scored word when there is a verdict. Rewriting costs 5 credits per original word in each chunk it successfully rewrites. A chunk is a paragraph or part of a long paragraph. Chunks under 10 words count as 10 for billing.

Reuse the same `idempotency_key`, text and options within 24 hours to avoid another charge for a completed check or accepted rewrite. The detect and rewrite tools generate a key when you omit it. If the error JSON includes `idempotencyKey`, pass it as `idempotency_key` when retrying the same request. The field lets you recover a submission that may have reached the server. It appears on transport failures (including timeouts), response decoding failures, invalid 2xx responses, HTTP 5xx errors and `in_progress` errors.

See the [API documentation](https://aicheckerhumanizer.com/docs/api) for response fields, errors and service limits.
