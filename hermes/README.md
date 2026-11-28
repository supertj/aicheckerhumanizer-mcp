# AI Checker Humanizer for Hermes Agent

Check a draft for AI writing with the [AI detection API](https://aicheckerhumanizer.com/ai-detector-api), or rewrite it with the [AI humanizer API](https://aicheckerhumanizer.com/ai-humanizer-api). The plugin can also retrieve a rewrite and check the credits left in your account.

## Install

To install from Git, use the repository URL with the plugin directory after `#`.

```sh
hermes plugins install 'https://github.com/supertj/aicheckerhumanizer-mcp#hermes'
```

Once the plugin is listed in the Hermes catalog, you can also install it by name.

```sh
hermes plugins install aicheckerhumanizer
```

1. Create an API key on your [account page](https://aicheckerhumanizer.com/account#api).
2. When the installer asks for the key, press Enter to skip it. That prompt echoes what you type.
3. Run `hermes plugins config aicheckerhumanizer` in the Hermes profile you want to use. Select `AICHECKERHUMANIZER_API_KEY` and enter the key at the masked prompt. On older versions without that command, set the key in the profile's Hermes `.env` file.
4. Run `hermes plugins enable aicheckerhumanizer` and start a new session.

## Tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `detect_ai_text` | `text` up to 10,000 words. Optional `idempotency_key`. | A 0–100 score and verdict when available, or a reason if there's no verdict. English and Portuguese can include sentence scores. Spanish and French get whole-text results. |
| `humanize_text` | `text` up to 3,000 words. The API splits the text into chunks and rejects requests over 200 chunks. Optional `mode` (`standard`, `academic`, `casual`, `concise`), `strength` (`light`, `medium`, `heavy`), and `idempotency_key`. Defaults are `standard` and `medium`. | The rewritten text when available. If the rewrite is still in progress or its status is unavailable, pass the returned `taskId` to `get_humanize_result`. |
| `get_humanize_result` | `task_id` from `humanize_text`. | The task status and any rewritten text. The result is available for about 70 minutes after submission. |
| `get_balance` | None. | Remaining credits and when each batch expires. |

`humanize_text` waits up to about 50 seconds for the rewrite. `skippedWords` counts words left unchanged because the selected mode doesn't support their language. If both the submission and the finished task report a count, use the finished task's count.

If every paragraph to rewrite is in English and the input has at least 150 English words that make up at least 80% of all words, the API returns a rewrite only if its detector rates the full text likely human. In that case the selected `mode` controls the rewrite and `strength` has no effect.

HTTP errors include `httpStatus`. For an expired rewrite, both `get_humanize_result` and a replay of the original submission return `httpStatus: 410`. The error can include `creditsCharged`, `expiredAt`, and the last task status in both `status` and `taskStatus`. See the [API documentation](https://aicheckerhumanizer.com/docs/api) for response fields and error codes.

## Credits

Detection costs one credit per scored word when there's a verdict. Each successfully rewritten chunk costs five credits per word in the original chunk, with a minimum of ten billable words per chunk. Task lookups and balance checks are free.

At $10 for 250,000 Starter credits, 1,000 scored words cost $0.04 and 1,000 billable rewrite words cost $0.20. Larger packs give more credits per dollar. Purchased credits expire after 365 days. Eligible accounts get 5,000 trial credits for 30 days when they create their first API key. The trial requires a verified, non-disposable email address and is available once per account and email address.

The plugin uses the same API key and credit balance as REST and the hosted MCP server. To avoid a second charge, reuse the `idempotency_key` with the same text and options within 24 hours. Use 1 to 128 visible ASCII characters (`!` through `~`). When you reuse the key, detection returns the saved result and a rewrite returns the original task ID.

`detect_ai_text` and `humanize_text` generate an idempotency key if you omit it. If a submission error includes `idempotency_key`, retry with that key. A timed-out request may have already used credits.

If a rewrite ends with reason `not_humanized`, no credits were charged. To retry after `not_humanized` or a failed or cancelled task, call `humanize_text` with a new `idempotency_key` or omit it. The result omits the old key because reusing it would return the same task.

## Disclosure

The plugin makes HTTPS requests only to `https://aicheckerhumanizer.com`. It sends the text you pass to the detection and rewrite tools, the task ID for result lookups, and the API key as a Bearer credential on each request. A rewrite that is still running when the tool returns can finish and use credits. The plugin stores nothing locally and sends no telemetry. Hermes manages the API key in the active profile.

## Run the tools directly

From the repository root, set `AICHECKERHUMANIZER_API_KEY` in your shell and run the example against a test account with credits available.

```sh
python -m hermes.examples.run_tools
```
