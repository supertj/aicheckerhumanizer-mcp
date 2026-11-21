# AI Checker Humanizer MCP server

A hosted MCP server for [AI Checker Humanizer](https://aicheckerhumanizer.com/). Your assistant can check a draft for AI-written text, rewrite it, and look up your credit balance. For a direct connection, use a Streamable HTTP client and set the `Authorization` header. Requests sent from web pages on other sites are rejected. If your client can launch a local stdio server instead, use the bridge described below.

- Endpoint: `https://aicheckerhumanizer.com/api/mcp`
- Transport: Streamable HTTP
- Auth: `Authorization: Bearer ach_live_...`

Create a key on your [account page](https://aicheckerhumanizer.com/account#api). Eligible accounts get 5,000 trial credits, valid for 30 days, when they create an API key. You need a verified email that isn't a disposable address, and the trial is available once per account and once per email address.

## Tools

| Tool | Arguments | Returns |
|---|---|---|
| `detect_ai_text` | `text` (up to 10,000 words), optional `idempotency_key` | When there's a verdict, a 0–100 score and one or more labels. English and Portuguese also get per-sentence scores. Spanish and French get an overall score and verdict for the whole text. |
| `humanize_text` | `text` (up to 3,000 words). Optional: `mode` (`standard` by default, or `academic`, `casual`, `concise`), `strength` (`medium` by default, or `light`, `heavy`), `idempotency_key` | The call waits up to 50 seconds for a new rewrite and returns the text if it's ready. If the rewrite is still running, pass the returned `taskId` as `task_id` to `get_humanize_result`. |
| `get_humanize_result` | `task_id` | The task status and any rewritten text. |
| `get_balance` | none | Credits left and when they expire. |

Retrying within 24 hours with the same `idempotency_key`, text and options doesn't charge you twice. Retrying a completed check returns the saved result. Retrying an accepted rewrite returns the same task ID. You can fetch the rewritten text for about 70 minutes after submission.

## Setup

### Claude Code

```bash
claude mcp add --transport http aicheckerhumanizer \
  https://aicheckerhumanizer.com/api/mcp \
  --header "Authorization: Bearer ach_live_..."
```

### Cursor and other Streamable HTTP clients

For Cursor, put this in `~/.cursor/mcp.json` for all projects. If you use a project's `.cursor/mcp.json` instead, set the header to `Bearer ${env:ACH_API_KEY}` so the key stays out of the repository.

```json
{
  "mcpServers": {
    "aicheckerhumanizer": {
      "url": "https://aicheckerhumanizer.com/api/mcp",
      "headers": { "Authorization": "Bearer ach_live_..." }
    }
  }
}
```

### VS Code (existing Local sessions)

If you already use the Local harness in VS Code Chat, add this to `.vscode/mcp.json` and enter your API key when prompted. For a new Copilot setup, add the server under MCP Servers in Chat Customizations.

```json
{
  "inputs": [
    { "type": "promptString", "id": "ach-api-key", "description": "AI Checker Humanizer API key", "password": true }
  ],
  "servers": {
    "aicheckerhumanizer": {
      "type": "http",
      "url": "https://aicheckerhumanizer.com/api/mcp",
      "headers": { "Authorization": "Bearer ${input:ach-api-key}" }
    }
  }
}
```

### Claude Desktop and other clients that launch local stdio servers

Make sure Claude Desktop can run `npx` from your Node.js installation. Add this to `claude_desktop_config.json` and restart the app. The [mcp-remote](https://github.com/geelen/mcp-remote) bridge runs over stdio and forwards your `Authorization` header:

```json
{
  "mcpServers": {
    "aicheckerhumanizer": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://aicheckerhumanizer.com/api/mcp", "--header", "Authorization:${ACH_AUTH}"],
      "env": { "ACH_AUTH": "Bearer ach_live_..." }
    }
  }
}
```

## Example prompts

- "Check this cover letter for AI writing and tell me which sentences scored highest."
- "Humanize the second paragraph in academic mode with light strength, then run the AI check on the full draft again."
- "How many API credits do I have left?"

## Credits

Detection costs 1 credit per scored word when the detector returns a verdict. For each chunk that's rewritten successfully, humanizing costs 5 credits per word in the original chunk. Chunks shorter than 10 words are billed as 10. At $10 for 250,000 Starter credits, that's $0.04 per 1,000 scored words or $0.20 per 1,000 billable rewrite words. Larger packs give you more credits per dollar. Purchased credits expire after 365 days.

MCP and REST share one API key and one credit balance. Word limits, concurrency limits and idempotency work the same way on both.

## Links

- [API documentation](https://aicheckerhumanizer.com/docs/api): REST endpoints, response fields, errors and limits
- [AI detector API](https://aicheckerhumanizer.com/ai-detector-api)
- [AI humanizer API](https://aicheckerhumanizer.com/ai-humanizer-api)
- Questions: support@aicheckerhumanizer.com

## License

The configuration examples in this repository are MIT licensed. The hosted service is covered by the [AI Checker Humanizer terms](https://aicheckerhumanizer.com/terms).
