# AI Checker Humanizer MCP server

A hosted MCP server for [AI Checker Humanizer](https://aicheckerhumanizer.com/). Your assistant can check a draft for AI-written text, rewrite it, and look up your credit balance. If your client can connect to a remote server over Streamable HTTP, add the URL and your API key. If it can't send a custom header, use the local bridge described below.

- Endpoint: `https://aicheckerhumanizer.com/api/mcp`
- Transport: Streamable HTTP
- Auth: `Authorization: Bearer ach_live_...`

Create a key on your [account page](https://aicheckerhumanizer.com/account#api). With a verified email, you get 5,000 trial credits when you create your first key. They expire after 30 days.

## Tools

| Tool | Arguments | Returns |
|---|---|---|
| `detect_ai_text` | `text` (up to 10,000 words), optional `idempotency_key` | A 0–100 score and a label. English and Portuguese also get per-sentence scores. Spanish and French get a whole-text verdict. |
| `humanize_text` | `text` (up to 3,000 words), `mode` (`standard`, `academic`, `casual`, `concise`), `strength` (`light`, `medium`, `heavy`), optional `idempotency_key` | The rewritten text. The call waits up to 50 seconds. If it returns a `taskId`, pass that value as `task_id` to `get_humanize_result`. |
| `get_humanize_result` | `task_id` | The task status and, once it's finished, the rewritten text. |
| `get_balance` | none | Credits left and when they expire. |

Retry within 24 hours with the same `idempotency_key`, text and options, and you aren't charged again: detection returns the earlier result, and humanizing returns the same task.

## Setup

### Claude Code

```bash
claude mcp add --transport http aicheckerhumanizer \
  https://aicheckerhumanizer.com/api/mcp \
  --header "Authorization: Bearer ach_live_..."
```

### Cursor and other clients that take a URL

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

### VS Code

Put this in `.vscode/mcp.json`. VS Code prompts for the key when the server first starts.

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

### Claude Desktop, or any client that can't send a custom header

Run [mcp-remote](https://github.com/geelen/mcp-remote) as a local bridge. It adds the header and talks to your client over stdio:

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
- "Humanize the second paragraph in academic mode with light strength, then check it again."
- "How many API credits do I have left?"

## Credits

Detection costs 1 credit per scored word when the detector returns a verdict. Humanizing costs 5 credits per word. At $10 for 250,000 Starter credits, that's $0.04 per 1,000 detected words or $0.20 per 1,000 humanized words. Larger packs give you more credits per dollar. Purchased credits expire after 365 days.

MCP and REST share one API key and one credit balance. Word limits, concurrency limits and idempotency work the same way on both.

## Links

- [API documentation](https://aicheckerhumanizer.com/docs/api): REST endpoints, response fields, errors and limits
- [AI detector API](https://aicheckerhumanizer.com/ai-detector-api)
- [AI humanizer API](https://aicheckerhumanizer.com/ai-humanizer-api)
- Questions: support@aicheckerhumanizer.com

## License

The configuration examples in this repository are MIT licensed. The hosted service is covered by the [AI Checker Humanizer terms](https://aicheckerhumanizer.com/terms).
