"""Register the AI Checker Humanizer tools with Hermes."""

from .tools import MODES, STRENGTHS, Tools

_IDEMPOTENCY_KEY = {
    "type": "string",
    "minLength": 1,
    "maxLength": 128,
    "pattern": "^[ -~]+$",
    "description": "Reuse the same key, text, and options within 24 hours to avoid a second charge.",
}


def register(ctx):
    from agent.secret_scope import UnscopedSecretError, get_secret

    def read_profile_secret(name):
        try:
            return get_secret(name)
        except UnscopedSecretError:
            return None

    tools = Tools(read_profile_secret)
    ctx.register_tool(
        name="detect_ai_text",
        toolset="aicheckerhumanizer",
        handler=tools.detect_ai_text,
        schema={
            "name": "detect_ai_text",
            "description": (
                "Check up to 10,000 words for AI writing. Returns a 0–100 score and verdict "
                "when available, or a reason if there is no verdict. English and Portuguese "
                "can include sentence scores. Spanish and French get whole-text results. "
                "Costs one credit per scored word only when there is a verdict. Replaying "
                "the same text and idempotency key within 24 hours won't charge you again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to check, up to 10,000 words."},
                    "idempotency_key": _IDEMPOTENCY_KEY,
                },
                "required": ["text"],
            },
        },
    )
    ctx.register_tool(
        name="humanize_text",
        toolset="aicheckerhumanizer",
        handler=tools.humanize_text,
        schema={
            "name": "humanize_text",
            "description": (
                "Rewrite up to 3,000 words. The API splits the text into chunks and rejects "
                "requests over 200 chunks. Returns the rewritten text when available. If the "
                "task is still running after the wait of up to 50 seconds, use its taskId with "
                "get_humanize_result. Each successfully rewritten chunk costs five "
                "credits per original word, with a ten-word minimum per chunk. Replaying "
                "the same text, options, and idempotency key within 24 hours won't charge "
                "you again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to rewrite, up to 3,000 words."},
                    "mode": {"type": "string", "enum": list(MODES), "default": "standard"},
                    "strength": {
                        "type": "string",
                        "enum": list(STRENGTHS),
                        "default": "medium",
                        "description": (
                            "Controls how much each chunk is rewritten. When every paragraph "
                            "to rewrite is English, the API ignores strength if the input has "
                            "at least 150 English words and at least 80% of its words are English."
                        ),
                    },
                    "idempotency_key": _IDEMPOTENCY_KEY,
                },
                "required": ["text"],
            },
        },
    )
    ctx.register_tool(
        name="get_humanize_result",
        toolset="aicheckerhumanizer",
        handler=tools.get_humanize_result,
        schema={
            "name": "get_humanize_result",
            "description": (
                "Look up a rewrite by task ID. Returns the task status and any rewritten "
                "text. Results are available for about 70 minutes after submission. "
                "Looking up a task costs no credits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "The taskId returned by humanize_text.",
                    },
                },
                "required": ["task_id"],
            },
        },
    )
    ctx.register_tool(
        name="get_balance",
        toolset="aicheckerhumanizer",
        handler=tools.get_balance,
        schema={
            "name": "get_balance",
            "description": "Show remaining API credits and when they expire. This call is free.",
            "parameters": {"type": "object", "properties": {}},
        },
    )
