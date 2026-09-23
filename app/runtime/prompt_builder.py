import json

from app.schemas.chat import ChatMessage, ToolCall
from app.server.errors import UnsupportedChatRoleError


class Mistral3PromptBuilder:
    """Builds a prompt string for `GGUFTokenizer.encode()` from a chat
    message list, matching the core structure of
    `mistralai/Ministral-3-3B-Instruct-2512`'s real (Jinja) chat template -
    confirmed against the real `chat_template.jinja` from that HF repo, not
    assumed: `[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]`, `[INST]...[/INST]` per
    user turn, assistant turns followed by `</s>`, plus M10's tool-calling
    support - `[AVAILABLE_TOOLS]<tools json>[/AVAILABLE_TOOLS]` once before
    the first user turn, `[TOOL_CALLS]<name>[ARGS]<args json>` per tool call
    on an assistant turn, `[TOOL_RESULTS]<content>[/TOOL_RESULTS]` for a
    "tool" role message.

    Deliberately still simplified from the real template: it doesn't
    implement the real template's consecutive-same-role message merging or
    role-ordering validation (real tool-calling flows don't send consecutive
    same-role turns in practice), and still injects no default system
    message when none is given - a vendor persona by default is wrong for a
    generic inference backend standing in for Ollama, not for being "Le
    Chat." Also out of scope: parsing a *generated* `[TOOL_CALLS]...[ARGS]...`
    span back out of the model's own output into a structured response
    `tool_calls` field - a real fine-tuned model emits `</s>` right after
    one (see the real template's own `{{- eos_token }}` placement), so it
    still terminates generation correctly, but the raw control-token text
    currently passes through as plain `content` rather than being
    restructured; a caller wanting to act on it must parse it itself for
    now.

    Each control-token substring emitted here (`[INST]`, `</s>`, etc.) is
    matched literally by GGUFTokenizer's control-token short-circuit, so the
    caller doesn't need to know their token ids - only their literal text.
    """

    def build(self, messages: list[ChatMessage], tools: list[dict] | None = None) -> str:
        parts = []
        tools_emitted = False
        for message in messages:
            if message.role == "user" and not tools_emitted:
                if tools:
                    parts.append(f"[AVAILABLE_TOOLS]{json.dumps(tools)}[/AVAILABLE_TOOLS]")
                tools_emitted = True
            parts.append(self._render_turn(message))
        return "".join(parts)

    def wants_bos(self, prompt: str) -> bool:
        """Always - this format has no BOS of its own (see app.runtime.chat_template)."""
        return True

    def _render_turn(self, message: ChatMessage) -> str:
        content = ("[IMG]" * len(message.images or [])) + message.content

        if message.role == "system":
            return f"[SYSTEM_PROMPT]{content}[/SYSTEM_PROMPT]"
        if message.role == "user":
            return f"[INST]{content}[/INST]"
        if message.role == "assistant":
            return content + self._render_tool_calls(message.tool_calls) + "</s>"
        if message.role == "tool":
            return f"[TOOL_RESULTS]{content}[/TOOL_RESULTS]"
        raise UnsupportedChatRoleError(f"Unsupported chat role: {message.role!r}")

    def _render_tool_calls(self, tool_calls: list[ToolCall] | None) -> str:
        if not tool_calls:
            return ""
        rendered = []
        for call in tool_calls:
            arguments = call.function.arguments
            if isinstance(arguments, str):
                arguments = arguments or "{}"
            else:
                arguments = json.dumps(arguments)
            rendered.append(f"[TOOL_CALLS]{call.function.name}[ARGS]{arguments}")
        return "".join(rendered)
