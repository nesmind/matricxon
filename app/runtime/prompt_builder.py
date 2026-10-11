import json

from app.schemas.chat import ChatMessage, ToolCall
from app.server.errors import UnsupportedChatRoleError


class Mistral3PromptBuilder:
    """Builds a prompt string for `GGUFTokenizer.encode` from a chat message list, matching the core
    structure of `mistralai/Ministral-3-3B-Instruct-2512`'s Jinja chat template (checked against its
    `chat_template.jinja`): `[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]`, `[INST]...[/INST]` per user turn,
    assistant turns followed by `</s>`, plus tool calling: `[AVAILABLE_TOOLS]<tools
    json>[/AVAILABLE_TOOLS]` once before the first user turn, `[TOOL_CALLS]<name>[ARGS]<args json>`
    per tool call on an assistant turn, `[TOOL_RESULTS]<content>[/TOOL_RESULTS]` for a "tool"
    message.

    Simplified: no consecutive-same-role merging or role-order validation (tool flows don't send
    consecutive same-role turns), and no default system message (a vendor persona is wrong for a
    generic backend standing in for Ollama). Parsing a generated `[TOOL_CALLS]...[ARGS]...` span
    back into `tool_calls` is out of scope: a fine-tuned model emits `</s>` right after one, so
    generation still ends, but the raw control-token text passes through as `content` for the caller
    to parse.

    Each control-token substring (`[INST]`, `</s>`, ...) is matched literally by GGUFTokenizer's
    control-token short-circuit, so callers need only the literal text.
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
        """Always: this format has no BOS of its own (app.runtime.chat_template)."""
        return True

    def _render_turn(self, message: ChatMessage) -> str:
        content = ("[IMG]" * len(message.images or [])) + message.content

        if message.role == "system":
            return f"[SYSTEM_PROMPT]{content}[/SYSTEM_PROMPT]"
        if message.role == "user":
            return f"[INST]{content}[/INST]"
        if message.role == "assistant":
            return content + _render_tool_calls(message.tool_calls) + "</s>"
        if message.role == "tool":
            return f"[TOOL_RESULTS]{content}[/TOOL_RESULTS]"
        raise UnsupportedChatRoleError(f"Unsupported chat role: {message.role!r}")


def _render_tool_calls(tool_calls: list[ToolCall] | None) -> str:
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


class LegacyMistralPromptBuilder:
    """Fallback for a GGUF with no `tokenizer.chat_template` whose `general.architecture` isn't
    `mistral3`: how Mistral-7B-Instruct v0.1/v0.2 and most community fine-tunes get tagged by
    llama.cpp conversion (close enough to Llama to be `"llama"`), many without chat_template
    metadata.

    Renders the older plain format they were trained on: `[INST]... [/INST]` per user turn, `</s>`
    after each assistant turn, no `[SYSTEM_PROMPT]` tag (a Mistral3-tokenizer-only control token
    these models only saw as text). On a Hebrew Mistral fine-tune, whose vocab has no
    `[SYSTEM_PROMPT]`/`[INST]` special tokens, Mistral3PromptBuilder's format made it echo
    `[/SYSTEM_PROMPT]` and invent closing tags. A system message is folded into the front of the
    next user turn: the official v0.1/v0.2 template has no system role, and community templates do
    the same.
    """

    def build(self, messages: list[ChatMessage], tools: list[dict] | None = None) -> str:
        parts = []
        pending_system: list[str] = []
        tools_emitted = False
        for message in messages:
            if message.role == "system":
                pending_system.append(message.content)
            elif message.role == "user":
                content = ("[IMG]" * len(message.images or [])) + message.content
                if pending_system:
                    content = "\n\n".join(pending_system) + "\n\n" + content
                    pending_system = []
                prefix = ""
                if tools and not tools_emitted:
                    prefix = f"[AVAILABLE_TOOLS]{json.dumps(tools)}[/AVAILABLE_TOOLS]"
                tools_emitted = True
                parts.append(f"{prefix}[INST] {content} [/INST]")
            elif message.role == "assistant":
                parts.append(message.content + _render_tool_calls(message.tool_calls) + "</s>")
            elif message.role == "tool":
                parts.append(f"[TOOL_RESULTS]{message.content}[/TOOL_RESULTS]")
            else:
                raise UnsupportedChatRoleError(f"Unsupported chat role: {message.role!r}")
        return "".join(parts)

    def wants_bos(self, prompt: str) -> bool:
        """Always: this format has no BOS of its own (app.runtime.chat_template)."""
        return True


class VicunaPromptBuilder:
    """Vicuna v1.1's conversation format (FastChat's `vicuna_v1` template, which LLaVA's
    Vicuna-based checkpoints are fine-tuned on): a plain-text system preamble, `USER: {msg}` /
    `ASSISTANT: {msg}` turns separated by newlines, `</s>` after each assistant turn. Checked
    against Ollama's raw template for this family and FastChat's default system string.

    Selected by tag/filename match (chat_template.PromptBuilderFactory.for_metadata,
    `_VICUNA_TAG_MARKER`), not GGUF metadata: a Vicuna GGUF has no `tokenizer.chat_template` and
    nothing says "vicuna" (a LLaVA-Vicuna model has `general.name = "LLaMA v2"`), so it would fall
    into LegacyMistralPromptBuilder's Mistral-shaped guess, which gave an empty 1-token
    immediate-EOS reply once a system prompt was included.
    """

    _DEFAULT_SYSTEM = (
        "A chat between a curious human and an artificial intelligence assistant. The assistant "
        "gives helpful, detailed, and polite answers to the human's questions."
    )

    def build(self, messages: list[ChatMessage], tools: list[dict] | None = None) -> str:
        # Falls back to _DEFAULT_SYSTEM only when the caller sent no system message: one with empty
        # content (pAIring's chat_service always sends a system message so this is distinguishable)
        # means "genuinely no preamble", not "no opinion", and stays blank.
        has_system_message = any(m.role == "system" for m in messages)
        system_parts = [m.content for m in messages if m.role == "system"]
        system = "\n\n".join(system_parts) if has_system_message else self._DEFAULT_SYSTEM
        parts = [system, "\n\n"]
        for message in messages:
            content = ("[IMG]" * len(message.images or [])) + message.content
            if message.role == "system":
                continue
            if message.role in ("user", "tool"):
                # Vicuna has no tool-result role: folded in as a user-shaped turn.
                # LegacyMistralPromptBuilder's [TOOL_RESULTS] wrapping avoids this only because
                # Mistral's template defines one.
                parts.append(f"USER: {content}\n\n")
            elif message.role == "assistant":
                tool_calls = _render_tool_calls(message.tool_calls)
                parts.append(f"ASSISTANT: {content}{tool_calls}</s>\n\n")
            else:
                raise UnsupportedChatRoleError(f"Unsupported chat role: {message.role!r}")
        parts.append("ASSISTANT:")
        return "".join(parts)

    def wants_bos(self, prompt: str) -> bool:
        """Always: this format has no BOS of its own (app.runtime.chat_template)."""
        return True
