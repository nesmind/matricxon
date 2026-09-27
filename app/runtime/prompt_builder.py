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
    `mistral3` - historically how Mistral-7B-Instruct v0.1/v0.2 and most of their community
    fine-tunes get tagged by llama.cpp conversion (structurally close enough to Llama to be
    classed as `general.architecture == "llama"`), and many such fine-tunes ship no
    chat_template metadata at all.

    Renders the older, plainer format these were actually trained on: `[INST] ... [/INST]` per
    user turn, `</s>` after each assistant turn, and no `[SYSTEM_PROMPT]` tag - that control
    token is Mistral3-tokenizer-only and these older models have never seen it as anything but
    ordinary text. Confirmed against Hebrew-Mistral-7B-Q5_K_M (2026-09-27): its vocab has no
    `[SYSTEM_PROMPT]`/`[INST]` special tokens, and Mistral3PromptBuilder's format made it echo
    `[/SYSTEM_PROMPT]` back and invent its own closing tags rather than answering. A system
    message, if present, is folded into the front of the next user turn instead - the official
    v0.1/v0.2 template has no system role at all, and this is the same workaround most
    community templates for those models use.
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
        """Always - this format has no BOS of its own (see app.runtime.chat_template)."""
        return True


class VicunaPromptBuilder:
    """Vicuna v1.1's own conversation format (FastChat's `vicuna_v1` conv_template, the same one
    LLaVA's Vicuna-based checkpoints - e.g. `llava-v1.6-vicuna-7b` - are fine-tuned on top of):
    a plain-text system preamble, `USER: {msg}` / `ASSISTANT: {msg}` turns separated by newlines,
    `</s>` after each assistant turn. Confirmed against Ollama's own published raw template for
    this exact model family (`{{ .System }}\\n\\nUSER: {{ .Prompt }}\\n\\nASSISTANT:`) and
    FastChat's default system string, not assumed from "it's llama-architecture" alone.

    Selected by tag/filename match (see app.runtime.chat_template.PromptBuilderFactory.for_metadata
    and its own `_VICUNA_TAG_MARKER`), not GGUF metadata - a Vicuna fine-tune's GGUF carries no
    `tokenizer.chat_template` and nothing in its metadata says "vicuna" (confirmed live,
    2026-09-27, llava-v1.6-vicuna-7b: `general.name = "LLaMA v2"`), so there is no metadata-only
    way to detect this - it would otherwise silently fall into LegacyMistralPromptBuilder's
    Mistral-shaped guess, which is wrong for this format and produced a real, empty (1-token,
    immediate-EOS) reply once a real system prompt was included."""

    _DEFAULT_SYSTEM = (
        "A chat between a curious human and an artificial intelligence assistant. The assistant "
        "gives helpful, detailed, and polite answers to the human's questions."
    )

    def build(self, messages: list[ChatMessage], tools: list[dict] | None = None) -> str:
        system_parts = [m.content for m in messages if m.role == "system"]
        system = "\n\n".join(system_parts) if system_parts else self._DEFAULT_SYSTEM
        parts = [system, "\n\n"]
        for message in messages:
            content = ("[IMG]" * len(message.images or [])) + message.content
            if message.role == "system":
                continue
            if message.role in ("user", "tool"):
                # Vicuna has no native tool-result role - folded in as a user-shaped turn, the
                # same "no real convention for this, so treat it as a message" choice
                # LegacyMistralPromptBuilder's own [TOOL_RESULTS] wrapping avoids only because
                # Mistral's real template actually defines one; Vicuna's doesn't.
                parts.append(f"USER: {content}\n\n")
            elif message.role == "assistant":
                tool_calls = _render_tool_calls(message.tool_calls)
                parts.append(f"ASSISTANT: {content}{tool_calls}</s>\n\n")
            else:
                raise UnsupportedChatRoleError(f"Unsupported chat role: {message.role!r}")
        parts.append("ASSISTANT:")
        return "".join(parts)

    def wants_bos(self, prompt: str) -> bool:
        """Always - this format has no BOS of its own (see app.runtime.chat_template)."""
        return True
