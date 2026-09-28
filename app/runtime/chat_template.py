"""Prompt building from a model's own chat template - the Jinja `tokenizer.chat_template` string
every modern instruct GGUF carries (the same template Hugging Face's `apply_chat_template`,
llama.cpp and Ollama all render). Replaces the old "every model gets Mistral's `[INST]` format"
behavior: a real Llama-3.2 chat through pAIring (2026-09-23) got `[INST]`-formatted prompts,
answered with empty replies or echoed `</s>[INST]...[/INST]` as plain text, never emitting its
real `<|eot_id|>` stop token.

`mistral3` keeps `Mistral3PromptBuilder` (hand-verified, including tool calling). No template and
not mistral3 falls back to `LegacyMistralPromptBuilder` (see its own docstring) -
Mistral3PromptBuilder's `[SYSTEM_PROMPT]` tag is Mistral-newer-tokenizer-specific and broke an
older Mistral fine-tune the same way `[INST]`-for-everything broke Llama 3.2
(Hebrew-Mistral-7B-Q5_K_M, 2026-09-27). A tag/filename matching "vicuna" gets
`VicunaPromptBuilder` instead - its own GGUF gives no metadata-only way to detect the format
(llava-v1.6-vicuna-7b: indistinguishable from countless other llama-arch models by metadata alone
- the wrong Mistral-shaped guess produced a real, empty, immediate-EOS reply).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from app.gguf.metadata import GGUFMetadata
from app.gguf.reader import GGUFReader
from app.runtime.prompt_builder import (
    LegacyMistralPromptBuilder,
    Mistral3PromptBuilder,
    VicunaPromptBuilder,
)
from app.runtime.vision_fusion import IMAGE_MARKER
from app.schemas.chat import ChatMessage
from app.server.errors import MatricxonError

# Name-based heuristic (see PromptBuilderFactory.for_metadata) - a Vicuna fine-tune's GGUF gives
# no metadata-only way to detect this (see module docstring), same spirit as
# app.models.capabilities._THINKING_MARKERS' own repo/filename substring match.
_VICUNA_TAG_MARKER = "vicuna"

# Built-in templates for real models whose GGUF ships no `tokenizer.chat_template`, keyed by
# `general.name`: (template, always add BOS). moondream2's own reference format (BOS first,
# system messages skipped - only confuse a 1.9B model) - the old Mistral fallback's first
# generated token was EOS here (empty reply, 2026-09-23).
_MOONDREAM_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'user' %}{{ m.image_markers + '\n\nQuestion: ' + m.text }}"
    "{% elif m.role == 'assistant' %}{{ '\n\nAnswer: ' + m.text }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '\n\nAnswer:' }}{% endif %}"
)
_BUILTIN_TEMPLATES = {"moondream2": (_MOONDREAM_TEMPLATE, True)}

# Llama 3.x's own template always opens with a system block, even with no system message/tools;
# Ollama's llama3 template only writes one when there is a system message (or tools) - for "Say
# hi." Ollama prefilled 13 tokens, Matricxon 40 (2026-09-23), most of a short reply's CPU latency.
# Used whenever there are no tools; with tools, the model's own template still renders them.
_LLAMA3_COMPACT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}"
    "{{ '<|start_header_id|>' + ('ipython' if m.role == 'tool' else m.role)"
    " + '<|end_header_id|>\n\n' + m.content + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)
_LLAMA3_TEMPLATE_MARKERS = ("<|start_header_id|>", "Cutting Knowledge Date")

# Real marker (`ggml-org/SmolVLM2-2.2B-Instruct-GGUF`'s own template:
# `message['content'][0]['type']`) for a template expecting `content` as a real list of
# `{"type": ...}` parts (HF's own multi-modal convention), not a flat string. Confirmed live: a
# flat string crashes on empty content (`content[0]` - "str object has no element 0") or, worse,
# silently renders empty on non-empty content (`content[0]` returns a 1-char string, not a dict,
# so `['type']` is Jinja `Undefined` - every `{% if line['type'] == ... %}` never matches).
_STRUCTURED_CONTENT_MARKERS = ("content'][0]", 'content"][0]')


class PromptBuilder(Protocol):
    def build(self, messages: list[ChatMessage], tools: list[dict] | None = None) -> str: ...

    def wants_bos(self, prompt: str) -> bool: ...


class ChatTemplateError(MatricxonError):
    status_code = 400


class ChatTemplatePromptBuilder:
    def __init__(
        self,
        template: str,
        bos_token: str,
        eos_token: str,
        add_bos: bool = False,
        no_tools_template: str | None = None,
        structured_content: bool = False,
    ) -> None:
        # Same Jinja settings HF's own apply_chat_template uses (whitespace control in particular).
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = self._tojson
        env.globals["raise_exception"] = self._raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self._template = env.from_string(template)
        # Rendered instead of `template` when a request has no tools (see _LLAMA3_COMPACT_TEMPLATE).
        self._no_tools_template = env.from_string(no_tools_template) if no_tools_template else None
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._add_bos = add_bos
        self._structured_content = structured_content

    @classmethod
    def from_metadata(cls, metadata: GGUFMetadata) -> "ChatTemplatePromptBuilder | None":
        template = metadata.get("tokenizer.chat_template")
        add_bos = bool(metadata.get("tokenizer.ggml.add_bos_token", False))
        if not template:
            builtin = _BUILTIN_TEMPLATES.get(str(metadata.get("general.name", "")).lower())
            if builtin is None:
                return None
            template, add_bos = builtin
        tokens: list[str] = metadata.get("tokenizer.ggml.tokens") or []

        def token_text(key: str) -> str:
            token_id = metadata.get(key)
            return tokens[token_id] if token_id is not None and token_id < len(tokens) else ""

        is_llama3 = all(marker in template for marker in _LLAMA3_TEMPLATE_MARKERS)
        structured_content = any(marker in template for marker in _STRUCTURED_CONTENT_MARKERS)
        return cls(
            template,
            token_text("tokenizer.ggml.bos_token_id"),
            token_text("tokenizer.ggml.eos_token_id"),
            add_bos,
            _LLAMA3_COMPACT_TEMPLATE if is_llama3 else None,
            structured_content,
        )

    @staticmethod
    def _tojson(value: Any, ensure_ascii: bool = False, indent: int | None = None, **_: Any) -> str:
        # Hugging Face's own override: Jinja's built-in tojson HTML-escapes <, > and &, which would
        # corrupt tool schemas rendered into a prompt.
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)

    @staticmethod
    def _raise_exception(message: str) -> None:
        raise ChatTemplateError(f"chat template rejected the conversation: {message}")

    def _message_dict(self, message: ChatMessage) -> dict[str, Any]:
        """`content` gets one `[IMG]` marker per attached image in front, exactly like
        Mistral3PromptBuilder - vision_fusion splits the *rendered* prompt on this literal
        substring, so it must survive rendering verbatim no matter what shape the real template
        expects. With `self._structured_content` (see `_STRUCTURED_CONTENT_MARKERS`), every part
        is still `{"type": "text", ...}`, even the image ones (`text: IMAGE_MARKER`) - never a
        real `{"type": "image"}`: a template's own image branch renders its *own* hardcoded
        literal (e.g. SmolVLM2's `'<image>'`), ignoring whatever we put there, so only the
        text branch (which always echoes `text` back verbatim) can carry our marker through.
        Real, confirmed-live cost: the prompt shows `[IMG]` instead of the template's own image
        literal, and a `content[0]['type']=='image'` punctuation check (SmolVLM2 has one) takes
        its no-image path - cosmetic, outweighed by the alternative (image embeddings landing
        after the generation prompt instead of in the turn at all, confirmed live).
        """
        data = message.model_dump(exclude_none=True, exclude={"images"})
        markers = IMAGE_MARKER * len(message.images or [])
        text = message.content or ""
        data["text"] = text
        data["image_markers"] = markers
        if self._structured_content:
            image_parts = [{"type": "text", "text": IMAGE_MARKER} for _ in (message.images or [])]
            data["content"] = [*image_parts, {"type": "text", "text": text}]
        else:
            data["content"] = markers + text
        return data

    def build(self, messages: list[ChatMessage], tools: list[dict] | None = None) -> str:
        compact = self._no_tools_template is not None and not tools
        template = self._no_tools_template if compact else self._template
        try:
            return template.render(
                messages=[self._message_dict(m) for m in messages],
                tools=tools or None,
                add_generation_prompt=True,
                bos_token=self._bos_token,
                eos_token=self._eos_token,
            )
        except TemplateError as exc:
            raise ChatTemplateError(f"could not render this model's chat template: {exc}") from exc

    def wants_bos(self, prompt: str) -> bool:
        """Only when the GGUF asks for a BOS (`tokenizer.ggml.add_bos_token` - e.g. Llama 3 yes,
        Granite no) and the template didn't already write one (Llama 3's `{{ bos_token }}`) -
        the same rule llama.cpp applies."""
        return self._add_bos and bool(self._bos_token) and not prompt.startswith(self._bos_token)


class PromptBuilderFactory:
    @staticmethod
    def for_metadata(metadata: GGUFMetadata, tag: str = "") -> PromptBuilder:
        if metadata.architecture == "mistral3":
            return Mistral3PromptBuilder()
        builder = ChatTemplatePromptBuilder.from_metadata(metadata)
        if builder is not None:
            return builder
        if _VICUNA_TAG_MARKER in tag.lower():
            return VicunaPromptBuilder()
        # No template and not a real mistral3-tokenizer model - e.g. an older Mistral-7B
        # v0.1/v0.2 fine-tune converted as general.architecture=="llama" with no chat_template
        # metadata (see LegacyMistralPromptBuilder's own docstring: Mistral3PromptBuilder's
        # [SYSTEM_PROMPT] tag isn't something these ever saw in training).
        return LegacyMistralPromptBuilder()


def _has_confirmed_template(metadata: GGUFMetadata, tag: str = "") -> bool:
    """True for exactly the cases PromptBuilderFactory.for_metadata above actually trusts - real
    mistral3, a real/builtin chat template, or a tag-matched Vicuna model - False whenever it
    would fall back to LegacyMistralPromptBuilder, our best-effort guess for a model we have no
    real confirmation about. Kept in sync with for_metadata by hand (a few lines of duplication,
    not worth a forced shared code path) rather than by construction - if that logic changes,
    change this too."""
    return (
        metadata.architecture == "mistral3"
        or ChatTemplatePromptBuilder.from_metadata(metadata) is not None
        or _VICUNA_TAG_MARKER in tag.lower()
    )


# Keyed by (path, mtime, size) - same reasoning and pattern as
# app.models.load_dtype.estimate_ram_gb's own cache: this parses the same GGUF header
# has_confirmed_chat_format below is called against on every GET /api/tags, and a file's own
# metadata never changes without the file itself changing.
_confirmed_chat_format_cache: dict[tuple[str, int, int, str], bool] = {}


def has_confirmed_chat_format(gguf_path: str | Path, tag: str = "") -> bool:
    """Whether `gguf_path` gets a chat template we actually know matches its real training format

    (see _has_confirmed_template above), for a caller with only a path, not already-parsed
    GGUFMetadata (see app.models.capabilities.effective_capabilities's "chat_format_unverified").
    False isn't proof of a base/non-chat model - just that this is a generic guess, which can
    still produce incoherent output regardless of which guess is used (Hebrew-Mistral-7B-Q5_K_M,
    2026-09-27)."""
    path = Path(gguf_path)
    stat = path.stat()
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size, tag)
    cached = _confirmed_chat_format_cache.get(cache_key)
    if cached is not None:
        return cached

    metadata = GGUFReader(path).read().metadata
    result = _has_confirmed_template(metadata, tag)
    _confirmed_chat_format_cache[cache_key] = result
    return result
