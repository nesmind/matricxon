"""Prompt building from a model's chat template: the Jinja `tokenizer.chat_template` string that
modern instruct GGUFs carry (what Hugging Face's `apply_chat_template`, llama.cpp and Ollama
render). It replaces "every model gets Mistral's `[INST]` format", under which a Llama-3.2 chat got
`[INST]` prompts, answered empty or echoed `</s>[INST]...[/INST]`, and never emitted its
`<|eot_id|>` stop token.

`mistral3` keeps `Mistral3PromptBuilder` (hand-verified, including tool calling). No template and
not mistral3 falls back to `LegacyMistralPromptBuilder`: Mistral3PromptBuilder's `[SYSTEM_PROMPT]`
tag is specific to Mistral's newer tokenizer and broke an older Mistral fine-tune, as
`[INST]`-for-everything broke Llama 3.2. A tag/filename matching "vicuna" gets
`VicunaPromptBuilder`: its GGUF gives no metadata-only way to detect the format (a LLaVA-Vicuna
model looks like countless llama-arch models), and the Mistral-shaped guess gave an empty
immediate-EOS reply.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from app.gguf.metadata import GGUFMetadata
from app.gguf.reader import GGUFReader
from app.runtime.chat_template_jinja_ext import GenerationTagExtension
from app.runtime.prompt_builder import (
    LegacyMistralPromptBuilder,
    Mistral3PromptBuilder,
    VicunaPromptBuilder,
)
from app.runtime.vision_fusion import IMAGE_MARKER
from app.schemas.chat import ChatMessage
from app.server.errors import MatricxonError

# Name-based heuristic (PromptBuilderFactory.for_metadata): Vicuna fine-tunes give no metadata-only
# signal (module docstring), in the spirit of _THINKING_MARKERS.
_VICUNA_TAG_MARKER = "vicuna"

# Built-in templates for models whose GGUF ships no `tokenizer.chat_template`, keyed by
# `general.name`: (template, always add BOS). moondream2's reference format (BOS first, system
# messages skipped: they only confuse a 1.9B model); the old Mistral fallback's first generated
# token was EOS here (empty reply).
_MOONDREAM_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'user' %}{{ m.image_markers + '\n\nQuestion: ' + m.text }}"
    "{% elif m.role == 'assistant' %}{{ '\n\nAnswer: ' + m.text }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '\n\nAnswer:' }}{% endif %}"
)
_BUILTIN_TEMPLATES = {"moondream2": (_MOONDREAM_TEMPLATE, True)}

# Llama 3.x's template always opens with a system block, even without a system message/tools;
# Ollama's llama3 template writes one only with a system message or tools (for "Say hi." Ollama
# prefilled 13 tokens, Matricxon 40: most of a short reply's CPU latency). Used when
# there are no tools; with tools, the model's template renders them.
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

# Marker (SmolVLM2's template: `message['content'][0]['type']`) for a template expecting `content`
# as a list of `{"type":...}` parts (HF's multi-modal convention), not a flat string. a flat
# string crashes on empty content ("str object has no element 0") or renders empty on non-empty
# content (Jinja `Undefined` never matches).
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
        # Same Jinja settings as HF's apply_chat_template, plus GenerationTagExtension.
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[GenerationTagExtension]
        )
        env.filters["tojson"] = self._tojson
        env.globals["raise_exception"] = self._raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self._template = env.from_string(template)
        # Rendered instead of `template` when a request has no tools (_LLAMA3_COMPACT_TEMPLATE).
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
        try:
            return cls(
                template,
                token_text("tokenizer.ggml.bos_token_id"),
                token_text("tokenizer.ggml.eos_token_id"),
                add_bos,
                _LLAMA3_COMPACT_TEMPLATE if is_llama3 else None,
                structured_content,
            )
        except TemplateError:
            # Called per model on every GET /api/tags: degrade, don't crash every other model.
            return None

    @staticmethod
    def _tojson(value: Any, ensure_ascii: bool = False, indent: int | None = None, **_: Any) -> str:
        # Hugging Face's override: Jinja's built-in tojson HTML-escapes <, > and &, which would
        # corrupt tool schemas rendered into a prompt.
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)

    @staticmethod
    def _raise_exception(message: str) -> None:
        raise ChatTemplateError(f"chat template rejected the conversation: {message}")

    def _message_dict(self, message: ChatMessage) -> dict[str, Any]:
        """`content` gets one `[IMG]` marker per attached image in front, like
        Mistral3PromptBuilder: vision_fusion splits the rendered prompt on this literal, so it must
        survive rendering verbatim. With `self._structured_content` (`_STRUCTURED_CONTENT_MARKERS`)
        every part is `{"type": "text",...}`, image ones included (`text: IMAGE_MARKER`), never a
        real `{"type": "image"}`: a template's image branch renders its own hardcoded literal (e.g.
        SmolVLM2's `'<image>'`), so only the text branch carries our marker. Cost: the prompt shows
        `[IMG]` instead of the template's literal, and a `content[0]['type']=='image'` check
        (SmolVLM2 has one) takes its no-image path; cosmetic, versus image embeddings landing after
        the generation prompt, outside the turn."""
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
        """Only when the GGUF asks for a BOS (`tokenizer.ggml.add_bos_token`: Llama 3 yes, Granite
        no) and the template didn't write one (Llama 3's `{{ bos_token }}`): llama.cpp's rule."""
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
        # No template and not a mistral3-tokenizer model, e.g. an older Mistral-7B v0.1/v0.2
        # fine-tune converted as general.architecture=="llama" with no chat_template
        # (LegacyMistralPromptBuilder: Mistral3PromptBuilder's [SYSTEM_PROMPT] tag isn't in their
        # training).
        return LegacyMistralPromptBuilder()


def _has_confirmed_template(metadata: GGUFMetadata, tag: str = "") -> bool:
    """True for exactly the cases PromptBuilderFactory.for_metadata trusts (mistral3, a real/builtin
    template, a tag-matched Vicuna model), False otherwise (the LegacyMistralPromptBuilder
    fallback). Kept in sync with for_metadata by hand: change both together."""
    return (
        metadata.architecture == "mistral3"
        or ChatTemplatePromptBuilder.from_metadata(metadata) is not None
        or _VICUNA_TAG_MARKER in tag.lower()
    )


# Keyed by (path, mtime, size), like app.models.load_dtype.estimate_ram_gb's cache:
# has_confirmed_chat_format parses this GGUF header on every GET /api/tags, and a file's metadata
# never changes without the file.
_confirmed_chat_format_cache: dict[tuple[str, int, int, str], bool] = {}


def has_confirmed_chat_format(gguf_path: str | Path, tag: str = "") -> bool:
    """Whether `gguf_path` gets a chat template matching its training format
    (_has_confirmed_template), for a caller with only a path (effective_capabilities's
    "chat_format_unverified"). False isn't proof of a base model, just an unconfirmed guess that can
    still produce incoherent output (a Hebrew Mistral fine-tune)."""
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
