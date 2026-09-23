"""Prompt building from a model's own chat template - the Jinja `tokenizer.chat_template` string
every modern instruct GGUF carries (the same template Hugging Face's `apply_chat_template`,
llama.cpp and Ollama all render). Replaces the old "every model gets Mistral's `[INST]` format"
behavior: a real Llama-3.2 chat through pAIring (2026-09-23) got `[INST]`-formatted prompts,
answered with empty replies or echoed `</s>[INST]...[/INST]` turns as plain text, and never
emitted its real `<|eot_id|>` stop token.

`mistral3` keeps `Mistral3PromptBuilder` (hand-verified against its real template, including
tool calling); a model without any template falls back to it too, which is exactly the old
behavior, so nothing that worked before changes.
"""

import json
from datetime import datetime
from typing import Any, Protocol

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from app.gguf.metadata import GGUFMetadata
from app.runtime.prompt_builder import Mistral3PromptBuilder
from app.runtime.vision_fusion import IMAGE_MARKER
from app.schemas.chat import ChatMessage
from app.server.errors import MatricxonError

# Built-in templates for real models whose GGUF ships no `tokenizer.chat_template`, keyed by
# `general.name`: (template, always add BOS). moondream2's own reference format
# (vikhyatk/moondream2's answer_question: "<image>\n\nQuestion: {q}\n\nAnswer:", BOS first) -
# under the old Mistral `[INST]` fallback its very first generated token was end-of-text (a real
# empty reply, 2026-09-23). System messages are skipped, as Ollama's own moondream template does -
# they only confuse a 1.9B model.
_MOONDREAM_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'user' %}{{ m.image_markers + '\n\nQuestion: ' + m.text }}"
    "{% elif m.role == 'assistant' %}{{ '\n\nAnswer: ' + m.text }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '\n\nAnswer:' }}{% endif %}"
)
_BUILTIN_TEMPLATES = {"moondream2": (_MOONDREAM_TEMPLATE, True)}


class PromptBuilder(Protocol):
    def build(self, messages: list[ChatMessage], tools: list[dict] | None = None) -> str: ...

    def wants_bos(self, prompt: str) -> bool: ...


class ChatTemplateError(MatricxonError):
    status_code = 400


class ChatTemplatePromptBuilder:
    def __init__(
        self, template: str, bos_token: str, eos_token: str, add_bos: bool = False
    ) -> None:
        # Same Jinja settings Hugging Face's own apply_chat_template uses - templates are written
        # against them (whitespace control in particular).
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = self._tojson
        env.globals["raise_exception"] = self._raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self._template = env.from_string(template)
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._add_bos = add_bos

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

        return cls(
            template,
            token_text("tokenizer.ggml.bos_token_id"),
            token_text("tokenizer.ggml.eos_token_id"),
            add_bos,
        )

    @staticmethod
    def _tojson(value: Any, ensure_ascii: bool = False, indent: int | None = None, **_: Any) -> str:
        # Hugging Face's own override: Jinja's built-in tojson HTML-escapes <, > and &, which would
        # corrupt tool schemas rendered into a prompt.
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)

    @staticmethod
    def _raise_exception(message: str) -> None:
        raise ChatTemplateError(f"chat template rejected the conversation: {message}")

    @staticmethod
    def _message_dict(message: ChatMessage) -> dict[str, Any]:
        """`content` gets one `[IMG]` marker per attached image in front, exactly like
        Mistral3PromptBuilder - app.runtime.vision_fusion splits the rendered prompt on them to
        splice in image embeddings. `text`/`image_markers` are the same two parts separately, for a
        template (like moondream's) that places the image somewhere other than right before the
        text."""
        data = message.model_dump(exclude_none=True, exclude={"images"})
        markers = IMAGE_MARKER * len(message.images or [])
        data["text"] = message.content or ""
        data["image_markers"] = markers
        data["content"] = markers + data["text"]
        return data

    def build(self, messages: list[ChatMessage], tools: list[dict] | None = None) -> str:
        try:
            return self._template.render(
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
    def for_metadata(metadata: GGUFMetadata) -> PromptBuilder:
        if metadata.architecture != "mistral3":
            builder = ChatTemplatePromptBuilder.from_metadata(metadata)
            if builder is not None:
                return builder
        return Mistral3PromptBuilder()
