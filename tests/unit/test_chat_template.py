"""Unit tests for app/runtime/chat_template.py - rendering a GGUF's own Jinja chat template, BOS
handling, and the per-architecture builder choice (see that module's docstring for the real
Llama-3.2 bug this fixes)."""

import pytest

from app.gguf.metadata import GGUFMetadata
from app.runtime.chat_template import (
    ChatTemplateError,
    ChatTemplatePromptBuilder,
    PromptBuilderFactory,
    _has_confirmed_template,
)
from app.runtime.prompt_builder import (
    LegacyMistralPromptBuilder,
    Mistral3PromptBuilder,
    VicunaPromptBuilder,
)
from app.schemas.chat import ChatMessage

# A trimmed-down version of Llama 3's real template structure.
_LLAMA3_TEMPLATE = (
    "{{- bos_token }}"
    "{% for m in messages %}"
    "<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)
_TOKENS = ["<|begin_of_text|>", "<|eot_id|>", "hello"]


def _metadata(
    architecture: str, template: str | None, add_bos: bool = True, name: str = "test-model"
) -> GGUFMetadata:
    values = {
        "general.name": name,
        "general.architecture": architecture,
        "tokenizer.ggml.tokens": _TOKENS,
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.add_bos_token": add_bos,
    }
    if template is not None:
        values["tokenizer.chat_template"] = template
    return GGUFMetadata(values)


def test_renders_the_models_own_template_with_a_generation_prompt() -> None:
    builder = PromptBuilderFactory.for_metadata(_metadata("llama", _LLAMA3_TEMPLATE))

    prompt = builder.build([ChatMessage(role="user", content="Hi")])

    assert prompt == (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def test_no_second_bos_when_the_template_already_wrote_one() -> None:
    builder = PromptBuilderFactory.for_metadata(_metadata("llama", _LLAMA3_TEMPLATE))
    prompt = builder.build([ChatMessage(role="user", content="Hi")])
    assert builder.wants_bos(prompt) is False


@pytest.mark.parametrize(("add_bos", "expected"), [(True, True), (False, False)])
def test_bos_follows_the_gguf_flag_when_the_template_has_none(
    add_bos: bool, expected: bool
) -> None:
    template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    builder = PromptBuilderFactory.for_metadata(_metadata("granite", template, add_bos=add_bos))
    assert builder.wants_bos(builder.build([ChatMessage(role="user", content="Hi")])) is expected


def test_mistral3_keeps_its_hand_verified_builder() -> None:
    builder = PromptBuilderFactory.for_metadata(_metadata("mistral3", _LLAMA3_TEMPLATE))
    assert isinstance(builder, Mistral3PromptBuilder)


def test_no_template_and_not_mistral3_architecture_gets_the_legacy_builder() -> None:
    """Not Mistral3PromptBuilder - that format's [SYSTEM_PROMPT] tag is Mistral3-tokenizer-only
    and broke a real older Mistral fine-tune the same way [INST]-for-everything broke Llama 3.2
    (see app/runtime/prompt_builder.py:LegacyMistralPromptBuilder's own docstring)."""
    builder = PromptBuilderFactory.for_metadata(_metadata("phi2", None))
    assert isinstance(builder, LegacyMistralPromptBuilder)


_LLAVA_VICUNA_TAG = "hf.co/second-state/Llava-v1.6-Vicuna-7B-GGUF:llava-v1.6-vicuna-7b-Q4_K_M"


def test_a_vicuna_tag_gets_the_vicuna_builder_instead_of_the_legacy_guess() -> None:
    """A tag-name match, not metadata - llava-v1.6-vicuna-7b's own GGUF gives no metadata-only
    way to detect this (see VicunaPromptBuilder's own docstring: LegacyMistralPromptBuilder's
    Mistral-shaped guess produced a real, empty, immediate-EOS reply for this exact model)."""
    builder = PromptBuilderFactory.for_metadata(_metadata("llama", None), tag=_LLAVA_VICUNA_TAG)
    assert isinstance(builder, VicunaPromptBuilder)


def test_vicuna_tag_match_is_case_insensitive() -> None:
    tag = "hf.co/org/repo:VICUNA-13b"
    builder = PromptBuilderFactory.for_metadata(_metadata("llama", None), tag=tag)
    assert isinstance(builder, VicunaPromptBuilder)


def test_a_non_vicuna_tag_still_gets_the_legacy_builder() -> None:
    tag = "hf.co/org/repo:some-model"
    builder = PromptBuilderFactory.for_metadata(_metadata("llama", None), tag=tag)
    assert isinstance(builder, LegacyMistralPromptBuilder)


def test_has_confirmed_template_is_true_for_a_vicuna_tag_match() -> None:
    assert _has_confirmed_template(_metadata("llama", None), tag=_LLAVA_VICUNA_TAG) is True


def test_has_confirmed_template_is_false_without_a_vicuna_tag_match() -> None:
    tag = "hf.co/org/some-model:Q4_K_M"
    assert _has_confirmed_template(_metadata("llama", None), tag=tag) is False


def test_raise_exception_in_a_template_becomes_a_400_error() -> None:
    builder = ChatTemplatePromptBuilder("{{ raise_exception('roles must alternate') }}", "", "")
    with pytest.raises(ChatTemplateError, match="roles must alternate") as exc_info:
        builder.build([ChatMessage(role="user", content="Hi")])
    assert exc_info.value.status_code == 400


def test_tojson_does_not_html_escape_tool_schemas() -> None:
    builder = ChatTemplatePromptBuilder("{{ tools | tojson }}", "", "")
    tools = [{"description": "a < b & c > d"}]
    assert builder.build([], tools) == '[{"description": "a < b & c > d"}]'


def test_moondream2_without_a_template_gets_its_builtin_question_answer_format() -> None:
    builder = PromptBuilderFactory.for_metadata(
        _metadata("phi2", None, add_bos=False, name="moondream2")
    )

    prompt = builder.build(
        [ChatMessage(role="system", content="Be nice."), ChatMessage(role="user", content="Hi")]
    )

    assert prompt == "\n\nQuestion: Hi\n\nAnswer:"
    assert builder.wants_bos(prompt) is True  # moondream's reference code always prepends BOS


def test_image_markers_lead_the_content_for_vision_fusion() -> None:
    builder = ChatTemplatePromptBuilder(
        "{% for m in messages %}{{ m.content }}|{{ m.text }}{% endfor %}", "", ""
    )
    prompt = builder.build([ChatMessage(role="user", content="What?", images=["a", "b"])])
    assert prompt == "[IMG][IMG]What?|What?"


# The real Llama 3.x template's always-on system block, reduced to what the compact switch keys on.
_LLAMA3_WITH_DATE_BLOCK = (
    "{{- bos_token }}<|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\n"
    "{% if tools %}TOOLS{% endif %}<|eot_id|>"
    "{% for m in messages %}"
    "<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)


def test_llama3_without_tools_skips_the_date_system_block_like_ollama() -> None:
    builder = PromptBuilderFactory.for_metadata(_metadata("llama", _LLAMA3_WITH_DATE_BLOCK))

    prompt = builder.build(
        [ChatMessage(role="system", content="Be brief."), ChatMessage(role="user", content="Hi")]
    )

    assert prompt == (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert builder.wants_bos(prompt) is False


# A trimmed-down version of the real ggml-org/SmolVLM2-2.2B-Instruct-GGUF template - expects
# `content` as a real list of {"type": ...} parts, not a flat string (confirmed live, 2026-09-30:
# a flat string either crashed on empty content or silently rendered as if the message had none).
_SMOLVLM2_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] | capitalize }}"
    "{% if message['content'][0]['type'] == 'image' %}{{ ':' }}{% else %}{{ ': ' }}{% endif %}"
    "{% for part in message['content'] %}"
    "{% if part['type'] == 'text' %}{{ part['text'] }}"
    "{% elif part['type'] == 'image' %}{{ '<image>' }}{% endif %}"
    "{% endfor %}"
    "<eot>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)


def test_structured_content_template_renders_real_text_not_silently_empty() -> None:
    """Before this fix: a flat `content` string meant `content[0]` indexed its first *character*

    (not a dict), so every `{% if part['type'] == ... %}` branch silently never matched - the
    real message text vanished from the prompt with no error at all."""
    builder = PromptBuilderFactory.for_metadata(_metadata("llama", _SMOLVLM2_TEMPLATE))
    prompt = builder.build([ChatMessage(role="user", content="hi")])
    assert prompt == "User: hi<eot>\nAssistant:"


def test_structured_content_template_does_not_crash_on_empty_content() -> None:
    """Before this fix: `content[0]` on an empty string raised `IndexError`, surfaced to the

    user as "could not render this model's chat template: str object has no element 0"."""
    builder = PromptBuilderFactory.for_metadata(_metadata("llama", _SMOLVLM2_TEMPLATE))
    prompt = builder.build([ChatMessage(role="user", content="")])
    assert prompt == "User: <eot>\nAssistant:"


def test_structured_content_template_puts_the_real_image_marker_before_text() -> None:
    """Real `[IMG]` (vision_fusion.IMAGE_MARKER), not the template's own `<image>` literal - see

    _message_dict's own docstring: a real `{"type": "image"}` part can never carry our marker
    through (the template ignores it and renders its own hardcoded literal instead), so every
    part here is `{"type": "text", ...}`, including the image one. Real, accepted cost: the
    template's own `content[0]['type'] == 'image'` punctuation check now takes its no-image
    path (a space after "User:") instead of its image one - cosmetic, not structural."""
    builder = PromptBuilderFactory.for_metadata(_metadata("llama", _SMOLVLM2_TEMPLATE))
    prompt = builder.build([ChatMessage(role="user", content="describe", images=["b64"])])
    assert prompt == "User: [IMG]describe<eot>\nAssistant:"


def test_structured_content_template_marker_lands_inside_the_turn_not_after_it() -> None:
    """Real, confirmed-live bug this guards: before _message_dict carried IMAGE_MARKER through a

    text-type part, a structured-content template's rendered prompt never contained `[IMG]` at
    all, so vision_fusion.build_prompt_with_images's own `prompt.split(IMAGE_MARKER)` found
    nothing to split on and appended the real image embeddings *after* the whole prompt
    (including the generation-prompt suffix) instead of inside the user's turn - a real,
    confirmed cause of an empty first reply (the model's next-token position was mid-image-patch,
    not right after "Assistant:")."""
    from app.runtime.vision_fusion import IMAGE_MARKER

    builder = PromptBuilderFactory.for_metadata(_metadata("llama", _SMOLVLM2_TEMPLATE))
    prompt = builder.build([ChatMessage(role="user", content="Hi", images=["b64"])])
    assert IMAGE_MARKER in prompt
    assert prompt.index(IMAGE_MARKER) < prompt.index("Assistant:")


def test_llama3_with_tools_keeps_the_models_own_template() -> None:
    builder = PromptBuilderFactory.for_metadata(_metadata("llama", _LLAMA3_WITH_DATE_BLOCK))
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

    prompt = builder.build([ChatMessage(role="user", content="Hi")], tools=tools)

    assert "Cutting Knowledge Date" in prompt and "TOOLS" in prompt
