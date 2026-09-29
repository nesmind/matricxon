"""app.runtime.chat_stop_tokens.extra_eos_token_ids - real, confirmed gap (2026-09-29): a real
DictaLM-3.0-1.7B-Thinking GGUF declares `eos_token_id=151643` (`<|endoftext|>`) but its own real
chat template is ChatML-shaped and every real turn actually ends with `<|im_end|>` instead.
"""

from app.gguf.metadata import GGUFMetadata
from app.runtime.chat_stop_tokens import extra_eos_token_ids

_CHATML_TEMPLATE = (
    "{{ '<|im_start|>system\\n' }}"
    "{% for m in messages %}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}"
    "{% endfor %}"
)
_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "hello"]


def _metadata(template: str | None, tokens: list[str] | None = _TOKENS) -> GGUFMetadata:
    values = {}
    if template is not None:
        values["tokenizer.chat_template"] = template
    if tokens is not None:
        values["tokenizer.ggml.tokens"] = tokens
    return GGUFMetadata(values)


def test_finds_the_real_im_end_token_id_for_a_chatml_template() -> None:
    metadata = _metadata(_CHATML_TEMPLATE)
    assert extra_eos_token_ids(metadata) == frozenset({2})


def test_empty_for_a_non_chatml_template() -> None:
    metadata = _metadata("{% for m in messages %}{{ m.content }}{% endfor %}")
    assert extra_eos_token_ids(metadata) == frozenset()


def test_empty_when_there_is_no_template_at_all() -> None:
    metadata = _metadata(None)
    assert extra_eos_token_ids(metadata) == frozenset()


def test_empty_when_the_template_is_chatml_shaped_but_the_token_is_missing_from_vocab() -> None:
    metadata = _metadata(_CHATML_TEMPLATE, tokens=["<|endoftext|>", "<|im_start|>", "hello"])
    assert extra_eos_token_ids(metadata) == frozenset()
