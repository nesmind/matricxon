"""Stop-token detection beyond a GGUF's single `tokenizer.ggml.eos_token_id`.

ChatML: a DictaLM GGUF declares `eos_token_id=151643` (`<|endoftext|>`, a legacy token), but its
template is ChatML (`<|im_start|>`/`<|im_end|>`) and every assistant turn ends with `<|im_end|>`
(151645). Without it generation ran past the turn and invented a new `<|im_start|>assistant` header,
shown as literal text.

Gemma 4: the GGUF declares `<turn|>` as EOS, but a tool-calling turn ends
`<|tool_call>...<tool_call|>` then `<|tool_response>` (the model waits for the tool), and a plain
end may be `<eos>`. Without those stops the model ran on after its call (a second identical call and
~250 tokens of "thinking"). llama.cpp stops on both too.

Deliberately narrow (ChatML and Gemma 4 only, detected by template-string substring like
`chat_template.py`'s `_LLAMA3_TEMPLATE_MARKERS`/`_STRUCTURED_CONTENT_MARKERS`) rather than a general
template solver; `app.runtime.special_token_filter` is the general safety net for a leaked special
token.
"""

from app.gguf.metadata import GGUFMetadata

_CHATML_TEMPLATE_MARKERS = ("<|im_start|>", "<|im_end|>")
_CHATML_EOS_TOKEN = "<|im_end|>"


_GEMMA4_TEMPLATE_MARKERS = ("<|tool_call>", "<|tool_response>")
_GEMMA4_STOP_TOKENS = ("<eos>", "<turn|>", "<|tool_response>")


def extra_eos_token_ids(metadata: GGUFMetadata) -> frozenset[int]:
    """Additional stop-token ids beyond `tokenizer.ggml.eos_token_id`, from this checkpoint's
    metadata; empty unless ChatML- or Gemma-4-shaped."""
    template = metadata.get("tokenizer.chat_template")
    if not template:
        return frozenset()
    tokens: list[str] = metadata.get("tokenizer.ggml.tokens") or []
    if all(marker in template for marker in _CHATML_TEMPLATE_MARKERS):
        names: tuple[str, ...] = (_CHATML_EOS_TOKEN,)
    elif all(marker in template for marker in _GEMMA4_TEMPLATE_MARKERS):
        names = _GEMMA4_STOP_TOKENS
    else:
        return frozenset()
    return frozenset(tokens.index(name) for name in names if name in tokens)
