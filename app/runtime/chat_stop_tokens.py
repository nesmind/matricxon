"""Real stop-token detection beyond a GGUF's own single `tokenizer.ggml.eos_token_id` - a real,
confirmed gap (2026-09-29): `DictaLM-3.0-1.7B-Thinking`'s own GGUF declares
`eos_token_id=151643` (`<|endoftext|>`, a generic legacy token), but its real chat template is
ChatML-shaped (`<|im_start|>`/`<|im_end|>`) and every real assistant turn actually ends with
`<|im_end|>` (151645) instead. Without this, matricxon never recognizes that as a stop condition,
so generation runs straight past the real end of the assistant's turn and starts hallucinating a
new `<|im_start|>assistant` turn header - confirmed live, visible as literal text in the reply.

Second case (2026-10-04): Gemma 4's GGUF declares `<turn|>` as EOS, but a tool-calling turn ends
`<|tool_call>...<tool_call|>` followed by `<|tool_response>` (model waits for the tool), and
a plain end may be `<eos>`. Without those as stops the model ran on after its call: a second
identical call and ~250 tokens of "thinking" - minutes of wasted decode on CPU. llama.cpp stops
on both too.

Deliberately narrow (ChatML and Gemma 4 only, detected the same way `chat_template.py`'s own
`_LLAMA3_TEMPLATE_MARKERS`/`_STRUCTURED_CONTENT_MARKERS` detect their own conventions - a real
template-string substring match, not a guess) rather than a general "parse the template to find
its real closing token" solver - see `app.runtime.special_token_filter` for the separate,
general safety net that catches a leaked special token regardless of *why* generation didn't
stop at it (this function existing, some other real convention this doesn't cover, anything).
"""

from app.gguf.metadata import GGUFMetadata

_CHATML_TEMPLATE_MARKERS = ("<|im_start|>", "<|im_end|>")
_CHATML_EOS_TOKEN = "<|im_end|>"


_GEMMA4_TEMPLATE_MARKERS = ("<|tool_call>", "<|tool_response>")
_GEMMA4_STOP_TOKENS = ("<eos>", "<turn|>", "<|tool_response>")


def extra_eos_token_ids(metadata: GGUFMetadata) -> frozenset[int]:
    """Additional real stop-token ids beyond `tokenizer.ggml.eos_token_id`, derived purely from
    this checkpoint's own real metadata - empty for anything neither ChatML- nor Gemma-4-shaped."""
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
