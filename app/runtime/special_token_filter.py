"""A general, model-agnostic safety net for a leaked special/control token in a live chat
stream - the same real, universal `<|...|>` bracket convention essentially every modern
tokenizer uses for its own special tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`,
`<|eot_id|>`, `<|start_header_id|>`, ...), regardless of *why* one leaked through (a real stop
token this project doesn't yet recognize for this checkpoint - see
`app.runtime.chat_stop_tokens` for the one confirmed, fixed case; a future model with some other
convention; anything else). Real, confirmed-live incident this exists for (2026-09-29):
`DictaLM-3.0-1.7B-Thinking` produced a visible `<|im_start|>assistant` in a chat reply.

Streaming-safe: a special token is usually one atomic vocab entry, so it normally arrives as one
whole chunk from `IncrementalTextDecoder.push()` - but this buffers on a real `<|` prefix
regardless, in case a future tokenizer/decoder path ever splits one across chunks, rather than
assuming the common case is the only case.
"""

import re

_TAG_START = "<|"
_TAG_PATTERN = re.compile(r"<\|[^|<>\s]{1,64}\|>")
_MAX_PENDING = 64 + len(_TAG_START) + 2  # longest real tag this recognizes, plus its own markers


class SpecialTokenTextFilter:
    """Feed decoded text chunks in; get back the same text with any `<|...|>`-shaped span
    removed. Buffers a `<|`-prefixed span that hasn't resolved into a complete tag (or been
    proven not to be one) yet - call `flush()` once the stream ends to release it as plain text
    (a lone `<|` with no closing `|>` was never a real tag, so it must not be dropped silently).
    """

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, text: str) -> str:
        self._pending += text
        out = []
        while True:
            start = self._pending.find(_TAG_START)
            if start == -1:
                out.append(self._pending)
                self._pending = ""
                break

            out.append(self._pending[:start])
            candidate = self._pending[start:]
            match = _TAG_PATTERN.match(candidate)
            if match:
                self._pending = candidate[match.end() :]
                continue

            # Not a complete match yet - is `candidate` still a *possible* prefix of one, or
            # has it already grown too long / hit a character a real tag can't contain?
            if len(candidate) <= _MAX_PENDING and re.fullmatch(r"<\|[^|<>\s]*", candidate):
                self._pending = candidate
                break

            # Never going to become a tag - emit the literal "<|" and keep scanning the rest
            # for another, later real `<|`.
            out.append(_TAG_START)
            self._pending = candidate[len(_TAG_START) :]

        return "".join(out)

    def flush(self) -> str:
        text, self._pending = self._pending, ""
        return text
