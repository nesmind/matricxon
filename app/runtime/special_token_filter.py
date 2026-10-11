"""A model-agnostic safety net for a leaked special/control token in a chat stream: the `<|...|>`
convention nearly every modern tokenizer uses (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`,
`<|eot_id|>`, `<|start_header_id|>`, ...), whatever the reason it leaked (a stop token not
recognized for this checkpoint, see `app.runtime.chat_stop_tokens`; a future convention; anything
else). Seen when a DictaLM model printed a visible `<|im_start|>assistant` in a reply.

Streaming-safe: a special token is usually one atomic vocab entry arriving as one chunk from
`IncrementalTextDecoder.push`, but this buffers on a `<|` prefix regardless, in case a
tokenizer/decoder splits one across chunks.
"""

import re

_TAG_START = "<|"
_TAG_PATTERN = re.compile(r"<\|[^|<>\s]{1,64}\|>")
_MAX_PENDING = 64 + len(_TAG_START) + 2


class SpecialTokenTextFilter:
    """Feed decoded text chunks in; get the same text with any `<|...|>`-shaped span removed.
    Buffers a `<|`-prefixed span not yet resolved into a complete tag (or proven not to be one);
    call `flush` at stream end to release it as plain text (a lone `<|` with no `|>` was never a
    tag and must not be dropped)."""

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

            # Not a complete match yet: is `candidate` still a *possible* prefix of one, or already
            # too long or holding a character a tag can't contain?
            if len(candidate) <= _MAX_PENDING and re.fullmatch(r"<\|[^|<>\s]*", candidate):
                self._pending = candidate
                break

            # Never going to become a tag: emit the literal "<|" and keep scanning the rest for a
            # later real `<|`.
            out.append(_TAG_START)
            self._pending = candidate[len(_TAG_START) :]

        return "".join(out)

    def flush(self) -> str:
        text, self._pending = self._pending, ""
        return text
