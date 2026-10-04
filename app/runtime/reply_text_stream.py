"""Turns a reply's generated token ids into the text the user sees plus any tool calls:
decode -> (hold back tool-call markup) -> drop stray special tokens."""

from app.runtime.special_token_filter import SpecialTokenTextFilter
from app.runtime.tokenizer import IncrementalTextDecoder
from app.runtime.tool_call_formats import ToolCallFormat
from app.runtime.tool_call_scanner import ToolCallScanner
from app.schemas.chat import ToolCall


class ReplyTextStream:
    def __init__(
        self,
        tokenizer: object,
        call_format: ToolCallFormat | None = None,
        tools: list[dict] | None = None,
    ) -> None:
        self._decoder = IncrementalTextDecoder(tokenizer)
        self._filter = SpecialTokenTextFilter()
        # Only scan when the request offered tools and the model has a known call format.
        self._scanner = ToolCallScanner(call_format, tools) if call_format and tools else None

    def push(self, token_id: int) -> str:
        text = self._decoder.push(token_id)
        if self._scanner is not None:
            text = self._scanner.feed(text)
        return self._filter.feed(text)

    def finish(self) -> tuple[str, list[ToolCall]]:
        """The last visible text, and the tool calls found in the reply."""
        text, calls = self._scanner.finish() if self._scanner else ("", [])
        return self._filter.feed(text) + self._filter.flush(), calls
