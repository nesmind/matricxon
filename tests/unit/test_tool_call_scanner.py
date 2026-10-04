"""The scanner passes plain text through, hides tool-call markup, and returns parsed calls."""

from app.runtime.tool_call_formats import LlamaJsonFormat, MistralFormat, TaggedBlockFormat
from app.runtime.tool_call_scanner import ToolCallScanner


def _run(scanner: ToolCallScanner, chunks: list[str]) -> tuple[str, list]:
    shown = "".join(scanner.feed(chunk) for chunk in chunks)
    tail, calls = scanner.finish()
    return shown + tail, calls


def _split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def test_plain_text_passes_through_even_with_a_marker_prefix() -> None:
    scanner = ToolCallScanner(TaggedBlockFormat(), [])
    shown, calls = _run(scanner, ["a <tool", " is a tag name, not a call"])
    assert shown == "a <tool is a tag name, not a call"
    assert calls == []


def test_text_before_the_call_is_shown_and_markup_hidden() -> None:
    text = 'Let me check.\n<tool_call>{"name": "w", "arguments": {"c": "P"}}</tool_call>'
    for size in (1, 3, 7, 1000):
        shown, calls = _run(ToolCallScanner(TaggedBlockFormat(), []), _split(text, size))
        assert shown == "Let me check.\n"
        assert [c.function.name for c in calls] == ["w"]
        assert calls[0].id and calls[0].id.startswith("call_")


def test_malformed_call_is_shown_as_text() -> None:
    text = "[TOOL_CALLS]oops"
    shown, calls = _run(ToolCallScanner(MistralFormat(), []), _split(text, 4))
    assert shown == text
    assert calls == []


def test_bare_json_reply_is_held_and_parsed() -> None:
    text = '{"name": "w", "parameters": {"c": "P"}}'
    shown, calls = _run(ToolCallScanner(LlamaJsonFormat(), []), _split(text, 5))
    assert shown == ""
    assert calls[0].function.arguments == {"c": "P"}


def test_bare_json_that_is_not_a_call_is_shown() -> None:
    text = '  {"answer": 42}'
    shown, calls = _run(ToolCallScanner(LlamaJsonFormat(), []), _split(text, 3))
    assert shown == text
    assert calls == []


def test_prose_reply_is_not_held_in_bare_json_mode() -> None:
    scanner = ToolCallScanner(LlamaJsonFormat(), [])
    assert scanner.feed("Hello") == "Hello"
    assert scanner.feed(" there") == " there"
