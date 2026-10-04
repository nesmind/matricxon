"""Each family's real tool-call text (from the real chat templates) parses into tool calls."""

import pytest

from app.runtime.gemma_tool_dsl import GemmaArgsParser
from app.runtime.tool_call_formats import (
    GemmaFormat,
    LlamaJsonFormat,
    MistralFormat,
    TaggedBlockFormat,
    ToolCallFormats,
    ToolSchemas,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"properties": {"id": {"type": "string"}, "n": {"type": "integer"}}},
        },
    }
]
SCHEMAS = ToolSchemas(TOOLS)


def _names_args(calls: list) -> list[tuple[str, object]]:
    return [(c.function.name, c.function.arguments) for c in calls]


class TestTaggedBlock:
    def test_json_body(self) -> None:
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        assert _names_args(TaggedBlockFormat().parse(text, SCHEMAS)) == [
            ("get_weather", {"city": "Paris"})
        ]

    def test_xml_body_keeps_string_params_and_parses_others(self) -> None:
        text = (
            "ok\n<tool_call>\n<function=lookup>\n<parameter=id>\n007\n</parameter>\n"
            "<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>"
        )
        assert _names_args(TaggedBlockFormat().parse(text, SCHEMAS)) == [
            ("lookup", {"id": "007", "n": 5})
        ]

    def test_two_calls(self) -> None:
        one = '<tool_call>{"name":"a","arguments":{}}</tool_call>'
        assert len(TaggedBlockFormat().parse(one + "\n" + one, SCHEMAS)) == 2

    def test_unclosed_block_still_parses(self) -> None:
        text = '<tool_call>{"name":"a","arguments":{"x":1}}'
        assert _names_args(TaggedBlockFormat().parse(text, SCHEMAS)) == [("a", {"x": 1})]

    def test_garbage_gives_no_calls(self) -> None:
        assert TaggedBlockFormat().parse("<tool_call>{oops</tool_call>", SCHEMAS) == []


class TestMistral:
    def test_name_args_form(self) -> None:
        text = '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}</s>'
        assert _names_args(MistralFormat().parse(text, SCHEMAS)) == [
            ("get_weather", {"city": "Paris"})
        ]

    def test_repeated_calls(self) -> None:
        text = '[TOOL_CALLS]a[ARGS]{"x":1}[TOOL_CALLS]b[ARGS]{"y":2}'
        assert [n for n, _ in _names_args(MistralFormat().parse(text, SCHEMAS))] == ["a", "b"]

    def test_json_array_form(self) -> None:
        text = '[TOOL_CALLS][{"name": "a", "arguments": {"x": 1}}]'
        assert _names_args(MistralFormat().parse(text, SCHEMAS)) == [("a", {"x": 1})]


class TestLlamaJson:
    def test_whole_reply_json(self) -> None:
        text = '{"name": "get_weather", "parameters": {"city": "Paris"}}'
        assert _names_args(LlamaJsonFormat().parse(text, SCHEMAS)) == [
            ("get_weather", {"city": "Paris"})
        ]

    @pytest.mark.parametrize("text", ["{not json", '{"answer": 42}', '{"name": 5}'])
    def test_other_json_is_not_a_call(self, text: str) -> None:
        assert LlamaJsonFormat().parse(text, SCHEMAS) == []


class TestGemma:
    def test_call_with_mixed_values(self) -> None:
        text = (
            '<|tool_call>call:lookup{id:<|"|>a,b<|"|>,n:3,flag:true,'
            'tags:[<|"|>x<|"|>,2]}<tool_call|>'
        )
        assert _names_args(GemmaFormat().parse(text, SCHEMAS)) == [
            ("lookup", {"id": "a,b", "n": 3, "flag": True, "tags": ["x", 2]})
        ]

    def test_nested_object_and_empty_args(self) -> None:
        assert GemmaArgsParser("{a:{b:1},c:{}}").parse_object() == {"a": {"b": 1}, "c": {}}

    def test_malformed_args_are_not_a_call(self) -> None:
        assert GemmaFormat().parse('<|tool_call>call:x{a:<|"|>oops}<tool_call|>', SCHEMAS) == []


class TestDetection:
    @pytest.mark.parametrize(
        ("template", "expected"),
        [
            ("... <|tool_call>call: ...", GemmaFormat),
            ("... <tool_call>\n<function= ...", TaggedBlockFormat),
            ("... [TOOL_CALLS] ...", MistralFormat),
            ('... {"name": f, "parameters": x} tool_calls', LlamaJsonFormat),
        ],
    )
    def test_from_template(self, template: str, expected: type) -> None:
        assert isinstance(ToolCallFormats.for_template(template), expected)

    def test_mistral3_architecture_and_unknown(self) -> None:
        assert isinstance(ToolCallFormats.for_template(None, "mistral3"), MistralFormat)
        assert ToolCallFormats.for_template("plain chat template") is None
        assert ToolCallFormats.for_template(None) is None
