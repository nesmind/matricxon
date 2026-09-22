"""Mistral3PromptBuilder against the real `mistralai/Ministral-3-3B-Instruct-2512`

chat_template.jinja's core structure (fetched from the real HF repo, not
assumed - see ROADMAP.md's M10 tool-calling entry): system/user/assistant
turns, plus [AVAILABLE_TOOLS]/[TOOL_CALLS]/[ARGS]/[TOOL_RESULTS].
"""

import json

import pytest

from app.runtime.prompt_builder import Mistral3PromptBuilder
from app.schemas.chat import ChatMessage, ToolCall, ToolFunctionCall
from app.server.errors import UnsupportedChatRoleError


class TestBasicTurns:
    def test_system_user_assistant_round_trip(self) -> None:
        messages = [
            ChatMessage(role="system", content="be nice"),
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]

        prompt = Mistral3PromptBuilder().build(messages)

        assert prompt == "[SYSTEM_PROMPT]be nice[/SYSTEM_PROMPT][INST]hi[/INST]hello</s>"

    def test_images_prefix_the_user_turn_as_img_tokens(self) -> None:
        messages = [ChatMessage(role="user", content="what is this", images=["b64", "b64"])]

        prompt = Mistral3PromptBuilder().build(messages)

        assert prompt == "[INST][IMG][IMG]what is this[/INST]"

    def test_unsupported_role_raises(self) -> None:
        messages = [ChatMessage(role="function", content="nope")]

        with pytest.raises(UnsupportedChatRoleError):
            Mistral3PromptBuilder().build(messages)


class TestToolCalling:
    def test_available_tools_rendered_once_before_the_first_user_turn(self) -> None:
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        messages = [
            ChatMessage(role="system", content="be nice"),
            ChatMessage(role="user", content="weather in paris?"),
        ]

        prompt = Mistral3PromptBuilder().build(messages, tools=tools)

        expected_tools = f"[AVAILABLE_TOOLS]{json.dumps(tools)}[/AVAILABLE_TOOLS]"
        assert prompt == (
            "[SYSTEM_PROMPT]be nice[/SYSTEM_PROMPT]"
            + expected_tools
            + "[INST]weather in paris?[/INST]"
        )

    def test_available_tools_never_repeated_on_a_second_user_turn(self) -> None:
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        messages = [
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="ok"),
            ChatMessage(role="user", content="second"),
        ]

        prompt = Mistral3PromptBuilder().build(messages, tools=tools)

        assert prompt.count("[AVAILABLE_TOOLS]") == 1

    def test_no_tools_declaration_when_none_given(self) -> None:
        messages = [ChatMessage(role="user", content="hi")]

        prompt = Mistral3PromptBuilder().build(messages)

        assert "[AVAILABLE_TOOLS]" not in prompt

    def test_assistant_tool_call_renders_name_and_json_arguments(self) -> None:
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        function=ToolFunctionCall(name="get_weather", arguments={"city": "Paris"})
                    )
                ],
            )
        ]

        prompt = Mistral3PromptBuilder().build(messages)

        assert prompt == '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}</s>'

    def test_multiple_tool_calls_in_one_assistant_turn_are_each_rendered(self) -> None:
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(function=ToolFunctionCall(name="a", arguments={})),
                    ToolCall(function=ToolFunctionCall(name="b", arguments={"x": 1})),
                ],
            )
        ]

        prompt = Mistral3PromptBuilder().build(messages)

        assert prompt == '[TOOL_CALLS]a[ARGS]{}[TOOL_CALLS]b[ARGS]{"x": 1}</s>'

    def test_string_arguments_pass_through_as_is(self) -> None:
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(function=ToolFunctionCall(name="f", arguments='{"raw": true}'))
                ],
            )
        ]

        prompt = Mistral3PromptBuilder().build(messages)

        assert prompt == '[TOOL_CALLS]f[ARGS]{"raw": true}</s>'

    def test_empty_string_arguments_become_an_empty_json_object(self) -> None:
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolCall(function=ToolFunctionCall(name="f", arguments=""))],
            )
        ]

        prompt = Mistral3PromptBuilder().build(messages)

        assert prompt == "[TOOL_CALLS]f[ARGS]{}</s>"

    def test_tool_result_message_renders_as_tool_results_block(self) -> None:
        messages = [ChatMessage(role="tool", content='{"temp_c": 15}')]

        prompt = Mistral3PromptBuilder().build(messages)

        assert prompt == '[TOOL_RESULTS]{"temp_c": 15}[/TOOL_RESULTS]'

    def test_full_tool_calling_round_trip(self) -> None:
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        messages = [
            ChatMessage(role="user", content="weather in paris?"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        function=ToolFunctionCall(name="get_weather", arguments={"city": "Paris"})
                    )
                ],
            ),
            ChatMessage(role="tool", content='{"temp_c": 15}'),
            ChatMessage(role="assistant", content="It's 15C in Paris."),
        ]

        prompt = Mistral3PromptBuilder().build(messages, tools=tools)

        assert prompt == (
            "[AVAILABLE_TOOLS]" + json.dumps(tools) + "[/AVAILABLE_TOOLS]"
            "[INST]weather in paris?[/INST]"
            '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}</s>'
            '[TOOL_RESULTS]{"temp_c": 15}[/TOOL_RESULTS]'
            "It's 15C in Paris.</s>"
        )
