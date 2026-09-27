"""Mistral3PromptBuilder against the real `mistralai/Ministral-3-3B-Instruct-2512`

chat_template.jinja's core structure (fetched from the real HF repo, not
assumed - see ROADMAP.md's M10 tool-calling entry): system/user/assistant
turns, plus [AVAILABLE_TOOLS]/[TOOL_CALLS]/[ARGS]/[TOOL_RESULTS].
"""

import json

import pytest

from app.runtime.prompt_builder import (
    LegacyMistralPromptBuilder,
    Mistral3PromptBuilder,
    VicunaPromptBuilder,
)
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


class TestLegacyMistralPromptBuilder:
    """No [SYSTEM_PROMPT] tag - see the class's own docstring for why
    (Hebrew-Mistral-7B-Q5_K_M, 2026-09-27)."""

    def test_system_is_folded_into_the_next_user_turn_not_tagged(self) -> None:
        messages = [
            ChatMessage(role="system", content="be nice"),
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]

        prompt = LegacyMistralPromptBuilder().build(messages)

        assert prompt == "[INST] be nice\n\nhi [/INST]hello</s>"

    def test_no_system_message_renders_a_plain_inst_block(self) -> None:
        messages = [ChatMessage(role="user", content="hi")]

        prompt = LegacyMistralPromptBuilder().build(messages)

        assert prompt == "[INST] hi [/INST]"

    def test_images_prefix_the_user_turn_as_img_tokens(self) -> None:
        messages = [ChatMessage(role="user", content="what is this", images=["b64"])]

        prompt = LegacyMistralPromptBuilder().build(messages)

        assert prompt == "[INST] [IMG]what is this [/INST]"

    def test_unsupported_role_raises(self) -> None:
        messages = [ChatMessage(role="function", content="nope")]

        with pytest.raises(UnsupportedChatRoleError):
            LegacyMistralPromptBuilder().build(messages)

    def test_second_user_turn_has_no_leftover_system_text(self) -> None:
        messages = [
            ChatMessage(role="system", content="be nice"),
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="ok"),
            ChatMessage(role="user", content="second"),
        ]

        prompt = LegacyMistralPromptBuilder().build(messages)

        assert prompt == "[INST] be nice\n\nfirst [/INST]ok</s>[INST] second [/INST]"

    def test_available_tools_rendered_once_before_the_first_user_turn(self) -> None:
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        messages = [ChatMessage(role="user", content="weather in paris?")]

        prompt = LegacyMistralPromptBuilder().build(messages, tools=tools)

        expected_tools = f"[AVAILABLE_TOOLS]{json.dumps(tools)}[/AVAILABLE_TOOLS]"
        assert prompt == expected_tools + "[INST] weather in paris? [/INST]"

    def test_tool_result_message_renders_as_tool_results_block(self) -> None:
        messages = [ChatMessage(role="tool", content='{"temp_c": 15}')]

        prompt = LegacyMistralPromptBuilder().build(messages)

        assert prompt == '[TOOL_RESULTS]{"temp_c": 15}[/TOOL_RESULTS]'

    def test_wants_bos_is_always_true(self) -> None:
        assert LegacyMistralPromptBuilder().wants_bos("anything") is True


class TestVicunaPromptBuilder:
    """FastChat's vicuna_v1 conv_template - see the class's own docstring for why this exists
    (llava-v1.6-vicuna-7b, 2026-09-27: a real, empty, immediate-EOS reply under
    LegacyMistralPromptBuilder's wrong Mistral-shaped guess)."""

    def test_uses_the_default_system_preamble_when_none_is_given(self) -> None:
        messages = [ChatMessage(role="user", content="hi")]

        prompt = VicunaPromptBuilder().build(messages)

        assert prompt == (
            "A chat between a curious human and an artificial intelligence assistant. The "
            "assistant gives helpful, detailed, and polite answers to the human's questions."
            "\n\nUSER: hi\n\nASSISTANT:"
        )

    def test_a_given_system_message_replaces_the_default_preamble(self) -> None:
        messages = [
            ChatMessage(role="system", content="Be terse."),
            ChatMessage(role="user", content="hi"),
        ]

        prompt = VicunaPromptBuilder().build(messages)

        assert prompt == "Be terse.\n\nUSER: hi\n\nASSISTANT:"

    def test_multiple_system_messages_are_joined(self) -> None:
        messages = [
            ChatMessage(role="system", content="Be terse."),
            ChatMessage(role="system", content="Never apologize."),
            ChatMessage(role="user", content="hi"),
        ]

        prompt = VicunaPromptBuilder().build(messages)

        assert prompt.startswith("Be terse.\n\nNever apologize.\n\nUSER: hi")

    def test_assistant_turn_ends_with_eos(self) -> None:
        messages = [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]

        prompt = VicunaPromptBuilder().build(messages)

        assert "ASSISTANT: hello</s>" in prompt

    def test_multi_turn_round_trip(self) -> None:
        messages = [
            ChatMessage(role="system", content="Be terse."),
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="ok"),
            ChatMessage(role="user", content="second"),
        ]

        prompt = VicunaPromptBuilder().build(messages)

        expected = "Be terse.\n\nUSER: first\n\nASSISTANT: ok</s>\n\nUSER: second\n\nASSISTANT:"
        assert prompt == expected

    def test_images_prefix_the_user_turn_as_img_tokens(self) -> None:
        messages = [ChatMessage(role="user", content="what is this", images=["b64"])]

        prompt = VicunaPromptBuilder().build(messages)

        assert "USER: [IMG]what is this" in prompt

    def test_tool_result_message_is_folded_in_as_a_user_turn(self) -> None:
        messages = [ChatMessage(role="tool", content='{"temp_c": 15}')]

        prompt = VicunaPromptBuilder().build(messages)

        assert 'USER: {"temp_c": 15}' in prompt

    def test_unsupported_role_raises(self) -> None:
        messages = [ChatMessage(role="function", content="nope")]

        with pytest.raises(UnsupportedChatRoleError):
            VicunaPromptBuilder().build(messages)

    def test_wants_bos_is_always_true(self) -> None:
        assert VicunaPromptBuilder().wants_bos("anything") is True
