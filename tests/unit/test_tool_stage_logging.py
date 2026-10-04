"""The tool-use trace in app/routers/chat_router.py: level 1 (INFO) milestones, level 2 (DEBUG)."""

import logging
from types import SimpleNamespace

from app.routers.chat_router import ChatRequestHandler
from app.schemas.chat import ChatMessage, ChatRequest, ToolCall, ToolFunctionCall

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]


def _request(tools: list[dict] | None) -> ChatRequest:
    messages = [ChatMessage(role="user", content="hi"), ChatMessage(role="tool", content="21C")]
    return ChatRequest(model="m", messages=messages, tools=tools)


def test_tools_offered_logs_a_summary_at_info_and_names_at_debug(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="app.routers.chat_router")
    handle = SimpleNamespace(tool_call_format=None)

    ChatRequestHandler._log_tools_offered(_request(TOOLS), handle)

    text = caplog.text
    assert "tools: 1 offered" in text and "1 tool results in the history" in text
    assert "tools offered: get_weather" in text


def test_nothing_is_logged_when_no_tools_are_offered(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="app.routers.chat_router")

    ChatRequestHandler._log_tools_offered(_request(None), SimpleNamespace(tool_call_format=None))
    ChatRequestHandler._log_tool_result(_request(None), "text", [])

    assert caplog.text == ""


def test_stop_token_log_says_whether_it_is_the_models_own_end(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.routers.chat_router")
    tokenizer = SimpleNamespace(decode=lambda ids: "<|tool_response>", eos_token_id=106)

    ChatRequestHandler._log_stop(50, SimpleNamespace(tokenizer=tokenizer), _request(TOOLS))

    assert "stopped on token 50 '<|tool_response>' (extra stop token)" in caplog.text


def test_tool_result_log_tells_a_call_from_plain_text(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.routers.chat_router")
    call = ToolCall(function=ToolFunctionCall(name="get_weather", arguments={"city": "Haifa"}))

    ChatRequestHandler._log_tool_result(_request(TOOLS), "", [call])
    ChatRequestHandler._log_tool_result(_request(TOOLS), "just words", [])

    assert "tool calls parsed: 1 (get_weather)" in caplog.text
    assert "no tool call in the reply (10 chars of text)" in caplog.text
