"""How each model family writes a tool call in its generated text, and how to read it back.

The formats come from the real chat templates (what the model saw in training):
  - Qwen / Hermes: `<tool_call>{"name":..,"arguments":{..}}</tool_call>` (Qwen3, Qwen2.5) or the
    XML form `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`
    (Qwen3.5) - both live inside the same tags, so one format handles both.
  - Mistral: `[TOOL_CALLS]name[ARGS]{json}` (repeatable) or
    `[TOOL_CALLS][{"name":..,"arguments":..}]`.
  - Llama 3.x: the whole reply is `{"name":..,"parameters":{..}}`.
  - Gemma 4: `<|tool_call>call:NAME{key:<|"|>text<|"|>,n:1}<tool_call|>` (see gemma_tool_dsl).
`parse` returns [] when the text is not really a call, so the caller can show it as plain text."""

import json
import re
from abc import ABC, abstractmethod

from app.runtime.gemma_tool_dsl import GemmaArgsParser
from app.schemas.chat import ToolCall, ToolFunctionCall


class ToolSchemas:
    """The request's tool definitions - used to keep a `string` parameter a string."""

    def __init__(self, tools: list[dict] | None) -> None:
        self._properties: dict[str, dict] = {}
        for tool in tools or []:
            function = tool.get("function", {})
            props = (function.get("parameters") or {}).get("properties") or {}
            self._properties[function.get("name", "")] = props

    def coerce(self, tool: str, key: str, raw: str) -> object:
        declared = (self._properties.get(tool, {}).get(key) or {}).get("type")
        if declared == "string":
            return raw
        try:
            return json.loads(raw)
        except ValueError:
            return raw


def _loads_prefix(text: str) -> object | None:
    """The JSON value at the start of `text`, ignoring whatever follows it."""
    try:
        return json.JSONDecoder().raw_decode(text.strip())[0]
    except ValueError:
        return None


def _call(name: str, arguments: object) -> ToolCall:
    return ToolCall(function=ToolFunctionCall(name=name, arguments=arguments or {}))


def _calls_from_json(data: object) -> list[ToolCall]:
    """One `{"name":..,"arguments"|"parameters":{..}}` object, or a list of them."""
    items = data if isinstance(data, list) else [data]
    calls = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return []
        args = item.get("arguments", item.get("parameters"))
        if args is not None and not isinstance(args, (dict, str)):
            return []
        calls.append(_call(item["name"], args))
    return calls


class ToolCallFormat(ABC):
    # Text that opens a call; None means the call is the whole reply (it starts with `{` or `[`).
    start_marker: str | None = None

    @abstractmethod
    def parse(self, text: str, schemas: ToolSchemas) -> list[ToolCall]: ...


class TaggedBlockFormat(ToolCallFormat):
    start_marker = "<tool_call>"
    _BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.S)
    _FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|\Z)", re.S)
    _PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?(?:</parameter>|\Z)", re.S)

    def parse(self, text: str, schemas: ToolSchemas) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for body in self._BLOCK.findall(text):
            if body.startswith("{"):
                calls += _calls_from_json(_loads_prefix(body))
                continue
            function = self._FUNCTION.search(body)
            if function is None:
                continue
            name = function.group(1)
            args = {
                key: schemas.coerce(name, key, value)
                for key, value in self._PARAMETER.findall(function.group(2))
            }
            calls.append(_call(name, args))
        return calls


class MistralFormat(ToolCallFormat):
    start_marker = "[TOOL_CALLS]"

    def parse(self, text: str, schemas: ToolSchemas) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for segment in text.split(self.start_marker)[1:]:
            segment = segment.strip()
            if segment.startswith("["):
                calls += _calls_from_json(_loads_prefix(segment))
            elif "[ARGS]" in segment:
                name, _, rest = segment.partition("[ARGS]")
                args = _loads_prefix(rest)
                if name.strip() and isinstance(args, dict):
                    calls.append(_call(name.strip(), args))
        return calls


class LlamaJsonFormat(ToolCallFormat):
    """Llama 3.x answers with bare JSON, so any reply that starts with `{` or `[` is held back
    until the end and only counts as a call if all of it is one."""

    def parse(self, text: str, schemas: ToolSchemas) -> list[ToolCall]:
        try:
            data = json.loads(text.strip())
        except ValueError:
            return []
        return _calls_from_json(data)


class GemmaFormat(ToolCallFormat):
    start_marker = "<|tool_call>"
    _END = "<tool_call|>"

    def parse(self, text: str, schemas: ToolSchemas) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for segment in text.split(self.start_marker)[1:]:
            segment = segment.split(self._END)[0]
            if not segment.startswith("call:") or "{" not in segment:
                continue
            name, _, rest = segment[len("call:") :].partition("{")
            try:
                args = GemmaArgsParser("{" + rest).parse_object()
            except ValueError:
                continue
            calls.append(_call(name.strip(), args))
        return calls


class ToolCallFormats:
    """Picks a model's format from its chat template source (None: the model has no known one)."""

    @staticmethod
    def for_template(template: str | None, architecture: str = "") -> ToolCallFormat | None:
        if architecture == "mistral3":
            return MistralFormat()
        if not template:
            return None
        if "<|tool_call>" in template:
            return GemmaFormat()
        if "<tool_call>" in template:
            return TaggedBlockFormat()
        if "[TOOL_CALLS]" in template:
            return MistralFormat()
        if '"parameters"' in template and "tool_calls" in template:
            return LlamaJsonFormat()
        return None
