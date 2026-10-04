"""Streaming side of tool calling: passes normal reply text through untouched, but holds back
anything that starts a tool call so its markup never reaches the visible message, then turns the
held text into structured `tool_calls` when the reply ends."""

import uuid

from app.runtime.tool_call_formats import ToolCallFormat, ToolSchemas
from app.schemas.chat import ToolCall


class ToolCallScanner:
    def __init__(self, call_format: ToolCallFormat, tools: list[dict] | None) -> None:
        self._format = call_format
        self._schemas = ToolSchemas(tools)
        self._pending = ""  # text that might still turn out to be the start of a call
        self._held = ""  # text from the start of a call onward
        self._capturing = False
        self._undecided = call_format.start_marker is None

    def feed(self, text: str) -> str:
        """Returns the part of `text` that is safe to show now."""
        if self._capturing:
            self._held += text
            return ""
        self._pending += text
        marker = self._format.start_marker
        if marker is None:
            return self._feed_bare_json()
        start = self._pending.find(marker)
        if start >= 0:
            visible, self._held = self._pending[:start], self._pending[start:]
            self._pending, self._capturing = "", True
            return visible
        return self._release_all_but_marker_prefix(marker)

    def finish(self) -> tuple[str, list[ToolCall]]:
        """The text still owed to the user, and the calls found in the held text."""
        if not self._capturing:
            leftover, self._pending = self._pending, ""
            return leftover, []
        calls = self._format.parse(self._held, self._schemas)
        for call in calls:
            call.id = f"call_{uuid.uuid4().hex[:8]}"
        # Not really a call after all (malformed): show it rather than swallow it.
        return ("" if calls else self._held), calls

    def _feed_bare_json(self) -> str:
        if self._undecided:
            stripped = self._pending.lstrip()
            if not stripped:
                return ""
            self._undecided = False
            self._capturing = stripped[0] in "{["
        if self._capturing:
            self._held, self._pending = self._pending, ""
            return ""
        visible, self._pending = self._pending, ""
        return visible

    def _release_all_but_marker_prefix(self, marker: str) -> str:
        keep = 0
        for size in range(min(len(marker) - 1, len(self._pending)), 0, -1):
            if marker.startswith(self._pending[-size:]):
                keep = size
                break
        visible = self._pending[: len(self._pending) - keep]
        self._pending = self._pending[len(self._pending) - keep :]
        return visible
