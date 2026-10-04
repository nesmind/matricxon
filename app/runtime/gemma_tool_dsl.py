"""Parser for Gemma 4's tool-call argument syntax: `{key:value,key:<|"|>text<|"|>,k:[..],k:{..}}`.

Strings are wrapped in the literal `<|"|>` marker, keys are bare, everything else is JSON-like
(numbers, true/false/null, nested `[...]` and `{...}`). Raises ValueError on anything malformed."""

QUOTE = '<|"|>'


class GemmaArgsParser:
    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0

    def parse_object(self) -> dict[str, object]:
        self._skip_space()
        self._expect("{")
        result: dict[str, object] = {}
        while True:
            self._skip_space()
            if self._peek() == "}":
                self._pos += 1
                return result
            key = self._read_until(":").strip()
            self._pos += 1
            result[key] = self._value()
            self._skip_space()
            if self._peek() == ",":
                self._pos += 1

    def _value(self) -> object:
        self._skip_space()
        if self._text.startswith(QUOTE, self._pos):
            self._pos += len(QUOTE)
            end = self._text.find(QUOTE, self._pos)
            if end == -1:
                raise ValueError("unterminated string")
            value = self._text[self._pos : end]
            self._pos = end + len(QUOTE)
            return value
        if self._peek() == "{":
            return self.parse_object()
        if self._peek() == "[":
            return self._parse_list()
        return self._bare(self._read_until(",}]").strip())

    def _parse_list(self) -> list[object]:
        self._pos += 1
        items: list[object] = []
        while True:
            self._skip_space()
            if self._peek() == "]":
                self._pos += 1
                return items
            items.append(self._value())
            self._skip_space()
            if self._peek() == ",":
                self._pos += 1

    @staticmethod
    def _bare(token: str) -> object:
        if token in ("true", "false"):
            return token == "true"
        if token in ("null", ""):
            return None
        for cast in (int, float):
            try:
                return cast(token)
            except ValueError:
                continue
        return token

    def _read_until(self, stops: str) -> str:
        start = self._pos
        while self._pos < len(self._text) and self._text[self._pos] not in stops:
            self._pos += 1
        if self._pos >= len(self._text):
            raise ValueError("unexpected end of arguments")
        return self._text[start : self._pos]

    def _peek(self) -> str:
        return self._text[self._pos] if self._pos < len(self._text) else ""

    def _skip_space(self) -> None:
        while self._peek().isspace():
            self._pos += 1

    def _expect(self, char: str) -> None:
        if self._peek() != char:
            raise ValueError(f"expected {char!r}")
        self._pos += 1
