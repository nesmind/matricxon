from typing import Any

from app.server.errors import MissingMetadataError


class GGUFMetadata:
    """Typed access over a GGUF file's parsed metadata key/value pairs."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def require(self, key: str) -> Any:
        if key not in self._values:
            raise MissingMetadataError(f"Missing required GGUF metadata key: {key}")
        return self._values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def get_str(self, key: str, default: str | None = None) -> str | None:
        return self._values.get(key, default)

    def get_u32(self, key: str, default: int | None = None) -> int | None:
        return self._values.get(key, default)

    def get_f32(self, key: str, default: float | None = None) -> float | None:
        return self._values.get(key, default)

    def get_bool(self, key: str, default: bool | None = None) -> bool | None:
        return self._values.get(key, default)

    def get_array(self, key: str, default: list | None = None) -> list | None:
        return self._values.get(key, default)

    def keys(self) -> list[str]:
        return list(self._values.keys())

    @property
    def architecture(self) -> str:
        return self.require("general.architecture")

    def arch_key(self, suffix: str) -> str:
        return f"{self.architecture}.{suffix}"
