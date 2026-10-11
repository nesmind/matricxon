import re
from dataclasses import dataclass

from app.server.errors import ModelResolutionError

_HF_TAG_PATTERN = re.compile(r"^hf\.co/(?P<repo_id>[^/]+/[^:]+):(?P<suffix>.+)$")


@dataclass(frozen=True)
class HFModelTag:
    """A `hf.co/<repo_id>:<suffix>` model tag, the only shape `/api/pull` accepts in v1. matricxon
    has no access to Ollama's closed registry protocol, so a plain Ollama-library tag like
    `llama3.2:1b` can never resolve to bytes here: rejected up front with a clear pull error instead
    of a confusing later one. A `hf.co/<repo>` GGUF listing needs no such protocol."""

    raw: str
    repo_id: str
    suffix: str

    @classmethod
    def parse(cls, tag: str) -> "HFModelTag":
        match = _HF_TAG_PATTERN.match(tag)
        if match is None:
            raise ModelResolutionError(
                f"matricxon can only pull hf.co/<repo>:<suffix> tags, got: {tag!r}"
            )
        return cls(raw=tag, repo_id=match.group("repo_id"), suffix=match.group("suffix"))

    def local_filename(self) -> str:
        """The `<filename>.gguf` this tag is stored under: the requested suffix itself, not HF's
        actual filename, so the on-disk path maps 1:1 back to this tag (ModelCatalog's path
        convention)."""
        return f"{self.suffix}.gguf"
