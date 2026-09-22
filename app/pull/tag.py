import re
from dataclasses import dataclass

from app.server.errors import ModelResolutionError

_HF_TAG_PATTERN = re.compile(r"^hf\.co/(?P<repo_id>[^/]+/[^:]+):(?P<suffix>.+)$")


@dataclass(frozen=True)
class HFModelTag:
    """A `hf.co/<repo_id>:<suffix>` model tag - the only shape `/api/pull`
    accepts in v1. matricxon has no access to Ollama's proprietary registry
    protocol (that's closed/undocumented), so a plain Ollama-library tag
    like `llama3.2:1b` can never be resolved to real bytes here - it's
    rejected up front with a clear pull error rather than failing later
    with a confusing one. Pulling straight from a `hf.co/<repo>` GGUF
    listing needs no such protocol and works the same for any caller.
    """

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
        """The `<filename>.gguf` this tag is stored/reconstructed under -
        the requested suffix itself, not whatever HF's actual filename is,
        so the on-disk path maps 1:1 back to this exact tag (see
        ModelCatalog's path convention)."""
        return f"{self.suffix}.gguf"
