from pathlib import Path

from app.architectures.base import ModelArchitecture
from app.architectures.bert import BertArchitecture
from app.architectures.command_r import CommandRArchitecture
from app.architectures.falcon import FalconArchitecture
from app.architectures.gemma4 import Gemma4Architecture
from app.architectures.granite import GraniteArchitecture
from app.architectures.granitemoe import GraniteMoeArchitecture
from app.architectures.llama import LlamaArchitecture
from app.architectures.mistral3 import Mistral3TextArchitecture
from app.architectures.nemotron_h import NemotronHArchitecture
from app.architectures.nomic_bert import NomicBertArchitecture
from app.architectures.phi2 import Phi2Architecture
from app.architectures.qwen2 import Qwen2Architecture
from app.architectures.qwen3 import Qwen3Architecture
from app.architectures.starcoder2 import Starcoder2Architecture
from app.gguf.metadata import GGUFMetadata
from app.gguf.reader import GGUFReader
from app.server.errors import UnsupportedArchitectureError


class ArchitectureRegistry:
    """Resolves a GGUF file's `general.architecture` to a ModelArchitecture
    subclass; fails closed (never attempts a best-effort/approximate forward
    pass for something unrecognized).
    """

    _ARCHITECTURES: list[type[ModelArchitecture]] = [
        Mistral3TextArchitecture,
        BertArchitecture,
        NomicBertArchitecture,
        LlamaArchitecture,
        Gemma4Architecture,
        Phi2Architecture,
        GraniteArchitecture,
        GraniteMoeArchitecture,
        NemotronHArchitecture,
        Qwen2Architecture,
        Qwen3Architecture,
        CommandRArchitecture,
        Starcoder2Architecture,
        FalconArchitecture,
    ]

    def resolve(self, metadata: GGUFMetadata) -> type[ModelArchitecture]:
        for architecture_cls in self._ARCHITECTURES:
            if architecture_cls.supports(metadata):
                return architecture_cls
        raise UnsupportedArchitectureError(
            f"Unsupported GGUF architecture: {metadata.architecture}"
        )

    def supported_names(self) -> list[str]:
        """Every real `general.architecture` string this registry currently

        resolves - each class's own `NAME`, as real data rather than
        something re-parsed back out of `supports()`'s comparison logic.
        Used by `GET /api/health` so a caller can ask "can you run this?"
        without hand-maintaining its own separate copy of this list.
        """
        return [architecture_cls.NAME for architecture_cls in self._ARCHITECTURES]

    def moe_supported_names(self) -> list[str]:
        """The subset of `supported_names()` with real, working Mixture-of-Experts support (see
        `ModelArchitecture.SUPPORTS_MOE`'s own docstring - `llama` for real Mixtral GGUFs,
        `granitemoe`) - used by `GET /api/health` alongside `supported_architectures` so a
        caller can tell "resolves this architecture name" apart from "and can actually run its
        MoE variant", the exact distinction `gemma4`'s own `unsupported_features` MoE flag
        exists to catch per-checkpoint (this is the same fact, surfaced architecture-wide)."""
        return [
            architecture_cls.NAME
            for architecture_cls in self._ARCHITECTURES
            if architecture_cls.SUPPORTS_MOE
        ]


# Keyed by (path, mtime, size) - same reasoning/pattern as has_confirmed_chat_format's own cache
# (app/runtime/chat_template.py): a cheap header-only GGUFReader parse, and a file's own metadata
# never changes without the file itself changing.
_unsupported_features_cache: dict[tuple[str, int, int], list[str]] = {}


def unsupported_features(gguf_path: str | Path) -> list[str]:
    """`gguf_path`'s own resolved architecture class's `unsupported_features(metadata)` (see
    that method's own docstring on `ModelArchitecture` for what this means and why it exists) -
    for a caller with only a path, not already-parsed `GGUFMetadata` (see
    `app.models.capabilities.effective_capabilities`'s own `"architecture_features_unsupported"`
    flag, the real consumer).

    `[]` (not a raised error) when `gguf_path` isn't `ArchitectureRegistry`-resolvable at all - a
    real, live regression this fixes (2026-09-29): a `clip` mmproj sidecar file (never
    resolvable on its own, see `capabilities._MMPROJ_ARCHITECTURE`'s own comment) crashed this
    for every installed model, not just itself, since `effective_capabilities` calls this once
    per installed tag inside `GET /api/tags`'s own loop - the exact same "one bad file must not
    take down the whole listing" failure shape `has_confirmed_chat_format`'s own except clause
    already guards against, just not copied here the first time."""
    path = Path(gguf_path)
    stat = path.stat()
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _unsupported_features_cache.get(cache_key)
    if cached is not None:
        return cached

    metadata = GGUFReader(path).read().metadata
    try:
        architecture_cls = ArchitectureRegistry().resolve(metadata)
    except UnsupportedArchitectureError:
        result: list[str] = []
    else:
        result = architecture_cls.unsupported_features(metadata)
    _unsupported_features_cache[cache_key] = result
    return result
