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
from app.architectures.qwen35 import Qwen35Architecture
from app.architectures.score_head import ScoreHead
from app.architectures.starcoder2 import Starcoder2Architecture
from app.gguf.metadata import GGUFMetadata
from app.gguf.reader import GGUFReader
from app.server.errors import UnsupportedArchitectureError


class ArchitectureRegistry:
    """Resolves a GGUF file's `general.architecture` to a ModelArchitecture subclass; fails closed
    (never a best-effort/approximate forward pass for something unrecognized)."""

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
        Qwen35Architecture,
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
        """Every `general.architecture` string this registry resolves: each class's `NAME`, as data
        rather than re-parsed from `supports`.
        Used by `GET /api/health` so a caller can ask "can you run this?" without keeping its own
        copy of this list.
        """
        return [architecture_cls.NAME for architecture_cls in self._ARCHITECTURES]

    def moe_supported_names(self) -> list[str]:
        """The subset of `supported_names` with working Mixture-of-Experts support
        (`ModelArchitecture.SUPPORTS_MOE`: `llama` for Mixtral
        GGUFs, `granitemoe`), used by `GET /api/health` so a caller can tell "resolves this name"
        from "and can run its MoE variant" (the
        per-checkpoint fact `gemma4`'s `unsupported_features` MoE flag catches, here
        architecture-wide).
        """
        return [
            architecture_cls.NAME
            for architecture_cls in self._ARCHITECTURES
            if architecture_cls.SUPPORTS_MOE
        ]

    def vision_supported_names(self) -> list[str]:
        """The subset of `supported_names` that fuses paired-mmproj image embeddings
        (`ModelArchitecture.SUPPORTS_VISION`), reported by `GET /api/health` so a caller can tell
        "resolves this name" from "and can see images"."""
        return [
            architecture_cls.NAME
            for architecture_cls in self._ARCHITECTURES
            if architecture_cls.SUPPORTS_VISION
        ]


# Keyed by (path, mtime, size), like has_confirmed_chat_format's cache
# (app/runtime/chat_template.py): a cheap header-only GGUFReader parse, and a file's metadata never
# changes without the file.
_unsupported_features_cache: dict[tuple[str, int, int], list[str]] = {}


def unsupported_features(gguf_path: str | Path) -> list[str]:
    """`unsupported_features(metadata)` of `gguf_path`'s architecture class (see
    `ModelArchitecture`), for a caller with only a path (the consumer:
    `app.models.capabilities.effective_capabilities`'s `"architecture_features_unsupported"`).

    `[]` (not an error) when `gguf_path` isn't resolvable. A `clip` mmproj sidecar (never resolvable
    alone; `capabilities._MMPROJ_ARCHITECTURE`) used to crash this for every installed model, since
    `effective_capabilities` calls it per tag inside `GET /api/tags`: one bad file must not take
    down the listing.
    """
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


_scoring_cache: dict[tuple[str, int, int], bool] = {}


def is_scoring_model(gguf_path: str | Path) -> bool:
    """Whether `gguf_path` is a scoring model (reranker/classifier; score_head.py). Header-only and
    cached like `unsupported_features`."""
    path = Path(gguf_path)
    stat = path.stat()
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    if cache_key not in _scoring_cache:
        parsed = GGUFReader(path).read()
        names = {info.name for info in parsed.tensor_infos}
        _scoring_cache[cache_key] = ScoreHead.is_scoring(parsed.metadata, names)
    return _scoring_cache[cache_key]
