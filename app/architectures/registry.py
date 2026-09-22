from app.architectures.base import ModelArchitecture
from app.architectures.bert import BertArchitecture
from app.architectures.command_r import CommandRArchitecture
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
from app.gguf.metadata import GGUFMetadata
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
