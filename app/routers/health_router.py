from fastapi import APIRouter

from app.architectures.registry import ArchitectureRegistry
from app.config import MATRICXON_VERSION
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.schemas.health import HealthResponse

router = APIRouter()


class HealthRequestHandler:
    """A genuine matricxon-only extension, not part of Ollama's own API

    (same category as the hf.co:-only pull syntax M7 established) - lets a
    caller (pAIring, or anyone else) ask "can you run this?" directly
    instead of hand-maintaining its own separate, driftable copy of
    matricxon's real supported-architecture/quantization lists. See
    ROADMAP.md's "Expose supported architectures/quantizations" entry.
    """

    def __init__(
        self, architecture_registry: ArchitectureRegistry, quant_registry: QuantStrategyRegistry
    ) -> None:
        self._architecture_registry = architecture_registry
        self._quant_registry = quant_registry

    def handle(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=MATRICXON_VERSION,
            supported_architectures=self._architecture_registry.supported_names(),
            supported_quantizations=self._quant_registry.supported_names(),
        )


@router.get("/api/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    handler = HealthRequestHandler(ArchitectureRegistry(), QuantStrategyRegistry())
    return handler.handle()
