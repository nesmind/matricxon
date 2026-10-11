from fastapi import APIRouter

from app.architectures.registry import ArchitectureRegistry
from app.config import MATRICXON_VERSION
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.schemas.health import HealthResponse

router = APIRouter()


class HealthRequestHandler:
    """A matricxon-only extension, not Ollama API (like the hf.co:-only pull syntax of M7): lets a
    caller (pAIring or others) ask "can you run this?" instead of keeping a driftable copy of
    matricxon's supported-architecture/quantization lists. ROADMAP.md "Expose supported
    architectures/quantizations"."""

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
            moe_supported_architectures=self._architecture_registry.moe_supported_names(),
            vision_supported_architectures=self._architecture_registry.vision_supported_names(),
            supported_quantizations=self._quant_registry.supported_names(),
        )


@router.get("/api/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    handler = HealthRequestHandler(ArchitectureRegistry(), QuantStrategyRegistry())
    return handler.handle()
