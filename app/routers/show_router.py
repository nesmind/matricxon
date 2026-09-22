from fastapi import APIRouter, Depends

from app.config import settings
from app.dependencies import get_model_catalog
from app.models.capabilities import effective_capabilities
from app.models.catalog import ModelCatalog
from app.models.load_dtype import QUANTIZED_NATIVE_WIRED_ARCHITECTURES, estimate_ram_gb
from app.schemas.show import ShowRequest, ShowResponse
from app.schemas.tags import ModelDetails

router = APIRouter()


class ShowRequestHandler:
    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog

    def handle(self, request: ShowRequest) -> ShowResponse:
        installed = self._catalog.get(request.model)
        has_paired_mmproj = self._catalog.find_paired_mmproj(installed.tag) is not None
        # See estimate_ram_gb's own docstring - only report the smaller quantized-native figure
        # for an architecture that's actually wired to use it (QUANTIZED_NATIVE_WIRED_
        # ARCHITECTURES), never just because the setting is on globally.
        quantized_native_enabled = (
            settings.enable_quantized_native_compute
            and installed.architecture in QUANTIZED_NATIVE_WIRED_ARCHITECTURES
        )

        return ShowResponse(
            capabilities=effective_capabilities(installed, has_paired_mmproj),
            details=ModelDetails(
                family=installed.family,
                parameter_size=installed.parameter_size,
                context_length=installed.context_length,
            ),
            estimated_ram_gb=estimate_ram_gb(
                installed.path,
                settings.memory_safety_margin,
                quantized_native_enabled,
                installed.architecture,
            ),
        )


@router.post("/api/show", response_model=ShowResponse)
def post_show(
    request: ShowRequest, catalog: ModelCatalog = Depends(get_model_catalog)
) -> ShowResponse:
    return ShowRequestHandler(catalog).handle(request)
