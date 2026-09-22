from fastapi import APIRouter, Depends

from app.config import settings
from app.dependencies import get_model_catalog
from app.models.capabilities import effective_capabilities
from app.models.catalog import ModelCatalog
from app.models.load_dtype import QUANTIZED_NATIVE_WIRED_ARCHITECTURES, estimate_ram_gb
from app.schemas.tags import ModelDetails, TagEntry, TagsResponse

router = APIRouter()


class TagsRequestHandler:
    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog

    def handle(self) -> TagsResponse:
        # See show_router.py's identical comment - only report the smaller quantized-native
        # figure for an architecture that's actually wired to use it, never just because the
        # setting is on globally.
        entries = [
            TagEntry(
                name=installed.tag,
                capabilities=effective_capabilities(
                    installed, self._catalog.find_paired_mmproj(installed.tag) is not None
                ),
                size=installed.size_bytes,
                details=ModelDetails(
                    family=installed.family,
                    parameter_size=installed.parameter_size,
                    context_length=installed.context_length,
                ),
                estimated_ram_gb=estimate_ram_gb(
                    installed.path,
                    settings.memory_safety_margin,
                    settings.enable_quantized_native_compute
                    and installed.architecture in QUANTIZED_NATIVE_WIRED_ARCHITECTURES,
                    installed.architecture,
                ),
            )
            for installed in self._catalog.list_installed()
        ]
        return TagsResponse(models=entries)


@router.get("/api/tags", response_model=TagsResponse)
def get_tags(catalog: ModelCatalog = Depends(get_model_catalog)) -> TagsResponse:
    return TagsRequestHandler(catalog).handle()
