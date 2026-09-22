from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends

from app.dependencies import get_model_catalog
from app.models.catalog import ModelCatalog
from app.schemas.create import CreateRequest
from app.server.ndjson import NDJSONResponse

router = APIRouter()


class CreateRequestHandler:
    """Handles `POST /api/create`'s FROM-only case: matricxon has no

    Modelfile parser (TEMPLATE/PARAMETER/SYSTEM directives, if sent, are
    silently not applied to inference - a real, documented v1 limitation,
    same spirit as Mistral3PromptBuilder dropping tool calls). `FROM
    <existing-tag>` is treated as a local re-tag, reusing the same
    underlying catalog duplication `/api/copy` uses. The actual work (a
    hardlink + a sidecar write) is fast enough to do synchronously before
    streaming starts - a real `UnknownModelError` for a bad FROM tag
    therefore surfaces as a clean pre-stream 404, not an in-band NDJSON
    error line (unlike `/api/pull`, which can genuinely fail mid-stream).
    """

    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog

    def handle(self, request: CreateRequest) -> NDJSONResponse:
        self._catalog.copy(request.from_, request.model)
        return NDJSONResponse(self._stream())

    async def _stream(self) -> AsyncIterator[dict]:
        yield {"status": "success"}


@router.post("/api/create", response_model=None)
def post_create(
    request: CreateRequest, catalog: ModelCatalog = Depends(get_model_catalog)
) -> NDJSONResponse:
    return CreateRequestHandler(catalog).handle(request)
