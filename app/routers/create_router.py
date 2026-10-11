from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends

from app.consensus.runtime import ConsensusRuntime
from app.dependencies import get_consensus_runtime, get_model_catalog
from app.models.catalog import ModelCatalog
from app.routers.pull_router import require_pull_lease
from app.schemas.create import CreateRequest
from app.server.admin_auth import require_admin
from app.server.ndjson import NDJSONResponse
from app.server.node_kind import require_compute

router = APIRouter()


class CreateRequestHandler:
    """Handles `POST /api/create`'s FROM-only case: matricxon has no Modelfile parser
    (TEMPLATE/PARAMETER/SYSTEM directives are silently ignored, a documented v1 limitation). `FROM
    <existing-tag>` is a local re-tag via the catalog aliasing `/api/copy` uses. The work (one small
    alias file) is quick enough to do before streaming, so a bad FROM tag is a clean pre-stream 404
    (`UnknownModelError`), not an in-band NDJSON error line like a mid-stream `/api/pull`
    failure."""

    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog

    def handle(self, request: CreateRequest) -> NDJSONResponse:
        self._catalog.copy(request.from_, request.model)
        return NDJSONResponse(self._stream())

    async def _stream(self) -> AsyncIterator[dict]:
        yield {"status": "success"}


@router.post(
    "/api/create",
    dependencies=[Depends(require_admin), Depends(require_compute)],
    response_model=None,
)
def post_create(
    request: CreateRequest,
    catalog: ModelCatalog = Depends(get_model_catalog),
    runtime: ConsensusRuntime | None = Depends(get_consensus_runtime),
) -> NDJSONResponse:
    require_pull_lease(runtime)
    return CreateRequestHandler(catalog).handle(request)
