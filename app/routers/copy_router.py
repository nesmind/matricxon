from fastapi import APIRouter, Depends, Response

from app.consensus.runtime import ConsensusRuntime
from app.dependencies import get_consensus_runtime, get_model_catalog
from app.models.catalog import ModelCatalog
from app.routers.pull_router import require_pull_lease
from app.schemas.copy import CopyRequest
from app.server.admin_auth import require_admin
from app.server.node_kind import require_compute

router = APIRouter()


class CopyRequestHandler:
    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog

    def handle(self, request: CopyRequest) -> None:
        self._catalog.copy(request.source, request.destination)


@router.post(
    "/api/copy",
    dependencies=[Depends(require_admin), Depends(require_compute)],
    status_code=200,
)
def post_copy(
    request: CopyRequest,
    catalog: ModelCatalog = Depends(get_model_catalog),
    runtime: ConsensusRuntime | None = Depends(get_consensus_runtime),
) -> Response:
    require_pull_lease(runtime)
    CopyRequestHandler(catalog).handle(request)
    return Response(status_code=200)
