from fastapi import APIRouter, Depends, Response

from app.consensus.runtime import ConsensusRuntime
from app.dependencies import get_consensus_runtime, get_model_catalog, get_model_manager
from app.models.catalog import ModelCatalog
from app.models.manager import ModelManager
from app.routers.pull_router import require_pull_lease
from app.schemas.delete import DeleteRequest
from app.server.admin_auth import require_admin
from app.server.node_kind import require_compute

router = APIRouter()


class DeleteRequestHandler:
    """Unloads the model first (if loaded) before touching its files: an open `mmap` can't always be
    cleanly deleted while a `ModelWorker` thread holds tensors backed by it."""

    def __init__(self, catalog: ModelCatalog, model_manager: ModelManager) -> None:
        self._catalog = catalog
        self._model_manager = model_manager

    def handle(self, request: DeleteRequest) -> None:
        self._catalog.check_deletable(request.model)
        self._model_manager.unload(request.model)
        self._catalog.delete(request.model)


@router.delete(
    "/api/delete",
    dependencies=[Depends(require_admin), Depends(require_compute)],
    status_code=200,
)
def delete_model(
    request: DeleteRequest,
    catalog: ModelCatalog = Depends(get_model_catalog),
    model_manager: ModelManager = Depends(get_model_manager),
    runtime: ConsensusRuntime | None = Depends(get_consensus_runtime),
) -> Response:
    # The models folder is shared: removing a model changes it like pulling one, so only the
    # pull-lease owner does it (which also needs a quorum).
    require_pull_lease(runtime)
    DeleteRequestHandler(catalog, model_manager).handle(request)
    return Response(status_code=200)
