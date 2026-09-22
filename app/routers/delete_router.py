from fastapi import APIRouter, Depends, Response

from app.dependencies import get_model_catalog, get_model_manager
from app.models.catalog import ModelCatalog
from app.models.manager import ModelManager
from app.schemas.delete import DeleteRequest

router = APIRouter()


class DeleteRequestHandler:
    """Unloads the model first (if it's loaded) before touching its files -
    an open `mmap` can't always be cleanly deleted while a `ModelWorker`
    thread still holds tensors backed by it.
    """

    def __init__(self, catalog: ModelCatalog, model_manager: ModelManager) -> None:
        self._catalog = catalog
        self._model_manager = model_manager

    def handle(self, request: DeleteRequest) -> None:
        self._model_manager.unload(request.model)
        self._catalog.delete(request.model)


@router.delete("/api/delete", status_code=200)
def delete_model(
    request: DeleteRequest,
    catalog: ModelCatalog = Depends(get_model_catalog),
    model_manager: ModelManager = Depends(get_model_manager),
) -> Response:
    DeleteRequestHandler(catalog, model_manager).handle(request)
    return Response(status_code=200)
