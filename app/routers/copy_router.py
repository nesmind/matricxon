from fastapi import APIRouter, Depends, Response

from app.dependencies import get_model_catalog
from app.models.catalog import ModelCatalog
from app.schemas.copy import CopyRequest

router = APIRouter()


class CopyRequestHandler:
    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog

    def handle(self, request: CopyRequest) -> None:
        self._catalog.copy(request.source, request.destination)


@router.post("/api/copy", status_code=200)
def post_copy(request: CopyRequest, catalog: ModelCatalog = Depends(get_model_catalog)) -> Response:
    CopyRequestHandler(catalog).handle(request)
    return Response(status_code=200)
