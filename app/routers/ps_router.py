from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends

from app.dependencies import get_model_manager
from app.models.manager import ModelManager
from app.schemas.ps import PsEntry, PsResponse

router = APIRouter()


class PsRequestHandler:
    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    def handle(self) -> PsResponse:
        clock_now = self._model_manager.now()
        wall_now = datetime.now(UTC)
        entries = [
            PsEntry(
                name=handle.tag,
                size=handle.size_bytes,
                size_vram=0,  # matricxon is CPU-only in v1 - see app/config.py's `device`
                expires_at=(wall_now + timedelta(seconds=handle.expires_in(clock_now))).isoformat(),
            )
            for handle in self._model_manager.list_loaded()
        ]
        return PsResponse(models=entries)


@router.get("/api/ps", response_model=PsResponse)
def get_ps(model_manager: ModelManager = Depends(get_model_manager)) -> PsResponse:
    return PsRequestHandler(model_manager).handle()
