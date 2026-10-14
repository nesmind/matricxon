import os

from fastapi import APIRouter, Depends

from app.config import settings
from app.dependencies import get_model_manager
from app.models.load_dtype import available_memory_bytes
from app.models.manager import ModelManager
from app.schemas.node_load import LoadedModelLoad, NodeLoadResponse

router = APIRouter()


class NodeLoadHandler:
    """This node's load for cluster routing: unlike a client's own in-flight count, `active` covers
    every client (several pAIring instances, other tools). Cheap: no disk or network work."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    def handle(self) -> NodeLoadResponse:
        models = [
            LoadedModelLoad(name=h.tag, size_bytes=h.size_bytes, active=h.worker.busy_count)
            for h in self._model_manager.list_loaded()
        ]
        return NodeLoadResponse(
            active=sum(m.active for m in models),
            models=models,
            threads=settings.torch_threads or os.cpu_count() or 1,
            free_memory_bytes=available_memory_bytes(),
            node_kind=settings.node_kind,
        )


@router.get("/api/node/load", response_model=NodeLoadResponse)
def get_node_load(model_manager: ModelManager = Depends(get_model_manager)) -> NodeLoadResponse:
    return NodeLoadHandler(model_manager).handle()
