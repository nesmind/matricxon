import asyncio

from fastapi import APIRouter, Depends

from app.dependencies import get_model_manager
from app.models.manager import ModelManager
from app.runtime.rerank_engine import RerankEngine
from app.schemas.rerank import RerankRequest, RerankResponse, RerankResult
from app.server.node_kind import require_compute

router = APIRouter()


class RerankRequestHandler:
    """Handles `POST /api/rerank`: scores each document against the query, best first."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    def handle(self, request: RerankRequest) -> RerankResponse:
        handle = self._model_manager.get_or_load(
            request.model, keep_alive_seconds=request.keep_alive
        )
        engine = RerankEngine(handle, request.max_tokens)
        results = [
            RerankResult(
                index=index,
                relevance_score=engine.score(request.query, document, request.instruction),
                document=document if request.return_documents else None,
            )
            for index, document in enumerate(request.documents)
        ]
        results.sort(key=lambda result: result.relevance_score, reverse=True)
        return RerankResponse(model=request.model, results=results[: request.top_n])


@router.post("/api/rerank", dependencies=[Depends(require_compute)], response_model=RerankResponse)
async def post_rerank(
    request: RerankRequest, model_manager: ModelManager = Depends(get_model_manager)
) -> RerankResponse:
    return await asyncio.to_thread(RerankRequestHandler(model_manager).handle, request)
