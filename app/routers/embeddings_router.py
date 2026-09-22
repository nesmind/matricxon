import asyncio

from fastapi import APIRouter, Depends

from app.dependencies import get_model_manager
from app.models.manager import ModelManager
from app.runtime.embedding_engine import EmbeddingEngine
from app.schemas.embeddings import EmbeddingsRequest, EmbeddingsResponse
from app.server.errors import PromptTooLongError


class EmbeddingsRequestHandler:
    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    def handle(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        handle = self._model_manager.get_or_load(request.model)

        token_ids = handle.tokenizer.encode(request.prompt)
        if len(token_ids) > request.options.num_ctx:
            raise PromptTooLongError(
                f"prompt has {len(token_ids)} tokens, exceeds num_ctx={request.options.num_ctx}"
            )

        engine = EmbeddingEngine(handle.architecture, handle.tokenizer)
        return EmbeddingsResponse(embedding=engine.embed(request.prompt))


router = APIRouter()


@router.post("/api/embeddings", response_model=EmbeddingsResponse)
async def post_embeddings(
    request: EmbeddingsRequest, model_manager: ModelManager = Depends(get_model_manager)
) -> EmbeddingsResponse:
    # Offloaded to a thread for the same reason as /api/chat: model loading
    # and the forward pass both block, and must not stall the event loop.
    handler = EmbeddingsRequestHandler(model_manager)
    return await asyncio.to_thread(handler.handle, request)
