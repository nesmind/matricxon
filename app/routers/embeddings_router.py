import asyncio
from collections.abc import Iterator

from fastapi import APIRouter, Depends

from app.dependencies import get_model_manager
from app.models.manager import ModelManager
from app.models.worker_call import WorkerCall
from app.runtime.embedding_engine import EmbeddingEngine
from app.runtime.rerank_engine import ModelKind
from app.schemas.embeddings import EmbeddingsRequest, EmbeddingsResponse
from app.server.errors import PromptTooLongError
from app.server.node_kind import require_compute


class EmbeddingsRequestHandler:
    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def handle(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        handle = await asyncio.to_thread(self._model_manager.get_or_load, request.model)
        ModelKind.require_generative(handle)
        engine = EmbeddingEngine(handle.architecture, handle.tokenizer)

        def embed_one() -> Iterator[list[float]]:
            token_ids = handle.tokenizer.encode(request.prompt)
            if len(token_ids) > request.options.num_ctx:
                raise PromptTooLongError(
                    f"prompt has {len(token_ids)} tokens, exceeds num_ctx={request.options.num_ctx}"
                )
            yield engine.embed(request.prompt)

        (embedding,) = await WorkerCall.collect(handle.worker, embed_one)
        return EmbeddingsResponse(embedding=embedding)


router = APIRouter()


@router.post(
    "/api/embeddings", dependencies=[Depends(require_compute)], response_model=EmbeddingsResponse
)
async def post_embeddings(
    request: EmbeddingsRequest, model_manager: ModelManager = Depends(get_model_manager)
) -> EmbeddingsResponse:
    # Loading runs in a thread and the forward pass on the model's worker, like /api/chat: neither
    # blocks the event loop, and the model counts as busy while in use.
    return await EmbeddingsRequestHandler(model_manager).handle(request)
