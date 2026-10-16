import asyncio
from collections.abc import Iterator

from fastapi import APIRouter, Depends

from app.dependencies import get_model_manager
from app.models.manager import ModelManager
from app.models.worker_call import WorkerCall
from app.runtime.embedding_engine import EmbeddingEngine
from app.runtime.rerank_engine import ModelKind
from app.schemas.embed import EmbedRequest, EmbedResponse
from app.server.errors import PromptTooLongError
from app.server.node_kind import require_compute

router = APIRouter()


class EmbedRequestHandler:
    """Handles `POST /api/embed` (plural), Ollama's newer batched embedding endpoint.
    `EmbeddingEngine` is v1 single-prompt-only (no padding/attention-mask support), so each input
    gets its own forward pass: correct, not batched. Revisit if it sees traffic; pAIring uses only
    the singular `/api/embeddings`."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def handle(self, request: EmbedRequest) -> EmbedResponse:
        handle = await asyncio.to_thread(self._model_manager.get_or_load, request.model)
        ModelKind.require_generative(handle)
        engine = EmbeddingEngine(handle.architecture, handle.tokenizer)

        def embed_all() -> Iterator[list[float]]:
            for text in request.inputs():
                token_ids = handle.tokenizer.encode(text)
                if len(token_ids) > request.options.num_ctx:
                    raise PromptTooLongError(
                        f"input has {len(token_ids)} tokens, "
                        f"exceeds num_ctx={request.options.num_ctx}"
                    )
                yield engine.embed(text)

        return EmbedResponse(embeddings=await WorkerCall.collect(handle.worker, embed_all))


@router.post("/api/embed", dependencies=[Depends(require_compute)], response_model=EmbedResponse)
async def post_embed(
    request: EmbedRequest, model_manager: ModelManager = Depends(get_model_manager)
) -> EmbedResponse:
    return await EmbedRequestHandler(model_manager).handle(request)
