import asyncio

from fastapi import APIRouter, Depends

from app.dependencies import get_model_manager
from app.models.manager import ModelManager
from app.runtime.embedding_engine import EmbeddingEngine
from app.schemas.embed import EmbedRequest, EmbedResponse
from app.server.errors import PromptTooLongError

router = APIRouter()


class EmbedRequestHandler:
    """Handles `POST /api/embed` (plural) - Ollama's newer batched embedding

    endpoint. `EmbeddingEngine` is v1 single-prompt-only (no padding/
    attention-mask support - see its own docstring), so each input string
    gets its own forward pass rather than one real padded batch; correct
    output, just not batched compute. Worth revisiting if this endpoint ever
    sees real traffic - no caller needs it today (pAIring only uses the
    singular `/api/embeddings`).
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    def handle(self, request: EmbedRequest) -> EmbedResponse:
        handle = self._model_manager.get_or_load(request.model)
        engine = EmbeddingEngine(handle.architecture, handle.tokenizer)

        embeddings = []
        for text in request.inputs():
            token_ids = handle.tokenizer.encode(text)
            if len(token_ids) > request.options.num_ctx:
                raise PromptTooLongError(
                    f"input has {len(token_ids)} tokens, exceeds num_ctx={request.options.num_ctx}"
                )
            embeddings.append(engine.embed(text))

        return EmbedResponse(embeddings=embeddings)


@router.post("/api/embed", response_model=EmbedResponse)
async def post_embed(
    request: EmbedRequest, model_manager: ModelManager = Depends(get_model_manager)
) -> EmbedResponse:
    handler = EmbedRequestHandler(model_manager)
    return await asyncio.to_thread(handler.handle, request)
