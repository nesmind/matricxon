import asyncio
import logging
import time
from collections.abc import AsyncIterator

import torch
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.dependencies import get_model_catalog, get_model_manager
from app.models.catalog import ModelCatalog
from app.models.handle import ModelHandle
from app.models.manager import ModelManager
from app.native.gemm import NativeGemm
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.prompt_builder import Mistral3PromptBuilder
from app.runtime.tokenizer import IncrementalTextDecoder
from app.runtime.vision_fusion import build_prompt_with_images
from app.schemas.chat import ChatChunk, ChatDoneChunk, ChatRequest, ChatStreamMessage
from app.server.errors import PromptTooLongError
from app.server.ndjson import NDJSONResponse

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequestHandler:
    """Handles `POST /api/chat`.

    All fallible work (model load, prompt building, tokenization, num_ctx
    validation) happens synchronously in `handle()`, before any
    `NDJSONResponse` is constructed - Starlette sends status/headers before
    pulling a streaming response's first item, so a validation error raised
    here surfaces as a clean pre-stream HTTP error (via the `MatricxonError`
    -> `JSONResponse` exception handler) instead of a broken in-stream one.
    """

    def __init__(self, model_manager: ModelManager, catalog: ModelCatalog) -> None:
        self._model_manager = model_manager
        self._catalog = catalog
        self._prompt_builder = Mistral3PromptBuilder()

    def handle(self, request: ChatRequest) -> JSONResponse | NDJSONResponse:
        if request.is_unload_call():
            self._model_manager.unload(request.model)
            return JSONResponse(status_code=200, content={})

        load_started = time.monotonic()
        handle = self._model_manager.get_or_load(
            request.model, keep_alive_seconds=request.keep_alive
        )
        load_duration = time.monotonic() - load_started
        logger.info("model %r ready in %.2fs (get_or_load)", request.model, load_duration)

        # A stop (unload) call for this exact model arrived while it had no handle yet - i.e.
        # while the get_or_load() call just above was still stuck inside loading/dequantization,
        # which alone can take longer than a caller waits before giving up. See
        # ModelManager.pop_cancelled_load's own docstring: without this check, that stop request
        # would otherwise be silently lost the instant loading finished, and generation would
        # start anyway - completely unaware anyone had already asked to stop.
        if self._model_manager.pop_cancelled_load(request.model):
            return JSONResponse(status_code=200, content={})

        stage_started = time.monotonic()
        # The model's own chat template (see app.runtime.chat_template) - Mistral3PromptBuilder
        # only for mistral3 itself or a model with no template at all.
        prompt_builder = handle.prompt_builder or self._prompt_builder
        prompt = prompt_builder.build(request.messages, request.tools)
        logger.info(
            "prompt built: %d messages -> %d chars in %.1fms",
            len(request.messages),
            len(prompt),
            (time.monotonic() - stage_started) * 1000,
        )

        # Real vision support (LlamaArchitecture only, e.g. LLaVA) - see
        # app.runtime.vision_fusion's own module docstring. A request with real images but no
        # paired mmproj (see ModelCatalog.find_paired_mmproj) falls through to the plain
        # tokenizer.encode() path below unchanged - matricxon's original "count and ignore" v1
        # stance for every architecture that isn't wired up for real fusion yet.
        all_images = [img for message in request.messages for img in (message.images or [])]
        mmproj = self._catalog.find_paired_mmproj(request.model) if all_images else None

        stage_started = time.monotonic()
        image_embeddings = None
        if mmproj is not None:
            prompt_token_ids, image_embeddings = build_prompt_with_images(
                prompt, all_images, handle.tokenizer, mmproj
            )
        else:
            prompt_token_ids = handle.tokenizer.encode(
                prompt, add_bos=prompt_builder.wants_bos(prompt)
            )
        logger.info(
            "tokenized: %d chars -> %d tokens in %.1fms (num_ctx=%d, images=%d)",
            len(prompt),
            len(prompt_token_ids),
            (time.monotonic() - stage_started) * 1000,
            request.options.num_ctx,
            len(all_images),
        )
        if len(prompt_token_ids) > request.options.num_ctx:
            raise PromptTooLongError(
                f"prompt has {len(prompt_token_ids)} tokens, "
                f"exceeds num_ctx={request.options.num_ctx}"
            )

        generation_request = self._build_generation_request(
            request, prompt_token_ids, image_embeddings
        )
        return NDJSONResponse(
            self._stream(handle, generation_request, len(prompt_token_ids), load_duration)
        )

    def _build_generation_request(
        self,
        request: ChatRequest,
        prompt_token_ids: list[int],
        image_embeddings: list[tuple[int, torch.Tensor]] | None = None,
    ) -> GenerationRequest:
        options = request.options
        return GenerationRequest(
            input_ids=torch.tensor([prompt_token_ids], dtype=torch.long),
            image_embeddings=image_embeddings,
            sampling=SamplingConfig(
                temperature=options.temperature,
                top_p=options.top_p,
                top_k=options.top_k,
                repeat_penalty=options.repeat_penalty,
                num_ctx=options.num_ctx,
                num_predict=options.num_predict,
                seed=options.seed,
            ),
        )

    async def _stream(
        self,
        handle: ModelHandle,
        generation_request: GenerationRequest,
        prompt_eval_count: int,
        load_duration: float,
    ) -> AsyncIterator[dict]:
        engine = ChatEngine(handle.architecture, eos_token_ids={handle.tokenizer.eos_token_id})
        decoder = IncrementalTextDecoder(handle.tokenizer)
        results = handle.worker.stream(
            lambda: engine.stream(generation_request, stop_check=handle.worker.should_stop)
        )

        eval_count = 0
        generation_started = time.monotonic()
        last_token_at = generation_started
        while True:
            item = await results.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            eval_count += 1
            now = time.monotonic()
            logger.info("token %d: id=%d elapsed=%.3fs", eval_count, item, now - last_token_at)
            last_token_at = now
            # The EOS token itself is a real, counted generation step (matches
            # ChatEngine's own token_ids/eval_count accounting) but its decoded
            # text (e.g. literal "</s>") is never meant to reach the visible
            # message - `done: true` is how a client learns generation ended.
            if item == handle.tokenizer.eos_token_id:
                continue
            text = decoder.push(item)
            if text:
                yield ChatChunk(message=ChatStreamMessage(content=text)).to_ndjson_dict()

        eval_duration = time.monotonic() - generation_started
        logger.info(
            "generation done: %d tokens in %.2fs (%.2fs/token)",
            eval_count,
            eval_duration,
            eval_duration / eval_count if eval_count else 0.0,
        )
        native = NativeGemm.active()
        if native is not None:
            calls, seconds = native.take_stats()
            logger.info(
                "native kernels: %d calls, %.2fs (%.0f%% of generation)",
                calls,
                seconds,
                100 * seconds / eval_duration if eval_duration else 0.0,
            )
        yield ChatDoneChunk(
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            load_duration=_seconds_to_ns(load_duration),
            eval_duration=_seconds_to_ns(eval_duration),
            total_duration=_seconds_to_ns(load_duration + eval_duration),
        ).to_ndjson_dict()


def _seconds_to_ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


@router.post("/api/chat", response_model=None)
async def post_chat(
    request: ChatRequest,
    model_manager: ModelManager = Depends(get_model_manager),
    catalog: ModelCatalog = Depends(get_model_catalog),
) -> JSONResponse | NDJSONResponse:
    # Offloaded to a thread: model loading (potentially seconds of blocking
    # disk/dequant work) and everything else ChatRequestHandler.handle() does
    # synchronously must never block the event loop - it also has to keep
    # answering /api/tags health probes while a chat call is in flight.
    handler = ChatRequestHandler(model_manager, catalog)
    return await asyncio.to_thread(handler.handle, request)
