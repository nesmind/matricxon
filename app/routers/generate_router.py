import asyncio
import time
from collections.abc import AsyncIterator

import torch
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.dependencies import get_model_manager
from app.models.handle import ModelHandle
from app.models.manager import ModelManager
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.tokenizer import IncrementalTextDecoder
from app.schemas.generate import GenerateChunk, GenerateDoneChunk, GenerateRequest
from app.server.errors import PromptTooLongError
from app.server.ndjson import NDJSONResponse

router = APIRouter()


class GenerateRequestHandler:
    """Handles `POST /api/generate` - a single raw-prompt completion, not a

    chat message list. Reuses ChatEngine/GenerationRequest unchanged (both
    are already prompt-agnostic, working purely in already-tokenized
    `input_ids`) and skips Mistral3PromptBuilder entirely: the caller's
    `prompt` string is tokenized as-is, with no `[INST]`/`[SYSTEM_PROMPT]`
    wrapping - matching real Ollama's `/api/generate` in "raw" mode, the
    simpler of its two real behaviors and the only one with no caller-facing
    ambiguity about which chat template got applied.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    def handle(self, request: GenerateRequest) -> JSONResponse | NDJSONResponse:
        if request.is_unload_call():
            self._model_manager.unload(request.model)
            return JSONResponse(status_code=200, content={})

        load_started = time.monotonic()
        handle = self._model_manager.get_or_load(
            request.model, keep_alive_seconds=request.keep_alive
        )
        load_duration = time.monotonic() - load_started

        if self._model_manager.pop_cancelled_load(request.model):
            return JSONResponse(status_code=200, content={})

        prompt_token_ids = handle.tokenizer.encode(request.prompt, add_bos=True)
        if len(prompt_token_ids) > request.options.num_ctx:
            raise PromptTooLongError(
                f"prompt has {len(prompt_token_ids)} tokens, "
                f"exceeds num_ctx={request.options.num_ctx}"
            )

        generation_request = self._build_generation_request(request, prompt_token_ids)
        return NDJSONResponse(
            self._stream(handle, generation_request, len(prompt_token_ids), load_duration)
        )

    def _build_generation_request(
        self, request: GenerateRequest, prompt_token_ids: list[int]
    ) -> GenerationRequest:
        options = request.options
        return GenerationRequest(
            input_ids=torch.tensor([prompt_token_ids], dtype=torch.long),
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
        results = handle.worker.stream(lambda: engine.stream(generation_request))

        eval_count = 0
        generation_started = time.monotonic()
        while True:
            item = await results.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            eval_count += 1
            if item == handle.tokenizer.eos_token_id:
                continue
            text = decoder.push(item)
            if text:
                yield GenerateChunk(response=text).to_ndjson_dict()

        eval_duration = time.monotonic() - generation_started
        yield GenerateDoneChunk(
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            load_duration=_seconds_to_ns(load_duration),
            eval_duration=_seconds_to_ns(eval_duration),
            total_duration=_seconds_to_ns(load_duration + eval_duration),
        ).to_ndjson_dict()


def _seconds_to_ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


@router.post("/api/generate", response_model=None)
async def post_generate(
    request: GenerateRequest, model_manager: ModelManager = Depends(get_model_manager)
) -> JSONResponse | NDJSONResponse:
    handler = GenerateRequestHandler(model_manager)
    return await asyncio.to_thread(handler.handle, request)
