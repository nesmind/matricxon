from collections.abc import Callable

import torch
from torch import nn

from app.architectures.base import ModelArchitecture
from app.gguf.metadata import GGUFMetadata
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.kv_cache import KVCache


class _AlwaysTokenOneDecoder(ModelArchitecture):
    """Deterministic stand-in for a real architecture: ignores its input and
    always makes token 1 the argmax. Still writes through a real (tiny)
    KVCache each call, so this exercises ChatEngine's cache/position
    plumbing - including matching the cache's dtype to the model's, which a
    real bug here (mismatched float32 cache vs a bf16 model) would surface
    as a crash inside `forward`, not a silent wrong answer.
    """

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.n_layer = 1
        self.n_head_kv = 1
        self.head_dim = 2
        self.kv_cache_layer_shapes = [(self.n_head_kv, self.head_dim)] * self.n_layer
        self._weight = nn.Parameter(torch.zeros(1, dtype=dtype))
        # Never goes through from_gguf/_defer_materialization - its one real
        # "weight" is set directly above, so there's nothing to lazily
        # materialize (see ModelArchitecture._ensure_materialized).
        self._materialized = True

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return True

    @classmethod
    def from_gguf(cls, loader: object) -> "_AlwaysTokenOneDecoder":
        raise NotImplementedError

    def _materialize_weights(self, loader: object) -> None:
        raise AssertionError("never deferred, so this should never be called")

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        position_ids: torch.Tensor | None = None,
        stop_check: Callable[[], bool] | None = None,
        image_embeddings: list[tuple[int, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        dtype = self._weight.dtype
        if kv_cache is not None:
            dummy_kv = torch.zeros(batch, self.n_head_kv, seq_len, self.head_dim, dtype=dtype)
            kv_cache.update(0, dummy_kv, dummy_kv)

        logits = torch.zeros(batch, seq_len, 4, dtype=dtype)
        logits[..., 1] = 10.0
        return logits


def _request(**sampling_overrides: object) -> GenerationRequest:
    input_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
    sampling = SamplingConfig(temperature=0.0, num_ctx=16, **sampling_overrides)
    return GenerationRequest(input_ids=input_ids, sampling=sampling)


class TestChatEngine:
    def test_stops_immediately_when_the_first_sampled_token_is_eos(self) -> None:
        engine = ChatEngine(_AlwaysTokenOneDecoder(), eos_token_ids={1})

        result = engine.generate(_request(num_predict=10))

        assert result.token_ids == [1]
        assert result.finish_reason == "stop"

    def test_stops_at_num_predict_when_no_eos_is_hit(self) -> None:
        engine = ChatEngine(_AlwaysTokenOneDecoder(), eos_token_ids=set())

        result = engine.generate(_request(num_predict=3))

        assert result.token_ids == [1, 1, 1]
        assert result.finish_reason == "length"

    def test_works_with_a_non_float32_model_dtype(self) -> None:
        engine = ChatEngine(_AlwaysTokenOneDecoder(dtype=torch.bfloat16), eos_token_ids=set())

        result = engine.generate(_request(num_predict=2))

        assert result.token_ids == [1, 1]
