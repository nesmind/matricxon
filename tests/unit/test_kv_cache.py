import pytest
import torch

from app.runtime.kv_cache import KVCache
from app.server.errors import PromptTooLongError


def _make_cache(max_seq_len: int = 8) -> KVCache:
    return KVCache(layer_shapes=[(2, 4), (2, 4)], max_seq_len=max_seq_len)


class TestKVCache:
    def test_starts_empty(self) -> None:
        cache = _make_cache()
        assert cache.length == 0

    def test_update_writes_at_current_length_for_every_layer(self) -> None:
        cache = _make_cache()
        k = torch.ones(1, 2, 3, 4)
        v = torch.full((1, 2, 3, 4), 2.0)

        k0, v0 = cache.update(0, k, v)
        k1, v1 = cache.update(1, k, v)
        cache.advance(3)

        assert k0.shape == (1, 2, 3, 4)
        assert torch.equal(k0, k)
        assert torch.equal(v1, v)
        assert cache.length == 3

    def test_advance_offsets_the_next_update(self) -> None:
        cache = _make_cache()
        prompt_k = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
        cache.update(0, prompt_k, prompt_k)
        cache.advance(3)

        next_k = torch.full((1, 2, 1, 4), 99.0)
        full_k, _ = cache.update(0, next_k, next_k)

        assert full_k.shape == (1, 2, 4, 4)
        assert torch.equal(full_k[:, :, :3, :], prompt_k)
        assert torch.equal(full_k[:, :, 3:, :], next_k)

    def test_exceeding_max_seq_len_raises(self) -> None:
        cache = _make_cache(max_seq_len=4)
        k = torch.zeros(1, 2, 3, 4)
        cache.update(0, k, k)
        cache.advance(3)

        with pytest.raises(PromptTooLongError):
            cache.update(0, torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 4))

    def test_supports_a_different_shape_per_layer(self) -> None:
        """Real requirement, not a hypothetical: Gemma4's local/sliding

        layers use 8 kv-heads of dim 256 while its global layers use 1 of
        dim 512 - see KVCache's own docstring.
        """
        cache = KVCache(layer_shapes=[(8, 256), (1, 512)], max_seq_len=8)
        local_k = torch.ones(1, 8, 2, 256)
        global_k = torch.ones(1, 1, 2, 512)

        local_out, _ = cache.update(0, local_k, local_k)
        global_out, _ = cache.update(1, global_k, global_k)

        assert local_out.shape == (1, 8, 2, 256)
        assert global_out.shape == (1, 1, 2, 512)
