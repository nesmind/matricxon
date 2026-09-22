import torch

from app.runtime.mamba_cache import NemotronHHybridCache


def _make_cache(layer_types: list[str], max_seq_len: int = 8) -> NemotronHHybridCache:
    return NemotronHHybridCache(
        layer_types=layer_types,
        attention_layer_shape=(2, 4),
        mamba_conv_state_shape=(3, 5),
        mamba_ssm_state_shape=(2, 4, 6),
        max_seq_len=max_seq_len,
    )


class TestAttentionLayersRouteThroughTheInnerKVCache:
    def test_update_writes_and_reads_back_correctly_by_real_layer_index(self) -> None:
        cache = _make_cache(["mamba", "attention", "mlp", "attention"])
        k = torch.ones(1, 2, 3, 4)
        v = torch.full((1, 2, 3, 4), 2.0)

        k1, v1 = cache.update_attention(1, k, v)
        k3, v3 = cache.update_attention(3, k, v)

        assert k1.shape == (1, 2, 3, 4)
        assert torch.equal(k1, k)
        assert torch.equal(v3, v)

    def test_two_real_attention_layer_indices_stay_independent(self) -> None:
        cache = _make_cache(["attention", "mamba", "attention"])
        k_a = torch.ones(1, 2, 1, 4)
        k_b = torch.full((1, 2, 1, 4), 99.0)

        cache.update_attention(0, k_a, k_a)
        cache.update_attention(2, k_b, k_b)
        cache.advance(1)

        out_a, _ = cache.update_attention(0, torch.zeros(1, 2, 0, 4), torch.zeros(1, 2, 0, 4))
        out_b, _ = cache.update_attention(2, torch.zeros(1, 2, 0, 4), torch.zeros(1, 2, 0, 4))
        assert torch.equal(out_a, k_a)
        assert torch.equal(out_b, k_b)


class TestMambaStatePersistence:
    def test_starts_zeroed(self) -> None:
        cache = _make_cache(["mamba", "attention"])
        conv_state, ssm_state = cache.mamba_state(0)
        assert torch.equal(conv_state, torch.zeros(1, 3, 5))
        assert torch.equal(ssm_state, torch.zeros(1, 2, 4, 6))

    def test_set_then_get_round_trips_by_real_layer_index(self) -> None:
        cache = _make_cache(["attention", "mamba", "mlp", "mamba"])
        new_conv = torch.full((1, 3, 5), 7.0)
        new_ssm = torch.full((1, 2, 4, 6), 9.0)

        cache.set_mamba_state(3, new_conv, new_ssm)
        conv_out, ssm_out = cache.mamba_state(3)

        assert torch.equal(conv_out, new_conv)
        assert torch.equal(ssm_out, new_ssm)

    def test_two_real_mamba_layer_indices_stay_independent(self) -> None:
        cache = _make_cache(["mamba", "mamba"])
        cache.set_mamba_state(0, torch.full((1, 3, 5), 1.0), torch.full((1, 2, 4, 6), 1.0))
        cache.set_mamba_state(1, torch.full((1, 3, 5), 2.0), torch.full((1, 2, 4, 6), 2.0))

        conv0, ssm0 = cache.mamba_state(0)
        conv1, ssm1 = cache.mamba_state(1)
        assert torch.equal(conv0, torch.full((1, 3, 5), 1.0))
        assert torch.equal(ssm1, torch.full((1, 2, 4, 6), 2.0))


class TestLengthAndAdvanceMatchPlainKVCacheSemantics:
    def test_starts_at_zero_and_advances(self) -> None:
        cache = _make_cache(["mamba", "attention", "mlp"])
        assert cache.length == 0

        cache.advance(5)
        assert cache.length == 5

        cache.advance(1)
        assert cache.length == 6
