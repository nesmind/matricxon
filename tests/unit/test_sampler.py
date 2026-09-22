import torch

from app.runtime.generation_request import SamplingConfig
from app.runtime.sampler import RepetitionPenaltyFilter, Sampler, TopKFilter, TopPFilter


class TestTemperatureGreedy:
    def test_zero_temperature_picks_argmax_regardless_of_other_options(self) -> None:
        logits = torch.tensor([0.1, 5.0, 0.2, -1.0])
        sampler = Sampler(SamplingConfig(temperature=0.0, top_k=1, top_p=0.01, seed=123))

        assert sampler.sample(logits, generated_ids=[]) == 1


class TestSamplerDeterminism:
    def test_same_seed_produces_the_same_sequence(self) -> None:
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0])
        config = SamplingConfig(temperature=1.0, top_k=0, top_p=1.0, seed=42)

        first = [Sampler(config).sample(logits, []) for _ in range(10)]
        second = [Sampler(config).sample(logits, []) for _ in range(10)]

        assert first == second

    def test_no_seed_can_diverge_across_samplers(self) -> None:
        logits = torch.full((100,), 1.0)
        config = SamplingConfig(temperature=1.0, top_k=0, top_p=1.0, seed=None)

        first = [Sampler(config).sample(logits, []) for _ in range(20)]
        second = [Sampler(config).sample(logits, []) for _ in range(20)]

        assert first != second


class TestRepetitionPenaltyFilter:
    def test_suppresses_a_positive_logit_for_an_already_generated_token(self) -> None:
        logits = torch.tensor([2.0, 2.0, 2.0])
        result = RepetitionPenaltyFilter(penalty=2.0).apply(logits, generated_ids=[1])

        assert result[1] < logits[1]
        assert result[0] == logits[0]

    def test_no_penalty_or_no_history_is_a_no_op(self) -> None:
        logits = torch.tensor([2.0, 2.0, 2.0])
        assert torch.equal(RepetitionPenaltyFilter(penalty=1.0).apply(logits, [1]), logits)
        assert torch.equal(RepetitionPenaltyFilter(penalty=2.0).apply(logits, []), logits)


class TestTopKFilter:
    def test_masks_everything_outside_the_top_k(self) -> None:
        logits = torch.tensor([1.0, 4.0, 3.0, 2.0])
        result = TopKFilter(k=2).apply(logits)

        assert result[1] == 4.0
        assert result[2] == 3.0
        assert result[0] == float("-inf")
        assert result[3] == float("-inf")

    def test_k_zero_is_a_no_op(self) -> None:
        logits = torch.tensor([1.0, 4.0, 3.0])
        assert torch.equal(TopKFilter(k=0).apply(logits), logits)


class TestTopPFilter:
    def test_keeps_at_least_the_top_token(self) -> None:
        logits = torch.tensor([10.0, -10.0, -10.0])
        result = TopPFilter(p=0.01).apply(logits)

        assert result[0] == 10.0
        assert result[1] == float("-inf")

    def test_p_one_is_a_no_op(self) -> None:
        logits = torch.tensor([1.0, 4.0, 3.0])
        assert torch.equal(TopPFilter(p=1.0).apply(logits), logits)
