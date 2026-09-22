"""Unit tests for app/models/load_dtype.py's mixed-precision planning helpers
(group_bytes_by_layer/plan_layer_dtypes) - the ModelManager._load()/Mistral3TextArchitecture
machinery that actually uses these is covered separately (tests/unit/test_model_manager_memory.py),
this file is pure-function-level: synthetic GGUFTensorInfo lists in, dict out, no GGUF file, no
torch model construction.
"""

import torch

from app.gguf.tensor_info import GGUFTensorInfo
from app.models.load_dtype import NON_LAYER_GROUP, group_bytes_by_layer, plan_layer_dtypes


def _tensor(name: str, n_elements: int) -> GGUFTensorInfo:
    # shape/ggml_type/offset are irrelevant to these two functions (only .name/.n_elements are
    # read) - filled with harmless placeholders rather than made optional on the real dataclass
    # just for this test file's convenience.
    return GGUFTensorInfo(name=name, shape=(n_elements,), ggml_type=0, offset=0)


class TestGroupBytesByLayer:
    def test_groups_tensors_by_blk_n_prefix(self) -> None:
        tensor_infos = [
            _tensor("blk.0.attn_q.weight", 100),
            _tensor("blk.0.attn_k.weight", 50),
            _tensor("blk.1.attn_q.weight", 100),
        ]
        result = group_bytes_by_layer(tensor_infos)
        assert result == {"blk.0": 150 * 4, "blk.1": 100 * 4}

    def test_tensors_with_no_blk_prefix_go_under_non_layer_group(self) -> None:
        tensor_infos = [_tensor("token_embd.weight", 100), _tensor("output_norm.weight", 10)]
        result = group_bytes_by_layer(tensor_infos)
        assert result == {NON_LAYER_GROUP: 110 * 4}

    def test_mixed_layer_and_non_layer_tensors(self) -> None:
        tensor_infos = [
            _tensor("token_embd.weight", 100),
            _tensor("blk.0.attn_q.weight", 50),
        ]
        result = group_bytes_by_layer(tensor_infos)
        assert result == {NON_LAYER_GROUP: 100 * 4, "blk.0": 50 * 4}

    def test_empty_input_returns_empty_dict(self) -> None:
        assert group_bytes_by_layer([]) == {}


class TestPlanLayerDtypes:
    def test_assigns_float32_to_as_many_layers_as_fit_the_budget(self) -> None:
        # 4 equal-sized layers, budget fits exactly 2 (safety_margin=1.0 for simple arithmetic).
        layer_bytes = {"blk.0": 100, "blk.1": 100, "blk.2": 100, "blk.3": 100}
        plan = plan_layer_dtypes(layer_bytes, available_bytes=200, safety_margin=1.0)
        assert plan == {
            "blk.0": torch.float32,
            "blk.1": torch.float32,
            "blk.2": torch.bfloat16,
            "blk.3": torch.bfloat16,
        }

    def test_non_layer_group_is_tried_first(self) -> None:
        # NON_LAYER_GROUP (100) + blk.0 (100) both fit (budget 150) only if NON_LAYER_GROUP goes
        # first and blk.0 doesn't - proves the priority order, not just that *something* fits.
        layer_bytes = {"blk.0": 100, NON_LAYER_GROUP: 100}
        plan = plan_layer_dtypes(layer_bytes, available_bytes=150, safety_margin=1.0)
        assert plan[NON_LAYER_GROUP] == torch.float32
        assert plan["blk.0"] == torch.bfloat16

    def test_layers_assigned_in_ascending_index_order(self) -> None:
        layer_bytes = {"blk.2": 100, "blk.0": 100, "blk.1": 100}
        plan = plan_layer_dtypes(layer_bytes, available_bytes=250, safety_margin=1.0)
        assert plan == {"blk.0": torch.float32, "blk.1": torch.float32, "blk.2": torch.bfloat16}

    def test_safety_margin_shrinks_the_effective_budget(self) -> None:
        layer_bytes = {"blk.0": 100}
        # Raw budget (150) covers blk.0, but 150 / 1.5 = 100 - exactly enough (<=), still fits.
        assert plan_layer_dtypes(layer_bytes, 150, safety_margin=1.5)["blk.0"] == torch.float32
        # 140 / 1.5 = 93.3 - doesn't fit anymore.
        assert plan_layer_dtypes(layer_bytes, 140, safety_margin=1.5)["blk.0"] == torch.bfloat16

    def test_none_available_bytes_assigns_float32_to_everything(self) -> None:
        layer_bytes = {"blk.0": 10**12, NON_LAYER_GROUP: 10**12}
        plan = plan_layer_dtypes(layer_bytes, available_bytes=None, safety_margin=1.5)
        assert plan == {"blk.0": torch.float32, NON_LAYER_GROUP: torch.float32}

    def test_nothing_fits_everything_falls_back_to_bf16(self) -> None:
        layer_bytes = {"blk.0": 100, "blk.1": 100}
        plan = plan_layer_dtypes(layer_bytes, available_bytes=1, safety_margin=1.0)
        assert plan == {"blk.0": torch.bfloat16, "blk.1": torch.bfloat16}

    def test_empty_input_returns_empty_dict(self) -> None:
        assert plan_layer_dtypes({}, available_bytes=1000, safety_margin=1.5) == {}
