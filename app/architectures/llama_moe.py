"""Mixtral support for `LlamaArchitecture`, split from llama.py (file-size cap).

llama.cpp has no `mixtral` architecture string: its converter registers `MixtralForCausalLM` on the
same `LlamaModel` class as Llama (`undo_permute = True` inherited), so a Mixtral GGUF's
`general.architecture` is `"llama"`, told apart only by
`llama.expert_count`/`llama.expert_used_count` metadata and
`ffn_gate_inp`/`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` tensors. `LlamaArchitecture` detects it
(`detect_moe`) and injects a `GraniteMoeFFN` into each `Mistral3DecoderLayer` instead of a
`SwiGLUMLP` (`build_ffn`). Attention/RoPE/`unpermute_rope_rows` are unchanged (one converter class).

`GraniteMoeFFN`'s router (top-k then softmax over the selected) isn't Granite-specific. Against HF's
`MixtralTopKRouter` (softmax over all, topk, renormalize) the shared normalizer cancels in the
renormalization, leaving the same per-expert weight; confirmed algebraically before reusing the
class.
"""

from collections.abc import Callable

import torch
from torch import nn

from app.architectures.granitemoe_layers import GraniteMoeFFN
from app.architectures.layers import SwiGLUMLP
from app.architectures.moe_experts import materialize_quantized_moe_experts
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata


def detect_moe(metadata: GGUFMetadata, arch: str) -> tuple[int | None, int | None]:
    """(num_experts, num_experts_per_tok), both None for a dense (non-Mixtral) `llama` file."""
    num_experts = metadata.get_u32(f"{arch}.expert_count")
    if num_experts is None:
        return None, None
    return num_experts, metadata.get_u32(f"{arch}.expert_used_count")


def build_ffn(
    n_embd: int,
    ffn_len: int,
    num_experts: int | None,
    num_experts_per_tok: int | None,
    dtype: torch.dtype,
) -> nn.Module:
    if num_experts is not None:
        return GraniteMoeFFN(n_embd, ffn_len, num_experts, num_experts_per_tok, dtype=dtype)
    return SwiGLUMLP(n_embd, ffn_len, dtype=dtype)


def materialize_moe_ffn(
    mlp: GraniteMoeFFN,
    loader: GGUFModelLoader,
    prefix: str,
    enabled: bool,
    mark_used: Callable[[GGUFModelLoader], None],
) -> None:
    """Router: a plain `.copy_` (too small to pack). Experts: `QuantizedMoEExperts`
    (`moe_experts.py`): per-expert
    `QuantizedLinear`s when `enabled` and a GEMV kernel exists; the router math is proven identical
    to Mixtral's (module docstring) either way.
    """
    mlp.router.weight.copy_(loader.load_tensor(prefix + "ffn_gate_inp.weight"))
    materialize_quantized_moe_experts(mlp.experts, loader, prefix, enabled, mark_used)
