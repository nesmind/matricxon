"""Mixtral support for `LlamaArchitecture` - split out of llama.py (file-size cap).

llama.cpp has no separate `mixtral` GGUF architecture string at all - confirmed directly from its
real `conversion/llama.py`: `MixtralForCausalLM` is registered onto the exact same `LlamaModel`
class plain Llama uses (`undo_permute = True` inherited unchanged), so a real Mixtral GGUF's
`general.architecture` is still `"llama"`, distinguished only by carrying
`llama.expert_count`/`llama.expert_used_count` metadata and `ffn_gate_inp`/`ffn_gate_exps`/
`ffn_up_exps`/`ffn_down_exps` tensors instead of plain `ffn_gate`/`ffn_up`/`ffn_down`.
`LlamaArchitecture` detects this per real metadata presence (`detect_moe` below) and injects a
`GraniteMoeFFN` (`granitemoe_layers.py`) into each `Mistral3DecoderLayer` instead of a plain
`SwiGLUMLP` (`build_ffn` below) - real attention/RoPE/`unpermute_rope_rows` handling is completely
unchanged either way (confirmed by the very fact both share one real converter class - if Mixtral
needed different attention wiring, llama.cpp couldn't reuse `LlamaModel` for it at all).

`GraniteMoeFFN`'s own router math (real top-k-*then*-softmax over just the selected experts) is
not Granite-specific despite the name - worked out symbolically against HF's own real
`modeling_mixtral.py` `MixtralTopKRouter` (softmax-over-all-experts-then-topk-then-renormalize-
the-selected-subset): the shared softmax normalizer cancels out of the renormalization step,
leaving the exact same final per-expert weight formula either order it's computed in - confirmed
algebraically, not assumed from surface resemblance, before reusing the class as-is rather than
duplicating it.
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
    """(num_experts, num_experts_per_tok), both None for a real dense (non-Mixtral) `llama` file."""
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
    """Router: always a plain `.copy_()` (small enough that packing it wouldn't meaningfully
    help). Experts: `QuantizedMoEExperts` (`app/architectures/moe_experts.py`) - real per-expert
    `QuantizedLinear`s when `enabled` and a real GEMV kernel exists, the exact same real router
    math proven algebraically identical to Mixtral's own (see this module's own docstring)
    either way."""
    mlp.router.weight.copy_(loader.load_tensor(prefix + "ffn_gate_inp.weight"))
    materialize_quantized_moe_experts(mlp.experts, loader, prefix, enabled, mark_used)
