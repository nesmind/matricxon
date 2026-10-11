import functools
import re
from collections.abc import Callable
from pathlib import Path

import torch

from app.gguf.dequant.quantized_gemv_registry import has_gemv_kernel
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.gguf.reader import GGUFReader
from app.gguf.tensor_info import GGUFTensorInfo

# Tensors with no "blk.N." prefix (token_embd, output_norm, untied output.weight): lets a caller
# size "everything outside the per-layer loop" like one layer. Not a GGUF tensor name; this module's
# grouping key.
NON_LAYER_GROUP = "__non_layer__"

_LAYER_PREFIX_RE = re.compile(r"^blk\.(\d+)\.")

# Every architecture whose _materialize_weights routes tensors through QuantizedLinear, shared by
# ModelManager._load (build one or not) and estimate_ram_gb's callers (report the smaller number or
# not); a smaller estimate for an unwired architecture would mislead. `bert`/`nomic-bert` are
# excluded: encoder-only and prefill-shaped, while QuantizedLinear's fast path helps only decode.
# `nemotron_h`/`granitemoe`/llama's Mixtral branch were once excluded on a wrong theory that their
# tensors bypass _load_projection: nemotron_h's ssm_in/ssm_out are plain 2D weights (MLP is 47.7% of
# a 12B checkpoint's elements, SSM 38.3%), and a 3D per-expert MoE tensor is a contiguous 2D slice
# needing no new kernel (`QuantizedMoEExperts`, `app/architectures/moe_experts.py`, now wired for
# `granitemoe`, llama's Mixtral branch and `gemma4`'s MoE variant).
QUANTIZED_NATIVE_WIRED_ARCHITECTURES = frozenset(
    {
        "mistral3",
        "llama",
        "gemma4",
        "phi2",
        "granite",
        "granitemoe",
        "qwen2",
        "qwen3",
        "qwen35",
        "command-r",
        "starcoder2",
        "nemotron_h",
    }
)

# The 5 (of 7) per-layer names Mistral3TextArchitecture._load_projection routes through
# QuantizedLinear, kept in sync by hand. `attn_q`/`attn_k` are excluded here since most
# architectures load them on the float path (counting them packed would under-estimate need);
# mistral3/llama add them (+ `token_embd`) below, as they apply `unpermute_rope_rows`/keep the
# embedding packed.
_QUANTIZED_NATIVE_TENSOR_SUFFIXES = (
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)

# Per-architecture eligible suffixes (not one list): gemma4 needs no unpermute_rope_rows, so its q/k
# are included; llama/phi2 have their own untied lm_head, mistral3/gemma4 don't (tied to
# token_embd); phi2's q/k/v come from one fused `attn_qkv.weight`, never a standalone
# `attn_v.weight`. The broader list for everyone would under-estimate those that don't route q/k
# that way.
_PACKED_QK_AND_EMBEDDING = ("attn_q.weight", "attn_k.weight", "token_embd.weight")

# The 3D per-expert tensors `QuantizedMoEExperts` packs (`app/architectures/moe_experts.py`); the
# router (`ffn_gate_inp.weight`) is excluded: always a plain `.copy_`, never packed, so listing it
# would under-estimate memory for any architecture not packing it, the one thing this guard must
# never do.
_MOE_EXPERT_TENSOR_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")

_QUANTIZED_NATIVE_TENSOR_SUFFIXES_BY_ARCH: dict[str, tuple[str, ...]] = {
    "mistral3": (*_QUANTIZED_NATIVE_TENSOR_SUFFIXES, *_PACKED_QK_AND_EMBEDDING),
    # A Mixtral GGUF (llama_moe.py) adds the expert suffixes too: a no-op for a dense llama file,
    # whose tensors never have those names.
    "llama": (
        *_QUANTIZED_NATIVE_TENSOR_SUFFIXES,
        *_PACKED_QK_AND_EMBEDDING,
        *_MOE_EXPERT_TENSOR_SUFFIXES,
        "output.weight",
    ),
    "gemma4": (
        "attn_q.weight",
        "attn_k.weight",
        *_QUANTIZED_NATIVE_TENSOR_SUFFIXES,
        *_MOE_EXPERT_TENSOR_SUFFIXES,
        # Gap: unlike mistral3/llama (_PACKED_QK_AND_EMBEDDING), these two were never
        # wired to QuantizedEmbedding: always the full dequantized table. Both are pure embedding
        # lookups (no untied lm_head: token_embd is gemma4's tied output projection too; gemma4.py),
        # and per_layer_token_embd.weight, active on gemma-4-E2B-it, is ~50.5% of that checkpoint's
        # elements (gemma4_ple.py).
        "token_embd.weight",
        "per_layer_token_embd.weight",
    ),
    "phi2": ("attn_output.weight", "ffn_up.weight", "ffn_down.weight", "output.weight"),
    # granite/qwen2/qwen3/command-r follow llama's `_load_projection` calls
    # (attn_v/attn_output/ffn_gate/ffn_up/ffn_down, + untied lm_head), confirmed per architecture,
    # not from family resemblance (qwen2.py says why that is dangerous). None route attn_q/attn_k
    # through _load_projection, so those stay excluded.
    "granite": (*_QUANTIZED_NATIVE_TENSOR_SUFFIXES, "output.weight"),
    "qwen2": (*_QUANTIZED_NATIVE_TENSOR_SUFFIXES, "output.weight"),
    "qwen3": (*_QUANTIZED_NATIVE_TENSOR_SUFFIXES, "output.weight"),
    "command-r": (*_QUANTIZED_NATIVE_TENSOR_SUFFIXES, "output.weight"),
    # qwen35 routes every 2D projection through _load_projection (no RoPE row permutation):
    # full-attention q/k/v/o and the linear-attention layers' fused qkv/gate/alpha/beta/out. Norms,
    # conv1d and the 1D ssm_* tensors stay a plain `.copy_`.
    "qwen35": (
        *_QUANTIZED_NATIVE_TENSOR_SUFFIXES,
        "attn_q.weight",
        "attn_k.weight",
        "attn_qkv.weight",
        "attn_gate.weight",
        "ssm_out.weight",
        "ssm_alpha.weight",
        "ssm_beta.weight",
        "output.weight",
    ),
    # starcoder2's MLP is plain (non-gated): no ffn_gate, so its list omits that suffix instead of
    # reusing _QUANTIZED_NATIVE_TENSOR_SUFFIXES.
    "starcoder2": (
        "attn_v.weight",
        "attn_output.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "output.weight",
    ),
    # nemotron_h's attn_q/attn_k are a plain.copy_ (no RoPE, no permutation reason to use
    # _load_projection); token_embd stays a plain nn.Embedding (no QuantizedEmbedding yet).
    # ssm_in/ssm_out ARE plain 2D nn.Linear weights (99.9% of a checkpoint's SSM elements,
    # ) and go through _load_projection; other ssm_* tensors aren't 2D and bypass it
    # (their names match no suffix below).
    "nemotron_h": (
        "attn_v.weight",
        "attn_output.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "output.weight",
        "ssm_in.weight",
        "ssm_out.weight",
    ),
    # granitemoe has no ffn_gate/ffn_up/ffn_down (no shared/always-on expert; GraniteMoeFFN): only
    # attn_v/attn_output/the expert tensors/untied lm_head.
    "granitemoe": (
        "attn_v.weight",
        "attn_output.weight",
        *_MOE_EXPERT_TENSOR_SUFFIXES,
        "output.weight",
    ),
}


@functools.lru_cache(maxsize=1)
def cpu_accelerates_bf16() -> bool:
    """A CUDA GPU (Ampere+) has native bf16 tensor cores; a CPU needs dedicated bf16 SIMD
    (`avx512_bf16`/`amx_bf16`); without either PyTorch falls back to a slow emulated path. Measured
    on a CPU lacking both: the same matmul took ~5s in bf16 vs ~82ms in float32, ~60x slower."""
    if torch.cuda.is_available():
        return True
    try:
        with open("/proc/cpuinfo") as f:
            flags_line = next((line for line in f if line.startswith("flags")), "")
    except OSError:
        return False
    return "avx512_bf16" in flags_line or "amx_bf16" in flags_line


def available_memory_bytes() -> int | None:
    """`MemAvailable` (not `MemFree`): accounts for reclaimable cache, so not pessimistic for a busy
    system. None ("assume it fits") off Linux's /proc/meminfo, rather than blocking every install on
    a heuristic that can't run there."""
    try:
        with open("/proc/meminfo") as f:
            line = next((line for line in f if line.startswith("MemAvailable:")), "")
    except OSError:
        return None
    parts = line.split()
    return int(parts[1]) * 1024 if len(parts) >= 2 else None


# Headroom beyond the raw weights for activations, the KV cache and whatever is already resident (a
# second loaded model, matricxon's process, the system).
MEMORY_SAFETY_MARGIN = 1.2


def exact_bf16_bytes(tensor_infos: list[GGUFTensorInfo]) -> int:
    """The exact bytes this model needs with every tensor dequantized to bf16 (2 bytes/element):
    `sum(t.n_elements for t in tensor_infos) * 2`, from the GGUF header's shapes (a cheap
    header-only GGUFReader parse), not approximated from on-disk size.

    Replaces `estimated_bf16_bytes(on_disk_bytes)`, which assumed a flat 4 bits/weight whatever the
    quant type and overestimated a Q4_K_M file (K-quants carry per-superblock scale/min overhead),
    enough to refuse a chat ("need ~12.9GB... only 8.1GB available") for a model that had loaded on
    the same machine.
    """
    return sum(t.n_elements for t in tensor_infos) * 2


def estimate_quantized_native_bytes(
    tensor_infos: list[GGUFTensorInfo],
    enabled: bool,
    architecture_name: str = "",
    packable: Callable[[int], bool] | None = None,
) -> int:
    """`exact_bf16_bytes`'s estimate adjusted for quantized-native compute
    (`Settings.enable_quantized_native_compute`) when `enabled`; otherwise identical.

    Without it, `ModelManager._load`'s `ensure_enough_memory_to_load` would refuse loads
    quantized-native compute could handle on a tight machine (a Ministral-3B Q4_K_M needs ~7.5GB the
    old way, near its ~2.1GB on-disk size once its largest per-layer tensors,
    `_QUANTIZED_NATIVE_TENSOR_SUFFIXES_BY_ARCH`, are never fully dequantized).

    A tensor whose name ends in one of `architecture_name`'s eligible suffixes (empty for an unwired
    name: a safe no-op) and whose GGUF type has a fused kernel (`has_gemv_kernel`) counts at its
    on-disk size via `QuantStrategy.byte_length`. Every other tensor is sized as in
    `exact_bf16_bytes`, so a tensor that won't go through `QuantizedLinear` is never
    under-estimated.
    """
    if not enabled:
        return exact_bf16_bytes(tensor_infos)
    eligible_suffixes = _QUANTIZED_NATIVE_TENSOR_SUFFIXES_BY_ARCH.get(architecture_name, ())
    registry = QuantStrategyRegistry()
    total = 0
    for t in tensor_infos:
        # `packable`: which types stay packed (default the CPU fused kernels; a packed-GPU load
        # passes `TorchDequantizer.supports`).
        can_pack = packable or has_gemv_kernel
        is_quantized_native = t.name.endswith(eligible_suffixes) and can_pack(t.ggml_type)
        if is_quantized_native:
            total += registry.get(t.ggml_type).byte_length(t.n_elements)
        else:
            total += t.n_elements * 2
    return total


# Keyed by (path, mtime, size,...), not path, so a file replaced in place (a re-pull) invalidates
# automatically; one os.stat is cheap next to the GGUFReader parse it avoids. Added because GET
# /api/tags recomputes this per installed model on every request, which with several models
# installed, under GIL contention from a chat/model-load, intermittently exceeded pAIring's request
# timeout and surfaced as a bodyless "Service Unavailable" on an idle machine.
_ram_estimate_cache: dict[tuple[str, int, int, bool, str], float] = {}


def estimate_ram_gb(
    gguf_path: str | Path,
    quantized_native_enabled: bool = False,
    architecture_name: str = "",
) -> float:
    """The RAM `gguf_path` needs to load: `estimate_quantized_native_bytes` of its tensor shapes in
    GB (equal to `exact_bf16_bytes` unless `quantized_native_enabled`), from a cheap header-only
    parse (GGUFReader.read).

    It does not apply `Settings.memory_safety_margin` (unlike
    `app.models.memory_guard.ensure_enough_memory_to_load`, the load-time gate): the padded figure
    would read as "this architecture needs more RAM than it does", though the padding is this app's
    load-time cushion.

    Shared by `/api/show` (one tag) and `/api/tags` (every tag, `tags_router.py`), so "how much RAM
    does Matricxon need" has one answer. pAIring had shown a generic on-disk-size estimate well
    under the real need (a Ministral-3B Q4_K_M needs ~7.5GB, not ~3GB). `quantized_native_enabled`
    is passed by the caller (show_router.py/tags_router.py) from the live setting, so the number
    moves with the lower-RAM path.

    Cached per file (`_ram_estimate_cache`): the parse is the expensive part and a file's shapes
    don't change without the file.
    """
    path = Path(gguf_path)
    stat = path.stat()
    cache_key = (
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
        quantized_native_enabled,
        architecture_name,
    )
    cached = _ram_estimate_cache.get(cache_key)
    if cached is not None:
        return cached

    tensor_infos = GGUFReader(path).read().tensor_infos
    real_bytes = estimate_quantized_native_bytes(
        tensor_infos, quantized_native_enabled, architecture_name
    )
    result = round(real_bytes / 1e9, 2)
    _ram_estimate_cache[cache_key] = result
    return result


def group_bytes_by_layer(tensor_infos: list[GGUFTensorInfo]) -> dict[str, int]:
    """Groups GGUF tensors by their llama.cpp "blk.N." prefix (shared by mistral3/llama/gemma4),
    summing each group's float32 bytes (n_elements * 4, not bf16's 2: this feeds
    `plan_layer_dtypes`). Tensors with no blk.N. prefix (token_embd, output_norm, ...) sum under
    `NON_LAYER_GROUP`, not dropped: the tied embedding/lm_head can be among the largest, and
    excluding it would waste budget headroom."""
    by_group: dict[str, int] = {}
    for t in tensor_infos:
        match = _LAYER_PREFIX_RE.match(t.name)
        key = f"blk.{match.group(1)}" if match else NON_LAYER_GROUP
        by_group[key] = by_group.get(key, 0) + t.n_elements * 4
    return by_group


def plan_layer_dtypes(
    layer_bytes: dict[str, int], available_bytes: int | None, safety_margin: float
) -> dict[str, torch.dtype]:
    """Greedily assigns float32 to as many groups from `group_bytes_by_layer` as fit the
    safety-margined budget (float32 is the fast path on any CPU; bf16 without hardware bf16 is ~60x
    slower, `cpu_accelerates_bf16`), bf16 to the rest. `NON_LAYER_GROUP` (the tied
    embedding/lm_head, usually the largest) goes first, then `blk.N` in ascending N: a fixed order,
    since which layers end up float32 doesn't affect output quality, only speed.

    None `available_bytes` ("unknown", `available_memory_bytes`) assigns float32 to every group: the
    "assume it fits" policy of `select_load_dtype`.
    """
    if available_bytes is None:
        return dict.fromkeys(layer_bytes, torch.float32)
    budget = available_bytes / safety_margin
    ordered_keys = sorted(
        layer_bytes,
        key=lambda k: (k != NON_LAYER_GROUP, int(k.split(".")[1]) if k != NON_LAYER_GROUP else -1),
    )
    plan: dict[str, torch.dtype] = {}
    used = 0
    for key in ordered_keys:
        size = layer_bytes[key]
        if used + size <= budget:
            plan[key] = torch.float32
            used += size
        else:
            plan[key] = torch.bfloat16
    return plan


def select_load_dtype(bf16_bytes: int, safety_margin: float = MEMORY_SAFETY_MARGIN) -> torch.dtype:
    """bf16 halves memory vs float32, why ModelManager loaded everything in it on RAM-constrained,
    GPU-less targets. But on hardware that can't accelerate bf16 (`cpu_accelerates_bf16`) float32 is
    only faster, not smaller, and its 2x memory is worth paying only when it fits: a 3.4B-param
    model in float32 (~13.6GB) on a 15GB machine ran matmuls ~60x faster but drove the system into
    heavy swapping (14GB/15GB RAM, 10GB swap), worse than the slowness. So float32 is chosen only
    when it demonstrably fits in available memory; bf16 is the safe fallback.

    `bf16_bytes` is the `exact_bf16_bytes` figure for this load, passed in so this decision and
    `_ensure_enough_memory_to_load` agree on one number. `safety_margin` is the caller's (possibly
    admin-configured, Settings.memory_safety_margin) value, for the same reason.
    """
    if cpu_accelerates_bf16():
        return torch.bfloat16
    available = available_memory_bytes()
    if available is None:
        return torch.float32
    estimated_fp32_bytes = bf16_bytes * 2
    if estimated_fp32_bytes * safety_margin <= available:
        return torch.float32
    return torch.bfloat16


def plan_mixed_precision_load(
    tensor_infos: list[GGUFTensorInfo],
    dtype: torch.dtype,
    safety_margin: float,
    *,
    enabled: bool,
    supports_mixed_precision: bool,
) -> tuple[dict[str, torch.dtype] | None, torch.dtype]:
    """Decides whether `ModelManager._load` mixes per-layer float32/bf16 (mistral3 only: the caller
    checks `architecture_cls is Mistral3TextArchitecture` to avoid a circular import, since
    mistral3.py imports `NON_LAYER_GROUP` from here). Returns `(None, dtype)` ("load uniformly at
    `dtype`") unless all hold: `enabled` (`Settings.enable_mixed_precision_loading`, off by default:
    it measured slower, ~26s/token vs ~9.2s/token uniform bf16, because a float32 budget with no
    headroom for KV cache, activations or the mmap page cache pushes memory near the 15GB ceiling);
    `supports_mixed_precision`; `dtype == torch.bfloat16` (pointless when float32 fits whole); and
    the CPU can't accelerate bf16 (`cpu_accelerates_bf16` False, else uniform bf16 is fast).

    When it applies, the second return value is `torch.float32`, the dequant kernels' native dtype,
    so the loader's `.copy_` does a single downcast into each parameter's dtype, with no
    intermediate bf16 rounding for tensors planned float32.
    """
    wants_mixed_precision = (
        enabled
        and supports_mixed_precision
        and dtype == torch.bfloat16
        and not cpu_accelerates_bf16()
    )
    if not wants_mixed_precision:
        return None, dtype
    layer_bytes = group_bytes_by_layer(tensor_infos)
    layer_dtypes = plan_layer_dtypes(layer_bytes, available_memory_bytes(), safety_margin)
    return layer_dtypes, torch.float32
