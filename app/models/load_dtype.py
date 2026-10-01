import functools
import re
from pathlib import Path

import torch

from app.gguf.dequant.quantized_gemv_registry import has_gemv_kernel
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.gguf.reader import GGUFReader
from app.gguf.tensor_info import GGUFTensorInfo

# Tensors with no "blk.N." prefix (token_embd, output_norm, untied output.weight) - lets a caller
# size "everything outside the per-layer loop" the same way as any one layer. Not a real GGUF
# tensor name, just this module's own grouping key.
NON_LAYER_GROUP = "__non_layer__"

_LAYER_PREFIX_RE = re.compile(r"^blk\.(\d+)\.")

# Every architecture whose _materialize_weights actually routes tensors through QuantizedLinear
# today - shared by ModelManager._load (build one or not) and estimate_ram_gb's callers (report
# the smaller number or not); reporting a smaller estimate for an unwired architecture would be
# live-misleading. `bert`/`nomic-bert` excluded: encoder-only, always prefill-shaped, and
# QuantizedLinear's fast path only helps decode. `nemotron_h`/`granitemoe`/llama's own Mixtral
# branch *used* to be excluded on a "these real tensors bypass _load_projection" theory - wrong
# both times (2026-09-29): nemotron_h's ssm_in/ssm_out are plain 2D weights despite living in the
# "SSM block" (MLP is 47.7% of a real 12B checkpoint's elements, SSM only 38.3%); a real 3D
# per-expert MoE tensor is a contiguous 2D slice of the bigger one, needing no new kernel - see
# `QuantizedMoEExperts` (`app/architectures/moe_experts.py`), now wired for every real MoE
# architecture here (`granitemoe`, `llama`'s Mixtral branch, `gemma4`'s MoE variant).
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

# The 5 (of 7) real per-layer names Mistral3TextArchitecture._load_projection routes through
# QuantizedLinear - kept in sync by hand. `attn_q`/`attn_k` excluded here since most architectures
# still load them on the float path (counting them as packed would under-estimate real need);
# mistral3/llama add them (+ `token_embd`) below - those two apply `unpermute_rope_rows`/keep the
# embedding packed.
_QUANTIZED_NATIVE_TENSOR_SUFFIXES = (
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)

# Per-architecture eligible suffixes (not one shared list) - eligibility genuinely differs:
# gemma4 needs no unpermute_rope_rows, so its q/k projections are included too; llama/phi2 have
# their own untied lm_head, mistral3/gemma4 don't (tied to token_embd); phi2's q/k/v come from one
# fused `attn_qkv.weight`, never a standalone `attn_v.weight`. Reusing the broader list for
# everyone would under-estimate whichever ones don't actually route q/k that way.
_PACKED_QK_AND_EMBEDDING = ("attn_q.weight", "attn_k.weight", "token_embd.weight")

# Real per-expert 3D tensors `QuantizedMoEExperts` packs (see `app/architectures/moe_experts.py`)
# - the router (`ffn_gate_inp.weight`) is deliberately excluded: it's always a plain `.copy_()`,
# never packed, so listing it here would under-estimate real memory need for any architecture
# that doesn't get it packed - the one thing this guard must never do.
_MOE_EXPERT_TENSOR_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")

_QUANTIZED_NATIVE_TENSOR_SUFFIXES_BY_ARCH: dict[str, tuple[str, ...]] = {
    "mistral3": (*_QUANTIZED_NATIVE_TENSOR_SUFFIXES, *_PACKED_QK_AND_EMBEDDING),
    # A real Mixtral GGUF (see llama_moe.py) adds the expert suffixes too - harmless no-op for a
    # plain dense llama file, since those tensor names simply never exist there.
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
        # Real, confirmed gap (2026-09-30): unlike mistral3/llama (_PACKED_QK_AND_EMBEDDING),
        # these two were never wired to QuantizedEmbedding at all - always the full dequantized
        # table regardless of this setting. Both are pure embedding lookups (no untied lm_head
        # here - token_embd is gemma4's own tied output projection too, see gemma4.py's own
        # docstring), and per_layer_token_embd.weight is real, active on gemma-4-E2B-it, ~50.5%
        # of that checkpoint's own total elements (see gemma4_ple.py).
        "token_embd.weight",
        "per_layer_token_embd.weight",
    ),
    "phi2": ("attn_output.weight", "ffn_up.weight", "ffn_down.weight", "output.weight"),
    # granite/qwen2/qwen3/command-r follow llama's exact _load_projection call list (attn_v/
    # attn_output/ffn_gate/ffn_up/ffn_down, + untied lm_head) - confirmed per-architecture, not
    # assumed from family resemblance (see qwen2.py's own docstring for why that assumption is
    # dangerous). None of the four route attn_q/attn_k through _load_projection either, so those
    # stay excluded here too.
    "granite": (*_QUANTIZED_NATIVE_TENSOR_SUFFIXES, "output.weight"),
    "qwen2": (*_QUANTIZED_NATIVE_TENSOR_SUFFIXES, "output.weight"),
    "qwen3": (*_QUANTIZED_NATIVE_TENSOR_SUFFIXES, "output.weight"),
    "command-r": (*_QUANTIZED_NATIVE_TENSOR_SUFFIXES, "output.weight"),
    # qwen35 routes every 2D projection through _load_projection (no RoPE row permutation
    # anywhere): full-attention q/k/v/o and the linear-attention layers' fused qkv/gate/alpha/
    # beta/out. Norms, conv1d and the 1D ssm_* tensors stay plain `.copy_()`.
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
    # starcoder2's MLP is plain (non-gated) - no ffn_gate tensor exists, so its list omits that
    # suffix rather than reusing _QUANTIZED_NATIVE_TENSOR_SUFFIXES unchanged.
    "starcoder2": (
        "attn_v.weight",
        "attn_output.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "output.weight",
    ),
    # nemotron_h's attn_q/attn_k are always a plain .copy_() (no RoPE at all, no permutation
    # reason to route them through _load_projection); token_embd stays a plain nn.Embedding (no
    # QuantizedEmbedding wiring yet). ssm_in/ssm_out ARE plain 2D nn.Linear weights (99.9% of a
    # real checkpoint's own SSM element count, confirmed 2026-09-29) and go through
    # _load_projection too; every other real ssm_* tensor genuinely isn't 2D-shaped and bypasses
    # it, left out automatically (its name never matches any suffix below).
    "nemotron_h": (
        "attn_v.weight",
        "attn_output.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "output.weight",
        "ssm_in.weight",
        "ssm_out.weight",
    ),
    # granitemoe has no ffn_gate/ffn_up/ffn_down at all (no shared/always-on expert, see
    # GraniteMoeFFN's own docstring) - only attn_v/attn_output/the expert tensors/untied lm_head.
    "granitemoe": (
        "attn_v.weight",
        "attn_output.weight",
        *_MOE_EXPERT_TENSOR_SUFFIXES,
        "output.weight",
    ),
}


@functools.lru_cache(maxsize=1)
def cpu_accelerates_bf16() -> bool:
    """A CUDA GPU (Ampere+) has native bf16 tensor cores; a CPU needs its own dedicated bf16 SIMD
    (`avx512_bf16`/`amx_bf16`) - without either, PyTorch falls back to a slow emulated path.
    Measured live on a CPU lacking both: the exact same matmul took ~5s in bf16 versus ~82ms in
    float32 - about 60x slower, not a rounding error (see ROADMAP.md)."""
    if torch.cuda.is_available():
        return True
    try:
        with open("/proc/cpuinfo") as f:
            flags_line = next((line for line in f if line.startswith("flags")), "")
    except OSError:
        return False
    return "avx512_bf16" in flags_line or "amx_bf16" in flags_line


def available_memory_bytes() -> int | None:
    """`MemAvailable` (not `MemFree`) - accounts for reclaimable cache, so it isn't pessimistic
    about a system that's just been busy. None (meaning "assume it fits") on anything that isn't
    Linux's /proc/meminfo, rather than blocking every install without it on a heuristic that can't
    even run there."""
    try:
        with open("/proc/meminfo") as f:
            line = next((line for line in f if line.startswith("MemAvailable:")), "")
    except OSError:
        return None
    parts = line.split()
    return int(parts[1]) * 1024 if len(parts) >= 2 else None


# Headroom beyond the raw weights themselves for activations, the KV cache, and whatever else is
# already resident (a second loaded model, matricxon's own process, the rest of the system).
MEMORY_SAFETY_MARGIN = 1.2


def exact_bf16_bytes(tensor_infos: list[GGUFTensorInfo]) -> int:
    """The real, exact number of bytes this model's weights need once every tensor is dequantized
    to bf16 (2 bytes/element) - `sum(t.n_elements for t in tensor_infos) * 2`, computed directly
    from the GGUF file's own header (shape per tensor, read by GGUFReader - a cheap, KB-scale parse
    that never touches the actual weight bytes, see that class's own docstring), not approximated
    from the file's on-disk size.

    Replaces an earlier version of this function (estimated_bf16_bytes(on_disk_bytes)) that assumed
    a flat worst-case 4-bits/weight regardless of the file's real quant type - confirmed live
    (2026-09-20) to overestimate a real Q4_K_M file: K-quants carry real per-superblock scale/min
    overhead on top of their nominal bit width, so a plain "on-disk bytes * 4" guess runs *higher*
    than this exact figure, not lower - the gap was large enough to refuse a real chat request
    ("need ~12.9GB ... only 8.1GB available") for a model that had already been confirmed, on the
    same machine, to load and answer successfully. This function has no failure mode the old one
    didn't already share (both need the file to exist and be readable) and is never less accurate,
    only ever more."""
    return sum(t.n_elements for t in tensor_infos) * 2


def estimate_quantized_native_bytes(
    tensor_infos: list[GGUFTensorInfo], enabled: bool, architecture_name: str = ""
) -> int:
    """`exact_bf16_bytes`'s own real-memory-need estimate, adjusted for real quantized-native
    compute (see `Settings.enable_quantized_native_compute`'s own docstring) when `enabled` -
    otherwise identical to `exact_bf16_bytes`, so this is a strict extension, not a second
    estimate that could disagree with it by accident.

    Without this, `ModelManager._load`'s own `ensure_enough_memory_to_load` circuit breaker would
    keep refusing a load that quantized-native compute could now actually handle on a genuinely
    tight machine - the real problem this whole feature exists to fix (see the plan behind this
    work: a real Ministral-3B Q4_K_M file needs ~7.5GB the old way, much closer to its own
    ~2.1GB on-disk size once its largest per-layer tensors - see
    `_QUANTIZED_NATIVE_TENSOR_SUFFIXES_BY_ARCH` - never get fully dequantized at all).

    For a tensor whose real name ends in one of `architecture_name`'s own eligible suffixes (see
    `_QUANTIZED_NATIVE_TENSOR_SUFFIXES_BY_ARCH` - empty for an unrecognized/unwired name, so this
    is always a safe, conservative no-op for anything not explicitly listed there) and whose real
    GGUF type has a fused kernel (`has_gemv_kernel`), counts its real on-disk (still quantized)
    byte size instead of the bf16-equivalent one - `QuantStrategy.byte_length` (the same real,
    exact figure `GGUFModelLoader.load_tensor` already uses) rather than a second, possibly-
    drifting approximation. Every other tensor is sized exactly as `exact_bf16_bytes` already
    would - this function must never *under*-estimate a tensor that won't actually go through
    `QuantizedLinear` for this specific architecture.
    """
    if not enabled:
        return exact_bf16_bytes(tensor_infos)
    eligible_suffixes = _QUANTIZED_NATIVE_TENSOR_SUFFIXES_BY_ARCH.get(architecture_name, ())
    registry = QuantStrategyRegistry()
    total = 0
    for t in tensor_infos:
        is_quantized_native = t.name.endswith(eligible_suffixes) and has_gemv_kernel(t.ggml_type)
        if is_quantized_native:
            total += registry.get(t.ggml_type).byte_length(t.n_elements)
        else:
            total += t.n_elements * 2
    return total


# Keyed by (path, mtime, size, ...) rather than just path so a file replaced in place (a re-pull
# landing at the same path) invalidates automatically instead of serving a stale answer forever -
# cheap to check (a single os.stat) next to the real cost this avoids, GGUFReader parsing the
# tensor_infos section fresh on every call. Added because GET /api/tags recomputes this for every
# installed model on every single request with no caching at all (confirmed live, 2026-09-27: with
# several models installed this made /api/tags slow enough, under GIL contention from an in-flight
# chat/model-load on the same process, to intermittently blow past pAIring's own request timeout
# and surface as a bodyless "Service Unavailable" even though the machine itself was idle).
_ram_estimate_cache: dict[tuple[str, int, int, bool, str], float] = {}


def estimate_ram_gb(
    gguf_path: str | Path,
    quantized_native_enabled: bool = False,
    architecture_name: str = "",
) -> float:
    """The real, bare RAM `gguf_path` needs to load - `estimate_quantized_native_bytes` of its real
    tensor shapes, in GB (see that function's own docstring - identical to `exact_bf16_bytes`
    unless `quantized_native_enabled`). A cheap, header-only parse (see `GGUFReader.read`'s own
    docstring, and `exact_bf16_bytes`'s own docstring for why this is exact, not an
    on-disk-size approximation).

    Deliberately does *not* apply `Settings.memory_safety_margin` (unlike
    `app.models.memory_guard.ensure_enough_memory_to_load`'s own, separate real load-time gate,
    which still does, via its own `safety_margin` param - see manager.py's own call to
    `estimate_quantized_native_bytes` for that) - confirmed live, 2026-09-30: showing the padded
    figure here read as "this architecture needs more RAM than it really does," when the padding
    is this app's own conservative load-time cushion, not a real per-tensor requirement.

    Shared by `/api/show` (one tag) and `/api/tags` (every installed tag at once, see
    `tags_router.py`) so a caller asking "how much RAM does Matricxon really need for this" always
    gets the same, consistently-computed answer regardless of which endpoint it asked - real,
    confirmed gap this closes (2026-09-21): a caller (pAIring) was showing a generic, engine-
    agnostic on-disk-size-based estimate that ran meaningfully under Matricxon's real requirement
    for the same real file (confirmed live: a real Ministral-3B Q4_K_M file needs ~7.5GB on
    Matricxon, not the ~3GB an Ollama-shaped estimate suggested).

    `quantized_native_enabled` closes a second, later real gap (2026-09-21): this used to always
    call `exact_bf16_bytes` regardless of `Settings.enable_quantized_native_compute`, so the
    displayed number never moved even once that real, lower-RAM path existed and was turned on -
    the caller (see show_router.py/tags_router.py) passes the live setting through.

    Cached per-file (see `_ram_estimate_cache` above) - the GGUFReader parse this does is the
    expensive part, and a file's own tensor shapes never change without the file itself
    changing."""
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
    """Groups GGUF tensors by their llama.cpp-convention "blk.N." prefix (shared by
    mistral3/llama/gemma4's own tensor names - see GGUFTensorInfo.name), summing each group's
    float32-dequantized byte count (n_elements * 4, not bf16's 2 - this feeds
    `plan_layer_dtypes`'s float32-vs-bf16 decision, so it needs float32's real footprint, not
    bf16's). Tensors with no blk.N. prefix (token_embd, output_norm, etc.) are summed under
    `NON_LAYER_GROUP` instead of being dropped - on a real model that tensor alone (the tied
    embedding/lm_head) can be one of the single largest, so excluding it from a caller's budget
    planning would waste real, meaningful headroom."""
    by_group: dict[str, int] = {}
    for t in tensor_infos:
        match = _LAYER_PREFIX_RE.match(t.name)
        key = f"blk.{match.group(1)}" if match else NON_LAYER_GROUP
        by_group[key] = by_group.get(key, 0) + t.n_elements * 4
    return by_group


def plan_layer_dtypes(
    layer_bytes: dict[str, int], available_bytes: int | None, safety_margin: float
) -> dict[str, torch.dtype]:
    """Greedily assigns float32 to as many groups from `group_bytes_by_layer` as fit within the
    safety-margined budget - float32 is the fast path on any CPU, bf16 without hardware bf16
    support is matricxon's ~60x-slower fallback (see `cpu_accelerates_bf16`) - bf16 for
    whatever's left. `NON_LAYER_GROUP` (the tied embedding/lm_head, usually the single largest
    individual tensor) is tried first, then `blk.N` groups in ascending N order - a fixed,
    deterministic order, not a size-based one, since which specific decoder layers end up float32
    doesn't affect output quality at all (bf16 is already used model-wide today; a layer that
    stays bf16 here is no less correct than every layer being bf16 is today) - only aggregate
    speed, which a fixed order already captures.

    None `available_bytes` (meaning "unknown", see `available_memory_bytes`) assigns float32 to
    every group - same "assume it fits" policy `select_load_dtype` already uses for the same
    input, rather than a second, differently-shaped way of saying the same thing.
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
    """bf16 halves memory versus float32 - the whole reason ModelManager loaded everything in it
    unconditionally before this (see ModelManager's own docstring), on this project's target of
    RAM-constrained, GPU-less machines. But float32 is only slower, not smaller-or-faster, on
    hardware that can't accelerate bf16 (see `cpu_accelerates_bf16`) - and even then, float32's
    2x memory cost is only worth paying when it actually fits: loading a 3.4B-param model in
    float32 (~13.6GB) on a 15GB machine measured as ~60x *faster* per matmul but drove the system
    into heavy swapping (14GB/15GB RAM, 10GB swap) instead, which is worse than the slowness it was
    meant to fix. So float32 is only chosen when it demonstrably fits in currently available
    memory; bf16 (slower, but the smallest footprint matricxon can currently load in - see M10's
    on-the-fly dequant stretch goal) is the safe fallback otherwise.

    `bf16_bytes` is the exact figure exact_bf16_bytes already computed for the same load - passed
    in rather than recomputed here so both this decision and _ensure_enough_memory_to_load's own
    circuit breaker always agree on the same real number for one load. `safety_margin` is likewise
    the caller's own (possibly admin-configured, see Settings.memory_safety_margin) value, not the
    module default, for the same reason.
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
    """Decides whether `ModelManager._load` should mix per-layer float32/bf16 (mistral3 only -
    `supports_mixed_precision` is `architecture_cls is Mistral3TextArchitecture`, checked by the
    caller rather than here to avoid a circular import: mistral3.py already imports
    `NON_LAYER_GROUP` from this module). Returns `(None, dtype)` - "load uniformly at `dtype`",
    today's plain behavior - unless every one of these holds: `enabled` (see
    `Settings.enable_mixed_precision_loading`'s own docstring - **off by default**, since real
    end-to-end verification, 2026-09-20, found this regresses generation speed rather than
    improving it on this project's target hardware: ~26s/token measured versus a ~9.2s/token
    uniform-bf16 control run under matched conditions, not swap contamination - the control,
    run back-to-back on the same degraded-swap machine state, reproduced the documented baseline
    almost exactly. Budgeting float32 layers against `available_memory_bytes()` alone, with no
    headroom reserved for the KV cache, activations, or the source GGUF's own mmap'd page cache,
    pushes this machine's resident footprint close enough to its real 15GB ceiling that
    memory-pressure overhead dwarfs the intended per-layer compute win. Kept, not deleted, since a
    properly-headroomed budget might still yield a real win - that redesign is real, needed
    follow-up work, not done here); `supports_mixed_precision`; `dtype == torch.bfloat16`
    (mixing is pointless when float32 already fits the whole model - `select_load_dtype` already
    picked it); and this CPU can't accelerate bf16 in hardware (`cpu_accelerates_bf16()` False -
    otherwise uniform bf16 is already the fast, correct choice).

    When it does apply, the second return value becomes `torch.float32` - the dequant kernels'
    own native output dtype (see `QuantStrategy.dequantize`'s docstring) - so the loader's own
    `.copy_()` does the one real downcast straight into each parameter's actual (possibly bf16)
    dtype, rather than an unnecessary intermediate bf16 rounding step for every tensor this plan
    assigns float32.
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
