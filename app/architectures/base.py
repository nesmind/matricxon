import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import torch
from torch import nn

from app.architectures.quantized_linear import QuantizedLinear
from app.gguf.dequant.quantized_gemv_registry import has_gemv_kernel
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class GenerationCancelledError(Exception):
    """Raised by a `_forward_impl`'s layer loop when `stop_check()` reports a stop mid-forward-pass
    (see `forward`'s own docstring). A plain `Exception`, not a `MatricxonError` - this must never
    reach a client as an HTTP error response; `ChatEngine.stream` catches it internally and ends
    generation cleanly, the same way an EOS token or a between-token stop already does.
    """


class ModelArchitecture(nn.Module, ABC):
    """Base class for a model built directly from a GGUF file's weights.

    Hyperparameters are read generically off GGUF's `<arch>.*` metadata
    namespace (see GGUFMetadata.arch_key) rather than hardcoded, so a new
    architecture only needs a new subclass, not changes here.

    M10's on-the-fly dequant: `from_gguf` no longer copies real weight data
    in eagerly - it defers that (see `_defer_materialization`/
    `_ensure_materialized`) until this model's first real forward pass,
    caching the result for every call after that. See this class's
    `forward`/`_ensure_materialized`/`close` and each subclass's own
    `_materialize_weights` for the mechanics.
    """

    #: The real `general.architecture` string this class resolves (e.g.
    #: `"mistral3"`) - each subclass's own `supports()` already compares
    #: against this same string, so it lives here once as real *data*
    #: (`GET /api/health` reads it directly, see app/routers/health_router.py)
    #: rather than being re-parsed back out of each `supports()` method's
    #: comparison logic.
    NAME: ClassVar[str]

    def __init__(self) -> None:
        super().__init__()
        self._pending_loader: GGUFModelLoader | None = None
        self._materialized = False
        self._materialize_lock = threading.Lock()
        # Set by `_materialize_weights` (via `_mark_quantized_native_used`) only when quantized-
        # native compute (see app.architectures.quantized_linear.QuantizedLinear) actually built
        # at least one real `QuantizedLinear` for this load - unlike every other tensor, those
        # hold a live reference to this loader's own raw (still-quantized) mmap bytes for the
        # model's *entire* lifetime, not just through materialization - see
        # `_ensure_materialized`'s own docstring for why this changes when the mmap gets closed.
        self._quantized_loader: GGUFModelLoader | None = None

    def _mark_quantized_native_used(self, loader: GGUFModelLoader) -> None:
        """Called once by `_materialize_weights` (idempotent - a real per-layer loop calls this
        once per quantized projection, not once total) the first time it builds a real
        `QuantizedLinear` from `loader` - tells `_ensure_materialized` to keep `loader`'s mmap
        open past materialization instead of closing it, since a `QuantizedLinear`'s own forward
        pass reads real raw bytes from it on every future call, not just this one."""
        self._quantized_loader = loader

    def _load_projection(
        self,
        loader: GGUFModelLoader,
        tensor_name: str,
        target: nn.Linear,
        dtype: torch.dtype,
        enabled: bool,
        bias_tensor_name: str | None = None,
    ) -> nn.Module:
        """Shared by every architecture's `_materialize_weights` for a real per-layer projection
        eligible for quantized-native compute (see `Settings.enable_quantized_native_compute`'s
        own docstring) - moved here once `Mistral3TextArchitecture` (the reference implementation)
        and later `llama`/`gemma4`/`phi2` all needed the identical real decision, rather than four
        separate copies of it.

        When `enabled` and `tensor_name`'s real GGUF type has a fused kernel
        (`quantized_gemv_registry.has_gemv_kernel`), returns a real `QuantizedLinear` built
        straight from this loader's raw bytes - `target` (the placeholder `nn.Linear` built in
        `__init__`) is discarded, never touched. Otherwise (disabled, or a real type with no fused
        kernel - e.g. an F16/F32 file) falls straight through to today's exact `.copy_()` path,
        returning `target` unchanged. `bias_tensor_name`, when given, loads a real bias either way
        (a `QuantizedLinear`'s own bias is always a real, small, fully-dequantized tensor - see
        that class's own docstring for why only the main weight stays packed).
        """
        if enabled:
            raw, ggml_type, shape = loader.raw_tensor_bytes_and_type(tensor_name)
            if has_gemv_kernel(ggml_type):
                self._mark_quantized_native_used(loader)
                out_features, in_features = shape
                bias = loader.load_tensor(bias_tensor_name) if bias_tensor_name else None
                return QuantizedLinear(
                    out_features, in_features, ggml_type, raw, bias=bias, dtype=dtype
                )
        target.weight.copy_(loader.load_tensor(tensor_name))
        if bias_tensor_name is not None:
            target.bias.copy_(loader.load_tensor(bias_tensor_name))
        return target

    @classmethod
    @abstractmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        """Whether this class can build the model described by `metadata`."""

    @classmethod
    @abstractmethod
    def from_gguf(cls, loader: GGUFModelLoader) -> "ModelArchitecture":
        """Builds the module graph (cheap: shapes/dtypes only, no tensor

        bytes read yet) and defers loading its real weights via `loader`
        until first use - see `_defer_materialization`.
        """

    @classmethod
    def _construct_without_init(cls, *args: object, **kwargs: object) -> "ModelArchitecture":
        """Builds `cls(*args, **kwargs)` without running any of nn.Linear/nn.Embedding's default
        random weight initialization - every real weight gets `.copy_()`'d in from the GGUF file
        immediately after construction (see each subclass's own from_gguf), so that default init
        is pure wasted work. Not a rounding error either: measured live building a real 3.4B-param
        model, plain construction (random-initializing every parameter) took over 90 seconds and
        never finished on a CPU without hardware bf16 support (nn.init's fill kernels hit the same
        slow emulated path as matmul does - see app.models.load_dtype.cpu_accelerates_bf16's own
        docstring); this construction-on-the-"meta"-device pattern (meta tensors carry shape/dtype
        only, no actual data, so submodules' own reset_parameters() calls become instant no-ops)
        completed the identical model in well under 4 seconds.

        `to_empty()` only allocates fresh (uninitialized) real storage for every parameter/buffer -
        it does not preserve values, which is exactly right for a weight that's about to be
        `.copy_()`'d in from the GGUF file, but silently wrong for anything __init__ computes a
        real value for on its own (e.g. a rotary embedding's derived inv_freq - see
        _rebuild_derived_buffers's own docstring). Callers with any such buffer MUST override that
        hook and call it after this returns; from_gguf is where that call belongs, not here, since
        this classmethod has no way to know which of `*args`/`**kwargs` a subclass would need to
        recompute them.
        """
        with torch.device("meta"):
            model = cls(*args, **kwargs)
        model.to_empty(device="cpu")
        return model

    def _rebuild_derived_buffers(self) -> None:
        """No-op by default - override only if __init__ computes a real buffer value that from_gguf
        never separately `.copy_()`'s in from the GGUF file itself (e.g. YarnRotaryEmbedding's
        inv_freq, derived from rope hyperparameters, not a GGUF tensor). A model built via
        _construct_without_init has that buffer as uninitialized garbage, not the real computed
        value, until this runs - from_gguf must call it explicitly right after construction."""

    @property
    def is_materialized(self) -> bool:
        """Whether real weight data has actually been copied in yet - False

        between `from_gguf` returning and this model's first real forward
        pass (see `_ensure_materialized`). Exposed mainly for testing the
        on-the-fly dequant mechanism itself; not used anywhere in the
        request-handling path.
        """
        return self._materialized

    def _defer_materialization(self, loader: GGUFModelLoader) -> None:
        """Called once by `from_gguf`, as its last step: hands this model the still-open `loader`

        it'll need to actually copy real weight data in later, without doing that copy now. The
        mmap behind `loader` has to stay open until either `_ensure_materialized` or `close` runs -
        `ModelManager` is responsible for keeping it alive that long (no `with` block around
        construction anymore) and for calling `close()` if this handle gets evicted/unloaded before
        ever running a real forward pass.
        """
        self._pending_loader = loader

    def _ensure_materialized(self, stop_check: Callable[[], bool] | None = None) -> None:
        """Copies every real weight in from the pending loader on the first call, a no-op on every

        call after that - the actual on-the-fly dequant mechanism (see this class's own docstring).
        Double-checked locking: the fast path (already materialized) never touches the lock, which
        matters since this runs on the front of every single forward pass, not just the first -
        only the first caller (and anyone racing it) pays the lock's cost. Real, not theoretical:
        `EmbeddingEngine`'s forward calls run on a raw `asyncio.to_thread` pool thread, not a
        dedicated per-model `ModelWorker` (see that class's own docstring on why `/api/chat` is
        different), so two concurrent requests against the same freshly-loaded embedding model
        genuinely can call `forward()` for the first time at once.

        `stop_check`, when given, reaches `_materialize_weights`'s own per-layer loop the same way
        `_forward_impl`'s does (see that method's own docstring) - real gap this closes
        (2026-09-21, confirmed live): a stop request arriving *during* the one-time
        materialization step (often 45s+ on this project's target hardware) previously had no way
        to take effect at all until materialization finished completely, since `_forward_impl` -
        the only place a stop was ever checked - hadn't even started running yet. If
        `_materialize_weights` raises `GenerationCancelledError` partway through, everything below
        that call deliberately does not run: `_pending_loader` stays open and `_materialized` stays
        False, so this model is left in a clean, retryable state - the next real call just
        re-materializes from scratch (wasting whatever partial progress was made, but never
        crashing on a half-loaded model - `.copy_()` is idempotent, redoing an already-copied
        layer produces the same correct result).
        """
        if self._materialized:
            return
        with self._materialize_lock:
            if self._materialized:
                return
            if self._pending_loader is None:
                raise RuntimeError(
                    f"{type(self).__name__} was evicted/closed before its weights were ever "
                    "materialized - this handle can no longer be used."
                )
            # INFO, not DEBUG: this is the real one-time dequantization cost (often 45s+ on this
            # project's target hardware - see ROADMAP.md's "Known issues") that previously had no
            # named stage in the logs at all - it silently ran inside whatever the caller's own
            # "prefill forward"/"model ready" timing already covered, invisible as its own step.
            logger.info("materializing weights (%s) - first use of this model", type(self).__name__)
            materialize_started = time.monotonic()
            with torch.no_grad():
                self._materialize_weights(self._pending_loader, stop_check)
            logger.info("materialization done in %.2fs", time.monotonic() - materialize_started)
            # Real quantized-native compute (see `_mark_quantized_native_used`'s own docstring)
            # needs this loader's mmap to stay open for the model's entire lifetime, not just
            # through this one materialization step - `close()` (called from every real eviction
            # path in ModelManager) is what closes it in that case instead, below.
            if self._quantized_loader is None:
                self._pending_loader.close()
            self._pending_loader = None
            self._materialized = True

    @abstractmethod
    def _materialize_weights(
        self, loader: GGUFModelLoader, stop_check: Callable[[], bool] | None = None
    ) -> None:
        """Copies every real weight this model needs in from `loader`, via `.copy_()` (this model's

        own parameters/buffers were allocated - uninitialized - by `_construct_without_init`).
        Exactly what `from_gguf` used to do eagerly before M10's on-the-fly dequant; now called
        once, lazily, by `_ensure_materialized`.

        `stop_check`, when given, is checked once between each real decoder layer's *complete*
        group of tensor copies (never mid-layer, so a layer's own weights are always either fully
        loaded or not started - no torn state) - see `_ensure_materialized`'s own docstring for
        what happens when it fires.
        """

    def close(self) -> None:
        """Releases the GGUF file's mmap if this model was evicted/unloaded before its weights

        were ever actually materialized (see `_ensure_materialized`) - a no-op once materialized
        (that already closed it) or if `from_gguf` was never called at all (e.g. a test building a
        model directly). `ModelManager` calls this from every path that drops a handle.

        Also closes `_quantized_loader` - real quantized-native compute (see
        `_mark_quantized_native_used`) keeps that one open *past* materialization, for every real
        `QuantizedLinear`'s own forward pass to keep reading from for the model's whole lifetime -
        this is the real eviction path that finally releases it, not `_ensure_materialized`.

        `_pending_loader` and `_quantized_loader` are always the *same* real loader object when
        both are set - confirmed live (2026-09-21): a stop mid-materialization (see
        `_materialize_weights`'s own per-layer `stop_check`) can raise `GenerationCancelledError`
        *after* some earlier layer already built a real `QuantizedLinear` (setting
        `_quantized_loader`) but *before* `_ensure_materialized` ever reaches its own
        `_pending_loader = None` line - leaving both non-None at once. Closing via `_pending_loader`
        first, unconditionally, crashed with the exact real `BufferError` `QuantizedLinear.release`
        exists to prevent - every already-built one's live memoryview was still blocking it. Fixed
        by always releasing first, regardless of which attribute the caller happens to be closing
        through - the release walk is a real no-op when no `QuantizedLinear` was ever built.
        """
        with self._materialize_lock:
            loader = self._pending_loader or self._quantized_loader
            if loader is not None:
                for module in self.modules():
                    if isinstance(module, QuantizedLinear):
                        module.release()
                loader.close()
            self._pending_loader = None
            self._quantized_loader = None

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        position_ids: torch.Tensor | None = None,
        stop_check: Callable[[], bool] | None = None,
        image_embeddings: list[tuple[int, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        self._ensure_materialized(stop_check)
        return self._forward_impl(input_ids, kv_cache, position_ids, stop_check, image_embeddings)

    @abstractmethod
    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        position_ids: torch.Tensor | None = None,
        stop_check: Callable[[], bool] | None = None,
        image_embeddings: list[tuple[int, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """input_ids: (batch, seq_len) -> logits: (batch, seq_len, vocab_size).

        `kv_cache`/`position_ids` are optional: omitting both runs a plain
        full-sequence forward pass (unit tests, the M3 oracle check); passing
        both enables incremental one-token-at-a-time decoding against a
        cache built by the caller (see ChatEngine).

        `image_embeddings` (2026-09-21, real LLaVA vision support): each `(start, embeds)` pair
        overwrites `embeds.shape[0]` consecutive positions of the token-embedding lookup at
        `input_ids[:, start:start+embeds.shape[0]]` with real projected image-patch embeddings
        (see `ClipVisionEncoder`/`app.runtime.vision_fusion`), before those positions ever enter
        the decoder layers - those input ids are placeholder values only (see
        `vision_fusion.build_prompt_with_images`), never meant to be looked up for real. Real
        support is `LlamaArchitecture`-only (LLaVA's own text half is llama-arch) - every other
        architecture accepts and ignores this param (`None` is always what they receive in
        practice today), matching this file's own `stop_check` precedent.

        `stop_check`, when given, is called once between each real decoder-layer iteration (not
        mid-layer - a layer's own matmuls always run to completion, so no torn/partial layer state
        is ever possible) - a real user-visible gap this closes (2026-09-21, confirmed live): a
        stop request (e.g. pAIring's "stop_model" unload call after a mid-stream message delete)
        previously only took effect *between tokens* (see `ModelWorker.request_stop`'s own
        docstring), so cancelling mid-layer-loop on this project's slow hardware could still mean
        waiting out an entire in-progress forward pass - many seconds to minutes on a large prompt
        - before the stop had any visible effect. `None` (the default - every non-`ChatEngine`
        caller: unit tests, the M3 oracle script, `EmbeddingEngine`) means "never stop early,"
        identical to today's behavior. Implementations raise `GenerationCancelledError` the moment
        `stop_check()` returns True, rather than returning a sentinel - a mid-stack `raise` can't be
        silently ignored by a caller that forgets to check a return value, and `ChatEngine.stream`
        already needs a `try`/`except` around its own forward calls to end the generator cleanly.
        """
