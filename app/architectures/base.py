import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import torch
from torch import nn

from app.architectures.device_placement import DevicePlacement
from app.architectures.packed_weights import PackedWeightLoading
from app.architectures.rope import RotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class GenerationCancelledError(Exception):
    """Raised by a `_forward_impl` layer loop when `stop_check` reports a stop mid-forward-pass
    (see `forward`). A
    plain `Exception`, not a `MatricxonError`: it never reaches a client as an HTTP error;
    `ChatEngine.stream`
    catches it and ends generation cleanly, like an EOS token or a between-token stop.
    """


class ModelArchitecture(DevicePlacement, PackedWeightLoading, nn.Module, ABC):
    """Base class for a model built directly from a GGUF file's weights.

    Hyperparameters are read generically from GGUF's `<arch>.*` metadata (GGUFMetadata.arch_key), so
    a new architecture needs only a subclass.

    On-the-fly dequant: `from_gguf` defers copying weight data
    (`_defer_materialization`/`_ensure_materialized`) until the first forward pass, then caches it.
    See `forward`, `_ensure_materialized`, `close` and each subclass's `_materialize_weights`.
    """

    # The `general.architecture` string this class resolves (e.g. `"mistral3"`). Each `supports`
    # compares against
    # it; kept here as data so `GET /api/health` (app/routers/health_router.py) reads it directly.
    NAME: ClassVar[str]
    # True for an architecture with working Mixture-of-Experts support. Not every `NAME`-matching
    # checkpoint is MoE
    # (a `llama` GGUF is dense unless its metadata activates Mixtral's experts,
    # `llama_moe.detect_moe`), but the
    # class can handle one. Read by `GET /api/health` (`ArchitectureRegistry.moe_supported_names`).
    SUPPORTS_MOE: ClassVar[bool] = False
    # True for an architecture whose `_forward_impl` splices paired-mmproj image embeddings
    # (`ArchitectureRegistry.vision_supported_names`; `capabilities.py` derives "vision" from the
    # same set).
    SUPPORTS_VISION: ClassVar[bool] = False
    # True for an architecture whose forward pass gives the same logits for a decode step run in a
    # batch of
    # sequences (`app.runtime.batch_decode`, `batch_cache`) as alone; checked per architecture in
    # tests/unit/test_batched_decode.py before it is set.
    SUPPORTS_BATCHED_DECODE: ClassVar[bool] = False
    _last_logits_only: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._pending_loader: GGUFModelLoader | None = None
        self._materialized = False
        self._materialize_lock = threading.Lock()
        # Set by `_materialize_weights` (via `_mark_quantized_native_used`) when quantized-native
        # compute
        # (quantized_linear.QuantizedLinear) built at least one real `QuantizedLinear`. Those hold a
        # live reference to the
        # loader's raw mmap bytes for the model's *whole* lifetime, which changes when the mmap
        # closes (see `_ensure_materialized`).
        self._quantized_loader: GGUFModelLoader | None = None

    # True when every layer caches keys already rotated by a plain `RotaryEmbedding` (whole head),
    # so
    # `context_shift_rotations` is exact. Off for hybrid caches and partial/multi-axis rotary.
    supports_context_shift: ClassVar[bool] = False

    def _rope_for_layer(self, layer_idx: int) -> RotaryEmbedding:
        return self.rope

    def context_shift_rotations(self, count: int) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """One `(cos, sin)` per KV-cache layer that moves its cached keys `count` positions earlier
        (`KVCache.drop_range`), or None if this architecture can't shift its cache."""
        if not self.supports_context_shift:
            return None
        return [
            self._rope_for_layer(i).shift_table(count)
            for i in range(len(self.kv_cache_layer_shapes))
        ]

    def build_cache(self, max_seq_len: int, dtype: torch.dtype) -> object:
        """Builds the cache `ChatEngine.stream` passes to every `forward`. Default: a plain
        per-attention-layer `KVCache` from `kv_cache_layer_shapes`. Override only when the cache
        isn't one K/V pair per layer: `nemotron_h`'s hybrid Mamba-2/attention/MLP layers
        (`NemotronHHybridCache`, `app/runtime/mamba_cache.py`) keep a fixed-size SSM state plus a
        short conv1d history, so it defines no `kv_cache_layer_shapes` (a per-subclass convention)
        and overrides this."""
        return KVCache(
            layer_shapes=self.kv_cache_layer_shapes,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=self.device,
        )

    @classmethod
    @abstractmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        """Whether this class can build the model described by `metadata`."""

    @classmethod
    def unsupported_features(cls, metadata: GGUFMetadata) -> list[str]:
        """Per-checkpoint feature flags `metadata` activates that this class does NOT implement,
        though `supports` is True. Closes a gap: Gemma4Architecture claimed
        "completion" for a gemma-4-E2B-it GGUF whose Per-Layer Embeddings and cross-layer KV reuse
        it silently ignored. `supports` checks only the architecture name, but some architectures
        gate another code path behind metadata (as `llama`'s Mixtral MoE detection,
        `llama_moe.detect_moe`).

        Empty (default) means no known gated-variant gap, not "fully verified". Overridden only for
        a documented gap (`Gemma4Architecture`). Surfaced via
        `app.architectures.registry.unsupported_features` -> `effective_capabilities`'s
        `"architecture_features_unsupported"` flag, so pAIring's model list can warn before a
        request silently produces wrong output.
        """
        return []

    @classmethod
    @abstractmethod
    def from_gguf(cls, loader: GGUFModelLoader) -> "ModelArchitecture":
        """Builds the module graph (cheap: shapes/dtypes only, no tensor bytes read) and defers
        loading its weights via
        `loader` until first use (`_defer_materialization`).
        """

    @classmethod
    def _construct_without_init(cls, *args: object, **kwargs: object) -> "ModelArchitecture":
        """Builds `cls(*args, **kwargs)` without nn.Linear/nn.Embedding's random init: every weight
        is `.copy_`'d from the GGUF right after (each subclass's from_gguf), so it is wasted work.
        Measured on a 3.4B-param model: random init took over 90 s and never finished on a CPU
        without hardware bf16 (nn.init's fill kernels hit the same slow emulated path as matmul; see
        app.models.load_dtype.cpu_accelerates_bf16); construction on the "meta" device (shape/dtype
        only, so `reset_parameters` is a no-op) took under 4 s.

        `to_empty` allocates uninitialized storage and keeps no values: right for weights about to
        be copied in, wrong for anything __init__ computes (e.g. a rotary embedding's inv_freq; see
        _rebuild_derived_buffers). Callers with such a buffer MUST override that hook and call it
        after this returns, in from_gguf, since this classmethod can't know which args a subclass
        needs.
        """
        with torch.device("meta"):
            model = cls(*args, **kwargs)
        model.to_empty(device="cpu")
        return model

    def _rebuild_derived_buffers(self) -> None:
        """No-op by default; override if __init__ computes a buffer that from_gguf never
        `.copy_`'s from the GGUF
        (e.g. YarnRotaryEmbedding's inv_freq, derived from rope hyperparameters). After
        _construct_without_init that buffer
        is uninitialized garbage until this runs; from_gguf must call it right after construction.
        """

    @property
    def is_materialized(self) -> bool:
        """Whether weight data has been copied in yet: False between `from_gguf` and the first
        forward pass
        (`_ensure_materialized`). For testing the on-the-fly dequant; unused on the request path.
        """
        return self._materialized

    def _defer_materialization(self, loader: GGUFModelLoader) -> None:
        """Called once by `from_gguf` as its last step: hands this model the open `loader` it will
        later copy weights
        from. Its mmap must stay open until `_ensure_materialized` or `close` runs; `ModelManager`
        keeps it alive and
        calls `close` if this handle is evicted before any forward pass.
        """
        self._pending_loader = loader

    def _ensure_materialized(self, stop_check: Callable[[], bool] | None = None) -> None:
        """Copies every weight in from the pending loader on the first call, a no-op after
        (on-the-fly dequant; see the class docstring). Double-checked locking: the fast path skips
        the lock, as this runs at the front of every forward pass. Needed because
        `EmbeddingEngine`'s forward runs on a raw `asyncio.to_thread` thread, not a per-model
        `ModelWorker`, so two concurrent requests on a freshly loaded embedding model can both reach
        `forward` first.

        `stop_check` reaches `_materialize_weights`'s per-layer loop. Closes a gap: a stop
        during the one-time materialization (often 45s+ on target hardware) had no
        effect until it finished. If `_materialize_weights` raises `GenerationCancelledError`,
        `_pending_loader` stays open and `_materialized` False, so the next call re-materializes
        from scratch (wasting progress, never crashing: `.copy_` is idempotent).
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
            # INFO, not DEBUG: the one-time dequantization cost (often 45s+ on target hardware;
            # ROADMAP.md "Known issues")
            # had no named log stage and hid inside the caller's "prefill forward"/"model ready"
            # timing.
            logger.info("materializing weights (%s) - first use of this model", type(self).__name__)
            materialize_started = time.monotonic()
            with torch.no_grad():
                self._materialize_weights(self._pending_loader, stop_check)
            if self._device_packed:
                self.settle_on_device()
            logger.info("materialization done in %.2fs", time.monotonic() - materialize_started)
            # Quantized-native compute (`_mark_quantized_native_used`) needs the loader's mmap open
            # for the model's whole
            # lifetime, not just this step; `close` (from every ModelManager eviction path) closes
            # it then, below.
            if self._quantized_loader is None:
                self._pending_loader.close()
            self._pending_loader = None
            self._materialized = True

    @abstractmethod
    def _materialize_weights(
        self, loader: GGUFModelLoader, stop_check: Callable[[], bool] | None = None
    ) -> None:
        """Copies every weight this model needs in from `loader` via `.copy_` (parameters/buffers
        were allocated uninitialized by `_construct_without_init`); called lazily by
        `_ensure_materialized`.

        `stop_check`, if given, is checked between each decoder layer's complete tensor copies
        (never mid-layer, so no torn state).
        """

    def close(self) -> None:
        """Releases the GGUF mmap if this model was evicted before materializing
        (`_ensure_materialized`); a no-op once materialized (that closed it) or if `from_gguf` never
        ran (a test building a model directly). `ModelManager` calls it from every path that drops a
        handle.

        Also closes `_quantized_loader`: quantized-native compute (`_mark_quantized_native_used`)
        keeps it open past materialization for every `QuantizedLinear`'s forward; eviction releases
        it.

        `_pending_loader` and `_quantized_loader` are the same object when both are set: a stop
        mid-materialization can raise `GenerationCancelledError` after an
        earlier layer built a `QuantizedLinear` (setting `_quantized_loader`) but before
        `_pending_loader = None`. Closing via `_pending_loader` first crashed with the `BufferError`
        `QuantizedLinear.release` prevents (built ones' live memoryviews block it). So always
        release first, whichever attribute closes; the walk is a no-op if no `QuantizedLinear` was
        built.
        """
        with self._materialize_lock:
            loader = self._pending_loader or self._quantized_loader
            if loader is not None:
                self._release_packed_modules()
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
        last_logits_only: bool = False,
    ) -> torch.Tensor:
        """`last_logits_only`: the caller reads only the last position's logits (`ChatEngine`), so a
        decoder
        architecture honoring `_last_logits_only` runs lm_head on that one position (a prefill
        otherwise computes, then discards, a full-vocabulary row per prompt token). Embedding models
        ignore it.
        """
        self._ensure_materialized(stop_check)
        self._last_logits_only = last_logits_only
        if self.device.type == "cpu":
            return self._forward_impl(
                input_ids, kv_cache, position_ids, stop_check, image_embeddings
            )
        # On a GPU: inputs follow the model and the result returns to the CPU (the sampler,
        # embedding pooling and callers read it there). Vision encoders stay on the CPU.
        images = [(start, e.to(self.device)) for start, e in image_embeddings or []] or None
        out = self._forward_impl(
            input_ids.to(self.device),
            kv_cache,
            self.to_compute_device(position_ids),
            stop_check,
            images,
        )
        return out.cpu()

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

        `kv_cache`/`position_ids` are optional: omitting both runs a plain full-sequence pass (unit
        tests, the M3 oracle check); passing both enables incremental decoding against the caller's
        cache (ChatEngine).

        `image_embeddings` (LLaVA): each `(start, embeds)` pair overwrites `embeds.shape[0]`
        positions of the token-embedding lookup at `input_ids[, start:start+embeds.shape[0]]` with
        projected image-patch embeddings (`ClipVisionEncoder`/`app.runtime.vision_fusion`) before
        the decoder layers; those ids are placeholders (`vision_fusion.build_prompt_with_images`).
        Only `LlamaArchitecture` supports it (LLaVA's text half is llama-arch); others accept and
        ignore it, like `stop_check`.

        `stop_check`, if given, is called once between decoder layers (not mid-layer, so no partial
        layer state). Closes a gap: a stop (e.g. pAIring's "stop_model" after a
        mid-stream message delete) took effect only between tokens (`ModelWorker.request_stop`), so
        cancelling on slow hardware could wait out a whole forward pass (many seconds to minutes on
        a large prompt). `None` (unit tests, the M3 oracle script, `EmbeddingEngine`) never stops
        early. Implementations raise `GenerationCancelledError` as soon as `stop_check` is True; a
        raise can't be ignored like a return value, and `ChatEngine.stream` already wraps its
        forward calls in `try`/`except`.
        """
