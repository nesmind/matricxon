import time
from collections.abc import Callable

from app.architectures.mistral3 import Mistral3TextArchitecture
from app.architectures.registry import ArchitectureRegistry
from app.gguf.loader import GGUFModelLoader
from app.gguf.reader import GGUFReader
from app.models.catalog import ModelCatalog
from app.models.handle import ModelHandle
from app.models.load_dtype import (
    MEMORY_SAFETY_MARGIN,
    QUANTIZED_NATIVE_WIRED_ARCHITECTURES,
    estimate_quantized_native_bytes,
    exact_bf16_bytes,
    plan_mixed_precision_load,
    select_load_dtype,
)
from app.models.memory_guard import ensure_enough_memory_to_load
from app.models.tokenizer_dispatch import build_tokenizer
from app.models.worker import ModelWorker


class ModelManager:
    """Loads GGUF models on demand and keeps up to `max_loaded` resident,
    evicting the least-recently-used one past its keep-alive.

    `max_loaded=2` (not 1) matches pAIring's real usage: its RAG path calls
    `embed()` then `chat_stream()` back-to-back on every turn, so 1 would
    thrash-reload a model on every single turn (see ROADMAP.md).

    Loads models in bf16 when the hardware can actually accelerate it (CUDA, or a CPU with
    `avx512_bf16`/`amx_bf16`), otherwise float32 if that demonstrably fits in currently available
    memory, else bf16 anyway - see `app.models.load_dtype.select_load_dtype`'s own docstring for
    the full reasoning:
    bf16 halves memory (e.g. ~6.9GB versus ~13.7GB for a real 3.85B-param model, see the
    hardware-constraints project memory) but is dramatically *slower* than float32 on a CPU
    without native bf16 support - and float32 being faster doesn't help if it swap-thrashes the
    machine instead.

    `max_loaded` itself has no upper bound baked in - the eviction/capacity
    logic below is written generically against `self._max_loaded`, not
    hardcoded to 2 - but raising it on a RAM-constrained box needs the real
    memory circuit breaker `_ensure_enough_memory_to_load` provides (see its
    own docstring): a load that clearly wouldn't fit now fails closed with a
    clear `InsufficientMemoryError` instead of risking an OS-level OOM kill.

    `clock` is injectable so eviction/expiry logic (and every `ModelHandle`
    it creates or touches) can be unit-tested deterministically with a fake
    clock instead of real sleeps.
    """

    def __init__(
        self,
        catalog: ModelCatalog,
        max_loaded: int = 2,
        default_keep_alive_seconds: int = 300,
        memory_safety_margin: float = MEMORY_SAFETY_MARGIN,
        enable_mixed_precision_loading: bool = False,
        enable_quantized_native_compute: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._catalog = catalog
        self._max_loaded = max_loaded
        self._default_keep_alive_seconds = default_keep_alive_seconds
        self._memory_safety_margin = memory_safety_margin
        # Off by default - see Settings.enable_mixed_precision_loading's own docstring for the real
        # 2026-09-20 end-to-end finding (measured slower, not faster, on this project's target
        # hardware) behind this flag.
        self._enable_mixed_precision_loading = enable_mixed_precision_loading
        # Off by default - see Settings.enable_quantized_native_compute's own docstring. A real,
        # permanent user choice (2026-09-21), not a rollout toggle.
        self._enable_quantized_native_compute = enable_quantized_native_compute
        self._clock = clock
        self._handles: dict[str, ModelHandle] = {}
        self._registry = ArchitectureRegistry()
        # Tags stop-requested while they had no handle yet - see unload()'s own docstring for why
        # that's a real, common case (not just a stray/duplicate call) on this project's slow-CPU
        # target: loading alone routinely takes tens of seconds, and a caller can ask to stop well
        # before that finishes.
        self._cancelled_loads: set[str] = set()

    def get_or_load(self, tag: str, keep_alive_seconds: int | None = None) -> ModelHandle:
        now = self._clock()
        self._evict_expired(now)

        handle = self._handles.get(tag)
        if handle is not None:
            handle.touch(now, keep_alive_seconds)
            return handle

        if len(self._handles) >= self._max_loaded:
            self._evict_least_recently_used()

        # A fresh load attempt always starts clean - any cancellation flag left over here belongs
        # to a *previous*, already-finished attempt for this same tag, not this one.
        self._cancelled_loads.discard(tag)
        handle = self._load(tag, keep_alive_seconds, now)
        self._handles[tag] = handle
        return handle

    def pop_cancelled_load(self, tag: str) -> bool:
        """True (and consumes the flag) if unload() was called for `tag` while it had no handle
        yet - i.e. while a get_or_load() call for it was still stuck inside `_load()`'s
        many-seconds dequantization step, well before there was any ModelWorker for unload()'s own
        request_stop() to reach. The one caller that needs this (ChatRequestHandler.handle(), right
        after its own get_or_load() call returns) uses it to skip straight to a "done, nothing
        generated" response instead of starting a generation nobody wants anymore - the
        freshly-loaded handle itself stays cached regardless: the real loading work already
        happened, so there's no reason to throw it away over one cancelled request."""
        if tag in self._cancelled_loads:
            self._cancelled_loads.discard(tag)
            return True
        return False

    def unload(self, tag: str) -> None:
        """Evicts `tag`'s handle and, if it has one, tells its worker to abort whatever job is
        currently running - see ModelWorker.request_stop's own docstring for why setting a flag is
        enough to actually halt the underlying compute, not just this manager's own bookkeeping.
        Popping the dict entry alone (the only thing this used to do) left a mid-generation worker
        thread completely unaware anything happened: it kept computing every remaining token up to
        max_new_tokens/EOS, orphaned from the manager and invisible to `/api/ps`, but still pegging
        the CPU - confirmed live via ChatRequest.is_unload_call()'s own "stop" contract (the same
        one app.services.matricxon_client.stop_model sends after e.g. a mid-stream message
        delete).

        `wait_until_idle()` runs before `architecture.close()`, not just `request_stop()` - real
        bug this closes (2026-09-22, confirmed live): `request_stop()` only sets a flag the worker
        notices at its next per-layer/per-token check, so without waiting, `close()` could run on
        this thread while the worker's own thread was still mid-`forward()` - with real
        quantized-native compute on, still actively reading a `QuantizedLinear`'s live raw-mmap
        view via a fused GEMV kernel call at that exact moment. `close()` released that view and
        closed the mmap out from under it, crashing with a real `BufferError: cannot close
        exported pointers exist`. Waiting first means `close()` only ever runs once the worker has
        genuinely stopped touching this model's weights, not just been asked to.

        If `tag` has no handle at all right now, that most likely means a get_or_load() call for
        it is still loading (see pop_cancelled_load's own docstring) - confirmed live as the actual
        dominant case, not a rare edge: on this project's target hardware, loading alone can easily
        take longer than a user waits before giving up and deleting their message, so the stop
        request routinely arrives before any handle - let alone any worker job - exists yet.
        Recording it here is what lets that in-flight load notice, the moment it finishes, that
        nobody wants the generation it's about to start."""
        handle = self._handles.pop(tag, None)
        if handle is not None:
            handle.worker.request_stop()
            handle.worker.wait_until_idle()
            handle.architecture.close()
        else:
            self._cancelled_loads.add(tag)

    def now(self) -> float:
        """Exposes the manager's own (possibly injected/fake) clock, so a
        caller computing a `ModelHandle`'s `expires_in()` reads the same
        clock its `last_used_at` was set from - a wall-clock `time.monotonic()`
        call at the API boundary would silently disagree under a fake clock.
        """
        return self._clock()

    def list_loaded(self) -> list[ModelHandle]:
        self._evict_expired(self._clock())
        return list(self._handles.values())

    def evict_expired(self) -> None:
        self._evict_expired(self._clock())

    def _evict_expired(self, now: float) -> None:
        # A handle whose worker is_busy() (still generating - see ModelWorker.is_busy's own
        # docstring for why this matters) is never evicted, expired keep-alive or not: nothing
        # about the timer running out makes it safe to yank the handle out from under a job that's
        # still actively consuming it - the worker thread would keep computing regardless, now
        # orphaned from this dict and invisible to /api/ps, with unload() (the ONE explicit way to
        # actually stop it, see that method's own docstring) unable to find it anymore either.
        for tag in [
            tag
            for tag, handle in self._handles.items()
            if handle.is_expired(now) and not handle.worker.is_busy()
        ]:
            self._handles.pop(tag).architecture.close()

    def _evict_least_recently_used(self) -> None:
        # Same "never evict a busy handle" rule as _evict_expired above - if every currently-loaded
        # handle is busy, this is a deliberate no-op: get_or_load's own caller then simply loads the
        # new model alongside them, temporarily over max_loaded rather than corrupting an in-flight
        # generation. A soft cap under real contention is the honest tradeoff; there is no safe way
        # to free a slot that's actually in use.
        idle = {tag: handle for tag, handle in self._handles.items() if not handle.worker.is_busy()}
        if not idle:
            return
        lru_tag = min(idle, key=lambda tag: idle[tag].last_used_at)
        self._handles.pop(lru_tag).architecture.close()

    def _load(self, tag: str, keep_alive_seconds: int | None, now: float) -> ModelHandle:
        installed = self._catalog.get(tag)
        gguf_path = self._catalog.gguf_path_for_tag(tag)

        # Parsed here (ahead of the memory check) rather than left to GGUFModelLoader's own
        # constructor below, purely so the exact tensor shapes it reads are available to compute
        # bf16_bytes from before deciding whether to attempt the load at all - GGUFReader.read()
        # is a cheap, KB-scale header-only parse (see its own docstring), so re-parsing the same
        # header a second time inside GGUFModelLoader.__init__ costs nothing worth avoiding.
        # `.metadata` (not just `.tensor_infos`) is kept too, to resolve the architecture class
        # below without a second parse of the same header.
        parsed = GGUFReader(gguf_path).read()
        tensor_infos = parsed.tensor_infos
        bf16_bytes = exact_bf16_bytes(tensor_infos)
        architecture_cls = self._registry.resolve(parsed.metadata)

        # See Settings.enable_quantized_native_compute's own docstring / estimate_quantized_
        # native_bytes's own docstring for why this circuit breaker needs its own, smaller,
        # real estimate when this mode is active - `select_load_dtype` below still uses the
        # full `bf16_bytes` figure (it only decides float32-vs-bf16 for the tensors that *do*
        # get fully materialized either way - norms/embeddings/attn_q/attn_k - a smaller number
        # there would risk wrongly picking float32 for those with less real headroom than it
        # assumes, not a safety problem `ensure_enough_memory_to_load` itself can afford).
        quantized_native_enabled = (
            self._enable_quantized_native_compute
            and architecture_cls.NAME in QUANTIZED_NATIVE_WIRED_ARCHITECTURES
        )
        real_bytes_needed = estimate_quantized_native_bytes(
            tensor_infos, quantized_native_enabled, architecture_cls.NAME
        )

        ensure_enough_memory_to_load(
            tag, real_bytes_needed, self._memory_safety_margin, list(self._handles)
        )
        dtype = select_load_dtype(bf16_bytes, self._memory_safety_margin)

        # Mixed per-layer float32/bf16 precision - see plan_mixed_precision_load's own docstring,
        # including why it's off (returns (None, dtype)) by default.
        layer_dtypes, loader_dtype = plan_mixed_precision_load(
            tensor_infos,
            dtype,
            self._memory_safety_margin,
            enabled=self._enable_mixed_precision_loading,
            supports_mixed_precision=architecture_cls is Mistral3TextArchitecture,
        )

        # No `with` block here on purpose: on-the-fly dequant means `from_gguf` defers reading any
        # tensor bytes at all until this model's first real forward pass (see
        # ModelArchitecture._ensure_materialized), so the mmap behind `loader` has to stay open past
        # this method returning - ownership transfers to `architecture` itself once from_gguf
        # succeeds. If anything fails before then, this closes it here instead of leaking it.
        loader = GGUFModelLoader(gguf_path, dtype=loader_dtype)
        try:
            # See Settings.enable_quantized_native_compute's own docstring - only architectures in
            # QUANTIZED_NATIVE_WIRED_ARCHITECTURES accept this kwarg today.
            quantized_native_kwargs = (
                {"enable_quantized_native": True} if quantized_native_enabled else {}
            )
            if layer_dtypes is not None:
                architecture = architecture_cls.from_gguf(
                    loader, dtype=dtype, layer_dtypes=layer_dtypes, **quantized_native_kwargs
                )
            else:
                architecture = architecture_cls.from_gguf(
                    loader, dtype=dtype, **quantized_native_kwargs
                )
            tokenizer = build_tokenizer(loader.metadata)
        except Exception:
            loader.close()
            raise
        architecture.eval()

        return ModelHandle(
            tag=tag,
            architecture=architecture,
            tokenizer=tokenizer,
            worker=ModelWorker(),
            capabilities=installed.capabilities,
            size_bytes=installed.size_bytes,
            keep_alive_seconds=(
                keep_alive_seconds
                if keep_alive_seconds is not None
                else self._default_keep_alive_seconds
            ),
            last_used_at=now,
        )
