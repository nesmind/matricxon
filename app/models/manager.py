import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from app.architectures.mistral3 import Mistral3TextArchitecture
from app.architectures.registry import ArchitectureRegistry
from app.gguf.dequant.torch_dequant import TorchDequantizer
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
from app.runtime.batch_decode import BatchDecoder
from app.runtime.cache_store import PersistentCacheStore
from app.runtime.cache_tier import ModelCacheTier
from app.runtime.chat_stop_tokens import extra_eos_token_ids
from app.runtime.chat_template import PromptBuilderFactory
from app.runtime.compute_device import ComputeDevice
from app.runtime.memory_trim import memory_trimmer
from app.runtime.prompt_cache import PromptCache
from app.runtime.tool_call_formats import ToolCallFormats
from app.server.errors import DeviceUnavailableError

# How many not-yet-started cancelled reply ids to remember (ModelManager.cancel_request).
_MAX_REMEMBERED_CANCELS = 1024


class ModelManager:
    """Loads GGUF models on demand and keeps up to `max_loaded` resident, evicting the
    least-recently-used one past its keep-alive.

    `max_loaded=2` matches pAIring's usage: its RAG path calls `embed` then `chat_stream` each turn,
    so 1 would reload a model every turn. There is no upper bound, but raising it on a
    RAM-constrained box relies on `_ensure_enough_memory_to_load`, which fails a load that clearly
    won't fit with `InsufficientMemoryError` instead of risking an OOM kill.

    Loads in bf16 when the hardware accelerates it (CUDA, or a CPU with `avx512_bf16`/`amx_bf16`),
    otherwise float32 if that demonstrably fits in available memory, else bf16
    (`app.models.load_dtype.select_load_dtype`): bf16 halves memory (~6.9GB vs ~13.7GB for a
    3.85B-param model) but is far slower on a CPU without native bf16, and float32 doesn't help if
    it thrashes swap.

    `clock` is injectable so eviction/expiry logic is testable with a fake clock.
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
        prompt_cache_slots: int = 4,
        prompt_cache_context_shift: bool = True,
        prompt_cache_budget_mb: int = 2048,
        max_decode_batch: int = 8,
        device: str = "cpu",
        gpu_weight_mode: str = "dequantized",
        cache_store: PersistentCacheStore | None = None,
    ) -> None:
        # Raises DeviceUnavailableError for a missing GPU (never a silent CPU fallback).
        self._device = ComputeDevice.resolve(device)
        self._gpu_weight_mode = gpu_weight_mode
        self._cache_store = cache_store
        self._catalog = catalog
        self._max_loaded = max_loaded
        self._default_keep_alive_seconds = default_keep_alive_seconds
        self._memory_safety_margin = memory_safety_margin
        # Off by default (Settings.enable_mixed_precision_loading: the end-to-end
        # finding, measured slower on the target hardware).
        self._enable_mixed_precision_loading = enable_mixed_precision_loading
        # Off by default (Settings.enable_quantized_native_compute). A permanent user choice
        # , not a rollout toggle.
        self._enable_quantized_native_compute = enable_quantized_native_compute
        self._clock = clock
        self._prompt_cache_slots = prompt_cache_slots
        self._prompt_cache_context_shift = prompt_cache_context_shift
        self._prompt_cache_budget_bytes = prompt_cache_budget_mb << 20
        self._max_decode_batch = max_decode_batch
        self._handles: dict[str, ModelHandle] = {}
        self._registry = ArchitectureRegistry()
        # Tags stop-requested while they had no handle yet (unload says why that is common, not a
        # stray call): on the slow-CPU target loading takes tens of seconds, and a caller can ask to
        # stop well before it finishes.
        self._cancelled_loads: set[str] = set()
        self._cancelled_requests: OrderedDict[str, None] = OrderedDict()
        # One lock per tag: requests for a model that is still loading (chats, a cluster warm-up)
        # wait for that load instead of loading a second copy.
        self._load_locks: dict[str, threading.Lock] = {}
        self._load_locks_guard = threading.Lock()

    def get_or_load(self, tag: str, keep_alive_seconds: int | None = None) -> ModelHandle:
        with self._load_locks_guard:
            lock = self._load_locks.setdefault(tag, threading.Lock())
        with lock:
            return self._get_or_load(tag, keep_alive_seconds)

    def _get_or_load(self, tag: str, keep_alive_seconds: int | None) -> ModelHandle:
        now = self._clock()
        self._evict_expired(now)

        handle = self._handles.get(tag)
        if handle is not None:
            handle.touch(now, keep_alive_seconds)
            return handle

        if len(self._handles) >= self._max_loaded:
            self._evict_least_recently_used()

        # A fresh load starts clean: a leftover cancellation flag belongs to a previous finished
        # attempt for this tag.
        self._cancelled_loads.discard(tag)
        handle = self._load(tag, keep_alive_seconds, now)
        handle.worker.on_job_done = lambda: handle.touch(self._clock())
        self._handles[tag] = handle
        # Loading leaves large temporary buffers behind.
        memory_trimmer.schedule()
        return handle

    def cancel_request(self, tag: str, request_id: str) -> None:
        """Stops one reply (`ChatRequest.request_id`) without unloading anything: the model, other
        users' replies and caches are untouched. If the reply isn't on a worker yet (still loading
        the model, or between router and worker) the id is remembered, so `pop_cancelled_request`
        can drop it when it would start. See `ModelWorker.cancel_job`."""
        handle = self._handles.get(tag)
        if handle is not None and handle.worker.cancel_job(request_id):
            return
        self._cancelled_requests[request_id] = None
        while len(self._cancelled_requests) > _MAX_REMEMBERED_CANCELS:
            self._cancelled_requests.popitem(last=False)

    def pop_cancelled_request(self, request_id: str) -> bool:
        """True (and consumes the note) if `cancel_request` ran for this id before it started."""
        return self._cancelled_requests.pop(request_id, 0) is None

    def pop_cancelled_load(self, tag: str) -> bool:
        """True (and consumes the flag) if unload was called for `tag` while it had no handle,
        i.e. while a get_or_load for it was in `_load`'s many-seconds dequantization step,
        before any ModelWorker existed for unload's request_stop to reach. The caller
        (ChatRequestHandler.handle, right after get_or_load returns) uses it to answer "done,
        nothing generated" instead of starting a generation nobody wants; the loaded handle stays
        cached (the work was done)."""
        if tag in self._cancelled_loads:
            self._cancelled_loads.discard(tag)
            return True
        return False

    def unload(self, tag: str) -> None:
        """Evicts `tag`'s handle and, if it has one, tells its worker to abort the running job
        (`ModelWorker.request_stop`). Popping the dict entry alone (the old behavior) left a
        mid-generation worker thread running: it kept computing every token up to
        max_new_tokens/EOS, invisible to `/api/ps` and still using the CPU. This is the "stop"
        contract of ChatRequest.is_unload_call (sent by app.services.matricxon_client.stop_model,
        e.g. after a mid-stream message delete).

        `wait_until_idle` runs before `architecture.close`: `request_stop` only sets a flag seen at
        the next per-layer/per-token check, so `close` could run mid-`forward` while
        quantized-native compute read a `QuantizedLinear`'s raw-mmap view through a fused GEMV
        kernel, crashing with `BufferError: cannot close exported pointers exist`. Waiting first
        means `close` runs after the worker has stopped touching the weights.

        If `tag` has no handle, a get_or_load for it is most likely still loading
        (`pop_cancelled_load`), the common case since loading can outlast a user's patience.
        Recording the stop lets that load notice, when it finishes, that nobody wants the
        generation.
        """
        handle = self._handles.pop(tag, None)
        if handle is not None:
            handle.worker.request_stop()
            handle.worker.wait_until_idle()
            self._retire(handle)
        else:
            self._cancelled_loads.add(tag)

    def _retire(self, handle: ModelHandle) -> None:
        """Drops a handle for good: its idle cached conversations go to the disk tier first (a
        background write; the slots are independent of the weights closed right after)."""
        handle.prompt_cache.persist_all()
        handle.worker.shutdown()
        handle.architecture.close()
        memory_trimmer.schedule()

    def holds_chat(self, tag: str) -> bool:
        """Is a conversation of the chat `tag` in RAM on any loaded model? (cluster routing)"""
        return any(handle.prompt_cache.holds(tag) for handle in self.list_loaded())

    def spill_chat(self, tag: str) -> int:
        """Writes the chat's idle RAM conversations to the shared disk tier and waits for the files,
        so another node can continue the chat; the models stay loaded. Returns how many."""
        moved = sum(handle.prompt_cache.spill_chat(tag) for handle in self.list_loaded())
        if moved and self._cache_store is not None:
            self._cache_store.flush()
        return moved

    def shutdown(self) -> None:
        """Server stop: halts every reply, saves every model's cached conversations to disk and
        waits for the writes, so a restart keeps long chats cheap to resume."""
        for tag in list(self._handles):
            self.unload(tag)
        if self._cache_store is not None:
            self._cache_store.flush()

    def set_limits(
        self,
        *,
        max_loaded: int | None = None,
        default_keep_alive_seconds: int | None = None,
        memory_safety_margin: float | None = None,
        prompt_cache_slots: int | None = None,
        prompt_cache_budget_mb: int | None = None,
        max_decode_batch: int | None = None,
        enable_quantized_native_compute: bool | None = None,
        gpu_weight_mode: str | None = None,
    ) -> None:
        """New values without a restart; None keeps one. Loaded models take the keep-alive, cache
        and concurrency limits now; the other values apply from the next load."""
        if max_loaded is not None:
            self._max_loaded = max_loaded
        if default_keep_alive_seconds is not None:
            self._default_keep_alive_seconds = default_keep_alive_seconds
        if memory_safety_margin is not None:
            self._memory_safety_margin = memory_safety_margin
        if prompt_cache_slots is not None:
            self._prompt_cache_slots = prompt_cache_slots
        if prompt_cache_budget_mb is not None:
            self._prompt_cache_budget_bytes = prompt_cache_budget_mb << 20
        if max_decode_batch is not None:
            self._max_decode_batch = max_decode_batch
        if enable_quantized_native_compute is not None:
            self._enable_quantized_native_compute = enable_quantized_native_compute
        if gpu_weight_mode is not None:
            self._gpu_weight_mode = gpu_weight_mode
        cache_changed = prompt_cache_slots is not None or prompt_cache_budget_mb is not None
        for handle in list(self._handles.values()):
            if default_keep_alive_seconds is not None:
                handle.keep_alive_seconds = default_keep_alive_seconds
            if cache_changed:
                budget = self._prompt_cache_budget_bytes
                if self.on_gpu:  # never above the VRAM cap set at load
                    budget = min(budget, handle.prompt_cache.budget_bytes)
                handle.prompt_cache.set_limits(self._prompt_cache_slots, budget)
            if max_decode_batch is not None:
                handle.worker.set_max_active(max_decode_batch)

    def _cache_budget_bytes(self, on_gpu: bool, free_before_load: int, weight_bytes: int) -> int:
        """Prompt-cache budget. On a GPU the cached KV lives in VRAM beside the weights, so it is
        capped at half the VRAM left once this model's weights are in (never above the setting; the
        weights may not be uploaded yet, hence the estimate)."""
        if not on_gpu:
            return self._prompt_cache_budget_bytes
        return min(self._prompt_cache_budget_bytes, max(0, free_before_load - weight_bytes) // 2)

    @property
    def on_gpu(self) -> bool:
        return self._device.is_gpu

    def now(self) -> float:
        """Exposes the manager's own (possibly fake) clock, so a caller computing a `ModelHandle`'s
        `expires_in` reads the same clock its `last_used_at` came from (a wall-clock
        `time.monotonic` at the API boundary would disagree under a fake clock)."""
        return self._clock()

    def list_loaded(self) -> list[ModelHandle]:
        self._evict_expired(self._clock())
        return list(self._handles.values())

    def evict_expired(self) -> None:
        self._evict_expired(self._clock())

    def _evict_expired(self, now: float) -> None:
        # A handle whose worker is_busy (ModelWorker.is_busy says why) is never evicted,
        # keep-alive expired or not: the timer running out doesn't make it safe to yank the handle
        # from a job still using it; the worker would keep computing, orphaned from this dict and
        # invisible to /api/ps, with unload (the one way to stop it) unable to find it.
        for tag in [
            tag
            for tag, handle in self._handles.items()
            if handle.is_expired(now) and not handle.worker.is_busy()
        ]:
            self._retire(self._handles.pop(tag))

    def _evict_least_recently_used(self) -> None:
        # Same "never evict a busy handle" rule as _evict_expired: if every loaded handle is busy
        # this is a deliberate no-op, and get_or_load's caller loads the new model alongside,
        # temporarily over max_loaded rather than corrupting an in-flight generation. A soft cap
        # under contention is the honest tradeoff; a slot in use can't be safely freed.
        idle = {tag: handle for tag, handle in self._handles.items() if not handle.worker.is_busy()}
        if not idle:
            return
        lru_tag = min(idle, key=lambda tag: idle[tag].last_used_at)
        self._retire(self._handles.pop(lru_tag))

    def _load(self, tag: str, keep_alive_seconds: int | None, now: float) -> ModelHandle:
        installed = self._catalog.get(tag)
        gguf_path = self._catalog.gguf_path_for_tag(tag)

        # Parsed here (before the memory check), not left to GGUFModelLoader's constructor, so the
        # tensor shapes are available to compute bf16_bytes before deciding whether to attempt the
        # load. GGUFReader.read is a cheap KB-scale header parse, so GGUFModelLoader.__init__
        # re-parsing costs nothing worth avoiding. `.metadata` is kept to resolve the architecture
        # class below without another parse.
        parsed = GGUFReader(gguf_path).read()
        tensor_infos = parsed.tensor_infos
        bf16_bytes = exact_bf16_bytes(tensor_infos)
        architecture_cls = self._registry.resolve(parsed.metadata)

        # Why this breaker needs its own smaller estimate in this mode:
        # Settings.enable_quantized_native_compute / estimate_quantized_native_bytes.
        # `select_load_dtype` below still uses the full `bf16_bytes`: it decides float32-vs-bf16
        # only for tensors fully materialized either way (norms/embeddings/attn_q/attn_k), and a
        # smaller number could wrongly pick float32 with less headroom than assumed. The CPU
        # packed-weight kernels read the mmap, so a GPU model either materializes in bf16 or, with
        # gpu_weight_mode="packed", uploads supported quant types still packed.
        on_gpu = self._device.is_gpu
        if on_gpu and not architecture_cls.SUPPORTS_GPU:
            raise DeviceUnavailableError(
                f"architecture {architecture_cls.NAME!r} has no GPU support yet (CPU only)"
            )
        wired = architecture_cls.NAME in QUANTIZED_NATIVE_WIRED_ARCHITECTURES
        gpu_packed = on_gpu and wired and self._gpu_weight_mode == "packed"
        quantized_native_enabled = gpu_packed or (
            self._enable_quantized_native_compute and not on_gpu and wired
        )
        real_bytes_needed = estimate_quantized_native_bytes(
            tensor_infos,
            quantized_native_enabled,
            architecture_cls.NAME,
            packable=TorchDequantizer.supports if gpu_packed else None,
        )

        free_before_load = (self._device.free_memory_bytes() or 0) if on_gpu else 0
        ensure_enough_memory_to_load(
            tag,
            real_bytes_needed,
            self._memory_safety_margin,
            list(self._handles),
            available_fn=self._device.free_memory_bytes if on_gpu else None,
            memory_kind="VRAM" if on_gpu else "memory",
        )
        # float32-vs-bf16 for what stays float: with quantized-native compute that is a small part
        # of the model, so size the decision from the real packed footprint, not the full unpacked
        # bf16 size, which made a 4B model always fall to bf16, ~10x slower here (no hardware bf16;
        # measured: Qwen3.5-4B decode 1.65 -> 0.18 tok/s).
        dtype = (
            self._device.load_dtype()
            if on_gpu
            else select_load_dtype(
                real_bytes_needed if quantized_native_enabled else bf16_bytes,
                self._memory_safety_margin,
            )
        )

        # Mixed per-layer float32/bf16 precision: plan_mixed_precision_load (including why it
        # returns (None, dtype) by default).
        layer_dtypes, loader_dtype = plan_mixed_precision_load(
            tensor_infos,
            dtype,
            self._memory_safety_margin,
            enabled=self._enable_mixed_precision_loading and not on_gpu,
            supports_mixed_precision=architecture_cls is Mistral3TextArchitecture,
        )

        # No `with` block on purpose: on-the-fly dequant means `from_gguf` defers reading tensor
        # bytes until the first forward pass (ModelArchitecture._ensure_materialized), so the mmap
        # behind `loader` must stay open past this method; ownership passes to `architecture` once
        # from_gguf succeeds. If anything fails first, this closes it here instead of leaking it.
        loader = GGUFModelLoader(gguf_path, dtype=loader_dtype)
        try:
            # See Settings.enable_quantized_native_compute: only architectures in
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
            prompt_builder = PromptBuilderFactory.for_metadata(loader.metadata, tag=tag)
        except Exception:
            loader.close()
            raise
        try:
            architecture.place_on(self._device.torch_device, packed_weights=gpu_packed)
        except Exception:
            architecture.close()
            raise
        architecture.eval()

        return ModelHandle(
            tag=tag,
            architecture=architecture,
            tokenizer=tokenizer,
            worker=ModelWorker(
                BatchDecoder(architecture, self._max_decode_batch), self._max_decode_batch
            ),
            capabilities=installed.capabilities,
            size_bytes=installed.size_bytes,
            keep_alive_seconds=(
                keep_alive_seconds
                if keep_alive_seconds is not None
                else self._default_keep_alive_seconds
            ),
            last_used_at=now,
            prompt_builder=prompt_builder,
            tool_call_format=ToolCallFormats.for_template(
                loader.metadata.get("tokenizer.chat_template"), loader.metadata.architecture
            ),
            extra_eos_token_ids=extra_eos_token_ids(loader.metadata),
            prompt_cache=PromptCache(
                self._prompt_cache_slots,
                self._cache_budget_bytes(on_gpu, free_before_load, real_bytes_needed),
                ModelCacheTier(self._cache_store, f"{tag}|{installed.size_bytes}")
                if self._cache_store is not None
                else None,
                self._prompt_cache_context_shift,
            ),
        )
