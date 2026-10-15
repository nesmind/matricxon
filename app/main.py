import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI

from app.config import settings
from app.consensus.events import ClusterEvents, enable_always_on
from app.dependencies import (
    get_cache_store,
    get_consensus_runtime,
    get_model_manager,
    get_shared_settings,
)
from app.native.fused_ops import FusedOps
from app.native.gemm import NativeGemm
from app.routers import (
    cache_chats_router,
    cache_persistence_router,
    capabilities_router,
    chat_router,
    cluster_router,
    consensus_router,
    copy_router,
    create_router,
    delete_router,
    embed_router,
    embeddings_router,
    generate_router,
    health_router,
    node_load_router,
    ps_router,
    pull_router,
    push_router,
    rerank_router,
    runtime_settings_router,
    show_router,
    storage_router,
    tags_router,
    version_router,
)
from app.runtime.compute_device import ComputeDevice
from app.server.errors import DeviceUnavailableError, ErrorHandlerRegistrar
from app.server.log_filters import LOG_LEVELS_BY_SETTING, quiet_cluster_logging
from app.server.pidfile import PidFile


class MatricxonApp:
    """Builds the FastAPI app: an Ollama-API-compatible inference server."""

    ROUTERS = (
        tags_router.router,
        chat_router.router,
        embeddings_router.router,
        pull_router.router,
        delete_router.router,
        show_router.router,
        ps_router.router,
        generate_router.router,
        embed_router.router,
        rerank_router.router,
        version_router.router,
        copy_router.router,
        create_router.router,
        push_router.router,
        health_router.router,
        capabilities_router.router,
        cache_persistence_router.router,
        cache_chats_router.router,
        storage_router.router,
        cluster_router.router,
        runtime_settings_router.router,
        consensus_router.router,
        node_load_router.router,
    )

    def __init__(self) -> None:
        # scripts/start.sh sets MATRICXON_PID_FILE to an absolute path; the default is relative to
        # cwd (matching start.sh's default) for anything importing this module directly (tests, a
        # bare `uvicorn app.main:app`).
        self._pid_file = PidFile(Path(os.environ.get("MATRICXON_PID_FILE", "run/matricxon.pid")))

        # What each level traces: Settings.log_level (0: warnings/errors only, as before any logging
        # existed; 1: per-request milestones; 2: adds a per-decoder-layer trace). basicConfig
        # here, once, is the init point every module's logger depends on.
        logging.basicConfig(
            level=LOG_LEVELS_BY_SETTING[settings.log_level],
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        quiet_cluster_logging(include_traffic=settings.log_consensus_traffic)
        enable_always_on()

        # PyTorch's thread-count heuristic under-detects (measured: 2 threads on a 4-core machine,
        # leaving half the CPU idle in every matmul of the hot path); explicit beats implicit. See
        # Settings.torch_threads; irrelevant on a non-CPU device.
        if settings.device == "cpu":
            torch.set_num_threads(settings.torch_threads or os.cpu_count() or 1)
        # The target hardware (Sandy Bridge) lacks the instruction set NNPACK needs
        # (torch.backends.nnpack.is_available is False, like cpu_accelerates_bf16's
        # avx512_bf16/amx_bf16 check). By default every Conv1d (nemotron_h's SSM causal conv,
        # NemotronHMamba2Mixer) re-probes NNPACK and logs a native C++ warning that Python's
        # `warnings` filters can't catch, once per generated token ("Could not initialize NNPACK!
        # Reason: Unsupported hardware"). Turning it off skips the probe; conv1d has a working
        # non-NNPACK fallback, checked on the same shapes.
        torch.backends.nnpack.set_flags(False)
        # See Settings.gemv_backend: a no-op unless "native"; a failed build only logs a warning.
        NativeGemm.configure(settings.gemv_backend, settings.torch_threads)
        FusedOps.configure(settings.enable_fused_ops)

    @asynccontextmanager
    async def _lifespan(self, _app: FastAPI) -> AsyncIterator[None]:
        # Built now, not on the first request: runs the cache dir cleanup and indexing at boot, and
        # a cluster node without its provisioned key fails here, not on its first chat.
        await asyncio.to_thread(get_cache_store)
        consensus = get_consensus_runtime()
        shared = get_shared_settings()
        if consensus is not None:
            consensus.start()
        if shared is not None:
            shared.start()
        if settings.cluster_role != "off" and self._pid_file.previous_run_died():
            ClusterEvents.previous_run_crashed()
        self._pid_file.write()
        try:
            yield
        finally:
            if shared is not None:
                shared.stop()
            if consensus is not None:
                consensus.stop()
            await asyncio.to_thread(get_model_manager().shutdown)
            self._pid_file.remove()

    @staticmethod
    def require_device() -> None:
        """Exits with a one-line log (no traceback) when MATRICXON_DEVICE names a GPU that isn't
        there: checked at import, before the server starts, not on the first request."""
        log = logging.getLogger(__name__)
        try:
            device = ComputeDevice.resolve(settings.device)
        except DeviceUnavailableError as error:
            log.critical("matricxon cannot start: %s", error.message)
            raise SystemExit(1) from None
        if device.is_gpu:
            log.warning(
                "experimental GPU mode on %s (weights: %s)", device.name, settings.gpu_weight_mode
            )

    def build(self) -> FastAPI:
        app = FastAPI(title="matricxon", lifespan=self._lifespan)
        for router in self.ROUTERS:
            app.include_router(router)
        ErrorHandlerRegistrar().register(app)
        return app


_matricxon = MatricxonApp()
_matricxon.require_device()
app = _matricxon.build()
