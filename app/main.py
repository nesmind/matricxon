import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI

from app.config import settings
from app.native.fused_ops import FusedOps
from app.native.gemm import NativeGemm
from app.routers import (
    capabilities_router,
    chat_router,
    copy_router,
    create_router,
    delete_router,
    embed_router,
    embeddings_router,
    generate_router,
    health_router,
    ps_router,
    pull_router,
    push_router,
    show_router,
    tags_router,
    version_router,
)
from app.runtime.compute_device import ComputeDevice
from app.server.errors import DeviceUnavailableError, ErrorHandlerRegistrar
from app.server.pidfile import PidFile

# Maps Settings.log_level (0-2, see its own docstring) onto Python's stdlib `logging` levels -
# every module's own `logging.getLogger(__name__)` call implicitly depends on this being
# configured once, here, before any of them ever logs anything.
_LOG_LEVELS_BY_SETTING = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}


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
        version_router.router,
        copy_router.router,
        create_router.router,
        push_router.router,
        health_router.router,
        capabilities_router.router,
    )

    def __init__(self) -> None:
        # scripts/start.sh sets MATRICXON_PID_FILE to an absolute path before launching this
        # process — defaults to a path relative to cwd (matching scripts/start.sh's own default)
        # for anything that imports this module directly (tests, a bare `uvicorn app.main:app`
        # with no wrapper script).
        self._pid_file = PidFile(Path(os.environ.get("MATRICXON_PID_FILE", "run/matricxon.pid")))

        # See Settings.log_level's own docstring for what each level actually traces (0: warnings/
        # errors only, matching matricxon's own history of having no logging at all before this;
        # 1: per-request pipeline milestones; 2: adds a per-decoder-layer trace). basicConfig()
        # here, once, is the single init point every module's own logger implicitly depends on.
        logging.basicConfig(
            level=_LOG_LEVELS_BY_SETTING[settings.log_level],
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        # PyTorch's own thread-count heuristic under-detects on real hardware (measured live: 2
        # threads on an actual 4-core machine, leaving half the CPU idle during every matmul in the
        # hot generation path) - explicit beats implicit here. See Settings.torch_threads's own
        # docstring; irrelevant once matricxon actually supports a non-CPU device.
        if settings.device == "cpu":
            torch.set_num_threads(settings.torch_threads or os.cpu_count() or 1)
        # This project's target hardware (Sandy Bridge) genuinely lacks whatever instruction set
        # NNPACK needs (confirmed: torch.backends.nnpack.is_available() is False here, same class
        # of gap as cpu_accelerates_bf16's own avx512_bf16/amx_bf16 check) - left at its PyTorch
        # default, every real Conv1d call (nemotron_h's own SSM causal conv, see
        # NemotronHMamba2Mixer) re-probes NNPACK and re-logs a native C++ warning that Python's own
        # `warnings` filters can't catch, once per generated token (confirmed live, 2026-09-29:
        # "Could not initialize NNPACK! Reason: Unsupported hardware", 11x in one nemotron_h chat).
        # Turning it off here skips the probe (and the warning) entirely - conv1d already has a
        # working non-NNPACK fallback path either way, confirmed correct on the same shapes.
        torch.backends.nnpack.set_flags(False)
        # See Settings.gemv_backend - a no-op unless "native"; a failed build only logs a warning.
        NativeGemm.configure(settings.gemv_backend, settings.torch_threads)
        FusedOps.configure(settings.enable_fused_ops)

    @asynccontextmanager
    async def _lifespan(self, _app: FastAPI) -> AsyncIterator[None]:
        self._pid_file.write()
        try:
            yield
        finally:
            self._pid_file.remove()

    @staticmethod
    def require_device() -> None:
        """Exits with a one-line log message (no traceback) when MATRICXON_DEVICE names a GPU that
        isn't there - checked at import, before the server starts, not on the first request."""
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
