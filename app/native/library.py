"""Builds and loads matricxon's own native kernel library (`app/native/src/*.c`, see ROADMAP.md's
"In-house native (C) quantized kernels" entry) - compiled on first use with the system C compiler,
so there's no CMake and no new pip dependency. `-march=native` targets whatever CPU this process
actually runs on (this project's machine: Sandy Bridge, AVX but no AVX2/FMA/F16C), which is also
why the library is built locally rather than shipped prebuilt.

Rebuilt automatically whenever any source file is newer than the built library. The build writes
to a temp file and renames it into place, so two matricxon instances starting at once (pAIring
can run several) can never load a half-written library.
"""

import ctypes
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_NATIVE_DIR = Path(__file__).resolve().parent


class NativeBuildError(RuntimeError):
    pass


class NativeKernelLibrary:
    LIBRARY_NAME = "libmatricxon_kernels.so"
    CFLAGS = ("-O3", "-march=native", "-fopenmp", "-fPIC", "-shared", "-Wall")

    def __init__(
        self,
        source_dir: Path = _NATIVE_DIR / "src",
        build_dir: Path = _NATIVE_DIR / "build",
        compiler: str | None = None,
    ) -> None:
        self._source_dir = source_dir
        self._build_dir = build_dir
        self._compiler = compiler or os.environ.get("CC", "gcc")

    @property
    def library_path(self) -> Path:
        return self._build_dir / self.LIBRARY_NAME

    def _sources(self) -> list[Path]:
        return sorted(self._source_dir.glob("*.c"))

    def _is_stale(self) -> bool:
        if not self.library_path.exists():
            return True
        built_at = self.library_path.stat().st_mtime
        inputs = [*self._sources(), *self._source_dir.glob("*.h")]
        return any(path.stat().st_mtime > built_at for path in inputs)

    def build(self) -> Path:
        """Compiles every `src/*.c` into one shared library. Raises NativeBuildError (with the
        compiler's own output) if no compiler is installed or compilation fails."""
        if shutil.which(self._compiler) is None:
            raise NativeBuildError(f"C compiler {self._compiler!r} not found on PATH")
        self._build_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self._build_dir, suffix=".so.tmp")
        os.close(fd)
        command = [
            self._compiler,
            *self.CFLAGS,
            "-o",
            tmp_name,
            *map(str, self._sources()),
            "-lm",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            Path(tmp_name).unlink(missing_ok=True)
            raise NativeBuildError(f"native kernel build failed:\n{result.stderr}")
        os.chmod(tmp_name, 0o755)  # mkstemp creates it private (0600); a normal library is 0755
        os.replace(tmp_name, self.library_path)
        logger.info("built native kernels: %s", self.library_path)
        return self.library_path

    def load(self) -> ctypes.CDLL:
        """Builds first if needed, then loads. ctypes releases the GIL for the whole native call,
        so a running kernel never blocks the asyncio event loop's own thread."""
        if self._is_stale():
            self.build()
        # Idle OpenMP threads sleep instead of busy-spinning between calls. libgomp reads this
        # once when it's first loaded, i.e. right here (PyTorch ships its own separate copy, which
        # is already running and unaffected). Real incident behind it (2026-09-23): spinning
        # native threads competing with PyTorch's own on 4 CPUs is the likely reason a benchmark
        # run pushed this project's laptop into a thermal shutdown.
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        return ctypes.CDLL(str(self.library_path))
