"""Which compiler and flags to build the native kernels with, per platform - split out of
app/native/library.py so the build itself stays a plain "try each variant in order" loop.

- Linux (gcc): `-march=native -fopenmp` - the only setup actually built on so far (this project's
  own machine).
- macOS: `gcc` is Apple's clang there, which rejects a bare `-fopenmp`; OpenMP needs Homebrew's
  `libomp` passed through `-Xpreprocessor`. Apple Silicon also wants `-mcpu=native` (some Apple
  clang versions reject `-march=native` on arm64). Written against Apple's and Homebrew's
  documented flags, NOT yet built on a real Mac.
- Every platform gets a single-threaded fallback last: without OpenMP the `#pragma omp` lines are
  ignored, so the kernels still build - one thread instead of several, but still native (far
  faster than the Numba kernels on the same core) instead of dropping back to Numba altogether.
"""

import platform
from dataclasses import dataclass
from pathlib import Path

_BASE = ("-O3", "-fPIC", "-shared", "-Wall")
# Homebrew's libomp prefix on Apple Silicon, then on Intel Macs.
_LIBOMP_PREFIXES = (Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp"))


@dataclass(frozen=True)
class BuildVariant:
    name: str
    flags: tuple[str, ...]
    link_flags: tuple[str, ...] = ()


class CompilerFlags:
    def __init__(self, system: str | None = None, machine: str | None = None) -> None:
        self._system = system or platform.system()
        self._machine = (machine or platform.machine()).lower()

    def default_compiler(self) -> str:
        """`cc` on macOS (always present with the Xcode command line tools), `gcc` elsewhere."""
        return "cc" if self._system == "Darwin" else "gcc"

    def _arch_flags(self) -> tuple[str, ...]:
        if self._system == "Darwin" and self._machine in ("arm64", "aarch64"):
            return ("-mcpu=native",)
        return ("-march=native",)

    @staticmethod
    def _libomp_prefix() -> Path | None:
        return next((p for p in _LIBOMP_PREFIXES if (p / "lib").is_dir()), None)

    def variants(self) -> list[BuildVariant]:
        """Build variants to try, best first."""
        flags = _BASE + self._arch_flags()
        result = []
        if self._system == "Darwin":
            prefix = self._libomp_prefix()
            if prefix is not None:
                result.append(
                    BuildVariant(
                        "openmp",
                        (*flags, "-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"),
                        (f"-L{prefix}/lib", "-lomp"),
                    )
                )
        else:
            result.append(BuildVariant("openmp", (*flags, "-fopenmp")))
        result.append(BuildVariant("single-threaded", flags))
        return result
