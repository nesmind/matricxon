"""Unit tests for app/native/compiler_flags.py (per-platform build variants) and
app/native/library.py's variant fallback. The macOS cases only check the flags chosen - nothing here
can actually build on a Mac; the fallback test compiles for real with this machine's gcc."""

import shutil

import pytest

from app.native.compiler_flags import BuildVariant, CompilerFlags
from app.native.gemm import NativeGemm
from app.native.library import NativeBuildError, NativeKernelLibrary


def test_linux_tries_openmp_first_then_single_threaded() -> None:
    flags = CompilerFlags(system="Linux", machine="x86_64")

    variants = flags.variants()

    assert [v.name for v in variants] == ["openmp", "single-threaded"]
    assert "-fopenmp" in variants[0].flags
    assert "-march=native" in variants[0].flags
    assert "-fopenmp" not in variants[1].flags
    assert flags.default_compiler() == "gcc"


def test_apple_silicon_without_libomp_builds_single_threaded_with_mcpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(CompilerFlags, "_libomp_prefix", staticmethod(lambda: None))
    flags = CompilerFlags(system="Darwin", machine="arm64")

    variants = flags.variants()

    assert [v.name for v in variants] == ["single-threaded"]
    assert "-mcpu=native" in variants[0].flags
    assert "-march=native" not in variants[0].flags
    assert flags.default_compiler() == "cc"


def test_mac_with_homebrew_libomp_uses_xpreprocessor_openmp(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(CompilerFlags, "_libomp_prefix", staticmethod(lambda: tmp_path))
    variants = CompilerFlags(system="Darwin", machine="x86_64").variants()

    assert variants[0].name == "openmp"
    assert ("-Xpreprocessor", "-fopenmp") == variants[0].flags[-3:-1]
    assert variants[0].link_flags == (f"-L{tmp_path}/lib", "-lomp")
    assert "-march=native" in variants[0].flags  # Intel Macs keep -march=native


class _FixedVariants(CompilerFlags):
    def __init__(self, variants: list[BuildVariant]) -> None:
        super().__init__(system="Linux", machine="x86_64")
        self._fixed = variants

    def variants(self) -> list[BuildVariant]:
        return self._fixed


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs a C compiler")
def test_a_failing_variant_falls_back_to_the_next_one(tmp_path) -> None:
    base = ("-O3", "-fPIC", "-shared", "-march=native")
    library = NativeKernelLibrary(
        build_dir=tmp_path,
        compiler="gcc",
        flags=_FixedVariants(
            [
                BuildVariant("broken", (*base, "--no-such-compiler-flag")),
                BuildVariant("single-threaded", base),
            ]
        ),
    )

    gemm = NativeGemm(library.load(), n_threads=2)

    assert "openmp=no" in gemm.build_info  # really the single-threaded build that loaded


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs a C compiler")
def test_every_variant_failing_reports_each_one(tmp_path) -> None:
    library = NativeKernelLibrary(
        build_dir=tmp_path,
        compiler="gcc",
        flags=_FixedVariants(
            [BuildVariant("first", ("--bad-flag-1",)), BuildVariant("second", ("--bad-flag-2",))]
        ),
    )

    with pytest.raises(NativeBuildError, match=r"\[first\][\s\S]*\[second\]"):
        library.build()
    assert not list(tmp_path.glob("*.tmp"))  # no half-written temp library left behind
