"""Experimental GPU mode, checked without a GPU: the `meta` device stands in for CUDA (it is a
non-CPU device with no data), so placement, caches and masks are exercised, not real kernels."""

import json
from pathlib import Path

import pytest
import torch

from app.architectures.mistral3_layers import _causal_mask
from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from app.models.manager import ModelManager
from app.models.memory_guard import ensure_enough_memory_to_load
from app.runtime.batch_cache import BatchedKVCache
from app.runtime.compute_device import ComputeDevice
from app.runtime.kv_cache import KVCache
from app.server.errors import DeviceUnavailableError, InsufficientMemoryError
from tests.tiny_gguf import build_tiny_mistral3_gguf

META = torch.device("meta")


class FakeGpu(ComputeDevice):
    """A 'GPU' backed by the meta device with a configurable amount of free VRAM."""

    def __init__(self, free_bytes: int = 1 << 40) -> None:
        super().__init__(META)
        self._free = free_bytes

    def load_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def free_memory_bytes(self) -> int | None:
        return self._free


def _install(models_dir: Path, tag: str = "m:latest") -> None:
    repo = models_dir / tag.replace(":", "_")
    repo.mkdir(parents=True)
    gguf = repo / "model.gguf"
    build_tiny_mistral3_gguf(gguf)
    installed = InstalledModel(
        tag=tag, path=str(gguf), architecture="mistral3", capabilities=["completion"],
        size_bytes=gguf.stat().st_size, family="mistral3", parameter_size="0.001B",
        context_length=32,
    )  # fmt: skip
    (repo / f"{gguf.stem}{ModelCatalog.SIDECAR_SUFFIX}").write_text(json.dumps(installed.__dict__))


class TestComputeDevice:
    def test_cpu_resolves(self) -> None:
        device = ComputeDevice.resolve("cpu")
        assert not device.is_gpu and device.free_memory_bytes() is None

    def test_cuda_without_a_gpu_raises_instead_of_falling_back(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(DeviceUnavailableError, match="no CUDA GPU"):
            ComputeDevice.resolve("cuda")

    def test_out_of_range_index_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        with pytest.raises(DeviceUnavailableError, match="only 1"):
            ComputeDevice.resolve("cuda:3")

    @pytest.mark.parametrize("spec", ["mps", "nonsense:::"])
    def test_other_devices_are_rejected(self, spec: str) -> None:
        with pytest.raises(DeviceUnavailableError):
            ComputeDevice.resolve(spec)


class TestDeviceAwareTensors:
    def test_kv_cache_allocates_and_forks_on_its_device(self) -> None:
        cache = KVCache([(2, 4)], max_seq_len=8, device=META)
        assert cache.device == META and cache._k[0].device == META
        assert cache.fork(0).device == META

    def test_batched_length_follows_the_caches(self) -> None:
        caches = [KVCache([(1, 2)], 4, device=META)]
        assert BatchedKVCache(caches).length.device == META

    def test_causal_mask_is_built_on_the_requested_device(self) -> None:
        assert _causal_mask(2, 3, 1, META).device == META
        assert _causal_mask(2, 3, 1).device.type == "cpu"


class TestMemoryGuardOnVram:
    def test_checks_the_supplied_free_memory(self) -> None:
        with pytest.raises(InsufficientMemoryError, match="VRAM"):
            ensure_enough_memory_to_load(
                "m", 1_000, 1.2, [], available_fn=lambda: 100, memory_kind="VRAM"
            )
        ensure_enough_memory_to_load("m", 1_000, 1.2, [], available_fn=lambda: 10_000)


class TestManagerOnAGpu:
    def _manager(self, tmp_path: Path, monkeypatch, gpu: FakeGpu) -> ModelManager:
        _install(tmp_path / "models")
        monkeypatch.setattr(ComputeDevice, "resolve", classmethod(lambda cls, spec: gpu))
        return ModelManager(ModelCatalog(tmp_path / "models"), device="cuda")

    def test_model_is_placed_on_the_device_with_gpu_dtype(self, tmp_path, monkeypatch) -> None:
        manager = self._manager(tmp_path, monkeypatch, FakeGpu())
        arch = manager.get_or_load("m:latest").architecture
        assert arch.device == META
        assert all(p.device == META for p in arch.parameters())
        assert arch.rope.inv_freq.device == META
        assert arch.build_cache(8, torch.bfloat16).device == META
        assert manager.on_gpu

    def test_load_is_refused_when_vram_is_short(self, tmp_path, monkeypatch) -> None:
        manager = self._manager(tmp_path, monkeypatch, FakeGpu(free_bytes=10))
        with pytest.raises(InsufficientMemoryError, match="VRAM"):
            manager.get_or_load("m:latest")

    def test_architecture_without_gpu_support_is_refused(self, tmp_path, monkeypatch) -> None:
        from app.architectures.mistral3 import Mistral3TextArchitecture

        monkeypatch.setattr(Mistral3TextArchitecture, "SUPPORTS_GPU", False)
        manager = self._manager(tmp_path, monkeypatch, FakeGpu())
        with pytest.raises(DeviceUnavailableError, match="no GPU support"):
            manager.get_or_load("m:latest")


class TestForwardRouting:
    def test_inputs_follow_the_model_and_logits_come_back_to_the_cpu(self) -> None:
        from app.architectures.mistral3 import Mistral3TextArchitecture

        seen: dict[str, torch.device] = {}

        class Probe(Mistral3TextArchitecture):
            def __init__(self) -> None:  # skip building real layers
                torch.nn.Module.__init__(self)
                self._materialized = True
                self._last_logits_only = False
                self._compute_device = META

            def _forward_impl(self, input_ids, kv_cache, position_ids, stop_check, images):
                seen["ids"], seen["pos"] = input_ids.device, position_ids.device
                return torch.zeros(1, 1, 4)

        out = Probe().forward(torch.zeros(1, 1, dtype=torch.long), position_ids=torch.zeros(1))
        assert seen == {"ids": META, "pos": META} and out.device.type == "cpu"


class CpuBackedGpu(FakeGpu):
    """A 'GPU' that really computes (on the CPU device), to run a packed-mode load end to end."""

    def __init__(self) -> None:
        ComputeDevice.__init__(self, torch.device("cpu"))
        self._free = 1 << 40

    def load_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def is_gpu(self) -> bool:
        return True


class TestPackedModeThroughTheManager:
    def test_unpackable_weights_fall_back_to_real_tensors(self, tmp_path, monkeypatch) -> None:
        # The tiny mistral3 file is F32 (no packable type): every placeholder is loaded by the
        # plain fallback path and settled on the device, then a forward must run.
        _install(tmp_path / "models")
        monkeypatch.setattr(ComputeDevice, "resolve", classmethod(lambda c, s: CpuBackedGpu()))
        manager = ModelManager(
            ModelCatalog(tmp_path / "models"), device="cuda", gpu_weight_mode="packed"
        )
        arch = manager.get_or_load("m:latest").architecture
        out = arch(torch.tensor([[1, 2, 3]]))
        assert out.shape[:2] == (1, 3)


class TestPromptCacheBudgetOnVram:
    def test_cpu_keeps_the_setting_and_gpu_is_capped_by_free_vram(self, tmp_path: Path) -> None:
        manager = ModelManager(ModelCatalog(tmp_path), prompt_cache_budget_mb=2048)
        assert manager._cache_budget_bytes(False, 0, 0) == 2048 << 20
        # 10 GB free, 8 GB of weights -> half of the 2 GB left (1 GB), below the 2 GB setting
        assert manager._cache_budget_bytes(True, 10 << 30, 8 << 30) == 1 << 30
        assert manager._cache_budget_bytes(True, 1 << 30, 8 << 30) == 0  # nothing left over


class TestStartupDeviceCheck:
    def test_a_missing_gpu_exits_with_one_log_line_and_no_traceback(
        self, monkeypatch, caplog
    ) -> None:
        from app import config
        from app.main import MatricxonApp

        monkeypatch.setattr(config.settings, "device", "cuda")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(SystemExit) as exit_info:
            MatricxonApp.require_device()
        assert exit_info.value.code == 1
        [record] = [r for r in caplog.records if r.levelname == "CRITICAL"]
        assert "matricxon cannot start: device 'cuda' requested" in record.getMessage()
        assert record.exc_info is None

    def test_cpu_passes(self) -> None:
        from app.main import MatricxonApp

        MatricxonApp.require_device()
