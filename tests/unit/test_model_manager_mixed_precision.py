"""Mixed per-layer float32/bf16 loading for Mistral3TextArchitecture (ModelManager._load's docstring
has the reasoning): this CPU has no hardware bf16, so bf16 for a whole model because full float32
doesn't fit makes every layer pay PyTorch's ~60x slower emulated path; loading as many layers in
float32 as the memory budget fits, bf16 for the rest, gets a proportional speedup. Separate from
test_model_manager_memory.py (near the 250-line cap), as that file's docstring."""

import json
from pathlib import Path

import torch

from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from app.models.manager import ModelManager
from app.runtime.kv_cache import KVCache
from tests.tiny_gguf import build_tiny_mistral3_gguf


def _install_multi_layer_tiny_model(models_dir: Path, tag: str, n_layer: int) -> InstalledModel:
    repo_dir = models_dir / tag.replace(":", "_")
    repo_dir.mkdir(parents=True)
    gguf_path = repo_dir / "model.gguf"
    build_tiny_mistral3_gguf(gguf_path, n_layer=n_layer)

    installed = InstalledModel(
        tag=tag,
        path=str(gguf_path),
        architecture="mistral3",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="mistral3",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar.write_text(json.dumps(installed.__dict__))
    return installed


def _layer_dtypes(architecture) -> list[torch.dtype]:
    return [layer.self_attn.q_proj.weight.dtype for layer in architecture.layers]


def _patch_memory_state(monkeypatch, *, available_bytes: int, cpu_accelerates_bf16: bool) -> None:
    """`app.models.load_dtype` (select_load_dtype and plan_mixed_precision_load resolve these names
    in their own globals) and `app.models.memory_guard` (its own `from app.models.load_dtype import
    available_memory_bytes` binding: the module-split gotcha) would silently disagree about
    "current" available memory unless every binding is patched together."""
    monkeypatch.setattr("app.models.memory_guard.available_memory_bytes", lambda: available_bytes)
    monkeypatch.setattr("app.models.load_dtype.available_memory_bytes", lambda: available_bytes)
    monkeypatch.setattr("app.models.load_dtype.cpu_accelerates_bf16", lambda: cpu_accelerates_bf16)


class TestMixedPrecisionLoading:
    def test_loads_as_many_layers_float32_as_fit_the_budget_rest_bf16(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """4 equal decoder layers (592 elements each -> 2368 float32 bytes/layer) plus one non-layer
        group (token_embd + output_norm, 2168 elements -> 8672 float32 bytes), numbers from
        tests/tiny_gguf.py's fixed dimensions. A 14000-byte budget (safety_margin=1.0) fits the
        non-layer group plus the first 2 layers (8672 + 2*2368 = 13408 <= 14000 < 15776 = 8672 +
        3*2368): an exact deterministic 2-float32/2-bf16 split, not "some layers differ"."""
        models_dir = tmp_path / "models"
        installed = _install_multi_layer_tiny_model(models_dir, "mixed:latest", n_layer=4)
        manager = ModelManager(
            ModelCatalog(models_dir), memory_safety_margin=1.0, enable_mixed_precision_loading=True
        )
        _patch_memory_state(monkeypatch, available_bytes=14000, cpu_accelerates_bf16=False)

        handle = manager.get_or_load(installed.tag)
        architecture = handle.architecture

        assert architecture.token_embd.weight.dtype == torch.float32
        assert _layer_dtypes(architecture) == [
            torch.float32,
            torch.float32,
            torch.bfloat16,
            torch.bfloat16,
        ]

    def test_disabled_by_default_even_when_it_would_otherwise_fire(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """End-to-end verification found this regresses generation speed on the target
        hardware (Settings.enable_mixed_precision_loading). `ModelManager`'s default constructor (no
        explicit `enable_mixed_precision_loading`) must not mix precision even where it would fire,
        matching dependencies.py's default-off wiring."""
        models_dir = tmp_path / "models"
        installed = _install_multi_layer_tiny_model(models_dir, "mixed:latest", n_layer=4)
        manager = ModelManager(ModelCatalog(models_dir), memory_safety_margin=1.0)
        _patch_memory_state(monkeypatch, available_bytes=14000, cpu_accelerates_bf16=False)

        handle = manager.get_or_load(installed.tag)
        architecture = handle.architecture

        assert architecture.token_embd.weight.dtype == torch.bfloat16
        assert _layer_dtypes(architecture) == [torch.bfloat16] * 4

    def test_forward_pass_succeeds_across_a_float32_bf16_layer_boundary(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A forward through mixed-dtype layers must not raise a dtype-mismatch RuntimeError:
        Mistral3TextArchitecture._forward_impl (RMSNorm preserves its *input*'s dtype, not its
        weight's, so the residual stream crossing a float32-to-bf16 boundary needs an explicit cast)
        and GroupedQueryAttention.forward's kv_cache-dtype comment (a real KVCache, not a cache-less
        forward, exercises the k/v side of the fix)."""
        models_dir = tmp_path / "models"
        installed = _install_multi_layer_tiny_model(models_dir, "mixed:latest", n_layer=4)
        manager = ModelManager(
            ModelCatalog(models_dir), memory_safety_margin=1.0, enable_mixed_precision_loading=True
        )
        _patch_memory_state(monkeypatch, available_bytes=14000, cpu_accelerates_bf16=False)

        handle = manager.get_or_load(installed.tag)
        architecture = handle.architecture
        assert _layer_dtypes(architecture) == [
            torch.float32,
            torch.float32,
            torch.bfloat16,
            torch.bfloat16,
        ]

        kv_cache = KVCache(
            layer_shapes=architecture.kv_cache_layer_shapes,
            max_seq_len=16,
            dtype=torch.float32,
        )
        input_ids = torch.tensor([[1, 2, 3]])
        position_ids = torch.arange(3, dtype=torch.long)

        with torch.no_grad():
            logits = architecture.forward(input_ids, kv_cache, position_ids)

        assert logits.shape == (1, 3, architecture.vocab_size)
        assert torch.isfinite(logits).all()

    def test_uniform_bf16_when_cpu_accelerates_bf16(self, tmp_path: Path, monkeypatch) -> None:
        """The mixed-precision branch must not fire on hardware that *can* accelerate bf16: bf16 is
        already the fast, correct choice there. Same tight budget as above but
        cpu_accelerates_bf16 True: every layer comes back uniformly bf16, as before this
        feature."""
        models_dir = tmp_path / "models"
        installed = _install_multi_layer_tiny_model(models_dir, "mixed:latest", n_layer=4)
        manager = ModelManager(ModelCatalog(models_dir), memory_safety_margin=1.0)
        _patch_memory_state(monkeypatch, available_bytes=14000, cpu_accelerates_bf16=True)

        handle = manager.get_or_load(installed.tag)
        architecture = handle.architecture

        assert architecture.token_embd.weight.dtype == torch.bfloat16
        assert _layer_dtypes(architecture) == [torch.bfloat16] * 4
