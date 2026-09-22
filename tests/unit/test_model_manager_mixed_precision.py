"""Mixed per-layer float32/bf16 loading for Mistral3TextArchitecture - see
ModelManager._load's own docstring for the full reasoning (this CPU has no hardware bf16
acceleration, so loading a whole model in bf16 because full float32 doesn't fit means every
layer pays PyTorch's ~60x-slower emulated bf16 path; loading as many individual layers in
float32 as actually fit the memory budget, bf16 for the rest, gets a proportional speedup
instead of an all-or-nothing one). A separate file from test_model_manager_memory.py (already
near the project's 250-line cap) rather than growing it further - same split-for-size
convention that file's own docstring already documents.
"""

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
    """`app.models.load_dtype` (both `select_load_dtype` and `plan_mixed_precision_load` resolve
    these two names against their own module's globals) and `app.models.memory_guard` (its own,
    separate `from app.models.load_dtype import available_memory_bytes` binding - see the
    project's own "module-split import-binding gotcha" precedent) would otherwise silently
    disagree about "current" available memory unless every real binding is patched together."""
    monkeypatch.setattr("app.models.memory_guard.available_memory_bytes", lambda: available_bytes)
    monkeypatch.setattr("app.models.load_dtype.available_memory_bytes", lambda: available_bytes)
    monkeypatch.setattr(
        "app.models.load_dtype.cpu_accelerates_bf16", lambda: cpu_accelerates_bf16
    )


class TestMixedPrecisionLoading:
    def test_loads_as_many_layers_float32_as_fit_the_budget_rest_bf16(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """4 equal-sized decoder layers (592 elements each -> 2368 float32 bytes/layer) plus one
        non-layer group (token_embd + output_norm, 2168 elements -> 8672 float32 bytes) - real
        numbers derived from tests/tiny_gguf.py's own fixed dimensions, not guessed. A 14000-byte
        budget (safety_margin=1.0, for clean arithmetic) fits the non-layer group plus exactly the
        first 2 of 4 layers (8672 + 2*2368 = 13408 <= 14000 < 15776 = 8672 + 3*2368), so this
        pins an exact, deterministic 2-float32/2-bf16 split, not just "some layers differ".
        """
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
        """Real end-to-end verification (2026-09-20) found this feature regresses generation speed
        on this project's target hardware - see Settings.enable_mixed_precision_loading's own
        docstring. `ModelManager`'s default constructor (no explicit
        `enable_mixed_precision_loading`) must not mix precision even under the exact memory/CPU
        conditions that would otherwise trigger it, matching dependencies.py's own default-off
        wiring."""
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
        """The real point of this test: a forward pass through mixed-dtype layers must not raise
        a dtype-mismatch RuntimeError - see Mistral3TextArchitecture._forward_impl's own docstring
        (RMSNorm preserves its *input*'s dtype, not necessarily its own weight's, so the residual
        stream crossing a float32-to-bf16 layer boundary needs an explicit cast) and
        GroupedQueryAttention.forward's kv_cache-dtype comment for why a real KVCache (not just a
        cache-less forward call) is needed to exercise the k/v-side of that same fix.
        """
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
        ]  # sanity: the boundary this test exists to cross actually exists

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
        """The mixed-precision branch must not fire on hardware that *can* accelerate bf16 - there
        bf16 is already the fast, correct choice for the whole model, so mixing in float32 would
        be pointless work at best. Same tight budget as the tests above, but with
        cpu_accelerates_bf16() True this time - every layer should come back uniformly bf16, the
        exact behavior from before this feature existed."""
        models_dir = tmp_path / "models"
        installed = _install_multi_layer_tiny_model(models_dir, "mixed:latest", n_layer=4)
        manager = ModelManager(ModelCatalog(models_dir), memory_safety_margin=1.0)
        _patch_memory_state(monkeypatch, available_bytes=14000, cpu_accelerates_bf16=True)

        handle = manager.get_or_load(installed.tag)
        architecture = handle.architecture

        assert architecture.token_embd.weight.dtype == torch.bfloat16
        assert _layer_dtypes(architecture) == [torch.bfloat16] * 4
