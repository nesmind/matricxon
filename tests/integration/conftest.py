import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_model_catalog, get_model_manager
from app.main import MatricxonApp
from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from app.models.manager import ModelManager
from tests.tiny_gguf import build_tiny_mistral3_gguf
from tests.tiny_gguf_bert import build_tiny_bert_gguf
from tests.tiny_gguf_command_r import build_tiny_command_r_gguf
from tests.tiny_gguf_falcon import build_tiny_falcon_gguf
from tests.tiny_gguf_gemma4 import build_tiny_gemma4_gguf
from tests.tiny_gguf_granite import build_tiny_granite_gguf
from tests.tiny_gguf_granitemoe import build_tiny_granitemoe_gguf
from tests.tiny_gguf_llama import build_tiny_llama_gguf
from tests.tiny_gguf_llama_moe import build_tiny_llama_moe_gguf
from tests.tiny_gguf_nemotron_h import build_tiny_nemotron_h_gguf
from tests.tiny_gguf_phi2 import build_tiny_phi2_gguf
from tests.tiny_gguf_qwen2 import build_tiny_qwen2_gguf
from tests.tiny_gguf_qwen3 import build_tiny_qwen3_gguf
from tests.tiny_gguf_qwen35 import build_tiny_qwen35_gguf
from tests.tiny_gguf_starcoder2 import build_tiny_starcoder2_gguf

# The already-downloaded ministral-3:3b GGUF from pAIring's Ollama blob store
# (tests/fixtures/README.md), symlinked (never copied) into a test models dir by
# `real_ministral_model`, so tests need no ~2GB-per-run copy and can't touch the file's bytes.
# Updated: pAIring re-pulled it under a new content hash (same tag/architecture/quant
# scheme, confirmed by a GGUF header read). The earlier hash (sha256-910e4bf4...) in ROADMAP.md's
# M2/M3 entries is history, not stale documentation; only this fixture path needed updating.
REAL_MINISTRAL_BLOB = Path(
    "/home/home/Code/Py/AI/pAIring/models/blobs/"
    "sha256-9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8"
)


class FakeInstalledModelFactory:
    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir

    def create(
        self,
        tag: str = "test-model:latest",
        architecture: str = "mistral3",
        capabilities: list[str] | None = None,
    ) -> InstalledModel:
        # Keyed off the tag (not a fixed "test-repo" constant) so several create calls in one test
        # (copy/create-router tests, a future max_loaded_models>2 eviction test) get distinct files
        # instead of overwriting each other's blob + sidecar.
        repo_dir = self._models_dir / "hf.co" / "test-org" / tag.replace("/", "_").replace(":", "_")
        repo_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = repo_dir / "test-model.gguf"
        # A real (if tiny) GGUF, not an empty stub: /api/show reads tensor shapes to compute
        # `estimated_ram_gb` (app/routers/show_router.py), which needs a parseable file. Always
        # mistral3-shaped whatever `architecture` is passed: callers only use this factory's
        # mistral3 default, and nothing cross-checks the file's architecture against the sidecar's.
        build_tiny_mistral3_gguf(gguf_path)

        installed = InstalledModel(
            tag=tag,
            path=str(gguf_path),
            architecture=architecture,
            capabilities=capabilities or ["completion"],
            size_bytes=0,
            family=architecture,
            parameter_size="3B",
            context_length=2048,
        )
        sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
        sidecar_path.write_text(json.dumps(installed.__dict__))
        return installed


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


@pytest.fixture
def model_factory(models_dir: Path) -> FakeInstalledModelFactory:
    return FakeInstalledModelFactory(models_dir)


@pytest.fixture
def model_manager(models_dir: Path) -> ModelManager:
    return ModelManager(ModelCatalog(models_dir))


@pytest.fixture
def client(models_dir: Path, model_manager: ModelManager) -> TestClient:
    app = MatricxonApp().build()
    app.dependency_overrides[get_model_catalog] = lambda: ModelCatalog(models_dir)
    # get_model_manager is @lru_cache'd and would build its own ModelManager around the real
    # (un-overridden) get_model_catalog, which it calls directly, not through FastAPI's DI;
    # overriding the catalog above wouldn't reach it without this too.
    app.dependency_overrides[get_model_manager] = lambda: model_manager
    return TestClient(app)


@pytest.fixture
def real_ministral_model(models_dir: Path) -> InstalledModel:
    """A real (symlinked, not copied) GGUF fixture with a hand-written but metadata-accurate
    sidecar: M5's "hand-placed real fixture files + sidecars (no pull yet)", versus the synthetic
    0-byte files `FakeInstalledModelFactory` uses. Skips without pAIring's blob store (e.g. CI)."""
    if not REAL_MINISTRAL_BLOB.exists():
        pytest.skip(f"real fixture not present on this machine: {REAL_MINISTRAL_BLOB}")

    repo_dir = models_dir / "registry.ollama.ai" / "library" / "ministral-3"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "3b.gguf"
    gguf_path.symlink_to(REAL_MINISTRAL_BLOB)

    installed = InstalledModel(
        tag="ministral-3:3b",
        path=str(gguf_path),
        architecture="mistral3",
        capabilities=["completion"],
        size_bytes=REAL_MINISTRAL_BLOB.stat().st_size,
        family="mistral3",
        parameter_size="3.8B",
        context_length=262144,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_mistral3_model(models_dir: Path) -> InstalledModel:
    """A tiny, complete, valid `mistral3` GGUF (tiny_gguf.py) that, unlike
    `FakeInstalledModelFactory`'s 0-byte files, loads through the real
    GGUFModelLoader/architecture/tokenizer pipeline, so `/api/chat` tests run the full stack fast
    and memory-safely."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-mistral3"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_mistral3_gguf(gguf_path)

    installed = InstalledModel(
        tag="tiny-mistral3:latest",
        path=str(gguf_path),
        architecture="mistral3",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="mistral3",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_bert_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `bert` GGUF (tiny_gguf_bert.py): the embeddings-side equivalent of
    `tiny_mistral3_model`, for fast `/api/embeddings` tests against
    BertArchitecture/WordPieceTokenizer/EmbeddingEngine."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-bert"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_bert_gguf(gguf_path)

    installed = InstalledModel(
        tag="tiny-bert:latest",
        path=str(gguf_path),
        architecture="bert",
        capabilities=["embedding"],
        size_bytes=gguf_path.stat().st_size,
        family="bert",
        parameter_size="0.001B",
        context_length=16,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_gemma4_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `gemma4` GGUF (tiny_gguf_gemma4.py) exercising local/sliding and global layer
    types, different head_dim per type, the `rope_freqs.weight` correction and final-logit
    softcapping together. Proves the wiring, not numerical correctness against the real
    `google/gemma-4-12b-it` (too large for this hardware)."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-gemma4"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_gemma4_gguf(gguf_path)

    installed = InstalledModel(
        tag="tiny-gemma4:latest",
        path=str(gguf_path),
        architecture="gemma4",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="gemma4",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_phi2_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `phi2` GGUF (tiny_gguf_phi2.py) exercising the fused-qkv split and
    partial-rotary handling (Phi-2's novelties; app/architectures/phi2.py), for fast `/api/chat`
    tests against Phi2Architecture/GGUFTokenizer."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-phi2"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_phi2_gguf(gguf_path)

    installed = InstalledModel(
        tag="tiny-phi2:latest",
        path=str(gguf_path),
        architecture="phi2",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="phi2",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_llama_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `llama` GGUF (tiny_gguf_llama.py): M10's plain-architecture equivalent of
    `tiny_mistral3_model`, for `/api/chat` tests against
    LlamaArchitecture/SentencePieceTokenizer."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-llama"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_llama_gguf(gguf_path)

    installed = InstalledModel(
        tag="tiny-llama:latest",
        path=str(gguf_path),
        architecture="llama",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="llama",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_granite_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `granite` GGUF (tests/tiny_gguf_granite.py): the dense-decoder equivalent of
    `tiny_llama_model`, for `/api/chat` tests against GraniteArchitecture/GGUFTokenizer, including
    its four "Power"-scaling multipliers."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-granite"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_granite_gguf(gguf_path)

    installed = InstalledModel(
        tag="tiny-granite:latest",
        path=str(gguf_path),
        architecture="granite",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="granite",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_granite_tied_embeddings_model(models_dir: Path) -> InstalledModel:
    """`tiny_granite_model` with no separate `output.weight`: the shape of real Granite checkpoints
    (`granite-3.0-2b-instruct`/`granite-3.0-1b-a400m-instruct` ship `tie_word_embeddings: true`),
    the *common* case (GraniteArchitecture)."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-granite-tied"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_granite_gguf(gguf_path, tied_embeddings=True)

    installed = InstalledModel(
        tag="tiny-granite-tied:latest",
        path=str(gguf_path),
        architecture="granite",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="granite",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_granitemoe_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `granitemoe` GGUF (tests/tiny_gguf_granitemoe.py): the sparse-MoE equivalent of
    `tiny_granite_model`, for `/api/chat` tests against
    GraniteMoeArchitecture/GraniteMoeFFN/GGUFTokenizer."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-granitemoe"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_granitemoe_gguf(gguf_path)

    installed = InstalledModel(
        tag="tiny-granitemoe:latest",
        path=str(gguf_path),
        architecture="granitemoe",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="granitemoe",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_granitemoe_tied_embeddings_model(models_dir: Path) -> InstalledModel:
    """`tiny_granitemoe_model` with no separate `output.weight`: real GraniteMoE checkpoints
    (`granite-3.0-1b-a400m-instruct`/`granite-3.0-3b-a800m-instruct`) ship `tie_word_embeddings:
    true`, like dense Granite."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-granitemoe-tied"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_granitemoe_gguf(gguf_path, tied_embeddings=True)

    installed = InstalledModel(
        tag="tiny-granitemoe-tied:latest",
        path=str(gguf_path),
        architecture="granitemoe",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="granitemoe",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_nemotron_h_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `nemotron_h` GGUF (tests/tiny_gguf_nemotron_h.py) exercising the hybrid
    Mamba-2/attention/MLP per-layer dispatch and `NemotronHHybridCache` end to end, for `/api/chat`
    tests against NemotronHArchitecture/GGUFTokenizer."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-nemotron-h"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_nemotron_h_gguf(gguf_path)

    installed = InstalledModel(
        tag="tiny-nemotron-h:latest",
        path=str(gguf_path),
        architecture="nemotron_h",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="nemotron_h",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_nemotron_h_tied_embeddings_model(models_dir: Path) -> InstalledModel:
    """`tiny_nemotron_h_model` with no separate `output.weight`."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-nemotron-h-tied"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_nemotron_h_gguf(gguf_path, tied_embeddings=True)

    installed = InstalledModel(
        tag="tiny-nemotron-h-tied:latest",
        path=str(gguf_path),
        architecture="nemotron_h",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="nemotron_h",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_qwen2_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `qwen2` GGUF (tests/tiny_gguf_qwen2.py) exercising q/k/v bias plus the
    `unpermute_rope_rows`-on-bias handling, for `/api/chat` tests against
    Qwen2Architecture/GGUFTokenizer."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-qwen2"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_qwen2_gguf(gguf_path)

    installed = InstalledModel(
        tag="tiny-qwen2:latest",
        path=str(gguf_path),
        architecture="qwen2",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="qwen2",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_qwen2_tied_embeddings_model(models_dir: Path) -> InstalledModel:
    """`tiny_qwen2_model` with no separate `output.weight`: Qwen2.5 checkpoints up to 3B tie
    embeddings (7B+ don't)."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-qwen2-tied"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_qwen2_gguf(gguf_path, tied_embeddings=True)

    installed = InstalledModel(
        tag="tiny-qwen2-tied:latest",
        path=str(gguf_path),
        architecture="qwen2",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="qwen2",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_qwen3_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `qwen3` GGUF (tests/tiny_gguf_qwen3.py) exercising QK-norm plus the
    `unpermute_rope_rows(..., n_head=1)` handling of its weight tensors, for `/api/chat` tests
    against Qwen3Architecture/GGUFTokenizer."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-qwen3"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_qwen3_gguf(gguf_path)

    installed = InstalledModel(
        tag="tiny-qwen3:latest",
        path=str(gguf_path),
        architecture="qwen3",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="qwen3",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_qwen3_tied_embeddings_model(models_dir: Path) -> InstalledModel:
    """`tiny_qwen3_model` with no separate `output.weight`: Qwen3-0.6B ships `tie_word_embeddings:
    true`."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-qwen3-tied"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_qwen3_gguf(gguf_path, tied_embeddings=True)

    installed = InstalledModel(
        tag="tiny-qwen3-tied:latest",
        path=str(gguf_path),
        architecture="qwen3",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="qwen3",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_command_r_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `command-r` GGUF (tests/tiny_gguf_command_r.py) exercising the parallel
    attention+FFN block, bias-free LayerNorm and logit_scale multiply, for `/api/chat` tests against
    CommandRArchitecture/GGUFTokenizer."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-command-r"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    # seed=1, not 0: seed=0's random weights make greedy decoding hit EOS after 2 tokens for this
    # architecture's logit_scale/parallel-block combination (correct behavior, not a bug) but defeat
    # this fixture's purpose of a multi-token generation.
    build_tiny_command_r_gguf(gguf_path, seed=1)

    installed = InstalledModel(
        tag="tiny-command-r:latest",
        path=str(gguf_path),
        architecture="command-r",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="command-r",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_command_r_tied_embeddings_model(models_dir: Path) -> InstalledModel:
    """`tiny_command_r_model` with no separate `output.weight`: Command-R checkpoints are always
    tied (llama.cpp never emits a separate output tensor), so this is the real-world case, not the
    edge (the lesson of `llama`'s tied-embeddings bug: detect dynamically)."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-command-r-tied"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_command_r_gguf(gguf_path, tied_embeddings=True)

    installed = InstalledModel(
        tag="tiny-command-r-tied:latest",
        path=str(gguf_path),
        architecture="command-r",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="command-r",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_starcoder2_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `starcoder2` GGUF (tests/tiny_gguf_starcoder2.py) exercising biased q/k/v/o, a
    biased plain (non-gated) tanh-GELU MLP, biased LayerNorm and the "no unpermute_rope_rows"
    finding, for `/api/chat` tests against Starcoder2Architecture/GGUFTokenizer."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-starcoder2"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_starcoder2_gguf(gguf_path, seed=1)

    installed = InstalledModel(
        tag="tiny-starcoder2:latest",
        path=str(gguf_path),
        architecture="starcoder2",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="starcoder2",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_starcoder2_tied_embeddings_model(models_dir: Path) -> InstalledModel:
    """`tiny_starcoder2_model` with no separate `output.weight`: `bigcode/starcoder2-3b/-7b/-15b`
    tie embeddings (`tie_word_embeddings=True`), detected dynamically
    (`loader.has_tensor("output.weight")`) like every architecture here."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-starcoder2-tied"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_starcoder2_gguf(gguf_path, tied_embeddings=True)

    installed = InstalledModel(
        tag="tiny-starcoder2-tied:latest",
        path=str(gguf_path),
        architecture="starcoder2",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="starcoder2",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_llama_moe_model(models_dir: Path) -> InstalledModel:
    """A tiny valid Mixtral-style `llama` GGUF (tiny_gguf_llama_moe.py) exercising the sparse MoE
    FFN dispatch through `LlamaArchitecture`'s MoE detection, for `/api/chat` tests.
    `general.architecture` stays `"llama"` (no `mixtral` string; app/architectures/llama_moe.py)."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-llama-moe"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_llama_moe_gguf(gguf_path)

    installed = InstalledModel(
        tag="tiny-llama-moe:latest",
        path=str(gguf_path),
        architecture="llama",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="llama",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_falcon_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `falcon` GGUF (tests/tiny_gguf_falcon.py) exercising the fused bias-free MQA
    `attn_qkv` split and the single biased parallel-residual norm, for `/api/chat` tests against
    FalconArchitecture/GGUFTokenizer."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-falcon"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_falcon_gguf(gguf_path, seed=1)

    installed = InstalledModel(
        tag="tiny-falcon:latest",
        path=str(gguf_path),
        architecture="falcon",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="falcon",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_falcon_tied_embeddings_model(models_dir: Path) -> InstalledModel:
    """`tiny_falcon_model` with no separate `output.weight`. `tiiuae/falcon-7b` always ships one
    (untied, checked on a real GGUF header), so this is the edge case, detected dynamically
    (`loader.has_tensor("output.weight")`) like every architecture."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-falcon-tied"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_falcon_gguf(gguf_path, tied_embeddings=True)

    installed = InstalledModel(
        tag="tiny-falcon-tied:latest",
        path=str(gguf_path),
        architecture="falcon",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="falcon",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_llama_tied_embeddings_model(models_dir: Path) -> InstalledModel:
    """`tiny_llama_model` with no separate `output.weight`: the shape of a
    `Llama-3.2-1B/3B-Instruct` GGUF (`tie_word_embeddings: true`), which crashed `/api/chat` with
    `UnknownModelError: GGUF file has no tensor named 'output.weight'` before `LlamaArchitecture`
    learned tied embeddings."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-llama-tied"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_llama_gguf(gguf_path, tied_embeddings=True)

    installed = InstalledModel(
        tag="tiny-llama-tied:latest",
        path=str(gguf_path),
        architecture="llama",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="llama",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


@pytest.fixture
def tiny_qwen35_model(models_dir: Path) -> InstalledModel:
    """A tiny valid `qwen35` GGUF (tests/tiny_gguf_qwen35.py) exercising the hybrid Gated
    DeltaNet/gated-attention stack and its recurrent cache through `/api/chat`."""
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-qwen35"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "tiny.gguf"
    build_tiny_qwen35_gguf(gguf_path)

    installed = InstalledModel(
        tag="tiny-qwen35:latest",
        path=str(gguf_path),
        architecture="qwen35",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="qwen35",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed
