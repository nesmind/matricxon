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
from tests.tiny_gguf_gemma4 import build_tiny_gemma4_gguf
from tests.tiny_gguf_llama import build_tiny_llama_gguf
from tests.tiny_gguf_phi2 import build_tiny_phi2_gguf

# The real, already-downloaded ministral-3:3b GGUF from pAIring's Ollama blob
# store - see tests/fixtures/README.md. Symlinked (never copied) into a test
# models dir by `real_ministral_model` below, so real-fixture tests don't
# need a ~2GB-per-run copy and can't ever touch this shared file's bytes.
# Updated 2026-09-19: pAIring re-pulled this model under a new content hash
# (same tag/architecture/quant scheme, confirmed via a real GGUF header read
# before updating this path, not assumed) - the previous hash
# (sha256-910e4bf4...) referenced in ROADMAP.md's M2/M3/etc. entries is a
# historical record of what was true at the time, not stale documentation to
# "fix"; only this current-state fixture path needed updating.
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
        # Keyed off the tag (not a fixed "test-repo" constant) so multiple
        # create() calls in one test - e.g. copy/create-router tests, or a
        # future max_loaded_models>2 eviction test - get distinct files
        # instead of silently overwriting each other's blob + sidecar.
        repo_dir = self._models_dir / "hf.co" / "test-org" / tag.replace("/", "_").replace(":", "_")
        repo_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = repo_dir / "test-model.gguf"
        # A real (if tiny) GGUF, not an empty stub - /api/show reads real tensor shapes to compute
        # `estimated_ram_gb` (see app/routers/show_router.py), which needs an actually-parseable
        # file. Always mistral3-shaped regardless of the real `architecture` string passed in -
        # every existing caller already only ever uses this factory's own mistral3 default, and
        # nothing here cross-checks the file's real architecture against the sidecar's stated one.
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
    # get_model_manager() is @lru_cache'd and would otherwise build its own
    # ModelManager around the real (un-overridden) get_model_catalog() - it
    # calls that directly, not through FastAPI's DI, so overriding the
    # catalog above wouldn't propagate into it without this too.
    app.dependency_overrides[get_model_manager] = lambda: model_manager
    return TestClient(app)


@pytest.fixture
def real_ministral_model(models_dir: Path) -> InstalledModel:
    """A real (symlinked, not copied) GGUF fixture with a hand-written but
    metadata-accurate sidecar - M5's "hand-placed real fixture files +
    sidecars (no pull yet)" scope, as opposed to the synthetic 0-byte files
    `FakeInstalledModelFactory` uses above. Skips on a machine without
    pAIring's local blob store (e.g. CI) rather than failing outright.
    """
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
    """A tiny but complete, valid, real `mistral3` GGUF (see tiny_gguf.py) -
    unlike `FakeInstalledModelFactory`'s 0-byte files, this actually loads
    through the real GGUFModelLoader/architecture/tokenizer pipeline, so
    `/api/chat` tests can exercise the full stack fast and memory-safely
    without needing gigabytes of real weights.
    """
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
    """A tiny but complete, valid, real `bert` GGUF (see tiny_gguf_bert.py) -
    the embeddings-side equivalent of `tiny_mistral3_model`, for fast/safe
    `/api/embeddings` tests against the real BertArchitecture/
    WordPieceTokenizer/EmbeddingEngine pipeline.
    """
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
    """A tiny but complete, valid, real `gemma4` GGUF (see tiny_gguf_gemma4.py)

    - exercises both local/sliding and global layer types, different
    head_dim per type, the `rope_freqs.weight` correction, and final-logit
    softcapping, all in one tiny model. Proves the wiring, not numerical
    correctness against the real (impractically large for this hardware,
    see ROADMAP.md) `google/gemma-4-12b-it` model.
    """
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
    """A tiny but complete, valid, real `phi2` GGUF (see tiny_gguf_phi2.py) - exercises the real
    fused-qkv split and partial-rotary handling (both real Phi-2 novelties, see
    app/architectures/phi2.py's own docstring), for fast/safe `/api/chat` tests against the real
    Phi2Architecture/GGUFTokenizer pipeline."""
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
    """A tiny but complete, valid, real `llama` GGUF (see tiny_gguf_llama.py)

    - M10's plain-architecture equivalent of `tiny_mistral3_model`, for
    fast/safe `/api/chat` tests against the real
    LlamaArchitecture/SentencePieceTokenizer pipeline.
    """
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
def tiny_llama_tied_embeddings_model(models_dir: Path) -> InstalledModel:
    """Same as `tiny_llama_model`, but with no separate `output.weight` tensor - the real shape of
    a real `Llama-3.2-1B/3B-Instruct` GGUF pull (`tie_word_embeddings: true`), which crashed
    `/api/chat` with `UnknownModelError: GGUF file has no tensor named 'output.weight'` before
    `LlamaArchitecture` learned to detect and handle tied embeddings (2026-09-22)."""
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
