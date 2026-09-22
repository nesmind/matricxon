import json
from pathlib import Path

import pytest

from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from app.server.errors import UnknownModelError


def _install(catalog: ModelCatalog, tag: str, content: bytes = b"weights") -> InstalledModel:
    repo_dir = catalog.models_dir / "hf.co" / "test-org" / tag.replace(":", "_")
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = repo_dir / "model.gguf"
    gguf_path.write_bytes(content)

    installed = InstalledModel(
        tag=tag,
        path=str(gguf_path),
        architecture="mistral3",
        capabilities=["completion"],
        size_bytes=len(content),
        family="mistral3",
        parameter_size="3B",
        context_length=2048,
    )
    sidecar_path = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar_path.write_text(json.dumps(installed.__dict__))
    return installed


class TestLoadSidecar:
    def test_list_installed_uses_the_sidecars_own_location_not_its_stale_stored_path(
        self, tmp_path: Path
    ) -> None:
        """Real bug found live (2026-09-22): moving a whole models_dir by hand (e.g. re-pointing
        pAIring's own admin-configurable Matricxon models path and `mv`ing the hf.co/ folder over)
        leaves every sidecar's own "path" field pointing at the old, now-nonexistent location -
        GGUFReader then raised a raw FileNotFoundError the moment /api/tags tried to open one,
        500ing the whole endpoint. list_installed must derive the real path from where the sidecar
        itself was actually found, ignoring a stale stored value entirely."""
        catalog = ModelCatalog(tmp_path / "models")
        installed = _install(catalog, "moved:latest")
        sidecar_path = catalog._sidecar_path_for(Path(installed.path))
        stale = json.loads(sidecar_path.read_text())
        stale["path"] = "/no/such/path/model.gguf"
        sidecar_path.write_text(json.dumps(stale))

        result = catalog.get("moved:latest")

        assert result.path == installed.path
        assert Path(result.path).read_bytes() == b"weights"


class TestDelete:
    def test_delete_removes_the_gguf_and_its_sidecar(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "models")
        installed = _install(catalog, "gone:latest")

        catalog.delete("gone:latest")

        assert not Path(installed.path).exists()
        assert not catalog._sidecar_path_for(Path(installed.path)).exists()

    def test_delete_prunes_every_now_empty_ancestor_folder(self, tmp_path: Path) -> None:
        """Real bug found live (2026-09-22): the old cleanup only ever removed the immediate <repo>
        folder - once every model under an org had been deleted, its own now-empty <org> folder (and
        any folder above it, up to models_dir) was left behind forever as an orphan. Confirmed live
        against a real models directory carrying ten such orphaned org folders."""
        catalog = ModelCatalog(tmp_path / "models")
        installed = _install(catalog, "gone:latest")
        repo_dir = Path(installed.path).parent
        org_dir = repo_dir.parent

        catalog.delete("gone:latest")

        assert not repo_dir.exists()
        assert not org_dir.exists()
        assert catalog.models_dir.exists()  # models_dir itself is never removed

    def test_delete_leaves_a_sibling_repos_folder_alone(self, tmp_path: Path) -> None:
        """The ancestor walk must stop the moment it hits a folder that still has something else in
        it - a second repo under the same org must survive deleting the first."""
        catalog = ModelCatalog(tmp_path / "models")
        gone = _install(catalog, "test-org:gone")
        kept = _install(catalog, "test-org:kept")
        # _install's own tag-derived directory naming means these two already share the same "hf.co/
        # test-org/" parent - confirmed by construction here rather than assumed.
        assert Path(gone.path).parent.parent == Path(kept.path).parent.parent

        catalog.delete("test-org:gone")

        assert not Path(gone.path).parent.exists()
        assert Path(kept.path).exists()
        assert Path(kept.path).parent.exists()
        assert Path(kept.path).parent.parent.exists()  # the shared org folder survives


class TestCopy:
    def test_copy_creates_a_second_tag_with_identical_metadata(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "models")
        source = _install(catalog, "source:latest")

        copied = catalog.copy("source:latest", "dest:latest")

        assert copied.tag == "dest:latest"
        assert copied.architecture == source.architecture
        assert copied.capabilities == source.capabilities
        assert copied.size_bytes == source.size_bytes
        assert catalog.get("dest:latest") == copied
        assert catalog.get("source:latest") == source  # source untouched

    def test_copied_blob_has_the_same_bytes_as_the_source(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "models")
        _install(catalog, "source:latest", content=b"real-weights-go-here")

        copied = catalog.copy("source:latest", "dest:latest")

        assert Path(copied.path).read_bytes() == b"real-weights-go-here"

    def test_copy_over_an_existing_destination_tag_overwrites_it(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "models")
        _install(catalog, "source:latest", content=b"new-weights")
        _install(catalog, "dest:latest", content=b"stale-weights")

        catalog.copy("source:latest", "dest:latest")

        assert Path(catalog.get("dest:latest").path).read_bytes() == b"new-weights"
        # exactly one "dest:latest" entry survives, not a stale second sidecar
        matching = [m for m in catalog.list_installed() if m.tag == "dest:latest"]
        assert len(matching) == 1

    def test_copy_of_unknown_source_raises(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "models")

        with pytest.raises(UnknownModelError):
            catalog.copy("nonexistent:latest", "dest:latest")

    def test_copy_sanitizes_slashes_and_colons_in_the_destination_tag(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "models")
        _install(catalog, "source:latest")

        copied = catalog.copy("source:latest", "hf.co/some/org:tag")

        assert catalog.get("hf.co/some/org:tag") == copied
