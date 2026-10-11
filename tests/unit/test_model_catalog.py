import json
import os
from pathlib import Path

import pytest

from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from app.server.errors import ModelInUseError, UnknownModelError


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
        """Bug: moving a whole models_dir by hand (e.g. re-pointing pAIring's
        Matricxon models path and `mv`ing the hf.co/ folder) left every sidecar's "path" at the old
        location; GGUFReader raised FileNotFoundError when /api/tags opened one, 500ing the
        endpoint. list_installed must derive the real path from where the sidecar was found,
        ignoring a stale stored value."""
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
        """Bug: the old cleanup removed only the immediate <repo> folder, so once
        every model under an org was deleted its empty <org> folder (and any above it, up to
        models_dir) stayed as an orphan (seen: ten such folders in a models directory)."""
        catalog = ModelCatalog(tmp_path / "models")
        installed = _install(catalog, "gone:latest")
        repo_dir = Path(installed.path).parent
        org_dir = repo_dir.parent

        catalog.delete("gone:latest")

        assert not repo_dir.exists()
        assert not org_dir.exists()
        assert catalog.models_dir.exists()

    def test_delete_leaves_a_sibling_repos_folder_alone(self, tmp_path: Path) -> None:
        """The ancestor walk must stop at a folder that still holds something else: a second repo
        under the same org survives deleting the first."""
        catalog = ModelCatalog(tmp_path / "models")
        gone = _install(catalog, "test-org:gone")
        kept = _install(catalog, "test-org:kept")
        # _install's tag-derived directory naming makes these two share the "hf.co/test-org/"
        # parent: confirmed by construction.
        assert Path(gone.path).parent.parent == Path(kept.path).parent.parent

        catalog.delete("test-org:gone")

        assert not Path(gone.path).parent.exists()
        assert Path(kept.path).exists()
        assert Path(kept.path).parent.exists()
        assert Path(kept.path).parent.parent.exists()


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
        assert catalog.get("source:latest") == source

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


class TestRelativeSidecarPath:
    def test_write_sidecar_stores_a_path_relative_to_models_dir(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "models")
        installed = _install(catalog, "a:1")
        catalog.write_sidecar(installed)

        stored = json.loads(catalog._sidecar_path_for(Path(installed.path)).read_text())
        assert stored["path"] == "hf.co/test-org/a_1/model.gguf"
        assert catalog.get("a:1").path == installed.path

    def test_a_copy_survives_a_moved_models_dir(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "models")
        _install(catalog, "a:1")
        catalog.copy("a:1", "b:1")

        (tmp_path / "models").rename(tmp_path / "moved")
        assert ModelCatalog(tmp_path / "moved").get("b:1").path.startswith(str(tmp_path / "moved"))


class TestAliases:
    def test_a_copy_is_an_alias_no_bytes_copied_and_no_hardlink(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "models")
        source = _install(catalog, "source:latest", content=b"weights")

        catalog.copy("source:latest", "dest:latest")

        models = [p for p in (tmp_path / "models").rglob("*.gguf") if p.is_file()]
        assert models == [Path(source.path)]
        assert os.stat(source.path).st_nlink == 1
        assert catalog.get("dest:latest").path == source.path

    def test_an_alias_of_an_alias_points_at_the_real_model(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "models")
        _install(catalog, "a:1")
        catalog.copy("a:1", "b:1")
        catalog.copy("b:1", "c:1")

        assert catalog._aliases() == {"b:1": "a:1", "c:1": "a:1"}

    def test_removing_an_alias_leaves_the_model_alone(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "models")
        source = _install(catalog, "a:1")
        catalog.copy("a:1", "b:1")

        catalog.delete("b:1")

        assert [m.tag for m in catalog.list_installed()] == ["a:1"]
        assert Path(source.path).exists()

    def test_a_model_that_has_aliases_cannot_be_removed(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "models")
        source = _install(catalog, "a:1")
        catalog.copy("a:1", "b:1")
        catalog.copy("a:1", "c:1")

        with pytest.raises(ModelInUseError, match="b:1, c:1"):
            catalog.delete("a:1")
        assert Path(source.path).exists() and len(catalog.list_installed()) == 3
        catalog.delete("b:1")
        catalog.delete("c:1")
        catalog.delete("a:1")
        assert catalog.list_installed() == []

    def test_copying_over_an_alias_repoints_it_and_a_dangling_alias_is_hidden(
        self, tmp_path: Path
    ) -> None:
        catalog = ModelCatalog(tmp_path / "models")
        _install(catalog, "a:1", content=b"one")
        _install(catalog, "z:1", content=b"two")
        catalog.copy("a:1", "b:1")
        catalog.copy("z:1", "b:1")
        assert Path(catalog.get("b:1").path).read_bytes() == b"two"

        for sidecar in (tmp_path / "models").rglob("*.gguf.json"):
            if "z_1" in str(sidecar):
                sidecar.unlink()
        assert {m.tag for m in catalog.list_installed()} == {"a:1"}
