import json
import os
import shutil
from pathlib import Path

from app.models.installed_model import InstalledModel
from app.server.errors import UnknownModelError


def _sanitize_tag(tag: str) -> str:
    """A tag can carry "/" and ":" (e.g. "hf.co/org/repo:suffix") - neither

    is safe as a single path segment, so both collapse to "_" for
    locally-created tags' directory name (`models_dir/local/<sanitized>/`).
    """
    return tag.replace("/", "_").replace(":", "_")


class ModelCatalog:
    """Maps opaque model tags to on-disk .gguf files + their sidecar metadata.

    Answerable without loading a model into memory: sidecars are written once
    at pull time from a header-only GGUF parse (see app/pull/job.py).
    """

    SIDECAR_SUFFIX = ".gguf.json"

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def list_installed(self) -> list[InstalledModel]:
        if not self._models_dir.exists():
            return []
        return [
            self._load_sidecar(sidecar_path)
            for sidecar_path in sorted(self._models_dir.rglob(f"*{self.SIDECAR_SUFFIX}"))
        ]

    def get(self, tag: str) -> InstalledModel:
        for installed in self.list_installed():
            if installed.tag == tag:
                return installed
        raise UnknownModelError(f"Unknown model: {tag}")

    def gguf_path_for_tag(self, tag: str) -> Path:
        return Path(self.get(tag).path)

    def find_paired_mmproj(self, tag: str) -> InstalledModel | None:
        """A real vision-capable text model (e.g. LLaVA) and its mmproj (vision tower +
        projector) install as two fully independent tags via matricxon's own `/api/pull` -
        confirmed live (2026-09-21): pulling `hf.co/second-state/Llava-v1.6-Vicuna-7B-GGUF`'s
        main model and its `:mmproj` suffix both land in the *same* real repo directory
        (`.../Llava-v1.6-Vicuna-7B-GGUF/`), with nothing else linking them - unlike real Ollama's
        own registry protocol, which has an explicit `projector` manifest layer for exactly this
        pairing. Convention-based instead of a new stored field: same directory, `architecture ==
        "clip"`. Computed here (not baked into a sidecar at pull time) so a text model pulled
        *before* its mmproj sibling becomes vision-capable the moment that sibling exists on disk,
        with no re-pull or migration needed - confirmed against this exact real case."""
        installed = self.get(tag)
        same_dir = Path(installed.path).parent
        for candidate in self.list_installed():
            # candidate.tag != tag: a clip file querying its own pairing (e.g. `effective_
            # capabilities` run against the mmproj tag itself, not just its paired text model)
            # must not match itself - confirmed live: without this it did, wrongly reporting
            # "vision" as one of the mmproj tag's own capabilities.
            if (
                candidate.tag != tag
                and candidate.architecture == "clip"
                and Path(candidate.path).parent == same_dir
            ):
                return candidate
        return None

    def delete(self, tag: str) -> None:
        gguf_path = self.gguf_path_for_tag(tag)
        self._sidecar_path_for(gguf_path).unlink(missing_ok=True)
        gguf_path.unlink(missing_ok=True)
        self._prune_empty_ancestors(gguf_path.parent)

    def _prune_empty_ancestors(self, directory: Path) -> None:
        """Walks up from `directory` removing every now-empty folder a delete left behind, stopping
        at models_dir itself (never removed) or the first still-nonempty ancestor. Confirmed live,
        2026-09-22: the old single-level check only ever cleaned up the immediate <repo> folder
        (e.g. "hf.co/org/repo/") - once every one of an org's models had been deleted, its own
        now-empty "hf.co/org/" folder was left behind forever, since nothing ever rechecked it after
        the repo folder's own removal could have emptied it in turn."""
        while directory != self._models_dir and directory.is_dir() and not any(directory.iterdir()):
            parent = directory.parent
            directory.rmdir()
            directory = parent

    def copy(self, source_tag: str, dest_tag: str) -> InstalledModel:
        """Duplicates an installed model under a new tag (`/api/copy`, and

        `/api/create`'s FROM-only case). Hardlinks the .gguf blob rather than
        copying its bytes when possible (same filesystem, and a real GGUF
        can be gigabytes), falling back to a real copy across filesystems.
        Overwrites an existing `dest_tag`, matching real Ollama's own
        behavior - not an error.
        """
        source = self.get(source_tag)
        source_path = Path(source.path)

        try:
            self.delete(dest_tag)
        except UnknownModelError:
            pass

        dest_dir = self._models_dir / "local" / _sanitize_tag(dest_tag)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / source_path.name

        partial_path = dest_path.with_name(dest_path.name + ".partial")
        partial_path.unlink(missing_ok=True)
        try:
            os.link(source_path, partial_path)
        except OSError:
            shutil.copyfile(source_path, partial_path)
        partial_path.rename(dest_path)

        installed = InstalledModel(
            tag=dest_tag,
            path=str(dest_path),
            architecture=source.architecture,
            capabilities=source.capabilities,
            size_bytes=source.size_bytes,
            family=source.family,
            parameter_size=source.parameter_size,
            context_length=source.context_length,
        )
        self._sidecar_path_for(dest_path).write_text(json.dumps(installed.__dict__))
        return installed

    def _sidecar_path_for(self, gguf_path: Path) -> Path:
        return gguf_path.parent / f"{gguf_path.stem}{self.SIDECAR_SUFFIX}"

    def _load_sidecar(self, sidecar_path: Path) -> InstalledModel:
        data = json.loads(sidecar_path.read_text())
        # The sidecar's own on-disk location is authoritative for where its real .gguf file is -
        # not the "path" field baked into the JSON at write time (see app/pull/job.py), which goes
        # stale the instant the file (or its whole models_dir) moves without something also
        # rewriting every sidecar's own stored path. Confirmed live: a manual `mv` of a whole
        # models directory left every sidecar's "path" pointing at the old, now-nonexistent
        # location, crashing /api/tags with a raw FileNotFoundError the moment anything tried to
        # actually open one. SIDECAR_SUFFIX guarantees the real .gguf always sits right next to its
        # own sidecar under the same stem, so there's no reason to trust a copy that can drift
        # instead of the location we just found it at via rglob.
        data["path"] = str(sidecar_path.parent / sidecar_path.stem)
        return InstalledModel(**data)
