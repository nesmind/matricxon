import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from app.models.installed_model import InstalledModel
from app.server.errors import ModelInUseError, UnknownModelError


def _sanitize_tag(tag: str) -> str:
    """A tag can carry "/" and ":" (e.g. "hf.co/org/repo:suffix"), unsafe as one path segment, so
    both become "_" for locally created tags' directory name (`models_dir/local/<sanitized>/`)."""
    return tag.replace("/", "_").replace(":", "_")


class ModelCatalog:
    """Maps opaque model tags to on-disk.gguf files + sidecar metadata. Answerable without loading
    a model: sidecars are written at pull time from a header-only GGUF parse (app/pull/job.py)."""

    SIDECAR_SUFFIX = ".gguf.json"
    ALIAS_SUFFIX = ".alias.json"

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def list_installed(self) -> list[InstalledModel]:
        """Every real model, then each alias as that model under its own tag (same file)."""
        if not self._models_dir.exists():
            return []
        real = [
            self._load_sidecar(sidecar_path)
            for sidecar_path in sorted(self._models_dir.rglob(f"*{self.SIDECAR_SUFFIX}"))
        ]
        by_tag = {m.tag: m for m in real}
        aliased = [
            replace(by_tag[target], tag=alias)
            for alias, target in self._aliases().items()
            if target in by_tag and alias not in by_tag
        ]
        return real + aliased

    def get(self, tag: str) -> InstalledModel:
        for installed in self.list_installed():
            if installed.tag == tag:
                return installed
        raise UnknownModelError(f"Unknown model: {tag}")

    def gguf_path_for_tag(self, tag: str) -> Path:
        return Path(self.get(tag).path)

    def find_paired_mmproj(self, tag: str) -> InstalledModel | None:
        """A vision-capable text model (e.g. LLaVA) and its mmproj (vision tower + projector)
        install as two independent tags via `/api/pull`: pulling
        `hf.co/second-state/Llava-v1.6-Vicuna-7B-GGUF`'s main model and its `:mmproj` suffix land in
        the same repo directory with nothing else linking them (Ollama's registry has a `projector`
        manifest layer). Pairing is by convention: same directory, `architecture == "clip"`.
        Computed here, not baked into a sidecar at pull time, so a text model pulled before its
        mmproj becomes vision-capable as soon as the sibling exists."""
        installed = self.get(tag)
        same_dir = Path(installed.path).parent
        for candidate in self.list_installed():
            # candidate.tag != tag: a clip file querying its own pairing (e.g.
            # `effective_capabilities` for the mmproj tag) must not match itself (it wrongly
            # reported "vision" among the mmproj tag's own capabilities).
            if (
                candidate.tag != tag
                and candidate.architecture == "clip"
                and Path(candidate.path).parent == same_dir
            ):
                return candidate
        return None

    def check_deletable(self, tag: str) -> None:
        """Raises ModelInUseError if aliases still use this model (check before unloading it)."""
        users = sorted(alias for alias, target in self._aliases().items() if target == tag)
        if users:
            raise ModelInUseError(f"{tag} is also used as {', '.join(users)}: remove those first")

    def delete(self, tag: str) -> None:
        """An alias just goes away. A real model that has aliases is refused (they use its file)."""
        if tag in self._aliases():
            self._alias_path(tag).unlink(missing_ok=True)
            return
        self.check_deletable(tag)
        gguf_path = self.gguf_path_for_tag(tag)
        self._sidecar_path_for(gguf_path).unlink(missing_ok=True)
        gguf_path.unlink(missing_ok=True)
        self._prune_empty_ancestors(gguf_path.parent)

    def _prune_empty_ancestors(self, directory: Path) -> None:
        """Walks up from `directory` removing every empty folder a delete left, stopping at
        models_dir (never removed) or the first non-empty ancestor. The old single-level check
        cleaned only the immediate <repo> folder: once all of an org's models were
        deleted, its empty "hf.co/org/" folder stayed forever."""
        while directory != self._models_dir and directory.is_dir() and not any(directory.iterdir()):
            parent = directory.parent
            directory.rmdir()
            directory = parent

    def copy(self, source_tag: str, dest_tag: str) -> InstalledModel:
        """Gives an installed model a second tag (`/api/copy`, and `/api/create`'s FROM-only case).
        No bytes are copied or linked: the new tag is an alias entry pointing at the real model's
        tag (an alias of an alias points at the real one). Overwrites an existing `dest_tag`, like
        Ollama."""
        source = self.get(source_tag)
        target = self._aliases().get(source_tag, source_tag)
        if dest_tag == source_tag:
            return source
        try:
            self.delete(dest_tag)
        except UnknownModelError:
            pass
        self._write_alias(dest_tag, target)
        return replace(source, tag=dest_tag)

    def _alias_path(self, tag: str) -> Path:
        return self._models_dir / "aliases" / f"{_sanitize_tag(tag)}{self.ALIAS_SUFFIX}"

    def _aliases(self) -> dict[str, str]:
        """alias tag -> the real model's tag."""
        folder = self._models_dir / "aliases"
        found: dict[str, str] = {}
        for path in sorted(folder.glob(f"*{self.ALIAS_SUFFIX}")) if folder.is_dir() else []:
            try:
                data = json.loads(path.read_text())
                found[data["tag"]] = data["target"]
            except (ValueError, KeyError, OSError):
                continue
        return found

    def _write_alias(self, tag: str, target: str) -> None:
        path = self._alias_path(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps({"tag": tag, "target": target}))
        os.replace(tmp, path)

    def write_sidecar(self, installed: InstalledModel) -> None:
        """Stores `path` relative to models_dir, so nodes mounting it elsewhere agree."""
        gguf_path = Path(installed.path)
        try:
            stored = str(gguf_path.relative_to(self._models_dir))
        except ValueError:
            stored = installed.path
        data = installed.__dict__ | {"path": stored}
        self._sidecar_path_for(gguf_path).write_text(json.dumps(data))

    def _sidecar_path_for(self, gguf_path: Path) -> Path:
        return gguf_path.parent / f"{gguf_path.stem}{self.SIDECAR_SUFFIX}"

    def _load_sidecar(self, sidecar_path: Path) -> InstalledModel:
        data = json.loads(sidecar_path.read_text())
        # The sidecar's on-disk location is authoritative for where its.gguf is, not the "path" in
        # the JSON at write time (app/pull/job.py), which goes stale when the file or models_dir
        # moves without rewriting every sidecar. a manual `mv` of a models directory left
        # every sidecar "path" stale and crashed /api/tags with a FileNotFoundError. SIDECAR_SUFFIX
        # guarantees the.gguf sits beside its sidecar under the same stem, so trust the location
        # just found via rglob.
        data["path"] = str(sidecar_path.parent / sidecar_path.stem)
        return InstalledModel(**data)
