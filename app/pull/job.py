import json
import threading
from collections.abc import Iterator
from pathlib import Path

from app.gguf.metadata import GGUFMetadata
from app.gguf.reader import GGUFReader
from app.models.capabilities import CapabilityInferer
from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from app.pull.download_coordinator import get_download_coordinator
from app.pull.downloader import HFDownloader
from app.pull.hf_resolver import HFRepoResolver
from app.pull.tag import HFModelTag
from app.schemas.pull import PullProgress
from app.server.errors import MatricxonError

_ALREADY_DOWNLOADED_STATUS = "already downloaded"


class PullJob:
    """Orchestrates one `/api/pull` call: resolve -> download -> write a
    sidecar from a header-only GGUF parse -> done.

    Yields Ollama-shaped NDJSON progress dicts; any failure becomes an
    in-band `{"error": ...}` line rather than an HTTP error - `/api/pull`
    has a looser contract than `/api/chat` (any real Ollama-API client,
    pAIring's admin UI included, reads that field directly instead of an
    HTTP status).

    `cancel_event`, when the caller sets it (see pull_router.PullRequestHandler, set the moment
    its own client disconnects), aborts the download in progress instead of letting it run to
    completion unseen - see app.pull.download_coordinator's own docstring for the real bug this
    closes. The download step is also serialized per-destination-file via DownloadCoordinator:
    two PullJobs racing for the same tag (a retry while an earlier, now-orphaned pull was still
    running, say) wait their turn instead of both writing the same `.partial` file, and the
    second one skips straight to "already downloaded" once it finds the first already finished.
    """

    def __init__(
        self,
        models_dir: Path,
        resolver: HFRepoResolver | None = None,
        downloader: HFDownloader | None = None,
        capability_inferer: CapabilityInferer | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self._models_dir = models_dir
        self._resolver = resolver or HFRepoResolver()
        self._downloader = downloader or HFDownloader()
        self._capability_inferer = capability_inferer or CapabilityInferer()
        self._cancel_event = cancel_event or threading.Event()

    def run(self, tag: str) -> Iterator[dict]:
        try:
            yield from self._run(tag)
        except MatricxonError as exc:
            yield PullProgress(status="error", error=exc.message).to_ndjson_dict()

    def _run(self, tag: str) -> Iterator[dict]:
        yield PullProgress(status="resolving manifest").to_ndjson_dict()
        model_tag = HFModelTag.parse(tag)
        resolved = self._resolver.resolve(model_tag.repo_id, model_tag.suffix)

        dest = self._models_dir / "hf.co" / model_tag.repo_id / model_tag.local_filename()
        yield PullProgress(
            status=f"pulling {resolved.filename}", total=resolved.size, digest=resolved.sha256
        ).to_ndjson_dict()

        with get_download_coordinator().guard(dest):
            if dest.exists():
                yield PullProgress(
                    status=_ALREADY_DOWNLOADED_STATUS,
                    completed=resolved.size,
                    total=resolved.size,
                    digest=resolved.sha256,
                ).to_ndjson_dict()
            else:
                downloaded = self._downloader.download(
                    resolved, dest, cancel_event=self._cancel_event
                )
                for completed, total in downloaded:
                    yield PullProgress(
                        status=f"pulling {resolved.filename}",
                        completed=completed,
                        total=total,
                        digest=resolved.sha256,
                    ).to_ndjson_dict()

        self._write_sidecar(dest, tag, model_tag.repo_id)
        yield PullProgress(status="success").to_ndjson_dict()

    def _write_sidecar(self, gguf_path: Path, tag: str, repo_id: str) -> None:
        metadata = GGUFReader(gguf_path).read().metadata
        installed = InstalledModel(
            tag=tag,
            path=str(gguf_path),
            architecture=metadata.architecture,
            capabilities=self._capability_inferer.infer(
                repo_id, gguf_path.name, metadata.architecture
            ),
            size_bytes=gguf_path.stat().st_size,
            family=metadata.architecture,
            parameter_size=self._format_parameter_size(metadata),
            context_length=metadata.get_u32(metadata.arch_key("context_length"), 0),
        )
        sidecar_path = gguf_path.parent / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
        sidecar_path.write_text(json.dumps(installed.__dict__))

    @staticmethod
    def _format_parameter_size(metadata: GGUFMetadata) -> str:
        count = metadata.get("general.parameter_count")
        if count is None:
            return "unknown"
        if count >= 1e9:
            return f"{count / 1e9:.1f}B"
        return f"{count / 1e6:.0f}M"
