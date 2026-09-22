import json
from pathlib import Path
from unittest.mock import Mock

from app.models.catalog import ModelCatalog
from app.pull.downloader import HFDownloader
from app.pull.hf_resolver import HFRepoResolver, ResolvedFile
from app.pull.job import PullJob
from tests.tiny_gguf import build_tiny_mistral3_gguf


def _fake_resolver(resolved: ResolvedFile) -> HFRepoResolver:
    resolver = Mock(spec=HFRepoResolver)
    resolver.resolve.return_value = resolved
    return resolver


def _fake_downloader(gguf_bytes: bytes) -> HFDownloader:
    downloader = Mock(spec=HFDownloader)

    def download(resolved, dest, cancel_event=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(gguf_bytes)
        yield len(gguf_bytes), len(gguf_bytes)

    downloader.download.side_effect = download
    return downloader


class TestPullJobHappyPath:
    def test_writes_the_gguf_and_a_matching_sidecar(self, tmp_path: Path) -> None:
        gguf_bytes = build_tiny_mistral3_gguf(tmp_path / "source.gguf").read_bytes()
        resolved = ResolvedFile(
            filename="tiny.gguf", size=len(gguf_bytes), sha256=None, download_url="https://x"
        )
        models_dir = tmp_path / "models"
        job = PullJob(
            models_dir,
            resolver=_fake_resolver(resolved),
            downloader=_fake_downloader(gguf_bytes),
        )

        lines = list(job.run("hf.co/test-org/tiny-repo:Q4_K_M"))

        assert not any("error" in line for line in lines)
        assert lines[-1]["status"] == "success"

        gguf_path = models_dir / "hf.co" / "test-org/tiny-repo" / "Q4_K_M.gguf"
        assert gguf_path.read_bytes() == gguf_bytes
        sidecar_path = gguf_path.parent / "Q4_K_M.gguf.json"
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["tag"] == "hf.co/test-org/tiny-repo:Q4_K_M"
        assert sidecar["architecture"] == "mistral3"
        assert sidecar["capabilities"] == ["completion"]

    def test_catalog_can_see_the_pulled_model_afterward(self, tmp_path: Path) -> None:
        gguf_bytes = build_tiny_mistral3_gguf(tmp_path / "source.gguf").read_bytes()
        resolved = ResolvedFile(
            filename="tiny.gguf", size=len(gguf_bytes), sha256=None, download_url="https://x"
        )
        models_dir = tmp_path / "models"
        job = PullJob(
            models_dir,
            resolver=_fake_resolver(resolved),
            downloader=_fake_downloader(gguf_bytes),
        )
        list(job.run("hf.co/test-org/tiny-repo:Q4_K_M"))

        installed = ModelCatalog(models_dir).get("hf.co/test-org/tiny-repo:Q4_K_M")
        assert installed.architecture == "mistral3"


class TestPullJobAlreadyDownloaded:
    """See app.pull.download_coordinator's own docstring: a second PullJob for the same tag, getting
    its turn at the lock only after an earlier one already finished, must not re-download."""

    def test_skips_the_download_when_the_file_already_exists(self, tmp_path: Path) -> None:
        gguf_bytes = build_tiny_mistral3_gguf(tmp_path / "source.gguf").read_bytes()
        resolved = ResolvedFile(
            filename="tiny.gguf", size=len(gguf_bytes), sha256=None, download_url="https://x"
        )
        models_dir = tmp_path / "models"
        gguf_path = models_dir / "hf.co" / "test-org/tiny-repo" / "Q4_K_M.gguf"
        gguf_path.parent.mkdir(parents=True)
        gguf_path.write_bytes(gguf_bytes)

        def _fail_if_called(*_a, **_kw):
            raise AssertionError("download should have been skipped - the file already exists")

        downloader = Mock(spec=HFDownloader)
        downloader.download.side_effect = _fail_if_called
        job = PullJob(models_dir, resolver=_fake_resolver(resolved), downloader=downloader)

        lines = list(job.run("hf.co/test-org/tiny-repo:Q4_K_M"))

        assert not any("error" in line for line in lines)
        assert lines[-1]["status"] == "success"
        downloader.download.assert_not_called()


class TestPullJobErrors:
    def test_bad_tag_format_yields_an_in_band_error_line_not_a_raised_exception(
        self, tmp_path: Path
    ) -> None:
        job = PullJob(tmp_path / "models")

        lines = list(job.run("plain-ollama-tag:latest"))

        assert lines[-1]["error"]

    def test_resolver_failure_yields_an_in_band_error_line(self, tmp_path: Path) -> None:
        from app.server.errors import ModelResolutionError

        resolver = Mock(spec=HFRepoResolver)
        resolver.resolve.side_effect = ModelResolutionError("no matching file")
        job = PullJob(tmp_path / "models", resolver=resolver)

        lines = list(job.run("hf.co/org/repo:Q8_0"))

        assert lines[-1]["error"] == "no matching file"
