import hashlib
import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from app.pull.downloader import HFDownloader
from app.pull.hf_resolver import ResolvedFile
from app.server.errors import ModelResolutionError, PullCancelledError, PullConnectionError

_CONTENT = b"fake gguf bytes" * 1000


def _downloader(content: bytes = _CONTENT) -> HFDownloader:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return HFDownloader(http_client=client)


class _DroppedConnectionStream(httpx.SyncByteStream):
    """A response body that yields a few real bytes, then fails the way
    httpx does when a peer closes mid-chunked-transfer."""

    def __iter__(self) -> Iterator[bytes]:
        yield b"partial bytes before the connection dropped"
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body (incomplete chunked read)"
        )

    def close(self) -> None:
        pass


def _downloader_with_dropped_connection() -> HFDownloader:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_DroppedConnectionStream())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return HFDownloader(http_client=client)


def _resolved(content: bytes = _CONTENT, sha256: str | None = None) -> ResolvedFile:
    return ResolvedFile(
        filename="model.gguf",
        size=len(content),
        sha256=sha256 if sha256 is not None else hashlib.sha256(content).hexdigest(),
        download_url="https://huggingface.co/org/repo/resolve/main/model.gguf",
    )


class TestHFDownloaderSuccess:
    def test_writes_the_full_content_to_dest(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        list(_downloader().download(_resolved(), dest))

        assert dest.read_bytes() == _CONTENT

    def test_removes_the_partial_file_after_a_successful_rename(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        list(_downloader().download(_resolved(), dest))

        assert not dest.with_suffix(".gguf.partial").exists()

    def test_final_progress_pair_reports_full_size(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        progress = list(_downloader().download(_resolved(), dest))

        completed, total = progress[-1]
        assert completed == len(_CONTENT)
        assert total == len(_CONTENT)


class TestHFDownloaderDigestVerification:
    def test_raises_and_removes_partial_on_digest_mismatch(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        resolved = _resolved(sha256="0" * 64)

        with pytest.raises(ModelResolutionError, match="sha256"):
            list(_downloader().download(resolved, dest))

        assert not dest.exists()
        assert not dest.with_suffix(".gguf.partial").exists()

    def test_skips_verification_when_sha256_is_unknown(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        resolved = _resolved(sha256=None)

        list(_downloader().download(resolved, dest))

        assert dest.read_bytes() == _CONTENT


class TestHFDownloaderConnectionDrop:
    def test_raises_pull_connection_error_on_mid_stream_drop(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        resolved = _resolved()

        with pytest.raises(PullConnectionError, match="model.gguf"):
            list(_downloader_with_dropped_connection().download(resolved, dest))

    def test_removes_the_partial_file_on_mid_stream_drop(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        resolved = _resolved()

        with pytest.raises(PullConnectionError):
            list(_downloader_with_dropped_connection().download(resolved, dest))

        assert not dest.exists()
        assert not dest.with_suffix(".gguf.partial").exists()


class TestHFDownloaderCancellation:
    """See app.pull.download_coordinator's own docstring for the real bug this - and the matching
    DownloadCoordinator fix - closes: a client disconnecting mid-pull left the download running on
    its own thread with no way to know its caller was gone."""

    def test_raises_pull_cancelled_error_once_the_event_is_set(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        resolved = _resolved()
        cancel_event = threading.Event()
        cancel_event.set()

        with pytest.raises(PullCancelledError, match="model.gguf"):
            list(_downloader().download(resolved, dest, cancel_event=cancel_event))

    def test_removes_the_partial_file_once_cancelled(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        resolved = _resolved()
        cancel_event = threading.Event()
        cancel_event.set()

        with pytest.raises(PullCancelledError):
            list(_downloader().download(resolved, dest, cancel_event=cancel_event))

        assert not dest.exists()
        assert not dest.with_suffix(".gguf.partial").exists()

    def test_an_unset_event_never_interrupts_a_normal_download(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        cancel_event = threading.Event()

        list(_downloader().download(_resolved(), dest, cancel_event=cancel_event))

        assert dest.read_bytes() == _CONTENT
