import hashlib
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

from app.pull.hf_resolver import ResolvedFile
from app.server.errors import ModelResolutionError, PullCancelledError, PullConnectionError

_PROGRESS_THROTTLE_SECONDS = 0.5
_CHUNK_SIZE = 1024 * 1024


class HFDownloader:
    """Streams a resolved file to `<dest>.partial`, yielding throttled
    `(completed, total)` progress pairs, then atomically renames to `dest`
    on success. Verifies the download against `ResolvedFile.sha256` (known
    upfront from HF's API, no extra request) before the rename - a
    corrupted/truncated download must never look like an installed model.
    A dropped connection mid-stream (`httpx.HTTPError`, e.g. the peer
    closing before the chunked body completes) is treated the same way:
    the `.partial` file is deleted and a `PullConnectionError` is raised
    instead of leaving a truncated file behind or letting the raw httpx
    exception propagate.

    `cancel_event`, when given, is checked once per chunk - set it (from the
    caller's own event loop thread) to abort an in-progress download early,
    same clean "delete `.partial`, raise instead of leaving a truncated file
    or running to completion unseen" treatment as a dropped connection. See
    app.pull.download_coordinator's own docstring for why this exists: a
    real pAIring bug report (2026-09-21) traced a `.partial` rename crash
    back to a browser refresh abandoning an in-flight pull - the download
    itself, running on its own thread, had no way to know its caller was
    gone and kept going regardless, until an unrelated retry raced it.

    `http_client` is injectable for the same reason as `HFRepoResolver`.
    """

    def __init__(self, http_client: httpx.Client | None = None) -> None:
        self._http_client = http_client or httpx.Client(timeout=30.0, follow_redirects=True)

    def download(
        self, resolved: ResolvedFile, dest: Path, cancel_event: threading.Event | None = None
    ) -> Iterator[tuple[int, int]]:
        partial_path = dest.with_suffix(dest.suffix + ".partial")
        dest.parent.mkdir(parents=True, exist_ok=True)

        digest = hashlib.sha256()
        completed = 0
        last_yield = 0.0

        try:
            with (
                self._http_client.stream("GET", resolved.download_url) as response,
                partial_path.open("wb") as f,
            ):
                response.raise_for_status()
                for chunk in response.iter_bytes(_CHUNK_SIZE):
                    if cancel_event is not None and cancel_event.is_set():
                        partial_path.unlink(missing_ok=True)
                        raise PullCancelledError(
                            f"pull of {resolved.filename!r} cancelled: the client disconnected"
                        )

                    f.write(chunk)
                    digest.update(chunk)
                    completed += len(chunk)

                    now = time.monotonic()
                    if now - last_yield >= _PROGRESS_THROTTLE_SECONDS:
                        last_yield = now
                        yield completed, resolved.size
        except httpx.HTTPError as exc:
            partial_path.unlink(missing_ok=True)
            raise PullConnectionError(f"download of {resolved.filename!r} failed: {exc}") from exc

        if resolved.sha256 is not None and digest.hexdigest() != resolved.sha256:
            partial_path.unlink(missing_ok=True)
            raise ModelResolutionError(
                f"downloaded file failed sha256 verification: {resolved.filename!r}"
            )

        partial_path.rename(dest)
        yield completed, resolved.size
