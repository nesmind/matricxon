from dataclasses import dataclass
from pathlib import Path

import httpx

from app.server.errors import ModelResolutionError


@dataclass(frozen=True)
class ResolvedFile:
    filename: str
    size: int
    sha256: str | None
    download_url: str


class HFRepoResolver:
    """Resolves a tag's `<suffix>` against a Hugging Face repo's real file
    listing (`GET /api/models/<repo_id>?blobs=true`, which conveniently
    already carries each LFS file's size and sha256 - no download needed to
    know either upfront).

    `http_client` is injectable so tests can point it at an `httpx.MockTransport`
    instead of the real network - matricxon's automated test suite must stay
    offline-runnable.
    """

    def __init__(self, http_client: httpx.Client | None = None) -> None:
        self._http_client = http_client or httpx.Client(timeout=30.0)

    def resolve(self, repo_id: str, suffix: str) -> ResolvedFile:
        response = self._http_client.get(
            f"https://huggingface.co/api/models/{repo_id}", params={"blobs": "true"}
        )
        if response.status_code == 404:
            raise ModelResolutionError(f"Hugging Face repo not found: {repo_id!r}")
        response.raise_for_status()

        siblings = response.json().get("siblings", [])
        gguf_files = [s for s in siblings if s["rfilename"].endswith(".gguf")]
        if not gguf_files:
            raise ModelResolutionError(f"{repo_id!r} has no .gguf files")

        match = self._match(suffix, gguf_files)
        return ResolvedFile(
            filename=match["rfilename"],
            size=match["size"],
            sha256=match.get("lfs", {}).get("sha256"),
            download_url=f"https://huggingface.co/{repo_id}/resolve/main/{match['rfilename']}",
        )

    def _match(self, suffix: str, gguf_files: list[dict]) -> dict:
        suffix_lower = suffix.lower()

        exact = [f for f in gguf_files if Path(f["rfilename"]).stem.lower() == suffix_lower]
        if len(exact) == 1:
            return exact[0]

        substring = [f for f in gguf_files if suffix_lower in f["rfilename"].lower()]
        if len(substring) == 1:
            return substring[0]

        candidates = ", ".join(f["rfilename"] for f in gguf_files)
        if not substring and not exact:
            raise ModelResolutionError(
                f"No .gguf file matching suffix {suffix!r} - available: {candidates}"
            )
        raise ModelResolutionError(
            f"Ambiguous suffix {suffix!r}, matches multiple files - available: {candidates}"
        )
