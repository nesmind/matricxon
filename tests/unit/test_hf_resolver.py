import httpx
import pytest

from app.pull.hf_resolver import HFRepoResolver
from app.server.errors import ModelResolutionError

_SIBLINGS = [
    {"rfilename": "README.md", "size": 100},
    {
        "rfilename": "nomic-embed-text-v1.5.Q8_0.gguf",
        "size": 146146432,
        "lfs": {"sha256": "abc123"},
    },
    {
        "rfilename": "nomic-embed-text-v1.5.Q4_K_M.gguf",
        "size": 84106624,
        "lfs": {"sha256": "def456"},
    },
]


def _resolver(siblings: list[dict] | None = None, status_code: int = 200) -> HFRepoResolver:
    def handler(request: httpx.Request) -> httpx.Response:
        if status_code != 200:
            return httpx.Response(status_code)
        body = siblings if siblings is not None else _SIBLINGS
        return httpx.Response(200, json={"siblings": body})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return HFRepoResolver(http_client=client)


class TestHFRepoResolverSubstringMatch:
    def test_resolves_a_unique_substring_match(self) -> None:
        resolved = _resolver().resolve("nomic-ai/nomic-embed-text-v1.5-GGUF", "Q8_0")

        assert resolved.filename == "nomic-embed-text-v1.5.Q8_0.gguf"
        assert resolved.size == 146146432
        assert resolved.sha256 == "abc123"
        assert resolved.download_url.endswith("nomic-embed-text-v1.5.Q8_0.gguf")


class TestHFRepoResolverExactMatch:
    def test_resolves_an_exact_case_insensitive_stem_match(self) -> None:
        siblings = [{"rfilename": "model.gguf", "size": 10, "lfs": {"sha256": "x"}}]
        resolved = _resolver(siblings).resolve("org/repo", "MODEL")

        assert resolved.filename == "model.gguf"


class TestHFRepoResolverErrors:
    def test_no_match_raises_with_candidates_listed(self) -> None:
        with pytest.raises(ModelResolutionError, match="Q8_0"):
            _resolver().resolve("nomic-ai/nomic-embed-text-v1.5-GGUF", "nonexistent-suffix")

    def test_ambiguous_match_raises(self) -> None:
        siblings = [
            {"rfilename": "modelA.gguf", "size": 1, "lfs": {"sha256": "a"}},
            {"rfilename": "modelB.gguf", "size": 1, "lfs": {"sha256": "b"}},
        ]
        with pytest.raises(ModelResolutionError, match="Ambiguous"):
            _resolver(siblings).resolve("org/repo", "model")

    def test_repo_not_found_raises(self) -> None:
        with pytest.raises(ModelResolutionError):
            _resolver(status_code=404).resolve("org/nonexistent-repo", "Q8_0")

    def test_repo_with_no_gguf_files_raises(self) -> None:
        with pytest.raises(ModelResolutionError):
            _resolver([{"rfilename": "README.md", "size": 1}]).resolve("org/repo", "Q8_0")
