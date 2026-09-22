import pytest

from app.pull.tag import HFModelTag
from app.server.errors import ModelResolutionError


class TestHFModelTagParse:
    def test_parses_repo_id_and_suffix(self) -> None:
        tag = HFModelTag.parse("hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0")

        assert tag.repo_id == "nomic-ai/nomic-embed-text-v1.5-GGUF"
        assert tag.suffix == "Q8_0"

    def test_rejects_a_plain_ollama_library_tag(self) -> None:
        with pytest.raises(ModelResolutionError):
            HFModelTag.parse("llama3.2:1b")

    def test_rejects_a_tag_missing_a_suffix(self) -> None:
        with pytest.raises(ModelResolutionError):
            HFModelTag.parse("hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF")


class TestHFModelTagLocalFilename:
    def test_uses_the_requested_suffix_not_any_resolved_hf_filename(self) -> None:
        tag = HFModelTag.parse("hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0")

        assert tag.local_filename() == "Q8_0.gguf"
