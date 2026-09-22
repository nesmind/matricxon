from app.models.capabilities import CapabilityInferer


class TestCapabilityInferer:
    def test_decoder_architecture_reports_completion(self) -> None:
        capabilities = CapabilityInferer().infer("org/repo", "model.gguf", "mistral3")

        assert capabilities == ["completion"]

    def test_bert_architecture_reports_embedding_not_completion(self) -> None:
        capabilities = CapabilityInferer().infer("org/repo", "model.gguf", "bert")

        assert capabilities == ["embedding"]

    def test_nomic_bert_architecture_reports_embedding_not_completion(self) -> None:
        capabilities = CapabilityInferer().infer("org/repo", "model.gguf", "nomic-bert")

        assert capabilities == ["embedding"]

    def test_thinking_marker_in_repo_id_is_detected(self) -> None:
        capabilities = CapabilityInferer().infer(
            "org/some-reasoning-model", "model.gguf", "mistral3"
        )

        assert "thinking" in capabilities

    def test_thinking_marker_in_filename_is_detected(self) -> None:
        capabilities = CapabilityInferer().infer("org/repo", "model-thinking.gguf", "mistral3")

        assert "thinking" in capabilities

    def test_no_marker_means_no_thinking_capability(self) -> None:
        capabilities = CapabilityInferer().infer("org/repo", "model.gguf", "mistral3")

        assert "thinking" not in capabilities

    def test_vision_is_never_reported_in_v1(self) -> None:
        capabilities = CapabilityInferer().infer("org/repo", "model.gguf", "mistral3")

        assert "vision" not in capabilities

    def test_clip_architecture_reports_no_capabilities(self) -> None:
        capabilities = CapabilityInferer().infer("org/repo", "mmproj.gguf", "clip")

        assert capabilities == []
