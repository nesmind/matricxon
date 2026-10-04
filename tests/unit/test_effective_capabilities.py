"""app/models/capabilities.py's effective_capabilities and the has_confirmed_chat_format check it
now folds in (app/runtime/chat_template.py) - the "chat_format_unverified" capability this powers,
added 2026-09-27 after Hebrew-Mistral-7B-Q5_K_M produced incoherent, non-chat-like output no matter
which prompt format it was given, with nothing in the model list distinguishing it from a normal,
well-behaved chat model - and "architecture_features_unsupported" (app.architectures.registry.
unsupported_features), added 2026-09-29 after a real gemma-4-E2B-it GGUF's Per-Layer Embeddings
and cross-layer KV reuse were silently never read at all by the dense-only Gemma4Architecture
class that otherwise claimed full "completion" support for it."""

from pathlib import Path

from app.gguf.constants import GGUFValueType
from app.gguf.reader import GGUFReader
from app.models.capabilities import effective_capabilities
from app.models.installed_model import InstalledModel
from app.runtime.chat_template import has_confirmed_chat_format
from scripts.make_tiny_gguf import GGUFBuilder
from tests.tiny_gguf import build_tiny_mistral3_gguf


def _build_untemplated_llama_gguf(path: Path) -> Path:
    """A minimal, metadata-only GGUF for an architecture that isn't mistral3 and carries no
    tokenizer.chat_template - the exact shape PromptBuilderFactory.for_metadata falls back to
    LegacyMistralPromptBuilder for (see that class's own docstring), which is what
    has_confirmed_chat_format should report as unconfirmed."""
    return (
        GGUFBuilder()
        .set_str("general.architecture", "llama")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, ["<unk>", "<s>", "</s>"])
        .add_tensor("token_embd.weight", [1], ggml_type=0, raw_bytes=b"\x00\x00\x00\x00")
        .write(path)
    )


def _installed(tag: str, path: Path, architecture: str, capabilities: list[str]) -> InstalledModel:
    return InstalledModel(
        tag=tag,
        path=str(path),
        architecture=architecture,
        capabilities=capabilities,
        size_bytes=0,
        family=architecture,
        parameter_size="?",
        context_length=0,
    )


def test_has_confirmed_chat_format_is_true_for_a_real_mistral3_model(tmp_path):
    gguf_path = build_tiny_mistral3_gguf(tmp_path / "model.gguf")

    assert has_confirmed_chat_format(gguf_path) is True


def test_has_confirmed_chat_format_is_false_for_an_untemplated_non_mistral3_model(tmp_path):
    gguf_path = _build_untemplated_llama_gguf(tmp_path / "model.gguf")

    assert has_confirmed_chat_format(gguf_path) is False


def test_has_confirmed_chat_format_is_true_for_a_vicuna_tagged_model(tmp_path):
    """A tag-name match, not metadata - see VicunaPromptBuilder's own docstring for why
    (llava-v1.6-vicuna-7b, 2026-09-27: no metadata-only way to detect this format)."""
    gguf_path = _build_untemplated_llama_gguf(tmp_path / "model.gguf")

    assert has_confirmed_chat_format(gguf_path, tag="hf.co/org/llava-v1.6-vicuna-7b:Q4_K_M") is True


def test_has_confirmed_chat_format_is_cached_per_file_and_tag(tmp_path, monkeypatch):
    gguf_path = _build_untemplated_llama_gguf(tmp_path / "model.gguf")
    real_read = GGUFReader.read
    call_count = 0

    def counting_read(self):
        nonlocal call_count
        call_count += 1
        return real_read(self)

    monkeypatch.setattr(GGUFReader, "read", counting_read)

    has_confirmed_chat_format(gguf_path)
    has_confirmed_chat_format(gguf_path)

    assert call_count == 1


def test_effective_capabilities_flags_an_unconfirmed_completion_model(tmp_path):
    gguf_path = _build_untemplated_llama_gguf(tmp_path / "model.gguf")
    installed = _installed("test:latest", gguf_path, "llama", ["completion"])

    capabilities = effective_capabilities(installed, has_paired_mmproj=False)

    assert "chat_format_unverified" in capabilities


def test_effective_capabilities_does_not_flag_a_confirmed_mistral3_model(tmp_path):
    gguf_path = build_tiny_mistral3_gguf(tmp_path / "model.gguf")
    installed = _installed("test:latest", gguf_path, "mistral3", ["completion"])

    capabilities = effective_capabilities(installed, has_paired_mmproj=False)

    assert "chat_format_unverified" not in capabilities


def test_effective_capabilities_does_not_flag_a_vicuna_tagged_model(tmp_path):
    gguf_path = _build_untemplated_llama_gguf(tmp_path / "model.gguf")
    tag = "hf.co/second-state/Llava-v1.6-Vicuna-7B-GGUF:llava-v1.6-vicuna-7b-Q4_K_M"
    installed = _installed(tag, gguf_path, "llama", ["completion"])

    capabilities = effective_capabilities(installed, has_paired_mmproj=False)

    assert "chat_format_unverified" not in capabilities


def test_effective_capabilities_does_not_flag_a_non_completion_model(tmp_path):
    """An embedding-only model has no "chat format" to be unverified about - the flag only makes
    sense for something actually meant to be chatted with."""
    gguf_path = _build_untemplated_llama_gguf(tmp_path / "model.gguf")
    installed = _installed("test:latest", gguf_path, "bert", ["embedding"])

    capabilities = effective_capabilities(installed, has_paired_mmproj=False)

    assert "chat_format_unverified" not in capabilities


def test_effective_capabilities_flags_a_checkpoint_with_a_real_gated_variant_gap(
    tmp_path, monkeypatch
):
    """Decoupled from any specific architecture's own real gap (those get fixed over time - see
    Gemma4Architecture's own MoE support, added 2026-09-29, the same day this flag was added for
    a *different* gap that GGUF happened to have) - proves effective_capabilities' own wiring to
    app.architectures.registry.unsupported_features, not any one architecture's current state."""
    gguf_path = _build_untemplated_llama_gguf(tmp_path / "model.gguf")
    installed = _installed("test:latest", gguf_path, "llama", ["completion"])
    monkeypatch.setattr("app.models.capabilities.unsupported_features", lambda path: ["fake-gap"])

    capabilities = effective_capabilities(installed, has_paired_mmproj=False)

    assert "architecture_features_unsupported" in capabilities


def test_effective_capabilities_does_not_flag_a_checkpoint_with_no_gap(tmp_path, monkeypatch):
    gguf_path = _build_untemplated_llama_gguf(tmp_path / "model.gguf")
    installed = _installed("test:latest", gguf_path, "llama", ["completion"])
    monkeypatch.setattr("app.models.capabilities.unsupported_features", lambda path: [])

    capabilities = effective_capabilities(installed, has_paired_mmproj=False)

    assert "architecture_features_unsupported" not in capabilities


def test_effective_capabilities_reports_tools_for_mistral3_but_not_an_untemplated_model(tmp_path):
    mistral = build_tiny_mistral3_gguf(tmp_path / "m.gguf")
    llama = _build_untemplated_llama_gguf(tmp_path / "l.gguf")

    with_tools = effective_capabilities(
        _installed("a:latest", mistral, "mistral3", ["completion"]), has_paired_mmproj=False
    )
    without = effective_capabilities(
        _installed("b:latest", llama, "llama", ["completion"]), has_paired_mmproj=False
    )

    assert "tools" in with_tools
    assert "tools" not in without
