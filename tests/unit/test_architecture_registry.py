from pathlib import Path

import pytest

from app.architectures.base import ModelArchitecture
from app.architectures.bert import BertArchitecture
from app.architectures.command_r import CommandRArchitecture
from app.architectures.falcon import FalconArchitecture
from app.architectures.gemma4 import Gemma4Architecture
from app.architectures.granite import GraniteArchitecture
from app.architectures.granitemoe import GraniteMoeArchitecture
from app.architectures.llama import LlamaArchitecture
from app.architectures.mistral3 import Mistral3TextArchitecture
from app.architectures.nemotron_h import NemotronHArchitecture
from app.architectures.nomic_bert import NomicBertArchitecture
from app.architectures.phi2 import Phi2Architecture
from app.architectures.qwen2 import Qwen2Architecture
from app.architectures.qwen3 import Qwen3Architecture
from app.architectures.qwen35 import Qwen35Architecture
from app.architectures.registry import ArchitectureRegistry, unsupported_features
from app.architectures.starcoder2 import Starcoder2Architecture
from app.gguf.constants import GGUFValueType
from app.gguf.metadata import GGUFMetadata
from scripts.make_tiny_gguf import GGUFBuilder


class TestNameMatchesSupports:
    """Regression guard for the real refactor this required: `supports()`

    used to compare `metadata.architecture` against a literal string
    duplicated in each subclass; now both `supports()` and `NAME` read
    from the same single source (see ModelArchitecture.NAME's own
    docstring) - this pins that they can never drift apart again.
    """

    def test_every_registered_architecture_declares_a_name(self) -> None:
        for architecture_cls in ArchitectureRegistry()._ARCHITECTURES:
            assert isinstance(architecture_cls.NAME, str)
            assert architecture_cls.NAME

    def test_name_matches_what_a_real_metadata_value_would_resolve_to(self) -> None:
        class _FakeMetadata:
            def __init__(self, architecture: str) -> None:
                self.architecture = architecture

        for architecture_cls in (
            Mistral3TextArchitecture,
            BertArchitecture,
            NomicBertArchitecture,
            LlamaArchitecture,
            Gemma4Architecture,
            Phi2Architecture,
            GraniteArchitecture,
            GraniteMoeArchitecture,
            NemotronHArchitecture,
            Qwen2Architecture,
            Qwen3Architecture,
            CommandRArchitecture,
            Starcoder2Architecture,
            FalconArchitecture,
        ):
            assert architecture_cls.supports(_FakeMetadata(architecture_cls.NAME)) is True
            assert architecture_cls.supports(_FakeMetadata("not-a-real-architecture")) is False


class TestSupportedNames:
    def test_lists_every_registered_architecture_by_name(self) -> None:
        names = ArchitectureRegistry().supported_names()

        assert names == [
            Mistral3TextArchitecture.NAME,
            BertArchitecture.NAME,
            NomicBertArchitecture.NAME,
            LlamaArchitecture.NAME,
            Gemma4Architecture.NAME,
            Phi2Architecture.NAME,
            GraniteArchitecture.NAME,
            GraniteMoeArchitecture.NAME,
            NemotronHArchitecture.NAME,
            Qwen2Architecture.NAME,
            Qwen3Architecture.NAME,
            Qwen35Architecture.NAME,
            CommandRArchitecture.NAME,
            Starcoder2Architecture.NAME,
            FalconArchitecture.NAME,
        ]

    def test_every_name_is_a_real_string_not_a_class_object(self) -> None:
        for name in ArchitectureRegistry().supported_names():
            assert isinstance(name, str)
            assert not isinstance(name, type)


class TestMoeSupportedNames:
    def test_lists_only_the_real_moe_capable_architectures(self) -> None:
        assert ArchitectureRegistry().moe_supported_names() == [
            LlamaArchitecture.NAME,
            Gemma4Architecture.NAME,
            GraniteMoeArchitecture.NAME,
        ]

    def test_every_moe_name_is_also_a_supported_name(self) -> None:
        registry = ArchitectureRegistry()
        assert set(registry.moe_supported_names()) <= set(registry.supported_names())


class TestVisionSupportedNames:
    def test_lists_only_the_architectures_that_really_fuse_images(self) -> None:
        assert ArchitectureRegistry().vision_supported_names() == [
            LlamaArchitecture.NAME,
            Phi2Architecture.NAME,
            Qwen35Architecture.NAME,
        ]

    def test_every_vision_name_is_also_a_supported_name(self) -> None:
        registry = ArchitectureRegistry()
        assert set(registry.vision_supported_names()) <= set(registry.supported_names())


def test_model_architecture_declares_name_as_a_class_var() -> None:
    # Every concrete subclass must set it - the base class deliberately leaves it
    # undeclared (no default) so a new architecture can't silently omit it.
    assert "NAME" not in ModelArchitecture.__dict__


class TestUnsupportedFeatures:
    """`ModelArchitecture.unsupported_features` (real gated-variant gaps a `supports()`-matched
    checkpoint's own metadata can still activate) - added 2026-09-29 after a real gemma-4-E2B-it
    GGUF's Per-Layer Embeddings/cross-layer KV reuse were silently never read at all by
    `Gemma4Architecture`, with nothing distinguishing it from a fully-supported checkpoint.
    `Gemma4Architecture` itself no longer has a known gap to test against (its own MoE variant,
    the original real example, was implemented 2026-09-29 too - see `gemma4_moe.py`) - the
    mechanism itself is still real and reusable, so it's exercised here through a minimal fake
    override rather than a real architecture's gap that may or may not exist at any given time.
    """

    def test_default_is_empty_for_every_registered_architecture(self) -> None:
        for architecture_cls in ArchitectureRegistry()._ARCHITECTURES:
            metadata = GGUFMetadata({"general.architecture": architecture_cls.NAME})
            assert architecture_cls.unsupported_features(metadata) == []

    def test_an_override_is_called_and_its_result_returned(self) -> None:
        class _FakeGapArchitecture(Gemma4Architecture):
            @classmethod
            def unsupported_features(cls, metadata: GGUFMetadata) -> list[str]:
                return ["fake-gap"]

        metadata = GGUFMetadata({"general.architecture": "gemma4"})
        assert _FakeGapArchitecture.unsupported_features(metadata) == ["fake-gap"]


def test_unsupported_features_resolves_the_real_architecture_from_a_gguf_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeGapArchitecture(Gemma4Architecture):
        @classmethod
        def unsupported_features(cls, metadata: GGUFMetadata) -> list[str]:
            return ["fake-gap"]

    monkeypatch.setattr(
        ArchitectureRegistry,
        "_ARCHITECTURES",
        [_FakeGapArchitecture, *ArchitectureRegistry._ARCHITECTURES],
    )
    path = (
        GGUFBuilder()
        .set_str("general.architecture", "gemma4")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, ["<unk>", "<s>", "</s>"])
        .add_tensor("token_embd.weight", [1], ggml_type=0, raw_bytes=b"\x00\x00\x00\x00")
        .write(tmp_path / "model.gguf")
    )

    assert unsupported_features(path) == ["fake-gap"]


def test_unsupported_features_is_empty_not_raised_for_an_unresolvable_architecture(
    tmp_path: Path,
) -> None:
    """A real, live regression (2026-09-29): a `clip` mmproj sidecar (never resolvable on its
    own - see app.models.capabilities._MMPROJ_ARCHITECTURE's own comment) crashed GET /api/tags
    for every installed model, not just itself, since effective_capabilities calls this once per
    installed tag inside that endpoint's own loop."""
    path = (
        GGUFBuilder()
        .set_str("general.architecture", "clip")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, ["<unk>", "<s>", "</s>"])
        .add_tensor("token_embd.weight", [1], ggml_type=0, raw_bytes=b"\x00\x00\x00\x00")
        .write(tmp_path / "mmproj.gguf")
    )

    assert unsupported_features(path) == []
