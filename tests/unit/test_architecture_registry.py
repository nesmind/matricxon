from app.architectures.base import ModelArchitecture
from app.architectures.bert import BertArchitecture
from app.architectures.gemma4 import Gemma4Architecture
from app.architectures.llama import LlamaArchitecture
from app.architectures.mistral3 import Mistral3TextArchitecture
from app.architectures.nomic_bert import NomicBertArchitecture
from app.architectures.phi2 import Phi2Architecture
from app.architectures.registry import ArchitectureRegistry


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
        ]

    def test_every_name_is_a_real_string_not_a_class_object(self) -> None:
        for name in ArchitectureRegistry().supported_names():
            assert isinstance(name, str)
            assert not isinstance(name, type)


def test_model_architecture_declares_name_as_a_class_var() -> None:
    # Every concrete subclass must set it - the base class deliberately leaves it
    # undeclared (no default) so a new architecture can't silently omit it.
    assert "NAME" not in ModelArchitecture.__dict__
