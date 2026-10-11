from app.architectures.registry import ArchitectureRegistry, unsupported_features
from app.models.installed_model import InstalledModel
from app.models.tool_support import supports_tools
from app.runtime.chat_template import has_confirmed_chat_format

_THINKING_MARKERS = ("thinking", "reasoning")
_ENCODER_ARCHITECTURES = ("bert", "nomic-bert")
# A `clip` mmproj file (a vision tower + projector, e.g. a LLaVA pull's `:mmproj` suffix) isn't
# `ArchitectureRegistry`-resolvable alone ("clip" isn't in `_ARCHITECTURES`): a `/api/chat` against
# it would fail, so it must never be reported "completion" (it previously fell into the "anything
# else registered" default of a text model).
_MMPROJ_ARCHITECTURE = "clip"
# Image-embedding fusion (LlamaArchitecture._forward_impl) is wired only for these architectures. A
# clip file paired with any other (mistral3/gemma4/bert/nomic-bert) has `image_embeddings` accepted
# and ignored, so reporting "vision" for it would be false. Extend this set, not the pairing
# detection, when a new architecture gets fusion. "phi2" was added once
# Phi2Architecture._forward_impl got the same splice (moondream2 is its reference vision model).
_VISION_FUSION_ARCHITECTURES = frozenset(ArchitectureRegistry().vision_supported_names())


class CapabilityInferer:
    """GGUF has no standard capabilities field (it is Ollama's manifest-layer convention), so they
    are inferred.

    "embedding" for the registered encoder architectures (`bert`, `nomic-bert`, matched by
    `general.architecture` string, not via `ArchitectureRegistry`, to avoid an import cycle),
    "completion" for anything else registered, empty for a `clip` mmproj (`_MMPROJ_ARCHITECTURE`).
    "thinking" is a small manual override by repo/filename substring (no local model needs it yet).
    "vision" is never reported here; see `effective_capabilities`.

    Takes a bare `architecture` string, so a caller with only that, e.g. `/api/infer-capabilities`
    (app.routers.capabilities_router, for pAIring's direct Hugging Face download), can use it
    without reimplementing this.
    """

    def infer(self, repo_id: str, filename: str, architecture: str) -> list[str]:
        if architecture == _MMPROJ_ARCHITECTURE:
            return []
        is_encoder = architecture in _ENCODER_ARCHITECTURES
        capabilities = ["embedding" if is_encoder else "completion"]
        haystack = f"{repo_id}/{filename}".lower()
        if any(marker in haystack for marker in _THINKING_MARKERS):
            capabilities.append("thinking")
        return capabilities


def effective_capabilities(installed: InstalledModel, has_paired_mmproj: bool) -> list[str]:
    """The request-time capability list for `installed`: its stored `capabilities` (computed at pull
    time by `CapabilityInferer.infer`) plus:

    - `"vision"` when `has_paired_mmproj` (`ModelCatalog.find_paired_mmproj`) is true and
      `installed.architecture` has image-embedding fusion (`_VISION_FUSION_ARCHITECTURES`): a paired
      mmproj proves a vision tower exists, not that this text architecture can use its output.
      Computed per request, not at pull time, since an mmproj can be pulled after the text model
      (`tags_router.py`/`show_router.py` pass the current catalog).
    - `"chat_format_unverified"` for a "completion" model with no confirmed chat template
      (`app.runtime.chat_template.has_confirmed_chat_format`: True for mistral3 or a model with its
      own/builtin template, False for `LegacyMistralPromptBuilder`'s generic guess), so pAIring's
      list can flag an unknown instruction format (a Hebrew Mistral fine-tune gave incoherent output
      with any prompt format yet looked like a normal model).
    - `"architecture_features_unsupported"` when the architecture class reports a gated-variant gap
      for this checkpoint (`app.architectures.registry.unsupported_features`; e.g. a gemma-4-E2B-it
      GGUF's Per-Layer Embeddings and cross-layer KV reuse were silently ignored by the dense-only
      `Gemma4Architecture`).
    - `"tools"` when the chat template shows a tool-call format matricxon can parse
      (`app.runtime.tool_call_formats`).
    """
    capabilities = installed.capabilities
    has_real_fusion = has_paired_mmproj and installed.architecture in _VISION_FUSION_ARCHITECTURES
    if has_real_fusion and "vision" not in capabilities:
        capabilities = [*capabilities, "vision"]
    format_confirmed = has_confirmed_chat_format(installed.path, tag=installed.tag)
    if "completion" in capabilities and not format_confirmed:
        capabilities = [*capabilities, "chat_format_unverified"]
    if "completion" in capabilities and supports_tools(installed.path):
        capabilities = [*capabilities, "tools"]
    if unsupported_features(installed.path):
        capabilities = [*capabilities, "architecture_features_unsupported"]
    return capabilities
