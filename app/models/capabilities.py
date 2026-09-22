from app.models.installed_model import InstalledModel

_THINKING_MARKERS = ("thinking", "reasoning")
_ENCODER_ARCHITECTURES = ("bert", "nomic-bert")
# A `clip` mmproj file (a real vision tower + projector, e.g. a real LLaVA pull's `:mmproj`
# suffix) is never `ArchitectureRegistry`-resolvable on its own (confirmed: "clip" isn't in its
# `_ARCHITECTURES` list) - a real `/api/chat` against one directly would just fail, so it must
# never be reported "completion" (a real bug this fixes: it previously fell into the same
# "anything else registered" default as an actual text model).
_MMPROJ_ARCHITECTURE = "clip"
# Real image-embedding fusion (see LlamaArchitecture._forward_impl's own docstring) is only wired
# up for this one architecture today - a same-directory clip file paired with any OTHER
# architecture (mistral3/gemma4/bert/nomic-bert) still gets `image_embeddings` accepted-and-ignored
# (see each one's own `del image_embeddings` comment), so reporting "vision" for one would be a
# real, live false claim - confirmed exactly the kind of gap worth guarding against explicitly
# rather than assuming "a paired mmproj exists" alone always means real support (2026-09-21, found
# while making this check robust for future models: e.g. a real gemma4 vision model paired with its
# own real mmproj would otherwise have been wrongly reported vision-capable here). Extend this set,
# not the pairing-detection logic itself, whenever a future architecture gets real fusion wiring.
# "phi2" added once Phi2Architecture._forward_impl grew the same real image_embeddings splice
# LlamaArchitecture's own already has (moondream2, its own real reference vision model).
_VISION_FUSION_ARCHITECTURES = frozenset({"llama", "phi2"})


class CapabilityInferer:
    """GGUF has no standardized capabilities field - this is Ollama's own
    manifest-layer convention, not a GGUF one - so it has to be inferred.

    "embedding" for the registered encoder architectures (`bert`,
    `nomic-bert` - matched by `general.architecture` string rather than
    resolving through `ArchitectureRegistry`, to avoid a needless import
    cycle for what's just a name check), "completion" for anything else
    registered (only `mistral3` so far), empty for a `clip` mmproj file (see
    `_MMPROJ_ARCHITECTURE`'s own comment). "thinking" is a small manual
    override by repo/filename substring, per ROADMAP.md - no real local
    model needs it yet, so this is deliberately minimal rather than a real
    classifier. "vision" is never reported *here* - see `effective_capabilities`
    for why that's a real, separate, dynamic concern rather than something
    inferable at pull time from one file's own metadata alone.

    Takes a bare `architecture` string rather than a full `GGUFMetadata` (the only field this
    ever read off one) - lets a caller with just that one string, not a whole parsed GGUF header,
    call this directly too, e.g. app.routers.capabilities_router's `/api/infer-capabilities`,
    added for pAIring's own direct-from-Hugging-Face download path (2026-09-21) so it can get a
    real capabilities verdict without reimplementing this logic on its own side.
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
    """The real, request-time-visible capability list for `installed` - its own stored
    `capabilities` (computed once, at pull time, by `CapabilityInferer.infer` above) plus
    `"vision"` when `has_paired_mmproj` (see `ModelCatalog.find_paired_mmproj`) is true *and*
    `installed.architecture` actually has real image-embedding fusion wired up (see
    `_VISION_FUSION_ARCHITECTURES`'s own comment) - a paired mmproj file existing is necessary
    but not sufficient: it only proves a vision tower is available to encode with, not that this
    specific text architecture knows how to consume the result.

    Deliberately *not* folded into `CapabilityInferer.infer` itself: a paired mmproj can be
    pulled *after* the text model (confirmed live, 2026-09-21, this exact real case - LLaVA's
    text half was pulled in an earlier session, its mmproj only during this one), so baking
    "vision" into the stored sidecar at pull time would miss it for every model already
    installed before its sibling exists - this is computed fresh by the caller (see
    `tags_router.py`/`show_router.py`) against the *current* catalog state instead.
    """
    has_real_fusion = has_paired_mmproj and installed.architecture in _VISION_FUSION_ARCHITECTURES
    if has_real_fusion and "vision" not in installed.capabilities:
        return [*installed.capabilities, "vision"]
    return installed.capabilities
