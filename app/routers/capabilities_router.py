from fastapi import APIRouter

from app.models.capabilities import CapabilityInferer
from app.schemas.capabilities import InferCapabilitiesRequest, InferCapabilitiesResponse

router = APIRouter()


class InferCapabilitiesRequestHandler:
    """A genuine matricxon-only extension, same category as /api/health (see health_router.py's
    own docstring) - lets a caller (pAIring's own direct-from-Hugging-Face download path, added
    2026-09-21 specifically for this) get a real capabilities verdict for a file it hasn't
    installed here without reimplementing CapabilityInferer's own logic on its own side, which
    would drift the moment this rule set changes. Pure computation, no I/O, no catalog
    dependency - unlike /api/pull, this never touches the filesystem or the network."""

    def __init__(self, inferer: CapabilityInferer) -> None:
        self._inferer = inferer

    def handle(self, request: InferCapabilitiesRequest) -> InferCapabilitiesResponse:
        capabilities = self._inferer.infer(request.repo_id, request.filename, request.architecture)
        return InferCapabilitiesResponse(capabilities=capabilities)


@router.post("/api/infer-capabilities", response_model=InferCapabilitiesResponse)
def post_infer_capabilities(request: InferCapabilitiesRequest) -> InferCapabilitiesResponse:
    return InferCapabilitiesRequestHandler(CapabilityInferer()).handle(request)
