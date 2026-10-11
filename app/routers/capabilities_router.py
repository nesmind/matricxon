from fastapi import APIRouter

from app.models.capabilities import CapabilityInferer
from app.schemas.capabilities import InferCapabilitiesRequest, InferCapabilitiesResponse

router = APIRouter()


class InferCapabilitiesRequestHandler:
    """A matricxon-only extension, like /api/health (health_router.py): lets a caller (pAIring's
    direct-from-Hugging-Face download path, for this) get a capabilities verdict
    for a file it hasn't installed here without reimplementing CapabilityInferer, which would drift
    when the rules change. Pure computation: no I/O, no catalog, unlike /api/pull."""

    def __init__(self, inferer: CapabilityInferer) -> None:
        self._inferer = inferer

    def handle(self, request: InferCapabilitiesRequest) -> InferCapabilitiesResponse:
        capabilities = self._inferer.infer(request.repo_id, request.filename, request.architecture)
        return InferCapabilitiesResponse(capabilities=capabilities)


@router.post("/api/infer-capabilities", response_model=InferCapabilitiesResponse)
def post_infer_capabilities(request: InferCapabilitiesRequest) -> InferCapabilitiesResponse:
    return InferCapabilitiesRequestHandler(CapabilityInferer()).handle(request)
