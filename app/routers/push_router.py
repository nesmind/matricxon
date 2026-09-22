from fastapi import APIRouter

from app.schemas.push import PushRequest
from app.server.errors import PushNotSupportedError

router = APIRouter()


@router.post("/api/push")
async def post_push(request: PushRequest) -> None:
    # Same reasoning as HFModelTag's own docstring for pull: matricxon has no
    # access to Ollama's proprietary registry protocol, and no registry of
    # its own to push to either - fail closed with a clear error rather than
    # silently pretending to succeed.
    raise PushNotSupportedError(
        f"matricxon has no registry to push {request.model!r} to - pull-only "
        "via hf.co:<repo>:<suffix> tags (see /api/pull)."
    )
