from fastapi import APIRouter, Depends

from app.schemas.push import PushRequest
from app.server.admin_auth import require_admin
from app.server.errors import PushNotSupportedError
from app.server.node_kind import require_compute

router = APIRouter()


@router.post("/api/push", dependencies=[Depends(require_admin), Depends(require_compute)])
async def post_push(request: PushRequest) -> None:
    # As HFModelTag for pull: matricxon has no access to Ollama's closed registry protocol and no
    # registry of its own to push to; fail closed with a clear error instead of pretending to
    # succeed.
    raise PushNotSupportedError(
        f"matricxon has no registry to push {request.model!r} to - pull-only "
        "via hf.co:<repo>:<suffix> tags (see /api/pull)."
    )
