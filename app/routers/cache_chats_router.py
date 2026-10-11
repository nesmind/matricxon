from fastapi import APIRouter, Depends

from app.dependencies import get_model_manager
from app.models.manager import ModelManager
from app.schemas.cache_chats import ChatHoldResponse, ChatSpillBody, ChatSpillResponse

router = APIRouter()


@router.get("/api/cache/holds", response_model=ChatHoldResponse)
def holds_chat(tag: str, manager: ModelManager = Depends(get_model_manager)) -> ChatHoldResponse:
    """Does this node hold chat `tag` (pAIring's conversation id) in RAM? Routes the next message to
    the node with its processed state."""
    return ChatHoldResponse(ram=manager.holds_chat(tag))


@router.post("/api/cache/spill", response_model=ChatSpillResponse)
def spill_chat(
    body: ChatSpillBody, manager: ModelManager = Depends(get_model_manager)
) -> ChatSpillResponse:
    """Moves the chat out of this node's RAM onto the shared disk (models stay loaded) so another
    node can continue it. Returns once the files are written."""
    return ChatSpillResponse(spilled=manager.spill_chat(tag=body.tag))
