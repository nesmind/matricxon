from fastapi import APIRouter, Depends

from app.config import settings
from app.dependencies import get_model_manager
from app.models.manager import ModelManager
from app.schemas.runtime_settings import RuntimeSettingsResult, RuntimeSettingsUpdate
from app.server.admin_auth import require_admin
from app.server.runtime_settings import RuntimeSettingsApplier

router = APIRouter()


@router.put(
    "/api/runtime-settings",
    dependencies=[Depends(require_admin)],
    response_model=RuntimeSettingsResult,
)
def put_runtime_settings(
    change: RuntimeSettingsUpdate, manager: ModelManager = Depends(get_model_manager)
) -> RuntimeSettingsResult:
    """Changes settings without a restart; signed like the other admin routes."""
    return RuntimeSettingsApplier(manager, settings).apply(change)
