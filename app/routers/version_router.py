from fastapi import APIRouter

from app.config import MATRICXON_VERSION
from app.schemas.version import VersionResponse

router = APIRouter()


@router.get("/api/version", response_model=VersionResponse)
async def get_version() -> VersionResponse:
    return VersionResponse(version=MATRICXON_VERSION)
