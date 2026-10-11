from fastapi import APIRouter, Depends

from app.dependencies import get_node_id, get_storage_paths
from app.schemas.storage import StoragePathInfo, StorageResponse
from app.storage.paths import StorageInspector, StoragePaths

router = APIRouter()


class StorageHandler:
    """Per-node storage inventory: every path, its cluster role, size and free space."""

    def __init__(self, paths: StoragePaths, node_id: str, inspector: StorageInspector) -> None:
        self._paths = paths
        self._node_id = node_id
        self._inspector = inspector

    def report(self) -> StorageResponse:
        infos = []
        for entry in self._paths.entries():
            usage = self._inspector.usage(entry.path)
            infos.append(
                StoragePathInfo(
                    name=entry.name,
                    path=str(entry.path),
                    role=entry.role.value,
                    setting=entry.setting,
                    exists=usage.exists,
                    size_bytes=usage.size_bytes,
                    free_bytes=usage.free_bytes,
                )
            )
        return StorageResponse(node_id=self._node_id, paths=infos)


@router.get("/api/storage", response_model=StorageResponse)
def get_storage(
    paths: StoragePaths = Depends(get_storage_paths), node_id: str = Depends(get_node_id)
) -> StorageResponse:
    return StorageHandler(paths, node_id, StorageInspector()).report()
