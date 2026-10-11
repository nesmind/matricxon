from pydantic import BaseModel


class StoragePathInfo(BaseModel):
    name: str
    path: str
    role: str
    setting: str | None
    exists: bool
    size_bytes: int
    free_bytes: int | None


class StorageResponse(BaseModel):
    node_id: str
    paths: list[StoragePathInfo]
