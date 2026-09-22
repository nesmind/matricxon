from pydantic import BaseModel


class PsEntry(BaseModel):
    name: str
    size: int
    size_vram: int
    expires_at: str


class PsResponse(BaseModel):
    models: list[PsEntry]
