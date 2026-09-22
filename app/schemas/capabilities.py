from pydantic import BaseModel


class InferCapabilitiesRequest(BaseModel):
    repo_id: str
    filename: str
    architecture: str


class InferCapabilitiesResponse(BaseModel):
    capabilities: list[str]
