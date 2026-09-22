from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    supported_architectures: list[str]
    supported_quantizations: list[str]
