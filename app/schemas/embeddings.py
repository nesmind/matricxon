from pydantic import BaseModel


class EmbeddingsOptions(BaseModel):
    num_ctx: int = 2048


class EmbeddingsRequest(BaseModel):
    model: str
    prompt: str
    options: EmbeddingsOptions = EmbeddingsOptions()


class EmbeddingsResponse(BaseModel):
    embedding: list[float]
