from pydantic import BaseModel, Field

from app.runtime.rerank_engine import DEFAULT_INSTRUCTION


class RerankRequest(BaseModel):
    model: str
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=1000)
    # Keep only the best `top_n`; None returns every document.
    top_n: int | None = Field(default=None, ge=1)
    instruction: str = DEFAULT_INSTRUCTION
    return_documents: bool = False
    # Max tokens for one query + document prompt.
    max_tokens: int = Field(default=8192, ge=64)
    keep_alive: int | None = None


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str | None = None


class RerankResponse(BaseModel):
    model: str
    # Best first.
    results: list[RerankResult]
