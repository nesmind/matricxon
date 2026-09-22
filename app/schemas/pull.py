from pydantic import BaseModel


class PullRequest(BaseModel):
    model: str
    stream: bool = True


class PullProgress(BaseModel):
    status: str
    completed: int | None = None
    total: int | None = None
    digest: str | None = None
    error: str | None = None

    def to_ndjson_dict(self) -> dict:
        return self.model_dump(exclude_none=True)
