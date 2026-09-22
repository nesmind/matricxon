from pydantic import BaseModel, Field


class CreateRequest(BaseModel):
    model: str
    from_: str = Field(alias="from")
    stream: bool = True
