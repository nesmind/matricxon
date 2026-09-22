from pydantic import BaseModel


class DeleteRequest(BaseModel):
    model: str
