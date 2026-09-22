from pydantic import BaseModel


class PushRequest(BaseModel):
    model: str
    stream: bool = True
