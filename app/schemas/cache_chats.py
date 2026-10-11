from pydantic import BaseModel


class ChatHoldResponse(BaseModel):
    ram: bool


class ChatSpillBody(BaseModel):
    tag: str


class ChatSpillResponse(BaseModel):
    spilled: int
