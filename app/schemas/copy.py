from pydantic import BaseModel


class CopyRequest(BaseModel):
    source: str
    destination: str
