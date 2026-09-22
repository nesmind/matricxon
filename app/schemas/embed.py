from pydantic import BaseModel


class EmbedOptions(BaseModel):
    num_ctx: int = 2048


class EmbedRequest(BaseModel):
    model: str
    input: str | list[str]
    options: EmbedOptions = EmbedOptions()

    def inputs(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else self.input


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
