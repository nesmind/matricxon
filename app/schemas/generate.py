from pydantic import BaseModel


class GenerateOptions(BaseModel):
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_ctx: int = 2048
    num_predict: int = -1
    seed: int | None = None


class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    options: GenerateOptions = GenerateOptions()
    stream: bool = True
    keep_alive: int | None = None

    def is_unload_call(self) -> bool:
        return not self.prompt and self.keep_alive == 0


class GenerateChunk(BaseModel):
    response: str
    done: bool = False

    def to_ndjson_dict(self) -> dict:
        return self.model_dump()


class GenerateDoneChunk(BaseModel):
    response: str = ""
    done: bool = True
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_duration: int | None = None
    eval_duration: int | None = None

    def to_ndjson_dict(self) -> dict:
        return self.model_dump(exclude_none=True)
