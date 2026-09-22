from pydantic import BaseModel


class ToolFunctionCall(BaseModel):
    name: str
    arguments: dict[str, object] | str = {}


class ToolCall(BaseModel):
    function: ToolFunctionCall


class ChatMessage(BaseModel):
    role: str
    content: str
    images: list[str] | None = None
    # Only meaningful on an "assistant" message (a prior turn where the model
    # itself called a tool) - Ollama/OpenAI's real wire shape, matched
    # 1:1 so a real tool-calling client's conversation history round-trips.
    tool_calls: list[ToolCall] | None = None


class ChatOptions(BaseModel):
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_ctx: int = 2048
    num_predict: int = -1
    seed: int | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    # Opaque passthrough (OpenAI/Ollama's real JSON-schema tool-definition
    # shape) - matricxon never validates a tool's schema itself, only
    # renders it verbatim into `[AVAILABLE_TOOLS]<json>[/AVAILABLE_TOOLS]`
    # (see Mistral3PromptBuilder), so there's no need to model it field by
    # field.
    tools: list[dict] | None = None
    options: ChatOptions = ChatOptions()
    stream: bool = True
    keep_alive: int | None = None

    def is_unload_call(self) -> bool:
        return not self.messages and self.keep_alive == 0


class ChatStreamMessage(BaseModel):
    content: str


class ChatChunk(BaseModel):
    message: ChatStreamMessage
    done: bool = False

    def to_ndjson_dict(self) -> dict:
        return self.model_dump()


class ChatDoneChunk(BaseModel):
    message: ChatStreamMessage = ChatStreamMessage(content="")
    done: bool = True
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_duration: int | None = None
    eval_duration: int | None = None

    def to_ndjson_dict(self) -> dict:
        return self.model_dump(exclude_none=True)
