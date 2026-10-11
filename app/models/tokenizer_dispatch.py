from app.gguf.metadata import GGUFMetadata
from app.models.handle import Tokenizer
from app.runtime.gemma_tokenizer import Gemma4Tokenizer
from app.runtime.sentencepiece_tokenizer import SentencePieceTokenizer
from app.runtime.tokenizer import GGUFTokenizer
from app.runtime.wordpiece_tokenizer import WordPieceTokenizer
from app.server.errors import UnsupportedArchitectureError


def build_tokenizer(metadata: GGUFMetadata) -> Tokenizer:
    """Dispatches on `tokenizer.ggml.model`, the key llama.cpp uses to pick a tokenizer: `"gpt2"`
    for the byte-level BPE decoder models use, `"bert"` for the WordPiece format of both local
    encoder models (checked against their GGUF metadata), `"llama"` for the
    SentencePiece-BPE of plain `llama`-arch models (M10, checked on a TinyLlama pull), `"gemma4"`
    for Gemma4's rank-based BPE without byte-remapping (M10, checked on a `google/gemma-4-12b-it`
    pull)."""
    tokenizer_model = metadata.require("tokenizer.ggml.model")
    if tokenizer_model == "gpt2":
        return GGUFTokenizer(metadata)
    if tokenizer_model == "bert":
        return WordPieceTokenizer(metadata)
    if tokenizer_model == "llama":
        return SentencePieceTokenizer(metadata)
    if tokenizer_model == "gemma4":
        return Gemma4Tokenizer(metadata)
    raise UnsupportedArchitectureError(f"Unsupported tokenizer.ggml.model: {tokenizer_model!r}")
