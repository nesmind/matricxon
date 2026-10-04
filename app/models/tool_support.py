"""Whether a model can do tool calling: its chat template shows a call format we can parse back."""

from pathlib import Path

from app.gguf.reader import GGUFReader
from app.runtime.tool_call_formats import ToolCallFormats

# Keyed by (path, mtime, size): read on every GET /api/tags, and a file's header never changes
# without the file itself changing (same pattern as has_confirmed_chat_format).
_cache: dict[tuple[str, int, int], bool] = {}


def supports_tools(gguf_path: str | Path) -> bool:
    path = Path(gguf_path)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _cache:
        metadata = GGUFReader(path).read().metadata
        template = metadata.get("tokenizer.chat_template")
        _cache[key] = ToolCallFormats.for_template(template, metadata.architecture) is not None
    return _cache[key]
