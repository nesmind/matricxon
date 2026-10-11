"""A custom Jinja2 tag for chat_template.py's `ImmutableSandboxedEnvironment`.

Newer chat templates (Llama 3.1+, IBM Granite 3.x, ...) wrap the assistant's turn in `{% generation
%}...{% endgeneration %}`. Hugging Face's `apply_chat_template` registers an extension for these
tags to mask training loss; vanilla Jinja2 raises `TemplateSyntaxError: Encountered unknown tag
'generation'` for any template using them (seen with a DictaLM GGUF). The template compiles once per
model on every `GET /api/tags` (chat_template.py `_has_confirmed_template`), so the crash took down
the whole model list.

matricxon only renders a prompt string and never trains on it, so the tag is a transparent
pass-through: the body renders as if the tags weren't there.
"""

from typing import Any

from jinja2 import nodes
from jinja2.ext import Extension


class GenerationTagExtension(Extension):
    tags = {"generation"}

    def parse(self, parser: Any) -> nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.CallBlock(self.call_method("_render", []), [], [], body).set_lineno(lineno)

    @staticmethod
    def _render(caller: Any) -> str:
        return caller()
