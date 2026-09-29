"""A single custom Jinja2 tag for chat_template.py's own `ImmutableSandboxedEnvironment`.

Some newer official chat templates (Llama 3.1+, IBM Granite 3.x, and others following the same
convention) wrap the assistant's own turn in `{% generation %}...{% endgeneration %}`. Hugging
Face's real `apply_chat_template` registers a Jinja2 extension for these tags so it can record
which rendered characters came from that span (used to build assistant-only loss masks for
training). Vanilla Jinja2 has no built-in notion of them at all - `env.from_string(template)`
raises `TemplateSyntaxError: Encountered unknown tag 'generation'` for ANY template using them,
confirmed live against a real DictaLM-3.0-24B-Thinking GGUF (2026-09-29). Since that GGUF's
template gets compiled once per model on every `GET /api/tags` (see chat_template.py's own
`_has_confirmed_template`), the crash wasn't scoped to that one model - it took down the whole
model list, for every installed model, until the request finished failing.

matricxon only ever renders a prompt string, never trains on the output, so there's no consumer
for the mask offsets HF's own extension tracks - this is a transparent pass-through: the tagged
block's body renders exactly as if the tags weren't there.
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
