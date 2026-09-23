"""The two engines scripts/benchmark_llamacpp.py compares, timed identically from the client side
(time to first token, then decode tokens/s) - split out of that script to keep it compact."""

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    ttft_s: float
    decode_tokens: int
    decode_s: float
    text: str

    @property
    def tokens_per_s(self) -> float:
        return self.decode_tokens / self.decode_s if self.decode_s > 0 else 0.0


class MatricxonRunner:
    def __init__(self, host: str, tag: str, options: dict) -> None:
        self._host = host.rstrip("/")
        self._tag = tag
        self._options = options

    def _post(self, path: str, body: dict) -> urllib.request.addinfourl:
        req = urllib.request.Request(
            f"{self._host}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=3600)

    def load(self) -> float:
        """Warm-up: a 1-token request so the timed runs below never include model load time."""
        started = time.monotonic()
        body = {
            "model": self._tag,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "keep_alive": 1800,
            "options": {**self._options, "num_predict": 1},
        }
        self._post("/api/chat", body).read()
        return time.monotonic() - started

    def chat(self, messages: list[dict]) -> RunResult:
        body = {
            "model": self._tag,
            "messages": messages,
            "stream": True,
            "keep_alive": 1800,
            "options": self._options,
        }
        started = time.monotonic()
        first_at = None
        tokens = 0
        parts = []
        with self._post("/api/chat", body) as resp:
            for line in resp:
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    tokens += 1  # matricxon streams one decoded token per chunk
                    parts.append(content)
                    first_at = first_at or time.monotonic()
                if chunk.get("done"):
                    tokens = chunk.get("eval_count", tokens)
                    break
        ended = time.monotonic()
        first_at = first_at or ended
        return RunResult(first_at - started, max(tokens - 1, 0), ended - first_at, "".join(parts))

    def unload(self) -> None:
        self._post("/api/generate", {"model": self._tag, "keep_alive": 0}).read()

    def abort(self) -> None:
        """Thermal-guard cutoff: unloading stops matricxon's in-flight generation (see
        ModelManager.unload -> ModelWorker.request_stop), which also ends the stream above."""
        self.unload()


class LlamaCppRunner:
    def __init__(self, gguf_path: Path, options: dict, threads: int) -> None:
        self._gguf_path = gguf_path
        self._options = options
        self._threads = threads
        self._llm = None
        self._aborted = False

    def abort(self) -> None:
        """Thermal-guard cutoff: ends the current stream loop at the next token."""
        self._aborted = True

    def load(self) -> float:
        from llama_cpp import Llama

        started = time.monotonic()
        self._llm = Llama(
            model_path=str(self._gguf_path),
            n_ctx=self._options["num_ctx"],
            n_threads=self._threads,
            seed=self._options["seed"],
            verbose=False,
        )
        return time.monotonic() - started

    def reset_cache(self) -> None:
        self._llm.reset()

    def chat(self, messages: list[dict]) -> RunResult:
        started = time.monotonic()
        first_at = None
        tokens = 0
        parts = []
        stream = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self._options["num_predict"],
            temperature=self._options["temperature"],
            seed=self._options["seed"],
            stream=True,
        )
        for chunk in stream:
            if self._aborted:
                break
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                tokens += 1  # llama-cpp-python streams one token per chunk
                parts.append(content)
                first_at = first_at or time.monotonic()
        ended = time.monotonic()
        first_at = first_at or ended
        return RunResult(first_at - started, max(tokens - 1, 0), ended - first_at, "".join(parts))
