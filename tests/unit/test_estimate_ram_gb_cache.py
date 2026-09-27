"""estimate_ram_gb's own per-file cache (app/models/load_dtype.py) - added because GET /api/tags
recomputed this from scratch (a real GGUFReader parse) for every installed model on every single
request, with no caching at all, which could make /api/tags intermittently slow under load (see
that cache's own docstring for the full, confirmed-live story). Asserted via a real GGUFReader.read
call count rather than comparing estimated_ram_gb's own output values - a tiny test fixture's real
tensor shapes are small enough that its rounded GB estimate is 0.0 regardless of layer count or
safety margin, which would make an output-based assertion pass even with caching totally broken."""

import os
from pathlib import Path

import pytest

from app.gguf.reader import GGUFReader
from app.models.load_dtype import estimate_ram_gb
from tests.tiny_gguf import build_tiny_mistral3_gguf


@pytest.fixture
def read_call_count(monkeypatch):
    real_read = GGUFReader.read
    counts = {"value": 0}

    def counting_read(self):
        counts["value"] += 1
        return real_read(self)

    monkeypatch.setattr(GGUFReader, "read", counting_read)
    return counts


def test_repeated_calls_for_the_same_unchanged_file_reuse_the_cached_result(
    tmp_path, read_call_count
):
    gguf_path = build_tiny_mistral3_gguf(tmp_path / "model.gguf")

    first = estimate_ram_gb(gguf_path, safety_margin=1.5)
    second = estimate_ram_gb(gguf_path, safety_margin=1.5)

    assert first == second
    assert read_call_count["value"] == 1


def test_a_different_safety_margin_triggers_its_own_real_read(tmp_path, read_call_count):
    gguf_path = build_tiny_mistral3_gguf(tmp_path / "model.gguf")

    estimate_ram_gb(gguf_path, safety_margin=1.1)
    estimate_ram_gb(gguf_path, safety_margin=1.8)

    assert read_call_count["value"] == 2


def test_a_file_replaced_at_the_same_path_invalidates_the_cache(tmp_path, read_call_count):
    gguf_path = build_tiny_mistral3_gguf(tmp_path / "model.gguf", n_layer=1)
    estimate_ram_gb(gguf_path, safety_margin=1.5)

    # A real re-pull landing at the same path won't usually collide mtimes down to the
    # nanosecond on a fast filesystem - nudge it forward so this test doesn't flake by coincidence
    # instead of actually exercising the invalidation path.
    build_tiny_mistral3_gguf(gguf_path, n_layer=4)
    stat = Path(gguf_path).stat()
    os.utime(gguf_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    estimate_ram_gb(gguf_path, safety_margin=1.5)

    assert read_call_count["value"] == 2
