# Integration test fixtures

Integration tests validate matricxon's GGUF loader and architectures against
real, already-downloaded model files rather than synthetic data. These are
Ollama blobs living outside this repo (never copied into git):

| Tag | Source blob | Arch | Quant | Size |
|---|---|---|---|---|
| `all-minilm:latest` | `~/Code/Py/AI/pAIring/models/blobs/sha256-797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662` | `bert` | F16 | 46 MB |
| `nomic-embed-text:latest` | `~/Code/Py/AI/pAIring/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6` | `nomic-bert` | F16 | 274 MB |
| `ministral-3:3b` | `~/Code/Py/AI/pAIring/models/blobs/sha256-9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` | `mistral3` | Q4_K/Q6_K | 2.0 GB |

As of M5, `tests/integration/conftest.py`'s `real_ministral_model` fixture
wires the `ministral-3:3b` row in above directly: it symlinks (never copies,
given the ~2GB size) the real blob into a temp test `models_dir` alongside
a hand-written but metadata-accurate sidecar, and skips automatically on a
machine without pAIring's local blob store rather than failing. See
`tests/integration/test_api_real_fixture.py`. The embedding rows
(`all-minilm`, `nomic-embed-text`) aren't wired up yet - that's M8's job,
once `BertArchitecture`/`EmbeddingEngine` exist to actually use them.
