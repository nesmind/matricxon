"""A fresh install's defaults (no `.env`, no env vars): the memory safety margin, quantized-native
compute with the C kernels, quiet logging. `_env_file=None` so a developer's local `.env` can't
mask a changed default."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_fresh_install_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.memory_safety_margin == 1.2
    assert settings.enable_quantized_native_compute is True
    assert settings.gemv_backend == "native"
    assert settings.log_level == 0


def test_memory_safety_margin_accepts_its_default_and_rejects_beyond_the_bounds() -> None:
    assert Settings(_env_file=None, memory_safety_margin=1.8).memory_safety_margin == 1.8
    for bad in (1.0, 1.9):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, memory_safety_margin=bad)
