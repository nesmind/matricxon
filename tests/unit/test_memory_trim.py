"""MemoryTrimmer hands memory an unloaded model left in glibc's pools back to the system."""

import threading

from app.runtime.memory_trim import MemoryTrimmer


class _CountingTrimmer(MemoryTrimmer):
    DELAY_SECONDS = 0.05

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.done = threading.Event()

        def fake_malloc_trim(_pad: int) -> int:
            self.calls += 1
            self.done.set()
            return 1

        self._malloc_trim = fake_malloc_trim


def test_a_burst_of_schedules_trims_once() -> None:
    trimmer = _CountingTrimmer()
    for _ in range(5):
        trimmer.schedule()
    assert trimmer.done.wait(2)
    assert trimmer.calls == 1


def test_it_can_trim_again_after_a_trim() -> None:
    trimmer = _CountingTrimmer()
    trimmer.schedule()
    assert trimmer.done.wait(2)
    trimmer.done.clear()
    trimmer.schedule()
    assert trimmer.done.wait(2)
    assert trimmer.calls == 2


def test_without_glibc_it_does_nothing() -> None:
    trimmer = MemoryTrimmer()
    trimmer._malloc_trim = None
    trimmer.schedule()
    assert trimmer._timer is None


def test_glibc_malloc_trim_is_found_here() -> None:
    assert MemoryTrimmer()._malloc_trim is not None
