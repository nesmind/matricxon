import threading
import time
from pathlib import Path

from app.pull.download_coordinator import DownloadCoordinator


class TestDownloadCoordinatorSerialization:
    def test_a_second_guard_for_the_same_dest_waits_for_the_first_to_finish(self) -> None:
        coordinator = DownloadCoordinator()
        dest = Path("/tmp/model.gguf")
        events: list[str] = []
        first_holding = threading.Event()
        release_first = threading.Event()

        def first() -> None:
            with coordinator.guard(dest):
                events.append("first-start")
                first_holding.set()
                release_first.wait(timeout=5)
                events.append("first-end")

        def second() -> None:
            first_holding.wait(timeout=5)
            with coordinator.guard(dest):
                events.append("second-start")

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        time.sleep(0.05)  # give `second` a real chance to block on the lock before `first` releases
        t2.start()
        release_first.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert events == ["first-start", "first-end", "second-start"]

    def test_two_different_dests_never_block_each_other(self) -> None:
        coordinator = DownloadCoordinator()
        both_holding = threading.Event()
        counter = {"holding": 0}
        lock = threading.Lock()

        def hold(dest: Path) -> None:
            with coordinator.guard(dest):
                with lock:
                    counter["holding"] += 1
                    if counter["holding"] == 2:
                        both_holding.set()
                assert both_holding.wait(timeout=5)

        t1 = threading.Thread(target=hold, args=(Path("/tmp/a.gguf"),))
        t2 = threading.Thread(target=hold, args=(Path("/tmp/b.gguf"),))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert both_holding.is_set()


class TestDownloadCoordinatorCleanup:
    def test_the_lock_slot_is_dropped_once_nothing_is_waiting(self) -> None:
        coordinator = DownloadCoordinator()
        dest = Path("/tmp/model.gguf")

        with coordinator.guard(dest):
            pass

        assert dest not in coordinator._locks
        assert dest not in coordinator._waiters

    def test_the_lock_slot_is_dropped_even_when_the_guarded_block_raises(self) -> None:
        coordinator = DownloadCoordinator()
        dest = Path("/tmp/model.gguf")

        try:
            with coordinator.guard(dest):
                raise ValueError("boom")
        except ValueError:
            pass

        assert dest not in coordinator._locks
