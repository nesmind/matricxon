"""CPU temperature guard for the manual benchmarks. Added after a native-kernel benchmark
(4 threads) caused a thermal shutdown on this laptop (Sandy Bridge CPU); long all-core runs need
a cutoff.

Reads the `coretemp` hwmon sensor (Linux); without one every check is a no-op.
"""

import json
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


class OverheatError(RuntimeError):
    pass


class ThermalGuard:
    def __init__(self, max_c: float = 90.0, resume_c: float = 72.0, poll_s: float = 1.0) -> None:
        self.max_c = max_c
        self.resume_c = resume_c
        self._poll_s = poll_s
        self._inputs = self._find_inputs()

    @staticmethod
    def _find_inputs() -> list[Path]:
        for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
            name = hwmon / "name"
            if name.exists() and name.read_text().strip() == "coretemp":
                return sorted(hwmon.glob("temp*_input"))
        return []

    def read_c(self) -> float | None:
        """Hottest coretemp reading in degrees C, or None without a sensor."""
        if not self._inputs:
            return None
        return max(int(path.read_text()) for path in self._inputs) / 1000.0

    def wait_until_cool(self) -> None:
        """Blocks until the CPU is at or below `resume_c`, so each timed run starts from the same
        thermal state."""
        temp = self.read_c()
        while temp is not None and temp > self.resume_c:
            print(f"  cooling down: {temp:.0f} C > {self.resume_c:.0f} C, waiting...", flush=True)
            time.sleep(10)
            temp = self.read_c()

    @contextmanager
    def watch(self, on_overheat: Callable[[], None]) -> Iterator[None]:
        """Polls in a background thread; at `max_c` calls `on_overheat` once (which must stop the
        work) and raises OverheatError when the block exits."""
        stop = threading.Event()
        tripped: list[float] = []

        def poll() -> None:
            while not stop.wait(self._poll_s):
                temp = self.read_c()
                if temp is not None and temp >= self.max_c:
                    tripped.append(temp)
                    on_overheat()
                    return

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
        if tripped:
            raise OverheatError(f"CPU reached {tripped[0]:.0f} C (limit {self.max_c:.0f} C)")


class MatricxonWatchdog:
    """`python tools/thermal_guard.py` runs beside a live matricxon and, at `max_c`, unloads every
    model: this stops in-flight generation (ModelManager.unload -> ModelWorker.request_stop) and
    frees the CPU, keeping the server up."""

    def __init__(self, guard: ThermalGuard, host: str) -> None:
        self._guard = guard
        self._host = host.rstrip("/")

    def _post(self, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self._host}{path}", data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read() or b"{}")

    def unload_all(self) -> list[str]:
        names = [m["name"] for m in self._post("/api/ps").get("models", [])]
        for name in names:
            self._post("/api/generate", {"model": name, "keep_alive": 0})
        return names

    def run(self) -> None:
        """After `max_c`, keeps unloading every second until the CPU is back at `resume_c` (one
        unload plus a passive wait let the next chat reload the model and hold 96-97 C for
        minutes)."""
        print(f"watching CPU temperature, unloading models at {self._guard.max_c:.0f} C (Ctrl+C)")
        overheated = False
        while True:
            temp = self._guard.read_c()
            if temp is not None:
                if temp >= self._guard.max_c:
                    overheated = True
                elif temp <= self._guard.resume_c:
                    overheated = False
            if overheated:
                try:
                    names = self.unload_all()
                except OSError as exc:
                    names = [f"(matricxon unreachable: {exc})"]
                if names:
                    print(
                        f"{time.strftime('%H:%M:%S')} {temp:.0f} C - unloaded {names}", flush=True
                    )
            time.sleep(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=MatricxonWatchdog.__doc__)
    parser.add_argument("--host", default="http://localhost:8420")
    parser.add_argument("--max-temp", type=float, default=92.0)
    parser.add_argument("--resume-temp", type=float, default=85.0)
    args = parser.parse_args()
    guard = ThermalGuard(max_c=args.max_temp, resume_c=args.resume_temp)
    MatricxonWatchdog(guard, args.host).run()
