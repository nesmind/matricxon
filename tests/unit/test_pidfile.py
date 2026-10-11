import os

from app.server.pidfile import PidFile


class TestPidFileWrite:
    def test_writes_this_processs_own_pid(self, tmp_path) -> None:
        path = tmp_path / "run" / "matricxon.pid"
        PidFile(path).write()

        assert path.read_text() == str(os.getpid())

    def test_creates_missing_parent_directories(self, tmp_path) -> None:
        path = tmp_path / "nested" / "run" / "matricxon.pid"
        PidFile(path).write()

        assert path.exists()

    def test_overwrites_an_existing_stale_file(self, tmp_path) -> None:
        path = tmp_path / "matricxon.pid"
        path.write_text("999999")

        PidFile(path).write()

        assert path.read_text() == str(os.getpid())


class TestPidFileRemove:
    def test_removes_an_existing_file(self, tmp_path) -> None:
        path = tmp_path / "matricxon.pid"
        path.write_text(str(os.getpid()))

        PidFile(path).remove()

        assert not path.exists()

    def test_is_a_noop_when_the_file_does_not_exist(self, tmp_path) -> None:
        path = tmp_path / "matricxon.pid"

        PidFile(path).remove()

        assert not path.exists()


def test_a_pid_file_left_by_a_process_that_is_gone_means_the_last_run_died(tmp_path):
    import subprocess
    import sys

    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    path = tmp_path / "pid"
    path.write_text(str(gone.pid))
    assert PidFile(path).previous_run_died() is True

    path.write_text(str(os.getpid()))
    assert PidFile(path).previous_run_died() is False
    path.unlink()
    assert PidFile(path).previous_run_died() is False
    path.write_text("not a pid")
    assert PidFile(path).previous_run_died() is False
