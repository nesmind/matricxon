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

        PidFile(path).remove()  # must not raise

        assert not path.exists()
