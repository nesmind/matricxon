import os

from fastapi.testclient import TestClient

from app.main import MatricxonApp


class TestMatricxonAppPidFileLifespan:
    def test_writes_pid_on_startup_and_removes_it_on_shutdown(self, tmp_path, monkeypatch) -> None:
        pid_path = tmp_path / "run" / "matricxon.pid"
        monkeypatch.setenv("MATRICXON_PID_FILE", str(pid_path))
        app = MatricxonApp().build()

        with TestClient(app):
            assert pid_path.read_text() == str(os.getpid())

        assert not pid_path.exists()
