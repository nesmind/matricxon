import torch

from app import config
from app.main import MatricxonApp


class TestMatricxonAppTorchThreads:
    def test_defaults_to_cpu_count_on_cpu_device(self, monkeypatch) -> None:
        original = torch.get_num_threads()
        try:
            monkeypatch.setattr(config.settings, "device", "cpu")
            monkeypatch.setattr(config.settings, "torch_threads", None)
            monkeypatch.setattr("os.cpu_count", lambda: 7)

            MatricxonApp()

            assert torch.get_num_threads() == 7
        finally:
            torch.set_num_threads(original)

    def test_honors_an_explicit_override(self, monkeypatch) -> None:
        original = torch.get_num_threads()
        try:
            monkeypatch.setattr(config.settings, "device", "cpu")
            monkeypatch.setattr(config.settings, "torch_threads", 3)

            MatricxonApp()

            assert torch.get_num_threads() == 3
        finally:
            torch.set_num_threads(original)

    def test_leaves_thread_count_untouched_on_a_non_cpu_device(self, monkeypatch) -> None:
        real_set_num_threads = torch.set_num_threads

        def fail_if_called(_n: int) -> None:
            raise AssertionError("must not touch thread count on a non-CPU device")

        original = torch.get_num_threads()
        try:
            monkeypatch.setattr(config.settings, "device", "cuda")
            monkeypatch.setattr(config.settings, "torch_threads", None)
            monkeypatch.setattr(torch, "set_num_threads", fail_if_called)

            MatricxonApp()
        finally:
            real_set_num_threads(original)
