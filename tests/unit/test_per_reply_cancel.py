"""Cancelling one reply must not touch another user's reply on the same model, and must never
unload the model (see ModelWorker.cancel_job / ModelManager.cancel_request)."""

import asyncio
import threading

import pytest

from app.models.worker import ModelWorker


async def _drain(results: "asyncio.Queue[object]") -> list[object]:
    items: list[object] = []
    while (item := await results.get()) is not None:
        items.append(item)
    return items


@pytest.mark.asyncio
async def test_cancelling_a_running_reply_leaves_the_queued_one_untouched() -> None:
    worker = ModelWorker()
    started, release = threading.Event(), threading.Event()

    def long_job():
        started.set()
        yield 0
        release.wait(5)  # still "generating" until the test lets it continue
        yield from range(1, 1000)

    first = worker.stream(long_job, request_id="a")
    second = worker.stream(lambda: iter(range(5)), request_id="b")
    assert await asyncio.to_thread(started.wait, 5)

    assert worker.cancel_job("a") is True
    release.set()

    assert len(await _drain(first)) <= 2  # halted right after the cancel
    assert await _drain(second) == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_a_reply_cancelled_while_queued_is_skipped_and_the_running_one_finishes() -> None:
    worker = ModelWorker()
    release = threading.Event()

    def blocker():
        release.wait(5)
        yield from range(3)

    running = worker.stream(blocker, request_id="a")
    queued = worker.stream(lambda: iter(range(5)), request_id="b")

    assert worker.cancel_job("b") is True
    release.set()

    assert await _drain(running) == [0, 1, 2]
    assert await _drain(queued) == []  # never ran


@pytest.mark.asyncio
async def test_stop_check_for_one_reply_ignores_another_replys_cancel() -> None:
    worker = ModelWorker()
    check_a, check_b = worker.stop_check_for("a"), worker.stop_check_for("b")
    worker.cancel_job("a")
    assert check_a() is True and check_b() is False


def test_cancelling_an_unknown_reply_reports_false() -> None:
    assert ModelWorker().cancel_job("nope") is False
