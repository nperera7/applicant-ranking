"""JobQueue: de-duplication, bounds, failure isolation and draining."""

from __future__ import annotations

import asyncio
import logging

import pytest

from applicant_ranking.service.jobs import JobFn, JobQueue


def recorder(log: list[str], name: str) -> JobFn:
    async def job() -> None:
        log.append(name)

    return job


async def test_runs_jobs_and_join_waits() -> None:
    queue = JobQueue(workers=2)
    await queue.start()
    done: list[str] = []

    assert queue.submit("a", recorder(done, "a"))
    assert queue.submit("b", recorder(done, "b"))
    await queue.join()

    assert sorted(done) == ["a", "b"]
    assert queue.completed == 2 and queue.failed == 0
    assert not queue.is_pending("a")
    await queue.stop()


async def test_dedupes_while_queued() -> None:
    queue = JobQueue()
    done: list[str] = []

    assert queue.submit("same", recorder(done, "first"))
    assert not queue.submit("same", recorder(done, "second"))
    assert queue.is_pending("same")

    await queue.start()
    await queue.join()
    assert done == ["first"]
    assert queue.submit("same", recorder(done, "third"))  # finished: accepted again
    await queue.join()
    assert done == ["first", "third"]
    await queue.stop()


async def test_dedupes_while_running() -> None:
    queue = JobQueue(workers=1)
    await queue.start()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow() -> None:
        started.set()
        await release.wait()

    assert queue.submit("key", slow)
    await started.wait()
    assert queue.queued == 0
    assert not queue.submit("key", slow)  # running, not queued: still a duplicate
    release.set()
    await queue.join()
    assert queue.submit("key", slow)
    await queue.join()
    await queue.stop()


async def test_full_queue_returns_false_and_does_not_mark_pending() -> None:
    queue = JobQueue(workers=1, maxsize=1)
    done: list[str] = []

    assert queue.submit("a", recorder(done, "a"))
    assert not queue.submit("b", recorder(done, "b"))
    assert not queue.is_pending("b")

    await queue.start()
    await queue.join()
    assert queue.submit("b", recorder(done, "b"))
    await queue.join()
    assert done == ["a", "b"]
    await queue.stop()


async def test_worker_survives_a_failing_job(caplog: pytest.LogCaptureFixture) -> None:
    queue = JobQueue(workers=1)
    await queue.start()
    done: list[str] = []

    async def boom() -> None:
        raise ValueError("resume says: Jane Doe, 555-0100")

    with caplog.at_level(logging.INFO, logger="applicant_ranking.service.jobs"):
        assert queue.submit("bad", boom)
        assert queue.submit("good", recorder(done, "good"))
        await queue.join()

    assert done == ["good"]
    assert queue.failed == 1 and queue.completed == 1
    assert "job bad failed: ValueError at test_jobs.py" in caplog.text
    assert "Jane Doe" not in caplog.text  # the message is never logged
    assert all(r.exc_info is None for r in caplog.records)
    assert queue.submit("bad", boom)  # its key was released
    await queue.join()
    await queue.stop()


async def test_jobs_run_concurrently_up_to_the_worker_count() -> None:
    queue = JobQueue(workers=2)
    await queue.start()
    both = asyncio.Barrier(2)
    done: list[str] = []

    async def meet(name: str) -> None:
        await both.wait()  # deadlocks unless both jobs run at once
        done.append(name)

    queue.submit("x", lambda: meet("x"))
    queue.submit("y", lambda: meet("y"))
    await asyncio.wait_for(queue.join(), timeout=5)

    assert sorted(done) == ["x", "y"]
    await queue.stop()


async def test_stop_drains_queued_jobs() -> None:
    queue = JobQueue(workers=1)
    await queue.start()
    done: list[str] = []

    async def step(name: str) -> None:
        await asyncio.sleep(0)
        done.append(name)

    for name in "abcde":
        assert queue.submit(name, lambda name=name: step(name))
    await queue.stop()

    assert done == list("abcde")
    assert queue.dropped == 0
    assert not queue.accepting
    assert not queue.submit("late", recorder(done, "late"))


async def test_stop_cancels_stuck_jobs_after_the_timeout(caplog: pytest.LogCaptureFixture,
                                                         ) -> None:
    queue = JobQueue(workers=1)
    await queue.start()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def stuck() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    done: list[str] = []
    queue.submit("stuck", stuck)
    queue.submit("waiting", recorder(done, "waiting"))
    await started.wait()

    with caplog.at_level(logging.WARNING, logger="applicant_ranking.service.jobs"):
        await queue.stop(drain_timeout=0)

    assert cancelled.is_set()
    assert done == []
    assert queue.dropped == 1
    assert not queue.is_pending("stuck") and not queue.is_pending("waiting")
    assert "cancelling" in caplog.text
    assert "1 queued job(s) dropped" in caplog.text
    await asyncio.wait_for(queue.join(), timeout=1)  # nothing left unfinished


async def test_submit_later_delivers_after_the_delay_and_holds_the_key() -> None:
    queue = JobQueue(workers=1)
    await queue.start()
    done: list[str] = []

    assert queue.submit_later("late", recorder(done, "late"), 0.02)
    assert queue.is_pending("late") and queue.queued == 0
    assert not queue.submit("late", recorder(done, "dup"))  # refused while waiting
    assert not queue.submit_later("late", recorder(done, "dup"), 0)
    assert done == []
    await asyncio.wait_for(queue.join(), timeout=2)

    assert done == ["late"]
    assert queue.completed == 1 and not queue.is_pending("late")
    await queue.stop()


async def test_join_waits_for_a_job_scheduled_by_a_running_job() -> None:
    queue = JobQueue(workers=1)
    await queue.start()
    done: list[str] = []

    async def first() -> None:
        done.append("first")
        assert queue.submit_later("second", recorder(done, "second"), 0)

    assert queue.submit("first", first)
    await asyncio.wait_for(queue.join(), timeout=2)

    assert done == ["first", "second"]
    await queue.stop()


async def test_stop_cancels_jobs_waiting_for_their_delay(caplog: pytest.LogCaptureFixture,
                                                         ) -> None:
    queue = JobQueue(workers=1)
    await queue.start()
    done: list[str] = []
    assert queue.submit_later("later", recorder(done, "later"), 60)

    with caplog.at_level(logging.WARNING, logger="applicant_ranking.service.jobs"):
        await asyncio.wait_for(queue.stop(), timeout=2)

    assert done == [] and queue.dropped == 1
    assert not queue.is_pending("later")
    assert "1 delayed job(s) cancelled" in caplog.text
    assert not queue.submit_later("x", recorder(done, "x"), 0)


async def test_stop_before_start_drops_queued_jobs() -> None:
    queue = JobQueue()
    queue.submit("a", recorder([], "a"))

    await queue.stop()

    assert queue.dropped == 1
    with pytest.raises(RuntimeError):
        await queue.start()


async def test_start_is_idempotent() -> None:
    queue = JobQueue(workers=3)
    await queue.start()
    await queue.start()
    done: list[str] = []
    queue.submit("a", recorder(done, "a"))
    await queue.join()
    await queue.stop()
    assert done == ["a"]


@pytest.mark.parametrize("kwargs", [{"workers": 0}, {"maxsize": 0}])
def test_rejects_bad_sizes(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        JobQueue(**kwargs)
