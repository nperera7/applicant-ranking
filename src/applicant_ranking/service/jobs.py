"""In-process job queue for webhook work: Lever needs a fast 2xx, scoring takes seconds
to minutes. Bounded ``asyncio.Queue`` + N worker tasks started/stopped by the FastAPI
lifespan. Jobs are de-duplicated by key while queued or running. Failures are logged
(no resume text) and never crash a worker.

Cloud Run note: background work after the response needs instance-based billing ("CPU
always allocated"); otherwise move this to Cloud Tasks/Pub/Sub.

Details:
- A failed job is logged with its key, the exception type and where it was raised, never
  the exception message (a validation error can carry model or resume text).
- ``stop`` stops accepting jobs at once, cancels jobs still waiting for their delay
  (``submit_later``), lets the workers finish what is queued for up to ``drain_timeout``
  seconds, then cancels whatever is still running and drops the rest (logged by count).
  A stopped queue cannot be restarted.
- ``submit_later`` enqueues a job after a delay (the webhook receiver re-queues an
  applicant whose resume Lever is still processing). Its key counts as pending from the
  moment it is scheduled, so the same key is refused meanwhile.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

JobFn = Callable[[], Awaitable[object]]


def _origin(exc: BaseException) -> str:
    """``file.py:123 in function`` of the innermost frame, for logs without the message."""
    tb = exc.__traceback__
    if tb is None:
        return "unknown location"
    while tb.tb_next is not None:
        tb = tb.tb_next
    code = tb.tb_frame.f_code
    return f"{Path(code.co_filename).name}:{tb.tb_lineno} in {code.co_name}"


class JobQueue:
    def __init__(self, *, workers: int = 2, maxsize: int = 1000) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self._workers = workers
        self._queue: asyncio.Queue[tuple[str, JobFn]] = asyncio.Queue(maxsize=maxsize)
        self._pending: set[str] = set()  # keys queued, running or waiting for a delay
        self._tasks: list[asyncio.Task[None]] = []
        self._timers: set[asyncio.Task[None]] = set()  # submit_later jobs still waiting
        self._closed = False
        self.completed = 0
        self.failed = 0
        self.dropped = 0

    @property
    def accepting(self) -> bool:
        return not self._closed

    @property
    def queued(self) -> int:
        """Jobs waiting for a worker (running jobs excluded)."""
        return self._queue.qsize()

    def is_pending(self, key: str) -> bool:
        """True while a job with ``key`` is queued or running."""
        return key in self._pending

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("JobQueue was stopped; create a new one")
        if self._tasks:
            return
        self._tasks = [asyncio.create_task(self._worker(), name=f"job-worker-{i}")
                       for i in range(self._workers)]
        logger.info("job queue started with %d worker(s)", self._workers)

    async def stop(self, *, drain_timeout: float = 30.0) -> None:
        self._closed = True
        timers, self._timers = list(self._timers), set()
        if timers:
            for timer in timers:
                timer.cancel()
            await asyncio.gather(*timers, return_exceptions=True)
            self.dropped += len(timers)
            logger.warning("job queue: %d delayed job(s) cancelled at shutdown", len(timers))
        self._queue.shutdown()  # workers exit once the queue is empty
        tasks, self._tasks = self._tasks, []
        if tasks:
            _, unfinished = await asyncio.wait(tasks, timeout=max(drain_timeout, 0.0))
            if unfinished:
                logger.warning("job queue: %d job(s) still running after %.1f s; cancelling",
                               len(unfinished), drain_timeout)
                for task in unfinished:
                    task.cancel()
                await asyncio.gather(*unfinished, return_exceptions=True)
        dropped = self._drop_remaining()
        if dropped:
            logger.warning("job queue: %d queued job(s) dropped at shutdown", dropped)
        self._pending.clear()
        logger.info("job queue stopped (%d completed, %d failed)", self.completed, self.failed)

    def submit(self, key: str, fn: JobFn) -> bool:
        """Enqueue unless a job with ``key`` is queued/running. False when duplicate or
        the queue is full (the caller still returns 2xx; re-running
        ``rank`` catches up)."""
        if self._closed:
            logger.warning("job %s not queued: the queue is stopping", key)
            return False
        if key in self._pending:
            logger.info("job %s not queued: already queued or running", key)
            return False
        try:
            self._queue.put_nowait((key, fn))
        except asyncio.QueueFull:
            logger.warning("job %s not queued: the queue is full (%d waiting)", key,
                           self._queue.qsize())
            return False
        except asyncio.QueueShutDown:
            return False
        self._pending.add(key)
        return True

    def submit_later(self, key: str, fn: JobFn, delay_seconds: float) -> bool:
        """``submit`` after ``delay_seconds``. The key is pending at once (a duplicate is
        refused meanwhile); a timer task sleeps, then enqueues the job. False when the key
        is already pending or the queue is stopping; a queue that is full or stopped when
        the delay elapses drops the job (logged, counted in ``dropped``)."""
        if self._closed:
            logger.warning("job %s not scheduled: the queue is stopping", key)
            return False
        if key in self._pending:
            logger.info("job %s not scheduled: already queued or running", key)
            return False
        self._pending.add(key)
        timer = asyncio.create_task(self._enqueue_later(key, fn, max(delay_seconds, 0.0)),
                                    name=f"job-timer-{key}")
        self._timers.add(timer)
        timer.add_done_callback(self._timers.discard)
        return True

    async def _enqueue_later(self, key: str, fn: JobFn, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self._pending.discard(key)
            raise
        if not self._closed:
            try:
                self._queue.put_nowait((key, fn))
                return
            except (asyncio.QueueFull, asyncio.QueueShutDown):
                pass
        self._pending.discard(key)
        self.dropped += 1
        logger.warning("job %s dropped after its delay: the queue is full or stopping", key)

    async def join(self) -> None:
        """Wait until nothing is queued, running or waiting for its delay (tests)."""
        while True:
            timers = list(self._timers)
            if timers:
                await asyncio.gather(*timers, return_exceptions=True)
            await self._queue.join()
            if not self._timers:
                return

    async def _worker(self) -> None:
        while True:
            try:
                key, fn = await self._queue.get()
            except asyncio.QueueShutDown:
                return
            try:
                await fn()
            except Exception as exc:
                self.failed += 1
                logger.error("job %s failed: %s at %s", key, type(exc).__name__, _origin(exc))
            else:
                self.completed += 1
            finally:
                self._pending.discard(key)
                self._queue.task_done()

    def _drop_remaining(self) -> int:
        dropped = 0
        while True:
            try:
                key, _ = self._queue.get_nowait()
            except (asyncio.QueueEmpty, asyncio.QueueShutDown):
                break
            self._pending.discard(key)
            self._queue.task_done()
            dropped += 1
        self.dropped += dropped
        return dropped
