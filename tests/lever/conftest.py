"""Fixtures for the Lever client, rate limiter and webhook tests.

``data/*.json`` are example payloads copied from Lever's Data API docs
(hire.lever.co/developer/documentation, snapshot of 2026-10-08); names and ids are Lever's
own fictional examples. Changes from the docs: ``application.json`` is trimmed and points
at the ``opportunity.json`` id; the docs' candidateStageChange delivery is missing a comma
(fixed); the applicationCreated envelope is assembled from the documented attribute values
because the docs show no example delivery for it.

Everything runs against ``httpx.MockTransport``; nothing here can reach the network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from applicant_ranking.lever.client import LeverClient
from applicant_ranking.lever.ratelimit import AsyncTokenBucket

DATA_DIR = Path(__file__).parent / "data"
API_KEY = "test-lever-key-do-not-log"
API_HOST = "api.lever.co"

Reply = httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]


class FakeClock:
    """Monotonic clock whose ``sleep`` advances time instantly (and records the delay)."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


class SleepRecorder:
    """Stands in for asyncio.sleep in the client's backoff: records, never waits."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


class CountingBucket(AsyncTokenBucket):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.acquired = 0

    async def acquire(self, tokens: float = 1.0) -> None:
        self.acquired += 1
        await super().acquire(tokens)


class FakeLever:
    """Scripted Lever API. ``on(method, path, *replies)`` queues replies for a route; the
    last reply repeats. Paths are relative to /v1 unless they are absolute https URLs."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._routes: dict[tuple[str, str, str], list[Reply]] = {}

    @staticmethod
    def _key(method: str, path: str) -> tuple[str, str, str]:
        if path.startswith(("https://", "http://")):
            url = httpx.URL(path)
            return method.upper(), url.host, url.path
        return method.upper(), API_HOST, "/v1" + path

    def on(self, method: str, path: str, *replies: Reply) -> None:
        assert replies
        self._routes[self._key(method, path)] = list(replies)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.host, request.url.path)
        queue = self._routes.get(key)
        if not queue:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(request)
        return reply

    def count(self, method: str, path: str) -> int:
        key = self._key(method, path)
        return sum(1 for r in self.requests
                   if (r.method, r.url.host, r.url.path) == key)


def load(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text())


@pytest.fixture
def payload() -> Callable[[str], Any]:
    return load


@pytest.fixture
def lever() -> FakeLever:
    return FakeLever()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def backoff() -> SleepRecorder:
    return SleepRecorder()


@pytest.fixture
def api_key() -> str:
    return API_KEY


@pytest.fixture
def bucket(clock: FakeClock) -> CountingBucket:
    return CountingBucket(1000.0, 20, clock=clock, sleep=clock.sleep)


@pytest.fixture
async def make_client(lever: FakeLever, bucket: CountingBucket,
                      backoff: SleepRecorder) -> AsyncIterator[Callable[..., LeverClient]]:
    """Factory for clients wired to ``lever``. Backoff sleeps are recorded in ``backoff``;
    jitter defaults to the full ceiling so delays are deterministic."""
    created: list[LeverClient] = []

    def _make(**overrides: Any) -> LeverClient:
        kwargs: dict[str, Any] = {
            "api_key": API_KEY,
            "transport": httpx.MockTransport(lever.handler),
            "bucket": bucket,
            "sleep": backoff,
            "jitter": lambda ceiling: ceiling,
            "backoff_base_seconds": 0.5,
            "backoff_cap_seconds": 8.0,
            "max_retries": 3,
        }
        kwargs.update(overrides)
        client = LeverClient(**kwargs)
        created.append(client)
        return client

    yield _make
    for client in created:
        await client.aclose()


@pytest.fixture
def client(make_client: Callable[..., LeverClient]) -> LeverClient:
    return make_client()
