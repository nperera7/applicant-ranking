from __future__ import annotations

import asyncio
from typing import Any

import pytest

from applicant_ranking.lever.ratelimit import AsyncTokenBucket


def _bucket(clock: Any, rate: float = 10.0, burst: int = 3) -> AsyncTokenBucket:
    return AsyncTokenBucket(rate, burst, clock=clock, sleep=clock.sleep)


async def test_starts_full_and_serves_a_burst_without_waiting(clock: Any) -> None:
    bucket = _bucket(clock, burst=3)
    for _ in range(3):
        await bucket.acquire()
    assert clock.sleeps == []
    assert bucket.available() == pytest.approx(0.0)


async def test_waits_for_refill_at_the_steady_rate(clock: Any) -> None:
    bucket = _bucket(clock, rate=10.0, burst=3)
    start = clock.now
    for _ in range(5):
        await bucket.acquire()
    # Two requests beyond the burst at 10/s: 0.1 s each.
    assert clock.now - start == pytest.approx(0.2)
    assert clock.sleeps == [pytest.approx(0.1), pytest.approx(0.1)]


async def test_refill_is_capped_at_burst(clock: Any) -> None:
    bucket = _bucket(clock, rate=10.0, burst=3)
    for _ in range(3):
        await bucket.acquire()
    clock.now += 3_600
    assert bucket.available() == pytest.approx(3.0)
    for _ in range(3):
        await bucket.acquire()
    assert clock.sleeps == []
    await bucket.acquire()
    assert clock.sleeps == [pytest.approx(0.1)]


async def test_partial_refill_shortens_the_wait(clock: Any) -> None:
    bucket = _bucket(clock, rate=10.0, burst=1)
    await bucket.acquire()
    clock.now += 0.06
    await bucket.acquire()
    assert clock.sleeps == [pytest.approx(0.04)]


async def test_concurrent_callers_are_served_in_arrival_order(clock: Any) -> None:
    bucket = _bucket(clock, rate=5.0, burst=2)
    served: list[int] = []

    async def worker(i: int) -> None:
        await bucket.acquire()
        served.append(i)

    tasks = [asyncio.create_task(worker(i)) for i in range(8)]
    await asyncio.gather(*tasks)
    assert served == list(range(8))
    # 6 beyond the burst at 5/s.
    assert clock.now - 1_000.0 == pytest.approx(6 / 5.0)


async def test_large_request_is_not_starved_by_later_small_ones(clock: Any) -> None:
    bucket = _bucket(clock, rate=1.0, burst=5)
    for _ in range(5):
        await bucket.acquire()
    order: list[str] = []

    async def take(name: str, tokens: float) -> None:
        await bucket.acquire(tokens)
        order.append(name)

    big = asyncio.create_task(take("big", 5))
    await asyncio.sleep(0)
    small = [asyncio.create_task(take(f"small{i}", 1)) for i in range(3)]
    await asyncio.gather(big, *small)
    assert order == ["big", "small0", "small1", "small2"]


async def test_cancelled_waiter_takes_no_tokens_and_does_not_block_others(clock: Any) -> None:
    blocked = asyncio.Event()

    async def never_returns(seconds: float) -> None:
        blocked.set()
        await asyncio.Event().wait()

    bucket = AsyncTokenBucket(1.0, 1, clock=clock, sleep=never_returns)
    await bucket.acquire()
    waiter = asyncio.create_task(bucket.acquire())
    await blocked.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    clock.now += 1.0
    await bucket.acquire()  # the lock was released and the refilled token is still there
    assert bucket.available() == pytest.approx(0.0)


async def test_accepts_a_synchronous_sleep(clock: Any) -> None:
    def sync_sleep(seconds: float) -> None:
        clock.now += seconds

    bucket = AsyncTokenBucket(4.0, 1, clock=clock, sleep=sync_sleep)
    await bucket.acquire()
    await bucket.acquire()
    assert clock.now == pytest.approx(1_000.25)


async def test_clock_going_backwards_never_removes_tokens(clock: Any) -> None:
    bucket = _bucket(clock, rate=10.0, burst=2)
    await bucket.acquire()
    clock.now -= 50
    assert bucket.available() == pytest.approx(1.0)


async def test_default_clock_and_sleep() -> None:
    bucket = AsyncTokenBucket(10.0, 2)
    await bucket.acquire()
    await bucket.acquire()
    assert bucket.capacity == 2
    assert bucket.rate_per_second == 10.0


@pytest.mark.parametrize("tokens", [0, -1, 4])
async def test_rejects_impossible_requests(clock: Any, tokens: float) -> None:
    bucket = _bucket(clock, burst=3)
    with pytest.raises(ValueError):
        await bucket.acquire(tokens)


@pytest.mark.parametrize(("rate", "burst"), [(0, 1), (-1, 1), (1, 0)])
def test_rejects_invalid_configuration(rate: float, burst: int) -> None:
    with pytest.raises(ValueError):
        AsyncTokenBucket(rate, burst)
