"""Client-side token bucket so we stay under Lever's per-key limit (10 rps, 20 burst)."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable

# Float slack so a waiter whose sleep was computed exactly isn't sent back to sleep for a
# rounding error's worth of tokens.
_EPSILON = 1e-9


class AsyncTokenBucket:
    def __init__(self, rate_per_second: float, burst: int, *,
                 clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], object] | None = None) -> None:
        """``clock`` (monotonic seconds) and ``sleep`` (an async sleep) are injectable for
        tests. The bucket starts full."""
        if not rate_per_second > 0:
            raise ValueError("rate_per_second must be > 0")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        self._rate = float(rate_per_second)
        self._capacity = float(burst)
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._tokens = self._capacity
        self._updated = self._clock()
        # asyncio.Lock wakes waiters in FIFO order: whoever queued first is served first,
        # and the holder keeps the lock while it sleeps for its tokens, so a stream of
        # small requests can't starve an earlier one.
        self._lock = asyncio.Lock()

    @property
    def rate_per_second(self) -> float:
        return self._rate

    @property
    def capacity(self) -> float:
        return self._capacity

    def available(self) -> float:
        """Tokens available right now (refills first)."""
        self._refill()
        return self._tokens

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until ``tokens`` are available, then take them. Safe under concurrency
        (callers are served without starvation)."""
        if not tokens > 0:
            raise ValueError("tokens must be > 0")
        if tokens > self._capacity:
            raise ValueError(f"cannot acquire {tokens} tokens from a bucket of {self._capacity:g}")
        async with self._lock:
            while True:
                self._refill()
                if self._tokens + _EPSILON >= tokens:
                    self._tokens = max(0.0, self._tokens - tokens)
                    return
                await self._wait((tokens - self._tokens) / self._rate)

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        # A clock that steps backwards (it shouldn't: monotonic) never removes tokens.
        self._updated = max(self._updated, now)

    async def _wait(self, seconds: float) -> None:
        result = self._sleep(seconds)
        if inspect.isawaitable(result):
            await result
