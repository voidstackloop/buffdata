import asyncio
import random
import time
from typing import Callable, Coroutine, TypeVar
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter, retry_if_not_exception_type

from buffdata.engine.client import ProviderError

T = TypeVar("T")

class AsyncRateLimiter:
    """Token bucket rate limiter with concurrency limits and jittered exponential retry."""

    def __init__(
        self,
        max_rpm: int = 60,
        concurrency: int = 10,
        max_retries: int = 5,
    ):
        self.max_rpm = max_rpm
        self.concurrency = concurrency
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(concurrency)
        self._lock = asyncio.Lock()
        self._timestamps: list[float] = []

    async def acquire(self):
        """Acquire permission under RPM and concurrency constraints."""
        await self._semaphore.acquire()
        while True:
            sleep_time = 0.0
            async with self._lock:
                now = time.monotonic()
                # Prune timestamps older than 60s
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]
                if len(self._timestamps) < self.max_rpm:
                    self._timestamps.append(time.monotonic())
                    return
                sleep_time = 60.0 - (now - self._timestamps[0]) + 0.05
            
            # Sleep OUTSIDE the lock to prevent massive traffic jams!
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def release(self):
        """Release concurrency slot."""
        self._semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()

    async def execute_with_retry(self, func: Callable[[], Coroutine[None, None, T]]) -> T:
        """Execute an async operation with automatic retry on transient or rate-limit errors.

        ProviderError (a missing API key, an unsupported provider, a network-policy block)
        is never retried -- none of those recover by waiting and trying again, so retrying
        them only adds up to ~15s of pointless exponential backoff before failing anyway.
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(initial=1.0, max=30.0, jitter=1.0),
            retry=retry_if_not_exception_type(ProviderError),
            reraise=True,
        ):
            with attempt:
                async with self:
                    return await func()
