import asyncio
import time
import pytest
from buffdata.engine.client import ProviderError
from buffdata.engine.limiter import AsyncRateLimiter

@pytest.mark.asyncio
async def test_rate_limiter_concurrency():
    limiter = AsyncRateLimiter(max_rpm=300, concurrency=2)
    active = 0
    max_observed = 0

    async def worker():
        nonlocal active, max_observed
        async with limiter:
            active += 1
            max_observed = max(max_observed, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*[worker() for _ in range(5)])
    assert max_observed <= 2


@pytest.mark.asyncio
async def test_execute_with_retry_does_not_retry_provider_error():
    # A missing API key, an unsupported provider, or a network-policy block will never
    # succeed on retry -- retrying them just burns ~15s of exponential backoff for nothing.
    limiter = AsyncRateLimiter(max_rpm=300, concurrency=2, max_retries=5)
    calls = {"n": 0}

    async def always_fails():
        calls["n"] += 1
        raise ProviderError("simulated policy block")

    started = time.perf_counter()
    with pytest.raises(ProviderError, match="simulated policy block"):
        await limiter.execute_with_retry(always_fails)
    elapsed = time.perf_counter() - started

    assert calls["n"] == 1  # no retries at all
    assert elapsed < 1.0  # would be ~15s+ if it went through the backoff schedule


@pytest.mark.asyncio
async def test_execute_with_retry_still_retries_other_exceptions():
    limiter = AsyncRateLimiter(max_rpm=300, concurrency=2, max_retries=3)
    calls = {"n": 0}

    async def fails_twice_then_succeeds():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("simulated transient failure")
        return "ok"

    result = await limiter.execute_with_retry(fails_twice_then_succeeds)
    assert result == "ok"
    assert calls["n"] == 3
