import asyncio
import pytest
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
