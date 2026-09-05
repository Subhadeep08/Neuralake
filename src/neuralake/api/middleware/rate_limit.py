import time
from collections import defaultdict

import redis.asyncio as aioredis
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from neuralake.config.settings import get_settings


class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self._windows: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> bool:
        now = time.monotonic()
        window = self._windows[key]
        window[:] = [t for t in window if now - t < 60]
        if len(window) >= self.rpm:
            return False
        window.append(now)
        return True


class RedisRateLimiter:
    def __init__(self, requests_per_minute: int = 60, redis_url: str = "redis://localhost:6379/0"):
        self.rpm = requests_per_minute
        self._redis = aioredis.from_url(redis_url, decode_responses=True)

    async def check(self, key: str) -> bool:
        redis_key = f"neuralake:ratelimit:{key}"
        pipe = self._redis.pipeline()
        pipe.incr(redis_key)
        pipe.ttl(redis_key)
        count, ttl = await pipe.execute()
        if ttl == -1:
            await self._redis.expire(redis_key, 60)
        return count <= self.rpm


def _get_limiter():
    settings = get_settings()
    if settings.redis.enabled:
        return RedisRateLimiter(redis_url=settings.redis.url)
    return InMemoryRateLimiter()


_limiter = None


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        global _limiter
        if _limiter is None:
            _limiter = _get_limiter()

        key = request.headers.get("X-API-Key") or request.client.host

        if isinstance(_limiter, RedisRateLimiter):
            allowed = await _limiter.check(key)
        else:
            allowed = _limiter.check(key)

        if not allowed:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return await call_next(request)
