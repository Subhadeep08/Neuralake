import time
from collections import defaultdict

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


_limiter = InMemoryRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        key = request.headers.get("X-API-Key") or request.client.host
        if not _limiter.check(key):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return await call_next(request)
