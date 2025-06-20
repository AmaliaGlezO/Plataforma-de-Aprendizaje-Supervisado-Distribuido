# src/api/middleware/rate_limiter.py

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time

class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit: int, time_window: int):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.requests = {}

    async def dispatch(self, request, call_next):
        client_ip = request.client.host
        current_time = time.time()

        # Limpiar las solicitudes antiguas
        self.requests.setdefault(client_ip, []).append(current_time)
        self.requests[client_ip] = [t for t in self.requests[client_ip] if t > current_time - self.time_window]

        if len(self.requests[client_ip]) > self.rate_limit:
            return JSONResponse({"detail": "Too Many Requests"}, status_code=429)

        response = await call_next(request)
        return response