"""
Middleware para la API REST.
Maneja autenticación, rate limiting y otras funcionalidades transversales.
"""

from .auth import AuthMiddleware, authenticate_request
from .rate_limiter import RateLimiterMiddleware

# Configuración de middleware
MIDDLEWARE_CONFIG = {
    "auth": {
        "secret_key": "your-secret-key-here",
        "algorithm": "HS256",
        "token_expire_minutes": 30
    },
    "rate_limit": {
        "requests_per_minute": 100,
        "burst_size": 20
    }
}

__all__ = [
    "AuthMiddleware",
    "authenticate_request",
    "RateLimiterMiddleware", 
    "MIDDLEWARE_CONFIG"
]