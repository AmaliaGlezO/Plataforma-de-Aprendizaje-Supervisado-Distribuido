# src/api/middleware/auth.py

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Aquí puedes agregar la lógica de autenticación
        token = request.headers.get("Authorization")
        
        if not token or not self.verify_token(token):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        
        response = await call_next(request)
        return response

    def verify_token(self, token: str) -> bool:
        # Lógica para verificar el token
        return token == "your_valid_token"  # Reemplaza con tu lógica de validación