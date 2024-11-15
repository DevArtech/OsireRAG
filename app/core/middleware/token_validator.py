from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.settings import get_settings


class TokenValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        accepted_paths = ["/ping", "/assets", "/gradio_api"]
        if get_settings().ENVIRONMENT != "local" and not any(
            path in request.url.path for path in accepted_paths
        ):
            token = request.query_params.get("token")

            if not token:
                token = request.headers.get("APIToken")
                if token and token.startswith("Bearer "):
                    token = token[len("Bearer ") :]

            if not token:
                token = request.cookies.get("apitoken")

            if not token or not self.validate_token(token):
                return JSONResponse(
                    status_code=401, content={"detail": "Invalid token"}
                )

        response = await call_next(request)
        return response

    def validate_token(self, token: str) -> bool:
        return token == get_settings().API_TOKEN
