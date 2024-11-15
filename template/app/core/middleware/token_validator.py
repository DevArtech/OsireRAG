import hashlib
from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.settings import get_settings


def validate_token(token: str) -> bool:
    salt = get_settings().SALT
    api_token = get_settings().API_TOKEN

    return hashlib.sha256((salt + token).encode()).hexdigest() == api_token


class TokenValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        accepted_paths = ["/ping", "/login", "/submit-token"]
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

            if not token:
                return RedirectResponse(url=get_settings().BASE_URL + "/login")

            if not validate_token(token):
                return JSONResponse(
                    status_code=401, content={"detail": "Invalid token"}
                )

        response = await call_next(request)
        return response
