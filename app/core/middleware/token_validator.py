"""
Module: token_validator.py
Middleware for validating API tokens.

Classes:
- TokenValidationMiddleware: Middleware for validating API tokens.

Functions:
- None

Usage:
- Add the TokenValidationMiddleware to the FastAPI app middleware stack.

Author: Adam Haile  
Date: 10/16/2024
"""

from fastapi import Request
from typing import Callable, Any
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.settings import get_settings


class TokenValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for validating API tokens.

    Attributes:
    - None

    Methods:
    - dispatch: Called by FastAPI when a request is received. Validates the API token in the request.
    - validate_token: Validates the API token in the request.

    Usage:
    - Add the TokenValidationMiddleware to the FastAPI app middleware stack.

    Author: Adam Haile
    Date: 10/16/2024
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        """
        Called by FastAPI when a request is received. Validates the API token in the request.
        """
        accepted_paths = ["/ping", "/assets", "/gradio_api"]
        if get_settings().ENVIRONMENT != "local" and not any(
            path in request.url.path for path in accepted_paths
        ):
            # Check for token in query params of the request
            token = request.query_params.get("token")

            # Check for token in the headers of the request if not defined by query params
            if not token:
                token = request.headers.get("APIToken")
                if token and token.startswith("Bearer "):
                    token = token[len("Bearer ") :]

            # Check for token in the cookies of the request if not defined by query params or headers
            if not token:
                token = request.cookies.get("apitoken")

            # Return 401 if token is not found or is invalid
            if not token or not self.validate_token(token):
                return JSONResponse(
                    status_code=401, content={"detail": "Invalid token"}
                )

        response = await call_next(request)
        return response

    def validate_token(self, token: str) -> bool:
        """
        Validates the API token in the request.
        """
        return token == get_settings().API_TOKEN
