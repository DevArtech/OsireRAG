"""
Module: main.py

The main module of the application. This module is responsible for creating the FastAPI 
application and mounting the Gradio interface to it.

Classes:
- None

Functions:
- health_check: A simple health check endpoint to verify the application is running.

Attributes:
- app: The FastAPI application object.
- rosie_path: The URL of the RosieRAG application.

Author: Adam Haile  
Date: 9/24/2024
"""

import os
os.makedirs("./.rosierag", exist_ok=True)

import gradio as gr
from fastapi import FastAPI, status, Form
from fastapi.responses import Response, RedirectResponse, HTMLResponse, JSONResponse

from app.api.api import api_router
from app.core.logger import logger
from app.core.interface.gradio import io
from app.core.settings import get_settings
from app.core.middleware.token_validator import TokenValidationMiddleware, validate_token


# Set the rosie path based on the environment
if get_settings().ENVIRONMENT == "local":
    rosie_path = "http://localhost:8080"
else:
    rosie_path = "https://dh-ood.hpc.msoe.edu" + get_settings().BASE_URL + "/"

# Create the FastAPI application
app = FastAPI(
    title="RosieRAG", root_path=get_settings().BASE_URL, redirect_slashes=True
)

# Add the token validation middleware
app.add_middleware(TokenValidationMiddleware)

# Add the API router which contains all the endpoints
app.include_router(api_router)


# Add the health check endpoint
@api_router.get("/ping/", tags=["admin"])
async def health_check():
    return Response(status_code=status.HTTP_200_OK)


@app.get("/login/", tags=["admin"], include_in_schema=False)
async def login_page():
    html_content = f"""
    <html>
        <head>
            <title>RosieRAG Login</title>
        </head>
        <body>
            <h2>Enter Your Password</h2>
            <form action="{get_settings().BASE_URL}/submit-token" method="post">
                <label for="token">Password:</label>
                <input type="password" id="token" name="token" required>
                <button type="submit">Submit</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(html_content)


# Custom endpoint to submit the password
@app.post("/submit-token/", tags=["admin"], include_in_schema=False)
async def submit_token(token: str = Form(...)):
    result = validate_token(token)
    if result:
        response = RedirectResponse(
            url=get_settings().BASE_URL + "/docs", status_code=302
        )
        response.set_cookie(
            key="apitoken",
            value=token,
            max_age=86400,
            secure=True,
            path=get_settings().BASE_URL + "/",
        )
        return response

    return JSONResponse(status_code=401, content={"detail": "Invalid token"})


# Mount the Gradio interface to the FastAPI application
app = gr.mount_gradio_app(
    app,
    io,
    path="/",
    root_path=get_settings().BASE_URL,
    app_kwargs={"redirect_slashes": True},
    favicon_path="./icon.png",
)


# Log the URL the application is running at
@app.on_event("startup")
async def startup_event():
    logger.info(f"RosieRAG is running at: {rosie_path}")
