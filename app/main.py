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
import gradio as gr
from fastapi import FastAPI, status
from fastapi.responses import Response

from app.api.api import api_router
from app.core.logger import logger
from app.core.interface.gradio import io
from app.core.settings import get_settings
from app.core.middleware.token_validator import TokenValidationMiddleware

# Validate the rosierag directory exists
os.makedirs("./.rosierag", exist_ok=True)

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
