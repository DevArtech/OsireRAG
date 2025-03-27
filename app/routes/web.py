"""
Module: web.py (Router)

This module contains the FastAPI router for the web module. The web module is a module that allows users
to upload web pages to the OsireRAG API.

Classes:
- None

Functions:
- upload_web: Uploads a new web page and saves it as a document to the OsireRAG API.

Attributes:
- router: The FastAPI router object.

Author: Adam Haile  
Date: 10/7/2024
"""

from typing import List
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.models.web import WebScraper

router = APIRouter(prefix="/web", tags=["web"])
web = WebScraper()


@router.post(
    "/{project_name}/upload/",
    responses={
        201: {"description": "Web page uploaded successfully."},
        404: {"description": "Project not found or webpage failed to download."},
    },
)
async def upload_web(project_name: str, web_pages: List[str]) -> JSONResponse:
    """
    Upload a new web page and save it as a document to the OsireRAG API.

    Args:
    - project_name (str): The name of the project.
    - web_pages (List[str]): A list of URLs to web pages to upload.

    Returns:
    - JSONResponse: The response message.

    Usage:
    - POST /web/{project_name}/upload/

    Author: Adam Haile
    Date: 10/7/2024
    """
    web.add_pages(project_name, web_pages)

    return JSONResponse(
        status_code=201,
        content={"response": f"{len(web_pages)} web pages uploaded successfully."},
    )
