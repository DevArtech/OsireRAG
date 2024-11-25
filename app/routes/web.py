import os
import re
from pydantic import BaseModel
import requests
from typing import List
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.models.web import WebScraper

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
    Upload a new web page and save it as a document to the RosieRAG API.
    """
    web.add_pages(project_name, web_pages)

    return JSONResponse(
        status_code=201,
        content={"response": f"{len(web_pages)} web pages uploaded successfully."},
    )
