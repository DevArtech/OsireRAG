import os
import re
from pydantic import BaseModel
import requests
from typing import List
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/web", tags=["web"])


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
    project_path = f"./.rosierag/{project_name}"
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    for page in web_pages:
        res = requests.get(page, allow_redirects=True)
        if res.status_code != 200:
            return JSONResponse(
                status_code=404,
                content={
                    "detail": f"Failed to download page {page}: {res.status_code}"
                },
            )

        title_match = re.search(r"<title>(.*?)</title>", res.text, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip().lower().replace(" ", "-")
        else:
            title = (
                page.split("/")[-1].split(".")[0]
                + "-"
                + str(len(os.listdir(project_path)) + 1)
            )

        safe_title = re.sub(r'[<>:"/\\|;&?*]', "", title)
        safe_title = re.sub(r"-+", "-", safe_title)
        file_path = project_path + f"/{safe_title}.html"

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            file.write(res.content)

    return JSONResponse(
        status_code=201,
        content={"response": f"{len(web_pages)} web pages uploaded successfully."},
    )
