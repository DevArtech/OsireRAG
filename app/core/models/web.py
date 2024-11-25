import io
import os
import re
import requests
from typing import List
from fastapi import UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse


class WebScraper(BaseModel):
    def add_pages(self, project_name: str, web_pages: List[str]) -> List[UploadFile]:
        project_path = f"./.rosierag/{project_name}"
        if not os.path.exists(project_path):
            return JSONResponse(
                status_code=404, content={"detail": "Project not found."}
            )

        page_files = []
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

            page_files.append(
                UploadFile(
                    file=io.BytesIO(open(file_path, "rb").read()),
                    filename=safe_title + ".html",
                )
            )

        return page_files
