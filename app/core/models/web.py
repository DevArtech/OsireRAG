"""
Module: web_scraper.py

Classes:
- WebScraper: A Pydantic model for downloading and saving web pages as files.

Functions:
- None

Usage:
- Import the WebScraper class to download and save web pages to a specific project directory.

Author: Adam Haile  
Date: 11/19/2024
"""

import io
import os
import re
import requests
from typing import List
from fastapi import UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse


class WebScraper(BaseModel):
    """
    A Pydantic model for downloading and saving web pages as files.

    Methods:
    - add_pages: Downloads web pages and saves them as HTML files in the project directory.

    Usage:
    - scraper = WebScraper()
    - page_files = scraper.add_pages(project_name="example_project", web_pages=["https://example.com"])

    Author: Adam Haile
    Date: 11/19/2024
    """

    def add_pages(self, project_name: str, web_pages: List[str]) -> List[UploadFile]:
        """
        Downloads web pages and saves them as HTML files in the project directory.

        Args:
        - `project_name (str)`: The name of the project to associate the saved files with.
        - `web_pages (List[str])`: A list of web page URLs to download.

        Returns:
        - List[UploadFile]: A list of FastAPI UploadFile objects representing the saved web pages.

        Raises:
        - JSONResponse: If the project directory does not exist or a page fails to download.

        Usage:
        - `page_files = scraper.add_pages(project_name="example_project", web_pages=["https://example.com"])`

        Author: Adam Haile
        Date: 10/7/2024
        """
        # Define the project directory path
        project_path = f"./.osirerag/{project_name}"

        # Check if the project directory exists
        if not os.path.exists(project_path):
            return JSONResponse(
                status_code=404, content={"detail": "Project not found."}
            )

        page_files = []  # List to store created UploadFile objects

        # Iterate through the provided web page URLs
        for page in web_pages:
            # Send a GET request to fetch the web page
            res = requests.get(page, allow_redirects=True)
            if res.status_code != 200:
                return JSONResponse(
                    status_code=404,
                    content={
                        "detail": f"Failed to download page {page}: {res.status_code}"
                    },
                )

            # Extract the title from the HTML content
            title_match = re.search(r"<title>(.*?)</title>", res.text, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip().lower().replace(" ", "-")
            else:
                # If no title is found, generate a filename based on the URL
                title = (
                    page.split("/")[-1].split(".")[0]
                    + "-"
                    + str(len(os.listdir(project_path)) + 1)
                )

            # Sanitize the title to ensure it's a valid filename
            safe_title = re.sub(r'[<>:"/\\|;&?*]', "", title)
            safe_title = re.sub(r"-+", "-", safe_title)
            file_path = project_path + f"/{safe_title}.html"

            # Ensure the directory structure exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save the web page content to a file
            with open(file_path, "wb") as file:
                file.write(res.content)

            # Create an UploadFile object for the saved file
            page_files.append(
                UploadFile(
                    file=io.BytesIO(open(file_path, "rb").read()),
                    filename=safe_title + ".html",
                )
            )

        return page_files
