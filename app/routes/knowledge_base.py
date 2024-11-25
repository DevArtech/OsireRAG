"""
Module: knowledge_base.py (Router)

This module contains the FastAPI router for the knowledge base. The knowledge base is a module that
allows users to create a knowledge base, add documents to it, and search it for relevant information.

Classes:
- None

Functions:
- create_kb: Create a new knowledge base.
- add_documents: Add a document to the knowledge base.
- add_webpages: Add a webpage to the knowledge base.
- add_project: Add a project to the knowledge base.
- search: Search the knowledge base.

Attributes:
- kb: The KnowledgeBase object.
- router: The FastAPI router object.

Author: Adam Haile
Date: 10/16/2024
"""

import json
from typing import List, Tuple
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from core.models.chunker import Chunk
from core.models.knowledge_base import KnowledgeBase, SearchParameters, DocumentArgs

kb = KnowledgeBase()
router = APIRouter(prefix="/knowledge-base", tags=["knowledge_base"])


@router.get(
    "/create-kb/",
    status_code=201,
    responses={
        201: {"description": "Knowledge base created successfully."},
        409: {"description": "Knowledge base already exists."},
    },
)
async def create_kb(
    project_name: str, vectorstore_name: str, model_name: str
) -> JSONResponse:
    """
    Create a new knowledge base.

    Args:
    - project_name (str): The name of the project.
    - vectorstore_name (str): The name of the vectorstore.
    - model_name (str): The name of the model.

    Returns:
    - JSONResponse: The response message.

    Usage:
    - GET /knowledge-base/create-kb/?project_name=example_project&vectorstore_name=example_vectorstore&model_name=example_model

    Author: Adam Haile
    Date: 10/16/2024
    """
    try:
        kb.create_kb(project_name, vectorstore_name, model_name)
        return JSONResponse(
            status_code=201, content={"response": "Knowledge base created successfully"}
        )
    except ValueError as e:
        return JSONResponse(status_code=409, content={"detail": str(e)})


@router.post(
    "/add-documents/",
    responses={
        200: {"description": "Documents added successfully."},
        404: {"description": "Project or vectorstore not found."},
    },
)
async def add_documents(
    project_name: str,
    vectorstore_name: str,
    model_name: str,
    documents: List[UploadFile],
) -> JSONResponse:
    """
    Add a document to the knowledge base.

    Args:
    - project_name (str): The name of the project.
    - vectorstore_name (str): The name of the vectorstore.
    - model_name (str): The name of the model.
    - documents (List[UploadFile]): The list of documents to add.

    Returns:
    - JSONResponse: The response message.

    Usage:
    - POST /knowledge-base/add-documents/

    Author: Adam Haile
    Date: 10/16/2024
    """
    try:
        return JSONResponse(
            status_code=200,
            content={
                "response": kb.add_documents(
                    DocumentArgs(
                        project_name=project_name,
                        vectorstore_name=vectorstore_name,
                        model_name=model_name,
                    ),
                    documents,
                )
            },
        )
    except ValueError as e:
        # Return a 404 response if the project, vectorstore, or model is not found
        return JSONResponse(status_code=404, content={"detail": str(e)})


@router.post(
    "/add-webpages/",
    responses={
        200: {"description": "Webpages added successfully."},
        404: {"description": "Project or vectorstore not found."},
    },
)
async def add_webpages(
    project_name: str,
    vectorstore_name: str,
    model_name: str,
    urls: List[str],
) -> JSONResponse:
    """
    Add a webpage to the knowledge base.

    Args:
    - project_name (str): The name of the project.
    - vectorstore_name (str): The name of the vectorstore.
    - model_name (str): The name of the model.

    Returns:
    - JSONResponse: The response message.

    Usage:
    - POST /knowledge-base/add-webpages/

    Author: Adam Haile
    Date: 10/16/2024
    """
    try:
        return JSONResponse(
            status_code=200,
            content={
                "response": kb.add_webpages(
                    project_name, vectorstore_name, model_name, urls
                )
            },
        )
    except ValueError as e:
        # Return a 404 response if the project, vectorstore, or model is not found
        return JSONResponse(status_code=404, content={"detail": str(e)})


@router.post(
    "/add-project/",
    responses={
        200: {"description": "Project added successfully."},
        404: {"description": "Project, vectorstore, or documents not found."},
    },
)
async def add_project(
    project_name: str, vectorstore_name: str, model_name: str
) -> JSONResponse:
    """
    Add a project to the knowledge base.

    Args:
    - project_name (str): The name of the project.
    - vectorstore_name (str): The name of the vectorstore.
    - model_name (str): The name of the model.

    Returns:
    - JSONResponse: The response message.

    Usage:
    - POST /knowledge-base/add-project/

    Author: Adam Haile
    Date: 10/16/2024
    """
    try:
        kb.add_project(project_name, vectorstore_name)
        return JSONResponse(
            status_code=200, content={"response": "Project added successfully"}
        )
    except ValueError as e:
        # Return a 404 response if the project or vectorstore is not found
        return JSONResponse(status_code=404, content={"detail": str(e)})


@router.post(
    "/search/",
    responses={
        200: {"model": List[Chunk], "description": "Search results."},
        404: {"description": "Project or vectorstore not found."},
    },
)
async def search(
    project_name: str,
    vectorstore_name: str,
    model_name: str,
    search_parameters: SearchParameters,
) -> StreamingResponse:
    """
    Search the knowledge base.

    Args:
    - project_name (str): The name of the project.
    - vectorstore_name (str): The name of the vectorstore.
    - model_name (str): The name of the model.

    Returns:
    - StreamingResponse: The top chunks found for the given query.

    Usage:
    - POST /knowledge-base/search/

    Author: Adam Haile
    Date: 10/16/2024
    """
    try:
        chunks = kb.search(
            project_name, vectorstore_name, model_name, search_parameters
        )

        # Asynchronously encode the chunks in JSON format and stream the response
        async def async_json_encoder(chunks: List[Tuple[Chunk, float]]):
            yield b'{"response": ['
            for i, chunk in enumerate(chunks):
                if i < len(chunks) - 1:
                    yield json.dumps([chunk[0].model_dump(), float(chunk[1])]).encode(
                        "utf-8"
                    ) + b","
                else:
                    yield json.dumps([chunk[0].model_dump(), float(chunk[1])]).encode(
                        "utf-8"
                    )
            yield b"]}\n"

        return StreamingResponse(
            async_json_encoder(chunks), media_type="application/json"
        )
    except ValueError as e:
        # Return a 404 response if the project, vectorstore, or model is not found
        return JSONResponse(status_code=404, content={"detail": str(e)})
