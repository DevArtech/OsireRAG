"""
Module: documents.py (Router)

This module contains the FastAPI router for the documents endpoints. The documents router
contains all the endpoints for creating, listing, uploading, retrieving, and deleting documents.

Classes:
- None

Functions:
- create_project: Create a new project to store documents.
- list_documents: List all documents currently stored in the RosieRAG API at the given project.
- upload_documents: Upload a new document.
- retrieve_document: Retrieve a specific document by index or name.
- delete_documents: Delete a document or a list of documents.

Attributes:
- router: The FastAPI router object

Author: Adam Haile  
Date: 10/7/2024
"""

import os
from typing import List
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse, Response

from app.core.models.documents import Document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get(
    "/{project_name}/create/",
    status_code=201,
    responses={
        201: {"description": "Project created successfully."},
        409: {"description": "Project already exists."},
    },
)
async def create_project(project_name: str) -> JSONResponse:
    """
    Create a new project to store documents.

    Args:
    - `project_name (str)`: The name of the project to create.

    Returns:
    - JSONResponse: The response object containing the status code and message.

    Raises:
    - 409: If the project already exists.

    Usage:
    - GET /documents/{project_name}/create/

    Author: Adam Haile
    Date: 10/7/2024
    """
    project_path = f"./.rosierag/{project_name}"
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    else:
        return JSONResponse(
            status_code=409, content={"detail": "Project already exists."}
        )

    return JSONResponse(
        status_code=201, content={"response": "Project created successfully."}
    )


@router.get(
    "/{project_name}/list/",
    responses={
        200: {
            "model": List[Document],
            "description": "List of documents stored in the project.",
        },
        404: {"description": "Project not found."},
    },
)
async def list_documents(project_name: str) -> JSONResponse:
    """
    List all documents currently stored in the RosieRAG API at the given project.

    Args:
    - `project_name (str)`: The name of the project to list documents from.

    Returns:
    - JSONResponse: The response object containing the status code and list of documents.

    Raises:
    - 404: If the project is not found.

    Usage:
    - GET /documents/{project_name}/list/

    Author: Adam Haile
    Date: 10/7/2024
    """
    documents = []
    project_path = os.path.join(os.path.abspath("./.rosierag"), project_name)
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    # Get all documents in the project directory and create a Document object for each
    documents.extend(
        [
            Document(directory=os.path.join(project_path, doc)).model_dump()
            for doc in os.listdir(project_path)
        ]
    )

    return JSONResponse(status_code=200, content={"response": documents})


@router.post(
    "/{project_name}/upload/",
    status_code=201,
    responses={
        201: {"description": "Document uploaded successfully."},
        404: {"description": "Project not found."},
        409: {"description": "Document of same name already exists."},
    },
)
async def upload_documents(project_name: str, file: UploadFile) -> JSONResponse:
    """
    Upload a new document.

    Args:
    - `project_name (str)`: The name of the project to upload the document to.
    - `file (UploadFile)`: The file to upload.

    Returns:
    - JSONResponse: The response object containing the status code and message.

    Raises:
    - 404: If the project is not found.

    Usage:
    - POST /documents/{project_name}/upload/

    Author: Adam Haile
    Date: 10/7/2024
    """

    # Validate the project exists
    project_path = os.path.join(os.path.abspath("./.rosierag"), project_name)
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    # Validate the document does not already exist
    file_path = os.path.join(project_path, file.filename)
    if os.path.exists(file_path):
        return JSONResponse(
            status_code=409, content={"detail": "Document of same name already exists."}
        )

    # Save the document to the project directory
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return JSONResponse(
        status_code=201, content={"response": "Document uploaded successfully."}
    )


@router.get(
    "/{project_name}/retrieve/{file}/",
    responses={
        200: {"description": "Document retrieved successfully."},
        404: {"description": "Project or document not found."},
    },
)
async def retrieve_document(project_name: str, file: str) -> JSONResponse:
    """
    Retrieve a specific document by file name and extension.

    Args:
    - `project_name (str)`: The name of the project to retrieve the document from.
    - `file (str)`: The name of the document to retrieve.

    Returns:
    - JSONResponse: The response object containing the status code and document content.

    Raises:
    - 404: If the project or document is not found.

    Usage:
    - GET /documents/{project_name}/retrieve/{file}/

    Author: Adam Haile
    Date: 10/7/2024
    """

    # Validate the project exists
    project_path = os.path.join(os.path.abspath("./.rosierag"), project_name)
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    # Get the document from the project directory
    document = [doc for doc in os.listdir(project_path) if doc == file]

    # Validate the document exists
    if not document:
        return JSONResponse(status_code=404, content={"detail": "Document not found."})

    # Create a Document object for the document
    document = Document(directory=os.path.join(project_path, document[0]))

    # Return the document content
    return JSONResponse(
        status_code=200,
        content={
            "response": {
                "directory": document.directory,
                "content": document.content,
                "metadata": document.metadata,
            }
        },
    )


@router.delete(
    "/{project_name}/delete/{file}/",
    status_code=204,
    responses={
        204: {"description": "Document deleted successfully."},
        404: {"description": "Project or document not found."},
    },
)
async def delete_documents(project_name: str, file: str) -> Response:
    """
    Delete a document.

    Args:
    - `project_name (str)`: The name of the project to delete the document from.
    - `file (str)`: The name of the document to delete.

    Returns:
    - Response: The response object containing the status code.

    Raises:
    - 404: If the project or document is not found.

    Usage:
    - DELETE /documents/{project_name}/delete/{file}/

    Author: Adam Haile
    Date: 10/7/2024
    """
    # Validate the project exists
    project_path = os.path.join(os.path.abspath("./.rosierag"), project_name)
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    # Validate the document exists and delete it
    document_path = os.path.join(project_path, file)
    if os.path.exists(document_path):
        os.remove(document_path)
    else:
        return JSONResponse(
            status_code=404, content={"response": "Document not found."}
        )

    return Response(status_code=204)
