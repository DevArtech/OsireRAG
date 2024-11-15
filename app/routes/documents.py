import os
from typing import List, Dict, Union, Any
from fastapi.responses import JSONResponse, Response
from fastapi import APIRouter, UploadFile

from core.models.documents import Document

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
    """
    documents = []
    project_path = f"./.rosierag/{project_name}"
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    documents.extend(
        [
            Document(directory=os.path.abspath(project_path + "/" + doc)).model_dump()
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
    """
    project_path = f"./.rosierag/{project_name}"
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    file_path = project_path + f"/{file.filename}"
    if os.path.exists(file_path):
        return JSONResponse(
            status_code=409, content={"detail": "Document of same name already exists."}
        )

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
    Retrieve a specific document by index or name.
    """
    project_path = f"./.rosierag/{project_name}"
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    document = [
        doc for doc in os.listdir(os.path.abspath(project_path + "/")) if doc == file
    ]

    if not document:
        return JSONResponse(status_code=404, content={"detail": "Document not found."})

    document = Document(directory=os.path.abspath(project_path + "/" + document[0]))

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
    Delete a document or a list of documents.
    """
    project_path = f"./.rosierag/{project_name}"
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    document_path = os.path.abspath(project_path + "/" + file)
    if os.path.exists(document_path):
        os.remove(document_path)
    else:
        return JSONResponse(
            status_code=404, content={"response": "Document not found."}
        )

    return Response(status_code=204)
