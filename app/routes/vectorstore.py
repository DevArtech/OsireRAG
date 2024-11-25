"""
Module: vectorstore.py (Router)

This module contains the FastAPI router for the vectorstore module. The vectorstore module is a module
that allows users to create, add, get, and search vectorstores.

Classes:
- None

Functions:
- create_vectorstore: Creates a new vectorstore.
- add_chunks: Adds chunks to a vectorstore.
- get_chunks: Gets chunks from a vectorstore.
- search: Searches for chunks in a vectorstore.

Attributes:
- router: The FastAPI router object.
- vs_manager: The VectorstoreManager object.

Author: Adam Haile
Date: 10/13/2024
"""
import os
import json
from typing import List, Tuple
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from core.models.chunker import Chunk
from core.models.vectorstore import VectorstoreManager, VectorstoreSearchParameters
from core.models.embedding import EmbeddedChunk

router = APIRouter(prefix="/vectorstore", tags=["vectorstore"])
vs_manager: VectorstoreManager = VectorstoreManager()


@router.post(
    "/{project_name}/create/{vectorstore_name}/",
    status_code=201,
    responses={
        201: {"description": "Vectorstore created successfully."},
        404: {"description": "Project not found."},
        409: {"description": "Vectorstore already exists."},
    },
)
async def create_vectorstore(project_name: str, vectorstore_name: str) -> JSONResponse:
    """
    Create a new vectorstore.

    Args:
    - project_name (str): The name of the project.
    - vectorstore_name (str): The name of the vectorstore.

    Returns:
    - JSONResponse: The response message.

    Usage:
    - POST /vectorstore/{project_name}/create/{vectorstore_name}/

    Author: Adam Haile
    Date: 10/13/2024
    """

    # Validate the project exists
    project_path = os.path.join(os.path.abspath("./.rosierag"), project_name)
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    # Validate the vectorstore does not already exist
    if os.path.exists(os.path.join(project_path, vectorstore_name)):
        return JSONResponse(
            status_code=409, content={"detail": "Vectorstore already exists."}
        )

    # Create the vectorstore
    user_vectorstore = vs_manager.create_vectorstore()
    vs_manager.save_vectorstore(
        user_vectorstore, os.path.abspath(project_path + "/" + vectorstore_name)
    )
    return JSONResponse(
        status_code=201, content={"response": "Vectorstore created successfully"}
    )


@router.post(
    "/{project_name}/add/{vectorstore_name}/",
    responses={
        200: {"model": List[str], "description": "List of IDs of the added chunks."},
        404: {"description": "Project or vectorstore not found."},
    },
)
async def add_chunks(
    project_name: str, vectorstore_name: str, chunks: List[EmbeddedChunk]
) -> JSONResponse:
    """
    Add chunks to a vectorstore.

    Args:
    - project_name (str): The name of the project.
    - vectorstore_name (str): The name of the vectorstore.
    - chunks (List[EmbeddedChunk]): The list of chunks to add.

    Returns:
    - JSONResponse: The response message.

    Usage:
    - POST /vectorstore/{project_name}/add/{vectorstore_name}/

    Author: Adam Haile
    Date: 10/13/2024
    """
    # Validate the project exists
    project_path = os.path.join(os.path.abspath("./.rosierag"), project_name)
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    # Validate the vectorstore exists
    vectorstore_path = os.path.join(project_path, vectorstore_name)
    if not os.path.exists(vectorstore_path):
        return JSONResponse(
            status_code=404, content={"detail": "Vectorstore not found."}
        )

    # Add the chunks to the vectorstore
    user_vectorstore = vs_manager.load_vectorstore(vectorstore_path)
    ids = vs_manager.add_chunks(user_vectorstore, chunks)
    vs_manager.save_vectorstore(user_vectorstore, vectorstore_path)

    return JSONResponse(status_code=200, content={"response": ids})


@router.post(
    "/{project_name}/get/{vectorstore_name}/",
    responses={
        200: {"model": List[Chunk], "description": "List of chunks."},
        404: {"description": "Project or vectorstore not found."},
    },
)
async def get_chunks(
    project_name: str, vectorstore_name: str, ids: List[str]
) -> StreamingResponse:
    """
    Get chunks from a vectorstore.

    Args:
    - project_name (str): The name of the project.
    - vectorstore_name (str): The name of the vectorstore.
    - ids (List[str]): The list of IDs of the chunks to get.

    Returns:
    - StreamingResponse: The response message.

    Usage:
    - POST /vectorstore/{project_name}/get/{vectorstore_name}/

    Author: Adam Haile
    Date: 10/13/2024
    """
    # Validate the project exists
    project_path = os.path.join(os.path.abspath("./.rosierag"), project_name)
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    # Validate the vectorstore exists
    vectorstore_path = os.path.join(project_path, vectorstore_name)
    if not os.path.exists(vectorstore_path):
        return JSONResponse(
            status_code=404, content={"detail": "Vectorstore not found."}
        )

    # Get the chunks from the vectorstore
    user_vectorstore = vs_manager.load_vectorstore(vectorstore_path)
    chunks = vs_manager.get_chunks(user_vectorstore, ids)

    # Stream the chunks as JSON encoded
    async def async_json_encoder(chunks: List[Chunk]):
        yield b'{"response": ['
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                yield json.dumps(chunk.model_dump()).encode("utf-8") + b","
            else:
                yield json.dumps(chunk.model_dump()).encode("utf-8")
        yield b"]}\n"

    return StreamingResponse(async_json_encoder(chunks), media_type="application/json")


# @router.delete("/{project_name}/delete/{vectorstore_name}")
# async def delete_chunks(project_name: str, vectorstore_name: str, ids: List[str]) -> JSONResponse:
#     project_path = f"./.rosierag/{project_name}"
#     if not os.path.exists(project_path):
#         return JSONResponse(status_code=404, content="Project not found.")

#     vectorstore_path = os.path.abspath(project_path + "/" + vectorstore_name)
#     if not os.path.exists(vectorstore_path):
#         return JSONResponse(status_code=404, content="Vectorstore not found.")

#     user_vectorstore = vs_manager.load_vectorstore(vectorstore_path)
#     new_vectorstore = vs_manager.delete_chunks(user_vectorstore, ids)
#     vs_manager.save_vectorstore(new_vectorstore, vectorstore_path)
#     return Response(status_code=204)


@router.post(
    "/{project_name}/search/{vectorstore_name}/",
    responses={
        200: {
            "model": List[Tuple[Chunk, float]],
            "description": "List of chunks and their similarity scores.",
        },
        404: {"description": "Project or vectorstore not found."},
    },
)
async def search(
    project_name: str,
    vectorstore_name: str,
    search_parameters: VectorstoreSearchParameters,
) -> StreamingResponse:
    """
    Search for chunks in a vectorstore.

    Args:
    - project_name (str): The name of the project.
    - vectorstore_name (str): The name of the vectorstore.
    - search_parameters (VectorstoreSearchParameters): The search parameters.

    Returns:
    - StreamingResponse: The response message.

    Usage:
    - POST /vectorstore/{project_name}/search/{vectorstore_name}/

    Author: Adam Haile
    Date: 10/13/2024
    """
    # Validate the project exists
    project_path = os.path.join(os.path.abspath("./.rosierag"), project_name)
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    # Validate the vectorstore exists
    vectorstore_path = os.path.join(project_path, vectorstore_name)
    if not os.path.exists(vectorstore_path):
        return JSONResponse(
            status_code=404, content={"detail": "Vectorstore not found."}
        )

    # Search the vectorstore
    user_vectorstore = vs_manager.load_vectorstore(vectorstore_path)
    chunks = vs_manager.search(
        vectorstore=user_vectorstore, search_params=search_parameters
    )

    # Stream the chunks as JSON encoded
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

    return StreamingResponse(async_json_encoder(chunks), media_type="application/json")
