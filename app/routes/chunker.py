"""
Module: chunker.py (Router)

This module contains the FastAPI router for the chunker endpoint. This endpoint is responsible for chunking a document into smaller pieces.

Classes:
- None

Functions:
- chunk_file: Chunk a document into smaller pieces.

Attributes:
- router: The FastAPI router object.
- chunker: The DocumentChunker object used to chunk documents.

Author: Adam Haile  
Date: 10/9/2024
"""

import os
import uuid
import json
import shutil
from typing import List
from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse

from app.core.models.documents import Document
from app.core.models.chunker import DocumentChunker, Chunk

router = APIRouter(prefix="/chunker", tags=["chunker"])
chunker = DocumentChunker()


@router.post(
    "/",
    responses={
        200: {"model": List[Chunk], "description": "Document chunked successfully."}
    },
)
async def chunk_file(
    documents: List[UploadFile], n: int = 3, max_length: int = 10000, overlap: int = 50
) -> StreamingResponse:
    """
    Chunk a document into smaller pieces.

    Args:
    - `documents (List[UploadFile])`: A list of documents to chunk.
    - `n (int)`: The number of sentences per chunk.
    - `max_length (int)`: The maximum character length of a chunk.
    - `overlap (int)`: The number of overlapping characters between chunks.

    Returns:
    - StreamingResponse: A streaming response containing the chunked document.

    Usage:
    - POST /chunker/

    Author: Adam Haile
    Date: 10/9/2024
    """

    # Create a temporary directory to store the documents
    temp_dir = f"./.osirerag/tmp_{uuid.uuid4()}"
    os.makedirs(temp_dir)

    tmp_documents = []

    # Save the documents to the temporary directory
    for document in documents:
        file_path = os.path.join(temp_dir, document.filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await document.read())

        tmp_documents.append(Document(directory=file_path))

    # Define an asynchronous JSON encoder to return the chunks as a streaming response in JSON format
    async def async_json_encoder(chunks: List[Chunk]):
        yield b'{"response": ['
        for i, chunk in enumerate(chunks):
            chunk.metadata.pop("directory", None)
            chunk_data = chunk.model_dump()
            if i < len(chunks) - 1:
                yield json.dumps(chunk_data).encode("utf-8") + b","
            else:
                yield json.dumps(chunk_data).encode("utf-8")
        yield b"]}\n"

    # Chunk the documents
    chunks = []
    for document in tmp_documents:
        chunks.extend(
            chunker.chunk_document(
                document, n=n, max_length=max_length, overlap=overlap
            )
        )

    # Cleanup and remove the temporary directory
    shutil.rmtree(temp_dir)

    # Return the chunks as a streaming response
    return StreamingResponse(
        async_json_encoder(chunks),
        media_type="application/json",
    )
