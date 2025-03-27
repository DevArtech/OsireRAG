"""
Module: embedding.py (Router)

This module contains the FastAPI router for the embedding endpoints. The embedding router
is responsible for handling requests related to embedding queries and chunks.

Classes:
- None

Functions:
- embed_query: Embed a query in the OsireRAG API.
- embed_chunks: Embed a set of chunks in the OsireRAG API.

Attributes:
- router: The FastAPI router object

Author: Adam Haile  
Date: 10/9/2024
"""

import json
from fastapi import APIRouter
from typing import List
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.models.chunker import Chunk
from app.core.models.embedding import embedder, EmbeddedChunk

router = APIRouter(prefix="/embedding", tags=["embedding"])


@router.post(
    "/query/",
    responses={
        200: {
            "model": List[float],
            "description": "List of floats representing the embedding of the query.",
        }
    },
)
async def embed_query(query: str) -> JSONResponse:
    """
    Embed a query in the OsireRAG API.

    Args:
    - `query (str)`: The query to embed.

    Returns:
    - JSONResponse: The response containing the embedded query.

    Usage:
    - POST /embedding/query/

    Author: Adam Haile
    Date: 10/9/2024
    """
    return JSONResponse(
        status_code=200, content={"response": embedder.embed_query(query)}
    )


@router.post(
    "/chunks/",
    responses={
        200: {"model": List[EmbeddedChunk], "description": "List of embedded chunks."}
    },
)
async def embed_chunks(chunks: List[Chunk]) -> StreamingResponse:
    """
    Embed a set of chunks in the OsireRAG API.

    Args:
    - `chunks (List[Chunk])`: The chunks to embed.

    Returns:
    - StreamingResponse: A streaming response containing the embedded chunks.

    Usage:
    - POST /embedding/chunks/

    Author: Adam Haile
    Date: 10/9/2024
    """

    # Define an asynchronous JSON encoder to stream the response in JSON format
    async def async_json_encoder(chunks: List[EmbeddedChunk]):
        yield b'{"response": ['
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                yield json.dumps(chunk.model_dump()).encode("utf-8") + b","
            else:
                yield json.dumps(chunk.model_dump()).encode("utf-8")
        yield b"]}\n"

    return StreamingResponse(
        async_json_encoder(embedder.embed_chunks(chunks)), media_type="application/json"
    )
