import os
import json
from fastapi import APIRouter
from typing import List
from fastapi.responses import JSONResponse, StreamingResponse

from core.models.chunker import Chunk
from core.models.embedding import DocumentEmbedder, EmbeddedChunk

embedder = DocumentEmbedder()
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
    Embed a query in the RosieRAG API.
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
    Embed a set of chunks in the RosieRAG API.
    """

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
