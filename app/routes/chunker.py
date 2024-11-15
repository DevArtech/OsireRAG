import os
import uuid
import json
import shutil
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse

from core.models.documents import Document
from core.models.chunker import DocumentChunker, Chunk

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
    """
    temp_dir = f"./.rosierag/tmp_{uuid.uuid4()}"
    os.makedirs(temp_dir)

    tmp_documents = []

    for document in documents:
        file_path = os.path.join(temp_dir, document.filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await document.read())

        tmp_documents.append(Document(directory=file_path))

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

    chunks = []
    for document in tmp_documents:
        chunks.extend(
            chunker.chunk_document(
                document, n=n, max_length=max_length, overlap=overlap
            )
        )

    shutil.rmtree(temp_dir)

    return StreamingResponse(
        async_json_encoder(chunks),
        media_type="application/json",
    )
