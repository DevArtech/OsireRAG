import json
from typing import List, Tuple
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from core.models.chunker import Chunk
from core.models.knowledge_base import KnowledgeBase, SearchParameters

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
    """
    try:
        return JSONResponse(
            status_code=200,
            content={
                "response": kb.add_documents(
                    project_name, vectorstore_name, model_name, documents
                )
            },
        )
    except ValueError as e:
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
    """
    try:
        kb.add_project(project_name, vectorstore_name)
        return JSONResponse(
            status_code=200, content={"response": "Project added successfully"}
        )
    except ValueError as e:
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
    """
    try:
        chunks = kb.search(
            project_name, vectorstore_name, model_name, search_parameters
        )

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
        return JSONResponse(status_code=404, content={"detail": str(e)})
