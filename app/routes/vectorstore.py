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
    project_path = f"./.rosierag/{project_name}"
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    if os.path.exists(os.path.abspath(project_path + "/" + vectorstore_name)):
        return JSONResponse(
            status_code=409, content={"detail": "Vectorstore already exists."}
        )

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
    project_path = f"./.rosierag/{project_name}"
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    vectorstore_path = os.path.abspath(project_path + "/" + vectorstore_name)
    if not os.path.exists(vectorstore_path):
        return JSONResponse(
            status_code=404, content={"detail": "Vectorstore not found."}
        )

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
    project_path = f"./.rosierag/{project_name}"
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    vectorstore_path = os.path.abspath(project_path + "/" + vectorstore_name)
    if not os.path.exists(vectorstore_path):
        return JSONResponse(
            status_code=404, content={"detail": "Vectorstore not found."}
        )

    user_vectorstore = vs_manager.load_vectorstore(vectorstore_path)
    chunks = vs_manager.get_chunks(user_vectorstore, ids)

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
    project_path = f"./.rosierag/{project_name}"
    if not os.path.exists(project_path):
        return JSONResponse(status_code=404, content={"detail": "Project not found."})

    vectorstore_path = os.path.abspath(project_path + "/" + vectorstore_name)
    if not os.path.exists(vectorstore_path):
        return JSONResponse(
            status_code=404, content={"detail": "Vectorstore not found."}
        )

    user_vectorstore = vs_manager.load_vectorstore(vectorstore_path)
    chunks = vs_manager.search(
        vectorstore=user_vectorstore, search_params=search_parameters
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

    return StreamingResponse(async_json_encoder(chunks), media_type="application/json")
