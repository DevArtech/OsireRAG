from fastapi import APIRouter

from routes import (
    documents,
    web,
    chunker,
    embedding,
    freq_retriever,
    vectorstore,
    rrf,
    reranker,
    knowledge_base,
    llm,
)

api_router = APIRouter()

api_router.include_router(documents.router)
api_router.include_router(web.router)
api_router.include_router(chunker.router)
api_router.include_router(embedding.router)
api_router.include_router(freq_retriever.router)
api_router.include_router(vectorstore.router)
api_router.include_router(rrf.router)
api_router.include_router(reranker.router)
api_router.include_router(knowledge_base.router)
api_router.include_router(llm.router)
