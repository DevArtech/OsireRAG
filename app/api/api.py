"""
Module: api.py
Connects all the routers into the main API router, which is connected to the main FastAPI app.

Classes:
- None

Functions:
- None

Usage:
- Import the api_router from this module into the main FastAPI app.

Author: Adam Haile  
Date: 10/16/2024
"""

from fastapi import APIRouter

from app.routes import (
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
