"""
Module: reranker.py (Router)

This module contains the FastAPI router for the reranker. The reranker is a module that allows users to
rerank a list of documents based on a query using a cross-encoder model.

Classes:
- None

Functions:
- cross_encoder_rerank: Reranks a list of documents based on a query using a cross-encoder model.

Attributes:
- router: The FastAPI router object.
- reranker: The Reranker object.

Author: Adam Haile  
Date: 11/3/2024
"""

from fastapi import APIRouter
from typing import List, Tuple

from app.core.models.chunker import Chunk
from app.core.models.reranker import Reranker

router = APIRouter(prefix="/reranker", tags=["reranker"])
reranker = Reranker()


@router.post("/cross-encoder/")
async def cross_encoder_rerank(
    query: str, documents: List[Chunk]
) -> List[Tuple[Chunk, float]]:
    """
    Perform a reranking of documents utilzing a cross-encoder model

    Args:
    - `query (str)`: The query to use for grounding the reranking.
    - `documents (List[Chunk])`: The list of documents to rerank.

    Returns:
    - List[Tuple[Chunk, float]]: The reranked documents with their corresponding scores.

    Usage:
    - POST /reranker/cross-encoder/

    Author: Adam Haile
    Date: 11/3/2024
    """
    scored_documents = reranker.cross_encode_rerank(query, documents)
    return scored_documents
