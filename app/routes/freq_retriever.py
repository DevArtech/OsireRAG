"""
Module: freq_retriever.py (Router)

This module contains the FastAPI router for the frequency retriever. The frequency retriever is a 
module that allows users to search for terms in a collection of documents and retrieve the frequency 
of those terms in each document.

Classes:
- None

Functions:
- tokenize: Tokenizes a list of chunks.
- create_model: Creates a BM25 model from a list of tokenized chunks.
- search: Searches for the top documents containing the same keywords as the query.

Attributes:
- router: The FastAPI router object.
- retriever: The BM25 model object.
- tokenizer: The ChunkTokenizer object.

Author: Adam Haile
Date: 10/23/2024
"""

from fastapi import APIRouter
from typing import List, Tuple

from core.models.chunker import Chunk
from core.models.term_freq_retriever import BM25Model, ChunkTokenizer, TokenizedChunk

router = APIRouter(prefix="/freq_retriever", tags=["freq_retriever"])
retriever = BM25Model()
tokenizer = ChunkTokenizer()


@router.post("/tokenize/")
async def tokenize(chunks: List[Chunk]) -> List[TokenizedChunk]:
    """
    Tokenizes a list of chunks.

    Args:
    - chunks (List[Chunk]): The list of chunks to tokenize.

    Returns:
    - List[TokenizedChunk]: The list of tokenized chunks.

    Usage:
    - POST /freq_retriever/tokenize/

    Author: Adam Haile
    Date: 10/23/2024
    """
    return tokenizer.tokenize_documents(chunks)


@router.post("/create/")
async def create_model(
    project_name: str,
    model_name: str,
    chunks: List[TokenizedChunk],
    k1: float = 1.5,
    b: float = 0.75,
    epsilon: float = 0.25,
) -> None:
    """
    Creates a BM25 model from a list of tokenized chunks.

    Args:
    - project_name (str): The project name.
    - model_name (str): The model name.
    - chunks (List[TokenizedChunk]): The list of tokenized chunks.
    - k1 (float): The k1 parameter for BM25.
    - b (float): The b parameter for BM25.
    - epsilon (float): The epsilon parameter for BM25.

    Returns:
    - None

    Usage:
    - POST /freq_retriever/create/

    Author: Adam Haile
    Date: 10/23/2024
    """
    retriever.create_model(
        project_name, model_name, chunks, k1=k1, b=b, epsilon=epsilon
    )


@router.post("/search/")
async def search(
    project_name: str, model_name: str, query: str, k: int = 10
) -> List[Tuple[Chunk, float]]:
    """
    Searches for the top documents containing the same keywords as the query.

    Args:
    - project_name (str): The project name.
    - model_name (str): The model name.
    - query (str): The query string.
    - k (int): The number of documents to return.

    Returns:
    - List[Tuple[Chunk, float]]: The list of documents and their scores.

    Usage:
    - POST /freq_retriever/search/

    Author: Adam Haile
    Date: 10/23/2024
    """
    chunks, bm25 = retriever.load_model(project_name, model_name)
    return retriever.search(query, bm25, chunks, k)
