"""
Module: rrf.py (Router)

This module contains the FastAPI router for the Reciprocal Rank Fusion (RRF) module. The RRF module
is a module that allows users to fuse multiple ranked lists of documents into a single ranked list.

Classes:
- None

Functions:
- rrf_ranks: Fuses multiple ranked lists of documents into a single ranked list.

Attributes:
- router: The FastAPI router object.
- rrf: The ReciprocalRankFusion object.

Author: Adam Haile
Date: 10/24/2024
"""
from fastapi import APIRouter
from typing import List, Tuple

from core.models.chunker import Chunk
from core.models.rrf import ReciprocalRankFusion

router = APIRouter(prefix="/rrf", tags=["rrf"])
rrf = ReciprocalRankFusion()


@router.post("/ranks/")
async def rrf_ranks(ranks: List[List[Tuple[Chunk, float]]], k: int = 60, n: int = 10) -> List[Tuple[Chunk, float]]:
    """
    Fuse multiple sets of documents together with a singular score using Reciprocal Rank Fusion.

    Args:
    - ranks (List[List[Tuple[Chunk, float]]]): A list of ranked lists of documents.
    - k (int): Weighting to provide to lower-ranked results
    - n (int): Number of results to return

    Returns:
    - List[Tuple[Chunk, float]]: The fused ranked list of documents.

    Usage:
    - POST /rrf/ranks/

    Author: Adam Haile
    Date: 10/24/2024
    """
    return rrf.ranks(ranks, k, n)
