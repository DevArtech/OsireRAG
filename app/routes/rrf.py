from fastapi import APIRouter
from typing import List, Tuple

from core.models.chunker import Chunk
from core.models.rrf import ReciprocalRankFusion

router = APIRouter(prefix="/rrf", tags=["rrf"])
rrf = ReciprocalRankFusion()


@router.post("/ranks/")
async def rrf_ranks(ranks: List[List[Tuple[Chunk, float]]], k: int = 60, n: int = 10):
    return rrf.ranks(ranks, k, n)
