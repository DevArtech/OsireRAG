from fastapi import APIRouter
from typing import List, Tuple

from core.models.chunker import Chunk
from core.models.reranker import Reranker

router = APIRouter(prefix="/reranker", tags=["reranker"])
reranker = Reranker()


@router.post("/cross-encoder/")
async def cross_encoder_rerank(
    query: str, documents: List[Chunk]
) -> List[Tuple[Chunk, float]]:
    scored_documents = reranker.cross_encode_rerank(query, documents)
    return scored_documents
