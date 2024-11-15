from fastapi import APIRouter
from typing import List, Tuple

from core.models.chunker import Chunk
from core.models.term_freq_retriever import BM25Model, ChunkTokenizer, TokenizedChunk

router = APIRouter(prefix="/freq_retriever", tags=["freq_retriever"])
retriever = BM25Model()
tokenizer = ChunkTokenizer()


@router.post("/tokenize/")
async def tokenize(chunks: List[Chunk]) -> List[TokenizedChunk]:
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
    retriever.create_model(
        project_name, model_name, chunks, k1=k1, b=b, epsilon=epsilon
    )


@router.post("/search/")
async def search(
    project_name: str, model_name: str, query: str, k: int = 10
) -> List[Tuple[Chunk, float]]:
    chunks, bm25 = retriever.load_model(project_name, model_name)
    return retriever.search(query, bm25, chunks, k)
