from typing import List, Tuple
from sentence_transformers import CrossEncoder

from core.models.chunker import Chunk
from core.settings import get_settings


class Reranker:
    def __init__(self):
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-TinyBERT-L-2-v2",
            max_length=512,
            device=get_settings().DEVICE,
        )

    def cross_encode_rerank(
        self, query: str, documents: List[Chunk]
    ) -> List[Tuple[Chunk, float]]:
        scored_documents = self.cross_encoder.rank(
            query=query,
            documents=[doc.content for doc in documents],
            return_documents=True,
        )

        max_score = max([doc["score"] for doc in scored_documents])
        min_score = min([doc["score"] for doc in scored_documents])

        normalized_scores = [
            (
                documents[doc["corpus_id"]],
                (doc["score"] - min_score) / (max_score - min_score),
            )
            for doc in scored_documents
        ]
        return normalized_scores
