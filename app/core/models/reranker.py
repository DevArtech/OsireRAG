"""
Module: reranker.py

Classes:
- Reranker: A Pydantic model for reranking documents based on a query.

Functions:
- None

Usage:
- Import the Reranker class from this module into other modules to use its cross-encoder reranking capabilities.

Author: Adam Haile  
Date: 11/3/2024
"""

from pydantic import BaseModel
from typing import List, Tuple, ClassVar
from sentence_transformers import CrossEncoder

from app.core.models.chunker import Chunk
from app.core.settings import get_settings


class Reranker(BaseModel):
    """
    A Pydantic model for reranking documents based on their relevance to a query.

    Attributes:
    - cross_encoder: CrossEncoder: A pre-trained cross-encoder model for scoring document-query pairs.

    Methods:
    - cross_encode_rerank: Reranks a list of documents based on a query using the cross-encoder.

    Usage:
    - Instantiate this class and call cross_encode_rerank to rerank documents based on their relevance to a query.

    Author: Adam Haile
    Date: 11/3/2024
    """

    cross_encoder: ClassVar[CrossEncoder] = CrossEncoder(
        "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        max_length=512,
        device=get_settings().DEVICE,
    )

    class Config:
        arbitrary_types_allowed = True

    def cross_encode_rerank(
        self, query: str, documents: List[Chunk]
    ) -> List[Tuple[Chunk, float]]:
        """
        Reranks a list of documents based on their relevance to a query.

        Args:
        - `query (str)`: The query string used for reranking.
        - `documents (List[Chunk])`: A list of Chunk objects to rerank.

        Returns:
        - List[Tuple[Chunk, float]]: A list of tuples containing the document Chunk and its normalized relevance sapp.core.

        Raises:
        - ValueError: If the list of scored documents is empty.

        Usage:
        - ```
        reranker = Reranker()
        reranked_docs = reranker.cross_encode_rerank(query="example query", documents=chunks)
        ```

        Author: Adam Haile
        Date: 11/3/2024
        """

        # Rank documents using the cross-encoder model and include original documents in results
        scored_documents = self.cross_encoder.rank(
            query=query,
            documents=[doc.content for doc in documents],
            return_documents=True,
        )

        # Retrieve the maximum and minimum scores for normalization
        max_score = max([doc["score"] for doc in scored_documents])
        min_score = min([doc["score"] for doc in scored_documents])

        # Normalize scores to a range of 0 to 1 and associate them with their respective documents
        normalized_scores = [
            (
                documents[doc["corpus_id"]],
                (doc["score"] - min_score) / (max_score - min_score),
            )
            for doc in scored_documents
        ]
        return normalized_scores
