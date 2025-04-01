"""
Module: rrf.py

Classes:
- ReciprocalRankFusion: A Pydantic model for combining rankings using Reciprocal Rank Fusion (RRF).

Functions:
- None

Usage:
- Import the ReciprocalRankFusion class from this module into other modules to merge and rerank document lists.

Author: Adam Haile  
Date: 10/24/2024
"""

from pydantic import BaseModel
from typing import List, Tuple

from app.core.models.chunker import Chunk
from app.core.logger import logger

class ReciprocalRankFusion(BaseModel):
    """
    A Pydantic model for combining rankings from multiple rankers using Reciprocal Rank Fusion (RRF).

    Methods:
    - ranks: Combines rankings and scores documents using the RRF algorithm.

    Usage:
    - Instantiate this class and call the ranks method to merge and rerank document lists.

    Author: Adam Haile
    Date: 10/24/2024
    """

    def ranks(
        self, ranks: List[List[Tuple[Chunk, float]]], k: int = 60, n: int = 10, threshold: float | None = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Combines rankings from multiple rankers using the RRF algorithm.

        Args:
        - `ranks (List[List[Tuple[Chunk, float]]])`: A list of ranked document lists, where each document is paired with its score.
        - `k (int)`: The RRF parameter for scaling the rank contribution. Default is 60.
        - `n (int)`: The maximum number of ranked results to return. Default is 10.
        - `threshold (float | None)`: Optional threshold to filter results by score. Default is None.

        Returns:
        - List[Tuple[Chunk, float]]: A list of the top `n` documents, paired with their normalized RRF scores.

        Raises:
        - ValueError: If the input ranks list is empty.

        Usage:
        - ```
        rrf = ReciprocalRankFusion()
        combined_ranks = rrf.ranks(ranks=[ranker1_results, ranker2_results], k=60, n=10)
        ```

        Author: Adam Haile
        Date: 10/24/2024
        """

        # Dictionary to track RRF scores: {doc_id: (doc, rrf_score)}
        doc_info = {}

        # Iterate over each ranking list
        for rank_list in ranks:
            for position, item in enumerate(rank_list):
                doc, score = item  # Unpack the tuple containing Chunk and score
                
                doc_id = id(doc)  # Use object id as a unique identifier
                
                # If the document has not been seen yet, add it
                if doc_id not in doc_info:
                    doc_info[doc_id] = (doc, 1 / (k + position + 1))
                else:
                    # Update RRF score for existing document
                    _, rrf_score = doc_info[doc_id]
                    doc_info[doc_id] = (doc, rrf_score + 1 / (k + position + 1))

        # Get all documents and their RRF scores
        all_docs = list(doc_info.values())
        
        # Normalize RRF scores to be between 0 and 1
        if all_docs:
            max_score = max(score for _, score in all_docs)
            min_score = min(score for _, score in all_docs)
            score_range = max_score - min_score
            
            if score_range > 0:
                normalized_docs = [(doc, (score - min_score) / score_range) for doc, score in all_docs]
            else:
                normalized_docs = [(doc, 1.0) for doc, _ in all_docs]
        else:
            normalized_docs = []

        # Sort documents by their normalized RRF scores
        sorted_docs = sorted(normalized_docs, key=lambda x: x[1], reverse=True)

        # Extract the documents with their normalized RRF scores
        result = []
        
        if threshold:
            # Filter by threshold on normalized RRF score
            result = [(doc, rrf_score) for doc, rrf_score in sorted_docs 
                        if rrf_score >= threshold][:n]
        else:
            # Just return top n with normalized RRF score
            result = [(doc, rrf_score) for doc, rrf_score in sorted_docs][:n]

        # Return the top `n` documents with their normalized RRF scores
        logger.info(f"RRF results: {[f'{doc.content[:100]} - {rrf_score}\n' for doc, rrf_score in result]}")
        return result
