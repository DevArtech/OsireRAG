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

from core.models.chunker import Chunk


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
        self, ranks: List[List[Tuple[Chunk, float]]], k: int = 60, n: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """
        Combines rankings from multiple rankers using the RRF algorithm.

        Args:
        - ranks: List[List[Tuple[Chunk, float]]]: A list of ranked document lists, where each document is paired with its score.
        - k: int: The RRF parameter for scaling the rank contribution. Default is 60.
        - n: int: The maximum number of ranked results to return. Default is 10.

        Returns:
        - List[Tuple[Chunk, float]]: A list of the top `n` documents, paired with their scores.

        Raises:
        - ValueError: If the input ranks list is empty.

        Usage:
        - rrf = ReciprocalRankFusion()
        - combined_ranks = rrf.ranks(ranks=[ranker1_results, ranker2_results], k=60, n=10)

        Author: Adam Haile
        Date: 10/24/2024
        """

        rrf_scores = {}

        # Iterate over each ranking list
        for rank_list in ranks:
            for position, doc in enumerate(rank_list):
                if isinstance(doc, tuple):  # Handle tuples containing Chunk and score
                    doc, score = doc

                # Check if the document is already in the scores dictionary
                found = False
                original_score = 0
                for existing_doc, existing_score in rrf_scores.keys():
                    if existing_doc == doc:
                        found = True
                        original_score = existing_score
                        break

                # Add the document with an initial score or update its RRF score
                if not found:
                    rrf_scores[(doc, score)] = 0
                else:
                    rrf_scores[(doc, original_score)] += 1 / (k + position + 1)

        # Sort documents by their aggregated RRF scores
        sorted_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Extract the documents with their original scores
        chunks_with_original_scores = [
            scored_chunk for scored_chunk, _ in sorted_scores
        ]

        # Return the top `n` documents
        return chunks_with_original_scores[:n]
