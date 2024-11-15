from typing import List, Tuple

from core.models.chunker import Chunk


class ReciprocalRankFusion:
    def ranks(
        self, ranks: List[List[Tuple[Chunk, float]]], k: int = 60, n: int = 10
    ) -> List[Tuple[Chunk, float]]:
        rrf_scores = {}
        for rank_list in ranks:
            for position, doc in enumerate(rank_list):
                if isinstance(doc, tuple):
                    doc, score = doc

                found = False
                og_score = 0
                for d, s in rrf_scores.keys():
                    if d == doc:
                        found = True
                        og_score = s
                        break

                if not found:
                    rrf_scores[(doc, score)] = 0
                else:
                    rrf_scores[(doc, og_score)] += 1 / (k + position + 1)

        sorted_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        chunks_with_original_scores = [
            scored_chunk for scored_chunk, _ in sorted_scores
        ]

        return chunks_with_original_scores[:n]
