from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sentence_transformers import CrossEncoder

from app.retriever import RetrievedItem


@dataclass
class RerankedItem:
    chunk_id: int
    source: str
    content: str
    score: float


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        retrieved_items: List[RetrievedItem],
        top_n: int = 5,
    ) -> List[RerankedItem]:
        if not retrieved_items:
            return []

        pairs = [(query, item.content) for item in retrieved_items]
        scores = self.model.predict(pairs)

        ranked = sorted(
            [
                RerankedItem(
                    chunk_id=item.chunk_id,
                    source=item.source,
                    content=item.content,
                    score=float(score),
                )
                for item, score in zip(retrieved_items, scores)
            ],
            key=lambda x: x.score,
            reverse=True,
        )

        return ranked[:top_n]