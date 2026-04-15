from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import json
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.ingest import ChunkRecord


@dataclass
class RetrievedItem:
    chunk_id: int
    source: str
    content: str
    score: float


class HybridRetriever:
    def __init__(self, chunks: List[ChunkRecord], model_name: str = "all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.embedding_model = SentenceTransformer(model_name)

        self.texts = [c.content for c in chunks]
        self.tokenized_texts = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_texts)

        embeddings = self.embedding_model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.embeddings = embeddings

    def bm25_search(self, query: str, top_k: int = 5) -> List[RetrievedItem]:
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedItem(
                chunk_id=self.chunks[i].chunk_id,
                source=self.chunks[i].source,
                content=self.chunks[i].content,
                score=float(scores[i]),
            )
            for i in top_idx
        ]

    def vector_search(self, query: str, top_k: int = 5) -> List[RetrievedItem]:
        q = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(q, top_k)
        return [
            RetrievedItem(
                chunk_id=self.chunks[i].chunk_id,
                source=self.chunks[i].source,
                content=self.chunks[i].content,
                score=float(scores[0][rank]),
            )
            for rank, i in enumerate(indices[0])
        ]

    def hybrid_search(self, query: str, top_k: int = 8) -> List[RetrievedItem]:
        bm25_items = self.bm25_search(query, top_k=top_k)
        vector_items = self.vector_search(query, top_k=top_k)

        merged: Dict[int, RetrievedItem] = {}

        for item in bm25_items:
            merged[item.chunk_id] = item

        for item in vector_items:
            if item.chunk_id in merged:
                merged[item.chunk_id].score += item.score
            else:
                merged[item.chunk_id] = item

        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return ranked[:top_k]


def save_vector_metadata(retriever: HybridRetriever, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(retriever.index, str(out / "faiss.index"))

    with open(out / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "content": c.content,
                }
                for c in retriever.chunks
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )