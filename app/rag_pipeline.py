from __future__ import annotations

from app.ingest import load_documents
from app.retriever import HybridRetriever
from app.reranker import CrossEncoderReranker
from app.llm import CitationLLM


class RAGPipeline:
    def __init__(self, data_dir: str = "data/raw"):
        chunks = load_documents(data_dir)
        self.retriever = HybridRetriever(chunks)
        self.reranker = CrossEncoderReranker()
        self.llm = CitationLLM()

    def ask(self, question: str, top_k: int = 8):
        retrieved = self.retriever.hybrid_search(question, top_k=top_k)
        reranked = self.reranker.rerank(
            question,
            retrieved,
            top_n=min(5, len(retrieved))
        )

        answer = self.llm.answer_with_citations(question, reranked)

        return {
            "answer": answer,
            "sources": [
                {
                    "source": r.source,
                    "chunk_id": r.chunk_id,
                    "content": r.content[:300],
                    "score": round(r.score, 4),
                }
                for r in reranked
            ],
        }