from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Cross-encoder reranker.
        Takes retrieved chunks and reorders them by true relevance to the query.
        """
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 3,
    ) -> List[Document]:
        """
        Rerank documents by relevance to query.

        Args:
            query: User question
            documents: Retrieved chunks from vector store
            top_n: How many to keep after reranking

        Returns:
            Top-n reranked documents
        """
        if not documents:
            return []

        # Create pairs of (query, chunk) for scoring
        pairs = [[query, doc.page_content] for doc in documents]

        # Score each pair
        scores = self.model.predict(pairs)

        # Sort by score descending
        scored_docs = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        # Return top_n
        top_docs = [doc for _, doc in scored_docs[:top_n]]

        print(f"Reranked {len(documents)} chunks → kept top {len(top_docs)}")
        return top_docs