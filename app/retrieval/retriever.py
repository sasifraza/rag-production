from typing import List
from langchain_core.documents import Document
from app.retrieval.vectorstore import load_vectorstore
from config.settings import settings


class Retriever:
    def __init__(self):
        self.vectorstore = load_vectorstore()
        self.top_k = settings.top_k

    def retrieve(self, query: str) -> List[Document]:
        """Find the most relevant chunks for a query."""
        results = self.vectorstore.similarity_search(
            query,
            k=self.top_k,
        )
        print(f"Retrieved {len(results)} chunk(s) for query: '{query}'")
        return results