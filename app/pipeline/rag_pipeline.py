from dataclasses import dataclass, field
from typing import List, Optional
from langchain_core.documents import Document
from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from app.generation.llm import get_llm
from app.generation.prompt import RAG_PROMPT, format_context


@dataclass
class RAGResponse:
    answer: str
    sources: List[Document]
    query: str


class RAGPipeline:
    def __init__(self, use_reranker: bool = True):
        self.retriever = Retriever()
        self.reranker = Reranker() if use_reranker else None
        self.llm = get_llm()
        self.chain = RAG_PROMPT | self.llm

    def run(self, query: str) -> RAGResponse:
        """Run a full RAG query — retrieve, rerank, generate."""

        # Step 1: Retrieve
        docs = self.retriever.retrieve(query)

        # Step 2: Rerank (if enabled)
        if self.reranker and docs:
            docs = self.reranker.rerank(query, docs)

        # Step 3: Format context
        context = format_context(docs)

        # Step 4: Generate
        response = self.chain.invoke({
            "context": context,
            "question": query,
        })

        return RAGResponse(
            answer=response.content,
            sources=docs,
            query=query,
        )