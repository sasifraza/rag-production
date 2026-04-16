from dataclasses import dataclass, field
from typing import List, Optional
from langchain_core.documents import Document
from app.retrieval.retriever import Retriever
from app.generation.llm import get_llm
from app.generation.prompt import RAG_PROMPT, format_context


@dataclass
class RAGResponse:
    answer: str
    sources: List[Document]
    query: str


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = get_llm()
        self.chain = RAG_PROMPT | self.llm

    def run(self, query: str) -> RAGResponse:
        """Run a full RAG query — retrieve, format, generate."""

        # Step 1: Retrieve relevant chunks
        docs = self.retriever.retrieve(query)

        # Step 2: Format into context
        context = format_context(docs)

        # Step 3: Generate answer
        response = self.chain.invoke({
            "context": context,
            "question": query,
        })

        return RAGResponse(
            answer=response.content,
            sources=docs,
            query=query,
        )