from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from app.ingestion.embedder import get_embeddings
from config.settings import settings


def build_vectorstore(documents: List[Document]) -> Chroma:
    """Embed documents and store them in Chroma."""
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=settings.vectorstore_path,
    )

    print(f"Indexed {len(documents)} chunk(s) into vectorstore")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing Chroma vectorstore from disk."""
    embeddings = get_embeddings()

    return Chroma(
        embedding_function=embeddings,
        persist_directory=settings.vectorstore_path,
    )