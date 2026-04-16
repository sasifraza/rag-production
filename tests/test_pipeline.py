from app.ingestion.chunker import chunk_documents
from langchain_core.documents import Document


def make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "test"})


def test_chunk_single_doc():
    doc = make_doc("Hello world. " * 100)
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1


def test_chunk_preserves_metadata():
    doc = make_doc("Some content " * 50)
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=0)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test"


def test_chunk_empty_list():
    chunks = chunk_documents([])
    assert chunks == []


def test_chunk_short_doc():
    doc = make_doc("Short.")
    chunks = chunk_documents([doc], chunk_size=512, chunk_overlap=64)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short."