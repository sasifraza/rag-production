import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.loader import load_directory
from app.ingestion.chunker import chunk_documents
from app.retrieval.vectorstore import build_vectorstore

def main():
    print("Loading documents from data/raw...")
    docs = load_directory("data/raw")

    if not docs:
        print("No documents found.")
        sys.exit(1)

    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()