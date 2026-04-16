from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

# Map file extensions to their loaders
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_document(file_path: str) -> List[Document]:
    """Load a single file and return list of Document objects."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    loader_cls = LOADERS.get(ext)

    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")

    loader = loader_cls(str(path))
    docs = loader.load()
    print(f"Loaded {len(docs)} page(s) from {path.name}")
    return docs


def load_directory(directory: str) -> List[Document]:
    """Load all supported files from a directory."""
    dir_path = Path(directory)
    all_docs: List[Document] = []

    for file_path in sorted(dir_path.rglob("*")):
        if file_path.suffix.lower() in LOADERS:
            try:
                docs = load_document(str(file_path))
                all_docs.extend(docs)
            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")

    print(f"Total pages loaded: {len(all_docs)}")
    return all_docs