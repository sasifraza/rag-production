from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import json

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkRecord:
    chunk_id: int
    source: str
    content: str


def load_documents(data_dir: str) -> List[ChunkRecord]:
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunk_records: List[ChunkRecord] = []
    chunk_id = 0

    for path in sorted(base.glob("*")):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()

            for d in docs:
                text = d.page_content
                text = text.replace("\n", " ")
                text = " ".join(text.split())
                d.page_content = text

        elif path.suffix.lower() in {".txt", ".md"}:
            docs = TextLoader(str(path), encoding="utf-8").load()

            for d in docs:
                text = d.page_content
                text = text.replace("\n", " ")
                text = " ".join(text.split())
                d.page_content = text
        else:
            continue

        split_docs = splitter.split_documents(docs)

        for doc in split_docs:
            text = doc.page_content.strip()
            if not text:
                continue

            page_num = doc.metadata.get("page", None)
            if page_num is not None:
                source_name = f"{path.name}_page_{page_num}"
            else:
                source_name = path.name

            chunk_records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    source=source_name,
                    content=text,
                )
            )
            chunk_id += 1

    if not chunk_records:
        raise ValueError("No supported documents found in data/raw")

    return chunk_records


def save_chunks_metadata(chunks: List[ChunkRecord], output_path: str) -> None:
    payload = [
        {
            "chunk_id": c.chunk_id,
            "source": c.source,
            "content": c.content,
        }
        for c in chunks
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)