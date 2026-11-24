import json
import math

from pathlib import Path

import chromadb

from dotenv import load_dotenv
from google import genai

from retrieval.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    embed_text,
)


def main() -> None:
    """Prepara una colección local para la primera prueba vectorial."""
    project_root = Path(__file__).resolve().parents[1]
    client = chromadb.PersistentClient(
        path=str(project_root / "data/processed/chroma")
    )

    collection = client.get_or_create_collection(
        name="nvidia_fixed_gemini001_768",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={
            "embedding_model": "gemini-embedding-001",
            "embedding_dimensions": 768,
            "chunk_strategy": "fixed",
        },
    )

    expected_metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "chunk_strategy": "fixed",
    }
    stored_metadata = collection.metadata or {}

    if any(
        stored_metadata.get(key) != value
        for key, value in expected_metadata.items()
    ):
        raise ValueError("La colección tiene una configuración incompatible.")

    chunks_path = (
        project_root
        / "data/processed/nvidia_2025.chunks.c1200.o200.jsonl"
    )
    with chunks_path.open(encoding="utf-8") as source:
        chunk = next(json.loads(line) for line in source if line.strip())

    load_dotenv(project_root / "corporate_reports/.env")
    with genai.Client(vertexai=False) as embedding_client:
        vector = embed_text(
            embedding_client,
            chunk["text"],
            task_type="RETRIEVAL_DOCUMENT",
        )

    collection.upsert(
        ids=[chunk["chunk_id"]],
        embeddings=[vector],
        documents=[chunk["text"]],
        metadatas=[{
            "source_file": chunk["source_file"],
            "pdf_page": chunk["pdf_page"],
            "start_char": chunk["start_char"],
            "end_char": chunk["end_char"],
        }],
    )

    print("Chunk guardado:", chunk["chunk_id"])
    print("Dimensiones:", len(vector))
    print("Norma:", math.hypot(*vector))
    print("Colección:", collection.name)
    print("Registros:", collection.count())


if __name__ == "__main__":
    main()
