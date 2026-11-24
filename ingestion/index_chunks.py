"""Indexación incremental de chunks fijos en una colección Chroma existente."""

import argparse
import json
import math
import time
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from google import genai

from retrieval.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    embed_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _validate_collection(collection: Collection) -> None:
    expected = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "chunk_strategy": "fixed",
    }
    metadata = collection.metadata or {}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("La colección tiene una configuración incompatible.")
    hnsw = collection.configuration.get("hnsw") or {}
    if hnsw.get("space") != "cosine":
        raise ValueError("La colección debe utilizar distancia cosine.")


def _chunk_metadata(chunk: dict, line_number: int) -> dict:
    """Valida el contrato de chunks fijos y obtiene los metadatos de origen."""
    if not isinstance(chunk, dict):
        raise ValueError(f"Se esperaba un objeto en la línea {line_number}.")
    source = chunk.get("source_file")
    page = chunk.get("pdf_page")
    start = chunk.get("start_char")
    end = chunk.get("end_char")
    text = chunk.get("text")
    if (
        not isinstance(source, str) or not source.strip()
        or type(page) is not int or page < 1
        or type(start) is not int or start < 0
        or type(end) is not int or end <= start
        or not isinstance(text, str) or not text
        or end - start != len(text)
    ):
        raise ValueError(f"Chunk no válido en la línea {line_number}.")
    if chunk.get("chunk_id") != f"{source}:p{page}:start{start}":
        raise ValueError(f"ID incompatible con chunks fijos en la línea {line_number}.")
    return {
        "source_file": source, "pdf_page": page,
        "start_char": start, "end_char": end,
    }


def index_chunks(
    client: genai.Client | None,
    collection: Collection,
    chunks_path: Path,
    max_new_embeddings: int = 5,
    *,
    dry_run: bool = False,
    request_interval: float = 1.0,
) -> dict[str, int]:
    """Procesa una tanda reanudable, sin borrar registros ni añadir reintentos.

    Omite texto idéntico y actualiza sus metadatos si es necesario. El límite
    cuenta chunks que necesitan un embedding, tanto nuevos como modificados.
    Los errores se propagan; las escrituras anteriores se conservan. No elimina
    IDs que hayan desaparecido del archivo: no es una sincronización completa.
    """
    if type(max_new_embeddings) is not int or max_new_embeddings < 1:
        raise ValueError("max_new_embeddings debe ser un entero mayor que cero.")
    if type(dry_run) is not bool:
        raise ValueError("dry_run debe ser un booleano.")
    if (
        type(request_interval) not in (int, float)
        or not math.isfinite(request_interval)
        or request_interval < 0
    ):
        raise ValueError("request_interval debe ser un número finito no negativo.")
    if not dry_run and client is None:
        raise ValueError("Se requiere un cliente de embeddings fuera de dry-run.")
    _validate_collection(collection)
    summary = {
        "reviewed": 0, "skipped": 0, "planned": 0,
        "saved": 0, "metadata_updated": 0, "empty_skipped": 0,
    }
    seen_ids = set()
    with chunks_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSON no válido en la línea {line_number}.") from error
            metadata = _chunk_metadata(chunk, line_number)
            chunk_id = chunk["chunk_id"]
            if chunk_id in seen_ids:
                raise ValueError(f"ID duplicado en la línea {line_number}.")
            seen_ids.add(chunk_id)
            summary["reviewed"] += 1
            if not chunk["text"].strip():
                summary["empty_skipped"] += 1
                continue
            existing = collection.get(
                ids=[chunk_id], include=["documents", "metadatas"]
            )
            if existing["ids"]:
                if existing["ids"] != [chunk_id]:
                    raise ValueError("Chroma devolvió un ID distinto del solicitado.")
                if existing["documents"][0] == chunk["text"]:
                    stored_metadata = existing["metadatas"][0] or {}
                    if (
                        any(stored_metadata.get(key) != value for key, value in metadata.items())
                        and not dry_run
                    ):
                        collection.update(ids=[chunk_id], metadatas=[metadata])
                        summary["metadata_updated"] += 1
                    summary["skipped"] += 1
                    continue

            summary["planned"] += 1
            if not dry_run:
                if summary["saved"] and request_interval:
                    time.sleep(request_interval)
                vector = embed_text(client, chunk["text"], "RETRIEVAL_DOCUMENT")
                collection.upsert(
                    ids=[chunk_id], embeddings=[vector],
                    documents=[chunk["text"]], metadatas=[metadata],
                )
                summary["saved"] += 1
            if summary["planned"] == max_new_embeddings:
                break

    return summary


def main() -> None:
    """Abre la colección existente y ejecuta una tanda limitada o su simulación."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks", type=Path,
        default=PROJECT_ROOT / "data/processed/nvidia_2025.chunks.c1200.o200.jsonl",
    )
    parser.add_argument(
        "--store", type=Path, default=PROJECT_ROOT / "data/processed/chroma"
    )
    parser.add_argument("--collection", default="nvidia_fixed_gemini001_768")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--request-interval", type=float, default=1.0,
        help="Pausa en segundos entre chunks que requieren embeddings (por defecto: 1).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    store = chromadb.PersistentClient(path=str(args.store))
    collection = store.get_collection(args.collection, embedding_function=None)
    if args.dry_run:
        summary = index_chunks(
            None, collection, args.chunks, args.limit,
            dry_run=True, request_interval=args.request_interval,
        )
    else:
        load_dotenv(PROJECT_ROOT / "corporate_reports/.env")
        with genai.Client(vertexai=False) as client:
            summary = index_chunks(
                client, collection, args.chunks, args.limit,
                request_interval=args.request_interval,
            )
    print("Modo:", "simulación sin generación ni escrituras" if args.dry_run else "indexación")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Registros actuales en la colección:", collection.count())


if __name__ == "__main__":
    main()
