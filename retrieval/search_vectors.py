"""Búsqueda semántica explícita en una colección Chroma existente."""

import argparse
import json
import math
import re
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from google import genai

from retrieval.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, embed_text


def search_chunks(
    client: genai.Client,
    collection: Collection,
    query: str,
    top_k: int = 3,
    *,
    source_file: str | None = None,
    document_id: str | None = None,
) -> list[dict[str, str | int | float]]:
    """Recupera chunks por distancia coseno; menor distancia implica cercanía.

    Conserva el origen y los offsets para construir contexto y evaluar después.
    No aplica umbral de relevancia ni genera respuestas. Los errores de API o
    almacenamiento se propagan. El llamador administra el cliente de Gemini.
    """
    if not isinstance(query, str):
        raise TypeError("query debe ser una cadena de texto.")
    if not query.strip():
        raise ValueError("query no puede estar vacía.")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k debe ser un entero mayor que cero.")
    if source_file is not None and (not isinstance(source_file, str) or not source_file.strip()):
        raise ValueError("source_file debe ser un nombre de documento no vacío.")
    if document_id is not None and (
        not isinstance(document_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", document_id) is None
    ):
        raise ValueError("document_id debe ser un identificador SHA256 válido.")
    metadata = collection.metadata or {}
    if metadata.get("text_corpus_sha256") and metadata.get("index_complete") is not True:
        raise ValueError("La colección multiinforme todavía está incompleta.")
    if (
        metadata.get("embedding_model") != EMBEDDING_MODEL
        or metadata.get("embedding_dimensions") != EMBEDDING_DIMENSIONS
    ):
        raise ValueError("El modelo o las dimensiones de la colección son incompatibles.")
    if (collection.configuration.get("hnsw") or {}).get("space") != "cosine":
        raise ValueError("La colección debe utilizar distancia cosine.")

    count = collection.count()
    if count == 0:
        return []
    vector = embed_text(client, query, "RETRIEVAL_QUERY")
    filters = []
    if source_file is not None:
        filters.append({"source_file": source_file})
    if document_id is not None:
        filters.append({"document_id": document_id})
    options = {}
    if filters:
        options["where"] = filters[0] if len(filters) == 1 else {"$and": filters}
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
        **options,
    )
    rows = []
    for field in ("ids", "documents", "metadatas", "distances"):
        groups = result.get(field)
        if groups is None or len(groups) != 1 or groups[0] is None:
            raise ValueError(f"Respuesta de Chroma no válida: {field}.")
        rows.append(groups[0])
    ids, documents, metadatas, distances = rows
    if len({len(row) for row in rows}) != 1:
        raise ValueError("Los campos de la respuesta de Chroma tienen tamaños distintos.")

    hits = []
    for chunk_id, text, origin, distance in zip(ids, documents, metadatas, distances):
        if (
            not isinstance(chunk_id, str) or not chunk_id.strip()
            or not isinstance(text, str) or not text.strip()
            or not isinstance(origin, dict)
        ):
            raise ValueError("El resultado no contiene ID, texto y metadatos válidos.")
        source = origin.get("source_file")
        page = origin.get("pdf_page")
        start = origin.get("start_char")
        end = origin.get("end_char")
        if (
            not isinstance(source, str) or not source.strip()
            or type(page) is not int or page < 1
            or type(start) is not int or start < 0
            or type(end) is not int or end - start != len(text)
            or isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or not math.isfinite(distance)
        ):
            raise ValueError("El resultado contiene origen, offsets o distancia no válidos.")
        if source_file is not None and source != source_file:
            raise ValueError("La búsqueda devolvió un documento fuera del filtro solicitado.")
        identity = {}
        if metadata.get("text_corpus_sha256") or "document_id" in origin or "source_sha256" in origin:
            source_hash = origin.get("source_sha256")
            if (
                not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
                or origin.get("document_id") != f"sha256:{source_hash}"
            ):
                raise ValueError("La versión del documento recuperado no es válida.")
            identity = {"document_id": origin["document_id"], "source_sha256": source_hash}
        if document_id is not None and origin.get("document_id") != document_id:
            raise ValueError("La búsqueda devolvió una versión fuera del filtro solicitado.")
        hits.append({
            **identity,
            "chunk_id": chunk_id, "source_file": source, "pdf_page": page,
            "start_char": start, "end_char": end, "text": text,
            "distance": float(distance),
        })
    return hits


def main() -> None:
    """Ejecuta una consulta sin modificar la colección."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--store", type=Path, default=project_root / "data/processed/chroma"
    )
    parser.add_argument("--collection", default="nvidia_fixed_gemini001_768")
    parser.add_argument("--source-file")
    parser.add_argument("--document-id")
    args = parser.parse_args()
    store = chromadb.PersistentClient(path=str(args.store))
    collection = store.get_collection(args.collection, embedding_function=None)
    load_dotenv(project_root / "corporate_reports/.env")
    with genai.Client(vertexai=False) as client:
        hits = search_chunks(client, collection, args.query, args.top_k,
                             source_file=args.source_file, document_id=args.document_id)
    print(json.dumps(hits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
