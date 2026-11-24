"""Indexación multiinforme con reutilización del baseline y presupuesto de API."""

import argparse
import json
import math
from numbers import Real
from pathlib import Path
import time

import chromadb
from chromadb.errors import NotFoundError
from dotenv import load_dotenv
from google import genai

from retrieval.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, embed_text
from retrieval.prepared_corpus import PreparedCorpus, load_prepared_corpus
from retrieval.report_catalog import PROJECT_ROOT


LEGACY_COLLECTION = "nvidia_fixed_gemini001_768"
CORPUS_COLLECTION = "nvidia_reports_fixed_gemini001_768"


def _collection(store, name):
    try:
        return store.get_collection(name, embedding_function=None)
    except NotFoundError:
        return None


def _check_collection(collection, expected: dict) -> None:
    metadata = collection.metadata or {}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("La colección no coincide con el corpus o la configuración de embeddings.")
    if (collection.configuration.get("hnsw") or {}).get("space") != "cosine":
        raise ValueError("La colección debe usar distancia cosine.")


def _reuse_vectors(collection) -> dict:
    """Carga solo el baseline conocido, creado con RETRIEVAL_DOCUMENT y L2.

    Su metadata antigua no incluía el task_type. Esta migración se restringe a
    la colección del ejercicio, cuya creación conocemos, no a caches arbitrarias.
    Se verifica además origen, texto y norma de cada vector reutilizable.
    """
    if collection is None:
        return {}
    if collection.name != LEGACY_COLLECTION:
        raise ValueError("La migración solo admite el baseline conocido de este proyecto.")
    _check_collection(collection, {
        "embedding_model": EMBEDDING_MODEL, "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "chunk_strategy": "fixed",
    })
    metadata = collection.metadata or {}
    if (
        metadata.get("embedding_task_type", "RETRIEVAL_DOCUMENT") != "RETRIEVAL_DOCUMENT"
        or metadata.get("embedding_normalization", "l2") != "l2"
    ):
        raise ValueError("La tarea o normalización del baseline son incompatibles.")
    if collection.count() == 0:
        return {}
    rows = collection.get(include=["documents", "metadatas", "embeddings"])
    columns = [rows.get(key) for key in ("ids", "documents", "metadatas", "embeddings")]
    if any(column is None for column in columns) or len({len(column) for column in columns}) != 1:
        raise ValueError("El baseline no contiene vectores y documentos alineados.")
    vectors = {}
    for chunk_id, text, origin, values in zip(*columns):
        if not isinstance(text, str) or not text.strip() or not isinstance(origin, dict):
            raise ValueError("El baseline contiene registros sin texto o procedencia.")
        source, page = origin.get("source_file"), origin.get("pdf_page")
        start, end = origin.get("start_char"), origin.get("end_char")
        if (
            not isinstance(source, str) or not source.strip()
            or type(page) is not int or page < 1 or type(start) is not int or start < 0
            or type(end) is not int or end - start != len(text)
            or chunk_id != f"{source}:p{page}:start{start}"
        ):
            raise ValueError("El baseline contiene una identidad de chunk no válida.")
        if (
            values is None or len(values) != EMBEDDING_DIMENSIONS
            or any(isinstance(value, bool) or not isinstance(value, Real) for value in values)
        ):
            raise ValueError("Dimensiones o valores no válidos en un vector reutilizado.")
        vector = [float(value) for value in values]
        if not all(math.isfinite(value) for value in vector) or not math.isclose(
            math.hypot(*vector), 1.0, rel_tol=0, abs_tol=1e-5
        ):
            raise ValueError("El vector reutilizado debe ser finito y tener norma unitaria.")
        vectors[(source, page, start, end, text)] = (chunk_id, vector)
    return vectors


def _chunk_metadata(chunk: dict, corpus_hash: str) -> dict:
    return {
        **{key: chunk[key] for key in (
            "source_file", "document_id", "source_sha256", "pdf_page", "start_char",
            "end_char", "chunk_strategy", "chunk_size", "chunk_overlap",
        )},
        "text_corpus_sha256": corpus_hash,
    }


def index_corpus(
    client: genai.Client | None,
    store,
    corpus: PreparedCorpus,
    *,
    collection_name: str = CORPUS_COLLECTION,
    max_new_embeddings: int = 5,
    request_interval: float = 1.0,
    reuse_legacy: bool = True,
    dry_run: bool = False,
) -> dict:
    """Reutiliza primero y limita solo embeddings nuevos; cero permite copiar.

    El llamador carga el corpus mediante load_prepared_corpus antes de entrar.
    No borra registros ni modifica el baseline. Las escrituras previas se
    conservan al fallar. No añadir reintentos para errores permanentes/cuotas.
    """
    if type(max_new_embeddings) is not int or max_new_embeddings < 0:
        raise ValueError("max_new_embeddings debe ser un entero no negativo.")
    if (
        type(request_interval) not in (int, float) or not math.isfinite(request_interval)
        or request_interval < 0
    ):
        raise ValueError("request_interval debe ser finito y no negativo.")
    if type(dry_run) is not bool or type(reuse_legacy) is not bool:
        raise ValueError("dry_run y reuse_legacy deben ser booleanos.")
    if not isinstance(collection_name, str) or not collection_name.strip() or collection_name == LEGACY_COLLECTION:
        raise ValueError("El destino debe ser una colección separada del baseline.")
    settings = {
        "embedding_model": EMBEDDING_MODEL, "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "embedding_task_type": "RETRIEVAL_DOCUMENT", "embedding_normalization": "l2",
        "chunk_strategy": corpus.config["chunk_strategy"],
        "text_corpus_sha256": corpus.corpus_sha256,
    }
    target = _collection(store, collection_name)
    if target is not None:
        _check_collection(target, settings)
    legacy = _collection(store, LEGACY_COLLECTION) if reuse_legacy else None
    reusable = _reuse_vectors(legacy)
    nonempty = {chunk["chunk_id"]: chunk for chunk in corpus.chunks if chunk["text"].strip()}
    existing_ids = set()
    if target is not None:
        existing = target.get(include=["documents", "metadatas"])
        columns = [existing.get(key) for key in ("ids", "documents", "metadatas")]
        if any(column is None for column in columns) or len({len(column) for column in columns}) != 1:
            raise ValueError("El destino contiene registros incompletos.")
        for chunk_id, text, origin in zip(*columns):
            expected = nonempty.get(chunk_id)
            if (
                expected is None or text != expected["text"] or not isinstance(origin, dict)
                or any(origin.get(key) != value for key, value in _chunk_metadata(expected, corpus.corpus_sha256).items())
            ):
                raise ValueError("Un registro del destino no coincide con el corpus verificado.")
            existing_ids.add(chunk_id)

    copies, fresh = [], []
    for chunk_id, chunk in nonempty.items():
        if chunk_id in existing_ids:
            continue
        key = (chunk["source_file"], chunk["pdf_page"], chunk["start_char"], chunk["end_char"], chunk["text"])
        if key in reusable:
            copies.append((chunk, reusable[key]))
        else:
            fresh.append(chunk)
    report = {
        "collection": collection_name, "text_corpus_sha256": corpus.corpus_sha256,
        "reviewed": len(corpus.chunks), "empty_skipped": len(corpus.chunks) - len(nonempty),
        "already_indexed": len(existing_ids), "reuse_collection_found": legacy is not None,
        "reusable_embeddings": len(copies), "new_embeddings_needed": len(fresh),
        "new_embeddings_in_batch": min(max_new_embeddings, len(fresh)),
        "reused": 0, "embedded": 0, "saved": 0,
    }
    if not dry_run:
        if client is None and max_new_embeddings > 0 and fresh:
            raise ValueError("Se requiere cliente para generar embeddings nuevos.")
        if target is None:
            target = store.create_collection(
                collection_name, embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={**settings, "index_complete": False},
            )
        elif len(existing_ids) < len(nonempty):
            target.modify(metadata={**(target.metadata or {}), "index_complete": False})
        for chunk, (legacy_id, vector) in copies:
            target.upsert(
                ids=[chunk["chunk_id"]], documents=[chunk["text"]], embeddings=[vector],
                metadatas=[{**_chunk_metadata(chunk, corpus.corpus_sha256),
                            "embedding_origin": LEGACY_COLLECTION, "embedding_reuse_id": legacy_id}],
            )
            report["reused"] += 1
            report["saved"] += 1
        for chunk in fresh[:max_new_embeddings]:
            if report["embedded"] and request_interval:
                time.sleep(request_interval)
            vector = embed_text(client, chunk["text"], "RETRIEVAL_DOCUMENT")
            target.upsert(
                ids=[chunk["chunk_id"]], documents=[chunk["text"]], embeddings=[vector],
                metadatas=[{**_chunk_metadata(chunk, corpus.corpus_sha256), "embedding_origin": EMBEDDING_MODEL}],
            )
            report["embedded"] += 1
            report["saved"] += 1
        ready = len(existing_ids) + report["saved"] == len(nonempty)
        target.modify(metadata={**(target.metadata or {}), "index_complete": ready})
    report["pending"] = len(nonempty) - len(existing_ids) - report["saved"]
    report["complete"] = report["pending"] == 0
    report["collection_count"] = target.count() if target is not None else 0
    return report


def main() -> None:
    """Simula, copia vectores existentes o indexa una tanda limitada."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/processed/report_corpus/corpus.fixed.c1200.o200.manifest.json")
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "data/raw/nvidia")
    parser.add_argument("--store", type=Path, default=PROJECT_ROOT / "data/processed/chroma")
    parser.add_argument("--collection", default=CORPUS_COLLECTION)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    corpus = load_prepared_corpus(args.manifest, args.source_dir)
    if args.dry_run and not args.store.is_dir():
        raise FileNotFoundError("No existe el almacén Chroma indicado para la simulación.")
    store = chromadb.PersistentClient(path=str(args.store))
    options = dict(collection_name=args.collection, max_new_embeddings=args.limit,
                   request_interval=args.request_interval, reuse_legacy=not args.no_reuse,
                   dry_run=args.dry_run)
    if args.dry_run or args.limit == 0:
        report = index_corpus(None, store, corpus, **options)
    else:
        load_dotenv(PROJECT_ROOT / "corporate_reports/.env")
        with genai.Client(vertexai=False) as client:
            report = index_corpus(client, store, corpus, **options)
    print("Modo:", "simulación sin embeddings ni escritura de registros" if args.dry_run else "indexación")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
