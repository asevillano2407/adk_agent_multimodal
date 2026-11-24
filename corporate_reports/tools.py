"""Herramientas de lectura del corpus corporativo local."""

import json
import logging
from pathlib import Path

import chromadb
from chromadb.errors import ChromaError
from google import genai
from google.genai.errors import APIError

from retrieval.format_context import format_context
from retrieval.report_catalog import load_report_catalog
from retrieval.search_vectors import search_chunks
from .config import COLLECTION_NAME, MULTIREPORT_COLLECTION_NAME, LEGACY_SOURCE_FILE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def search_corporate_reports(query: str, source_file: str = "") -> dict:
    """Busca evidencia en la colección textual configurada.

    Args:
        query: Pregunta o términos de búsqueda, en español o inglés.
        source_file: Nombre exacto del informe solicitado; vacío busca sin filtro.
            El documento debe estar disponible en la colección activa.

    Returns:
        Estado y hasta tres fragmentos documentales no confiables, con archivo,
        página PDF desde 1 y offsets. Encontrar fragmentos no garantiza respuesta.
        Un error de recuperación no significa que la información no exista.
    """
    if not isinstance(query, str) or not query.strip():
        return {"status": "invalid_query", "evidence": [],
                "message": "La consulta debe contener texto."}
    if not isinstance(source_file, str) or (
        source_file and (not source_file.strip() or any(c in source_file for c in ("/", "\\", ":", "\x00")))
    ):
        return {"status": "invalid_document", "evidence": [],
                "message": "Usa el nombre exacto del documento del catálogo."}
    multi_report = COLLECTION_NAME == MULTIREPORT_COLLECTION_NAME
    if not multi_report and source_file and source_file != LEGACY_SOURCE_FILE:
        return {"status": "document_not_indexed", "evidence": [],
                "message": "La búsqueda textual activa solo permite consultar el informe de 2025."}
    try:
        options = {}
        if source_file:
            options["source_file"] = source_file
            if multi_report:
                catalog = load_report_catalog(PROJECT_ROOT / "data/processed/page_images")
                if source_file not in catalog:
                    return {"status": "unknown_document", "evidence": [],
                            "message": "Consulta el catálogo antes de seleccionar el documento."}
                options["document_id"] = f"sha256:{catalog[source_file]['source_sha256']}"
        store = chromadb.PersistentClient(
            path=str(PROJECT_ROOT / "data/processed/chroma")
        )
        collection = store.get_collection(COLLECTION_NAME, embedding_function=None)
        if multi_report and (collection.metadata or {}).get("index_complete") is not True:
            return {"status": "index_not_ready", "evidence": [],
                    "message": "El índice multiinforme todavía no está disponible."}
        with genai.Client(vertexai=False) as client:
            hits = search_chunks(client, collection, query, top_k=3, **options)
        return {
            "status": "ok" if hits else "no_results",
            "content_is_untrusted": True,
            "evidence": json.loads(format_context(hits)),
        }
    except (APIError, ChromaError, OSError, ValueError) as error:
        # No registrar la consulta, los documentos ni mensajes del proveedor.
        logger.warning("Fallo de recuperación corporativa: %s", type(error).__name__)
        return {
            "status": "retrieval_error",
            "evidence": [],
            "message": "No se pudo consultar el índice. No deduzcas ausencia de información.",
        }
