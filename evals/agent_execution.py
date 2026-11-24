"""Controles y trazas mínimas para evaluaciones con herramientas reales."""

from pathlib import Path

import chromadb

from retrieval.prepared_corpus import load_prepared_corpus


def check_agent_index(project_root: Path, active_collection: str) -> dict:
    """Impide evaluar el agente multiinforme sobre un índice distinto o parcial."""
    expected = "nvidia_reports_fixed_gemini001_768"
    if active_collection != expected:
        raise ValueError("La colección multiinforme todavía no está activada en el agente.")
    corpus = load_prepared_corpus(
        project_root / "data/processed/report_corpus/corpus.fixed.c1200.o200.manifest.json",
        project_root / "data/raw/nvidia",
    )
    store = chromadb.PersistentClient(path=str(project_root / "data/processed/chroma"))
    collection = store.get_collection(expected, embedding_function=None)
    metadata = collection.metadata or {}
    if (
        metadata.get("index_complete") is not True
        or metadata.get("text_corpus_sha256") != corpus.corpus_sha256
        or collection.count() != sum(bool(chunk["text"].strip()) for chunk in corpus.chunks)
    ):
        raise ValueError("El índice del agente no está completo o no corresponde al corpus.")
    return {"collection": expected, "text_corpus_sha256": corpus.corpus_sha256}


def collect_tool_events(event: object) -> list[dict]:
    """Extrae nombres, filtros y procedencia sin guardar consultas, textos ni imágenes."""
    content = getattr(event, "content", None)
    records = []
    for part in getattr(content, "parts", None) or []:
        call = getattr(part, "function_call", None)
        if call is not None:
            record = {"kind": "call", "tool_name": call.name}
            args = call.args or {}
            for key in ("source_file", "pdf_page"):
                if key in args:
                    record[key] = args[key]
            records.append(record)
        response = getattr(part, "function_response", None)
        if response is not None:
            # set_model_response también contiene la respuesta del usuario;
            # la conservamos en su archivo propio, no duplicada en la traza.
            data = response.response or {}
            records.append({
                "kind": "response", "tool_name": response.name,
                "status": data.get("status"),
                "evidence": [
                    {key: hit[key] for key in ("source_file", "pdf_page", "chunk_id", "document_id")
                     if key in hit}
                    for hit in data.get("evidence", [])
                ] if response.name == "search_corporate_reports" else [],
            })
    return records


def validate_agent_execution(response: object, case: dict, trace: list[dict]) -> None:
    """Comprueba uso de búsqueda, filtro explícito y citas; no juzga fidelidad semántica."""
    source = case["source_file"]
    calls = [event for event in trace
             if event["kind"] == "call" and event["tool_name"] == "search_corporate_reports"]
    if not calls or any(call.get("source_file") != source for call in calls):
        raise ValueError("El agente debe buscar en el documento solicitado explícitamente.")
    results = [event for event in trace
               if event["kind"] == "response" and event["tool_name"] == "search_corporate_reports"]
    if not results or any(event["status"] not in ("ok", "no_results") for event in results):
        raise ValueError("La búsqueda del agente falló; no confundirlo con falta de evidencia.")
    evidence = [hit for event in results for hit in event["evidence"]]
    if any(hit.get("source_file") != source for hit in evidence):
        raise ValueError("La herramienta devolvió evidencia de otro documento.")
    if response.status == "answered" and (
        not evidence or not response.sources or any(source not in citation for citation in response.sources)
    ):
        raise ValueError("La respuesta requiere evidencia y citas del documento solicitado.")
