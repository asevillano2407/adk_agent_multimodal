"""Evaluación local de recuperación contra páginas de referencia conocidas."""

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from retrieval.search_pages import search_pages, search_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def evidence_metrics(retrieved: list[dict], evidence: list[dict], top_k: int) -> dict:
    """Comprueba anclas revisadas dentro de un mismo chunk, sin juez LLM.

    Es una comprobación determinista de evidencia conocida, no una medida
    general de suficiencia semántica ni de corrección de una respuesta.
    """
    for rank, hit in enumerate(retrieved[:top_k], 1):
        for reference in evidence:
            if _page_key(hit) == _page_key(reference) and all(
                _normalize(anchor) in _normalize(hit["text"])
                for anchor in reference["all_of"]
            ):
                return {"evidence_hit_at_k": 1, "first_evidence_rank": rank}
    return {"evidence_hit_at_k": 0, "first_evidence_rank": None}


def _page_key(page: dict) -> tuple[str, int]:
    """Identifica una página por documento y posición PDF, desde uno."""
    if not isinstance(page, dict):
        raise ValueError("Una referencia de página debe ser un objeto.")
    source_file = page.get("source_file")
    pdf_page = page.get("pdf_page")
    if (
        not isinstance(source_file, str)
        or not source_file.strip()
        or type(pdf_page) is not int
        or pdf_page < 1
    ):
        raise ValueError("La referencia requiere source_file y pdf_page positivo.")
    return source_file, pdf_page


def retrieval_metrics(
    retrieved: list[dict],
    relevant_pages: list[dict],
    top_k: int = 3,
) -> dict:
    """Calcula Hit@K y RR@K contra referencias parciales, sin juzgar la prosa."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k debe ser un entero mayor que cero.")
    if not isinstance(relevant_pages, list) or not relevant_pages:
        raise ValueError("Se requiere al menos una página de referencia.")
    reference_keys = {_page_key(page) for page in relevant_pages}
    for rank, page in enumerate(retrieved[:top_k], start=1):
        if _page_key(page) in reference_keys:
            return {
                "hit_at_k": 1,
                "reciprocal_rank_at_k": 1 / rank,
                "first_relevant_rank": rank,
            }
    return {
        "hit_at_k": 0,
        "reciprocal_rank_at_k": 0.0,
        "first_relevant_rank": None,
    }


def evaluate_retrieval(
    cases_path: Path,
    index_path: Path,
    top_k: int = 3,
    *,
    search_fn: Callable[[str, int], list[dict]] | None = None,
    retriever_name: str = "lexical_unique_terms_v1",
    index_records: list[dict] | None = None,
    search_case_fn: Callable[[dict, int], list[dict]] | None = None,
) -> dict:
    """Ejecuta consultas y devuelve métricas por caso y sus medias."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k debe ser un entero mayor que cero.")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("El conjunto de evaluación debe ser una lista no vacía.")

    # Comprobar las etiquetas antes de ejecutar: una ruta/página errónea no es
    # un fallo de búsqueda. search_pages validará el esquema del índice.
    index_keys = set()
    if search_fn is not None and search_case_fn is not None:
        raise ValueError("Selecciona una única función de búsqueda.")
    if index_records is not None and search_fn is None and search_case_fn is None:
        raise ValueError("Un corpus en memoria requiere una función de búsqueda explícita.")
    if index_records is None:
        with index_path.open(encoding="utf-8") as index:
            index_records = [json.loads(line) for line in index if line.strip()]
    index_units = set()
    for record in index_records:
        index_keys.add(_page_key(record))
        index_units.add("chunk" if "chunk_id" in record else "page")
    if len(index_units) > 1:
        raise ValueError("El índice mezcla páginas completas y chunks.")
    retrieval_unit = next(iter(index_units), "page")
    seen_ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Cada caso debe ser un objeto.")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
            raise ValueError("Cada caso debe tener un id no vacío y único.")
        seen_ids.add(case_id)
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"El caso {case_id} requiere una consulta no vacía.")
        references = case.get("relevant_pages")
        unanswerable = case.get("answerable", True) is False
        if not isinstance(references, list) or (not references and not unanswerable):
            raise ValueError(f"El caso {case_id} requiere páginas de referencia.")
        if unanswerable and references:
            raise ValueError(f"El caso negativo {case_id} no debe tener páginas relevantes.")
        if any(_page_key(page) not in index_keys for page in references):
            raise ValueError(f"El caso {case_id} referencia una página ausente del índice.")
        source = case.get("source_file")
        if source is not None and (
            not isinstance(source, str) or not source.strip()
            or any(page["source_file"] != source for page in references)
            or source not in {key[0] for key in index_keys}
        ):
            raise ValueError(f"Filtro de documento no válido para {case_id}.")
        for evidence in case.get("evidence", []):
            anchors = evidence.get("all_of")
            if (
                _page_key(evidence) not in {_page_key(page) for page in references}
                or not isinstance(anchors, list) or not anchors
                or any(not isinstance(a, str) or not a.strip() for a in anchors)
            ):
                raise ValueError(f"Evidencia no válida para {case_id}.")
            if not evidence_metrics(index_records, [evidence], len(index_records))["evidence_hit_at_k"]:
                raise ValueError(f"Anclas de evidencia ausentes del índice para {case_id}.")

    results = []
    for case in cases:
        retrieved = search_case_fn(case, top_k) if search_case_fn is not None else (
            search_pages(case["query"], index_path, top_k)
            if search_fn is None else search_fn(case["query"], top_k)
        )
        results.append({
            "id": case["id"],
            "query": case["query"],
            "requested_source_file": case.get("source_file"),
            "split": case.get("split", "development"),
            "answerable": case.get("answerable", True),
            "missing_information": case.get("missing_information"),
            "relevant_pages": case["relevant_pages"],
            "retrieved_pages": [
                {
                    key: page[key]
                    for key in (
                        "source_file", "pdf_page", "score", "distance",
                        "chunk_id", "start_char", "end_char", "document_id",
                    )
                    if key in page
                }
                for page in retrieved
            ],
            "returned_units": len(retrieved),
            "unique_pages": len({_page_key(page) for page in retrieved}),
            "retrieved_text_characters": sum(len(page["text"]) for page in retrieved),
            **(retrieval_metrics(retrieved, case["relevant_pages"], top_k)
               if case.get("answerable", True) else {
                   "hit_at_k": None, "reciprocal_rank_at_k": None,
                   "first_relevant_rank": None,
               }),
            **(evidence_metrics(retrieved, case["evidence"], top_k)
               if case.get("evidence") else {}),
        })

    positives = [case for case in results if case["answerable"]]
    checked = [case for case in positives if "evidence_hit_at_k" in case]
    return {
        "retriever": retriever_name,
        "retrieval_unit": retrieval_unit,
        "relevance_granularity": "page",
        "top_k": top_k,
        "cases_file": str(cases_path),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "index_file": str(index_path),
        "case_count": len(results),
        "answerable_count": len(positives),
        "unanswerable_count": len(results) - len(positives),
        "splits": {
            split: {
                "answerable_count": len(group),
                "mean_hit_at_k": sum(c["hit_at_k"] for c in group) / len(group),
                "mrr_at_k": sum(c["reciprocal_rank_at_k"] for c in group) / len(group),
            }
            for split in sorted({c["split"] for c in positives})
            if (group := [c for c in positives if c["split"] == split])
        },
        "mean_hit_at_k": (sum(c["hit_at_k"] for c in positives) / len(positives)
                          if positives else None),
        "mrr_at_k": (sum(c["reciprocal_rank_at_k"] for c in positives) / len(positives)
                     if positives else None),
        "mean_evidence_hit_at_k": (sum(c["evidence_hit_at_k"] for c in checked) / len(checked)
                                   if checked else None),
        "mean_retrieved_text_characters": sum(
            case["retrieved_text_characters"] for case in results
        ) / len(results),
        "label_scope": "Páginas relevantes conocidas; etiquetas no exhaustivas.",
        "cases": results,
    }


def main() -> None:
    """Evalúa el índice local y conserva un informe por ejecución."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=PROJECT_ROOT / "evals/retrieval_cases.json"
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--index", type=Path,
    )
    input_group.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "data/raw/nvidia")
    parser.add_argument("--validate-only", action="store_true",
                        help="Comprueba las etiquetas locales sin búsquedas ni API.")
    parser.add_argument("--filter-by-document", action="store_true",
                        help="Aplica el source_file solicitado explícitamente en cada caso.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--retriever", choices=("lexical", "vector"), default="lexical")
    parser.add_argument(
        "--store", type=Path, default=PROJECT_ROOT / "data/processed/chroma"
    )
    parser.add_argument("--collection", default="nvidia_fixed_gemini001_768")
    args = parser.parse_args()
    records = None
    corpus = None
    if args.corpus_manifest:
        from retrieval.prepared_corpus import load_prepared_corpus

        corpus = load_prepared_corpus(args.corpus_manifest, args.source_dir)
        records = corpus.chunks
        args.index = args.corpus_manifest
    elif args.index is None:
        args.index = PROJECT_ROOT / "data/processed/nvidia_2025.pages.jsonl"
    if args.filter_by_document and args.retriever == "lexical" and records is None and not args.validate_only:
        parser.error("El filtro léxico requiere --corpus-manifest.")
    if args.validate_only:
        report = evaluate_retrieval(
            args.cases, args.index, args.top_k,
            index_records=records, search_fn=lambda query, k: [],
        )
        print(f"PASS: {report['case_count']} casos con referencias locales válidas. "
              "Sin búsquedas ni llamadas al modelo; no se han medido métricas.")
        return

    def requested_source(case: dict) -> str | None:
        if not args.filter_by_document or not case.get("answerable", True):
            return None
        source = case.get("source_file")
        if source is None:
            raise ValueError("El filtro requiere un único documento explícito por caso.")
        return source

    if args.retriever == "lexical":
        if records is None:
            report = evaluate_retrieval(args.cases, args.index, args.top_k)
        else:
            report = evaluate_retrieval(
                args.cases, args.index, args.top_k, index_records=records,
                search_case_fn=lambda case, k: search_records(
                    case["query"], records, k, source_file=requested_source(case),
                ),
            )
    else:
        import chromadb
        from dotenv import load_dotenv
        from google import genai

        from retrieval.search_vectors import search_chunks

        store = chromadb.PersistentClient(path=str(args.store))
        collection = store.get_collection(args.collection, embedding_function=None)
        if corpus is not None and (
            (collection.metadata or {}).get("text_corpus_sha256") != corpus.corpus_sha256
            or (collection.metadata or {}).get("index_complete") is not True
        ):
            raise ValueError("La colección no está completa o no corresponde al corpus verificado.")

        def search_case(case: dict, k: int) -> list[dict]:
            return search_chunks(client, collection, case["query"], k,
                                 source_file=requested_source(case))

        # El archivo pasado con --index debe describir el corpus de la colección.
        load_dotenv(PROJECT_ROOT / "corporate_reports/.env")
        with genai.Client(vertexai=False) as client:
            report = evaluate_retrieval(
                args.cases, args.index, args.top_k,
                index_records=records, search_case_fn=search_case,
                retriever_name="gemini_chroma_cosine_v1",
            )
        report["retrieval_unit"] = "chunk"
        report["vector_store"] = str(args.store)
        report["collection"] = args.collection
        report["collection_count"] = collection.count()
        report["collection_metadata"] = collection.metadata
    report["filter_by_document"] = args.filter_by_document
    if corpus is not None:
        report["text_corpus_sha256"] = corpus.corpus_sha256
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report["timestamp_utc"] = run_id
    output_dir = PROJECT_ROOT / "evals/runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"retrieval_{run_id}.json"
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")

    for case in report["cases"]:
        pages = [page["pdf_page"] for page in case["retrieved_pages"]]
        if not case["answerable"]:
            print(f"{case['id']}: negativo; páginas={pages}; revisión manual, sin Hit/RR")
            continue
        print(
            f"{case['id']}: páginas={pages}; Hit@{args.top_k}={case['hit_at_k']}; "
            f"RR@{args.top_k}={case['reciprocal_rank_at_k']:.3f}; "
            f"páginas distintas={case['unique_pages']}; "
            f"caracteres={case['retrieved_text_characters']}; "
            f"evidencia={case.get('evidence_hit_at_k', 'sin anclas')}"
        )
    print(f"Unidad recuperada: {report['retrieval_unit']}; etiquetas: páginas conocidas")
    print(f"Hit@{args.top_k} medio (positivos): {report['mean_hit_at_k']}")
    print(f"MRR@{args.top_k} (positivos): {report['mrr_at_k']}")
    print(f"Hit de evidencia@{args.top_k}: {report['mean_evidence_hit_at_k']}")
    print(f"Informe: {output_path}")


if __name__ == "__main__":
    main()
