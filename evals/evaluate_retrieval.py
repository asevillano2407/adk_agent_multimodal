"""Evaluación local de recuperación contra páginas de referencia conocidas."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from retrieval.search_pages import search_pages


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
) -> dict:
    """Ejecuta consultas y devuelve métricas por caso y sus medias."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k debe ser un entero mayor que cero.")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("El conjunto de evaluación debe ser una lista no vacía.")

    # Comprobar las etiquetas antes de ejecutar: una ruta/página errónea no es
    # un fallo de búsqueda. search_pages validará el esquema del índice.
    with index_path.open(encoding="utf-8") as index:
        index_keys = {
            _page_key(json.loads(line)) for line in index if line.strip()
        }
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
        if not isinstance(references, list) or not references:
            raise ValueError(f"El caso {case_id} requiere páginas de referencia.")
        if any(_page_key(page) not in index_keys for page in references):
            raise ValueError(f"El caso {case_id} referencia una página ausente del índice.")

    results = []
    for case in cases:
        retrieved = search_pages(case["query"], index_path, top_k)
        results.append({
            "id": case["id"],
            "query": case["query"],
            "relevant_pages": case["relevant_pages"],
            "retrieved_pages": [
                {
                    "source_file": page["source_file"],
                    "pdf_page": page["pdf_page"],
                    "score": page["score"],
                }
                for page in retrieved
            ],
            **retrieval_metrics(retrieved, case["relevant_pages"], top_k),
        })

    return {
        "retriever": "lexical_unique_terms_v1",
        "top_k": top_k,
        "cases_file": str(cases_path),
        "index_file": str(index_path),
        "case_count": len(results),
        "mean_hit_at_k": sum(case["hit_at_k"] for case in results) / len(results),
        "mrr_at_k": sum(case["reciprocal_rank_at_k"] for case in results) / len(results),
        "label_scope": "Páginas relevantes conocidas; etiquetas no exhaustivas.",
        "cases": results,
    }


def main() -> None:
    """Evalúa el índice local y conserva un informe por ejecución."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=PROJECT_ROOT / "evals/retrieval_cases.json"
    )
    parser.add_argument(
        "--index", type=Path,
        default=PROJECT_ROOT / "data/processed/nvidia_2025.pages.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    report = evaluate_retrieval(args.cases, args.index, args.top_k)
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
        print(
            f"{case['id']}: páginas={pages}; Hit@{args.top_k}={case['hit_at_k']}; "
            f"RR@{args.top_k}={case['reciprocal_rank_at_k']:.3f}"
        )
    print(f"Hit@{args.top_k} medio: {report['mean_hit_at_k']:.3f}")
    print(f"MRR@{args.top_k}: {report['mrr_at_k']:.3f}")
    print(f"Informe: {output_path}")


if __name__ == "__main__":
    main()
