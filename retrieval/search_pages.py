"""Recuperación local de páginas mediante coincidencia de palabras."""

import json
import re
from pathlib import Path


def search_pages(
    query: str,
    index_path: Path,
    top_k: int = 3,
) -> list[dict[str, str | int]]:
    """Devuelve páginas con coincidencias, conservando origen y puntuación.

    Cuenta términos distintos compartidos, sin distinguir mayúsculas.
    Los empates se ordenan por archivo y página. No traduce ni usa embeddings.
    """
    if not isinstance(query, str):
        raise TypeError("query debe ser una cadena de texto.")
    query_terms = set(re.findall(r"\w+", query.casefold()))
    if not query_terms:
        raise ValueError("query debe contener al menos una palabra o número.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k debe ser un entero mayor que cero.")

    candidates = []
    with index_path.open(encoding="utf-8") as index:
        for line_number, line in enumerate(index, start=1):
            if not line.strip():
                continue
            try:
                page = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON no válido en la línea {line_number} del índice."
                ) from error

            if not isinstance(page, dict):
                raise ValueError(f"Se esperaba un objeto en la línea {line_number}.")
            source_file = page.get("source_file")
            pdf_page = page.get("pdf_page")
            text = page.get("text")
            if (
                not isinstance(source_file, str)
                or not source_file.strip()
                or type(pdf_page) is not int
                or pdf_page < 1
                or not isinstance(text, str)
            ):
                raise ValueError(
                    f"Registro de página no válido en la línea {line_number}: "
                    "se requieren source_file, pdf_page positivo y text."
                )

            page_terms = set(re.findall(r"\w+", text.casefold()))
            score = len(query_terms & page_terms)
            if score > 0:
                candidates.append({**page, "score": score})

    candidates.sort(
        key=lambda page: (-page["score"], page["source_file"], page["pdf_page"])
    )
    return candidates[:top_k]
