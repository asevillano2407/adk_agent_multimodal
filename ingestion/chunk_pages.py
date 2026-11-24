import re
from collections.abc import Iterator
from typing import Literal


def chunk_page(
    page: dict,
    chunk_size: int = 1200,
    overlap: int = 200,
    strategy: Literal["fixed", "natural"] = "fixed",
) -> Iterator[dict]:
    """Divide una página en ventanas de caracteres con solapamiento.

    Las posiciones son offsets del texto original: inicio inclusivo y fin
    exclusivo. El texto vacío no genera chunks. Los cortes no respetan
    necesariamente palabras, frases ni tablas con la estrategia fija.
    La estrategia natural intenta respetar límites textuales; recurre al corte
    fijo si el límite elegido impediría avanzar manteniendo el solapamiento.
    """
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size debe ser un entero mayor que cero.")
    if type(overlap) is not int or not 0 <= overlap < chunk_size:
        raise ValueError("overlap debe ser un entero entre cero y chunk_size - 1.")
    if strategy not in ("fixed", "natural"):
        raise ValueError("strategy debe ser 'fixed' o 'natural'.")
    if not isinstance(page, dict):
        raise TypeError("page debe ser un diccionario.")

    source_file = page.get("source_file")
    pdf_page = page.get("pdf_page")
    text = page.get("text")
    if not isinstance(source_file, str) or not source_file.strip():
        raise ValueError("source_file debe ser una cadena no vacía.")
    if type(pdf_page) is not int or pdf_page < 1:
        raise ValueError("pdf_page debe ser un entero positivo.")
    if not isinstance(text, str):
        raise ValueError("text debe ser una cadena; usa una cadena vacía si no hay texto.")
    identity = {}
    if "document_id" in page or "source_sha256" in page:
        source_hash = page.get("source_sha256")
        if (
            not isinstance(source_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
            or page.get("document_id") != f"sha256:{source_hash}"
        ):
            raise ValueError("document_id y source_sha256 deben identificar la misma versión.")
        identity = {
            "document_id": page["document_id"], "source_sha256": source_hash,
            "chunk_strategy": strategy, "chunk_size": chunk_size, "chunk_overlap": overlap,
        }

    start = 0
    while start < len(text):
        maximum_end = min(start + chunk_size, len(text))
        end = (
            choose_chunk_end(text, start, chunk_size)
            if strategy == "natural" else maximum_end
        )
        if end < len(text) and end - start <= overlap:
            end = maximum_end
        chunk_id = f"{source_file}:p{pdf_page}:start{start}"
        if strategy == "natural":
            chunk_id = (
                f"{source_file}:p{pdf_page}:natural:"
                f"c{chunk_size}:o{overlap}:start{start}"
            )
        if identity:
            chunk_id = (
                f"{source_file}:{identity['document_id']}:{strategy}:"
                f"c{chunk_size}:o{overlap}:p{pdf_page}:start{start}"
            )
        yield {
            **identity,
            "chunk_id": chunk_id,
            "source_file": source_file,
            "pdf_page": pdf_page,
            "start_char": start,
            "end_char": end,
            "text": text[start:end],
        }
        if end == len(text):
            break
        start = end - overlap

def choose_chunk_end(
    text: str,
    start: int,
    chunk_size: int,
) -> int:
    """Prioriza párrafo, frase y espacio en el último 40 % de la ventana.

    Devuelve un offset absoluto y exclusivo, sin modificar el texto original.
    La detección de frases es heurística: no interpreta abreviaturas ni tablas.
    """
    if not isinstance(text, str):
        raise TypeError("text debe ser una cadena.")
    if type(start) is not int or start < 0:
        raise ValueError("start debe ser un entero no negativo.")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size debe ser un entero mayor que cero.")
    if start > len(text):
        raise ValueError("start no puede superar la longitud del texto.")

    end = min(start + chunk_size, len(text))

    if end == len(text):
        return end

    minimum_end = start + max(1, chunk_size * 3 // 5)
    # Un carácter adicional permite comprobar el espacio después de un punto
    # situado justo en el límite, sin permitir que el corte supere ese límite.
    search_end = min(end + 1, len(text))
    patterns = (
        r"\r?\n[ \t]*\r?\n",  # Párrafo, incluyendo saltos CRLF.
        r"[.!?](?=\s)",       # Puntuación seguida de espacio en blanco.
        r"\s+",               # Espacios, tabulaciones o saltos de línea.
    )
    for pattern in patterns:
        candidates = (
            match.end()
            for match in re.compile(pattern).finditer(text, start, search_end)
            if minimum_end <= match.end() <= end
        )
        boundary = max(candidates, default=None)
        if boundary is not None:
            return boundary

    return end
