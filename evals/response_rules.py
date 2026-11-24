"""Reglas deterministas por caso; no sustituyen la revisión de fidelidad."""

import re
from typing import Any


def validate_response_rules(response: Any, case: dict) -> None:
    """Rechaza expresiones conocidas, notas no solicitadas y cálculos no esperados.

    Las frases prohibidas se comparan literalmente ignorando mayúsculas y
    espacios. Es un detector acotado: puede omitir paráfrasis y rechazar una
    cita o negación que reproduzca la frase. No constituye un juez semántico.
    El control opcional de notas reconoce «Nota/Note» seguido de un número;
    no valida semánticamente las citas ni cubre todas sus posibles variantes.
    """
    if case.get("expect_no_calculation") and response.revenue_share_calculation is not None:
        raise ValueError("El caso requiere revenue_share_calculation: null.")
    text = " ".join(" ".join([response.answer, *response.limitations]).casefold().split())
    if case.get("forbid_note_numbers"):
        references = " ".join([text, *response.sources]).casefold()
        if re.search(r"\b(?:nota|note)\s+\d+\b", references):
            raise ValueError("Las referencias del caso deben usar archivo y página, sin números de nota.")
    for phrase in case.get("forbidden_phrases", []):
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError("Cada frase de regresión debe contener texto.")
        if " ".join(phrase.casefold().split()) in text:
            raise ValueError("La respuesta contiene una expresión no respaldada del caso de regresión.")
