"""Generación explícita y validación de embeddings para recuperación."""

import math
from typing import Literal

from google import genai
from google.genai import types


EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768


def embed_text(
    client: genai.Client,
    text: str,
    task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"],
) -> list[float]:
    """Genera un vector finito de 768 dimensiones y norma unitaria.

    El cliente y sus credenciales pertenecen al llamador. Los errores del
    proveedor se propagan sin añadir reintentos ni registrar el texto enviado.
    """
    if not isinstance(text, str):
        raise TypeError("text debe ser una cadena.")
    if not text.strip():
        raise ValueError("text no puede estar vacío.")
    if task_type not in ("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"):
        raise ValueError("task_type debe ser RETRIEVAL_DOCUMENT o RETRIEVAL_QUERY.")

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )

    if result.embeddings is None or len(result.embeddings) != 1:
        raise ValueError("Se esperaba exactamente un embedding en la respuesta.")
    values = result.embeddings[0].values
    if not isinstance(values, (list, tuple)) or len(values) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"El embedding debe tener {EMBEDDING_DIMENSIONS} dimensiones.")

    vector = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("El embedding debe contener únicamente números finitos.")
        try:
            numeric_value = float(value)
        except OverflowError as error:
            raise ValueError("Un valor del embedding supera el rango numérico.") from error
        if not math.isfinite(numeric_value):
            raise ValueError("El embedding debe contener únicamente números finitos.")
        vector.append(numeric_value)

    # hypot evita el subdesbordamiento de elevar al cuadrado valores pequeños.
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("La norma del embedding debe ser finita y mayor que cero.")
    return [value / norm for value in vector]
