"""Utilidades de identidad para archivos de la ingesta local."""

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Calcula el hash leyendo el archivo por bloques, sin cargarlo entero."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
