from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfReader


def extract_pages(pdf_path: Path) -> Iterator[dict[str, str | int]]:
    """Extrae texto por página conservando el archivo de origen."""
    reader = PdfReader(pdf_path)

    for pdf_page, page in enumerate(reader.pages, start=1):
        yield {
            "source_file": pdf_path.name,
            "pdf_page": pdf_page,
            "text": page.extract_text() or "",
        }