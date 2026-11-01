import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from ingestion.extract_pdf import extract_pages


def export_pdf(pdf_path: Path, output_path: Path) -> int:
    """Publica un JSONL completo y devuelve el número de páginas exportadas."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_count = 0

    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for record in extract_pages(pdf_path):
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                page_count += 1

        # Windows requiere cerrar el temporal antes de reemplazar el destino.
        temporary_path.replace(output_path)
        return page_count
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
