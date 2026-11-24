import json
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

from ingestion.extract_pdf import extract_pages


def export_pdf(pdf_path: Path, output_path: Path) -> int:
    """Publica un JSONL completo y devuelve el número de páginas exportadas."""
    return write_jsonl(extract_pages(pdf_path), output_path)


def write_jsonl(records: Iterable[dict], output_path: Path) -> int:
    """Escribe registros con reemplazo final y devuelve su cantidad."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0

    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            # Un prefijo fijo evita que el temporal supere MAX_PATH en Windows
            # cuando la ruta del destino todavía cabe en el límite clásico.
            prefix=".tmp-",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_count += 1

        # Windows requiere cerrar el temporal antes de reemplazar el destino.
        temporary_path.replace(output_path)
        return record_count
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
