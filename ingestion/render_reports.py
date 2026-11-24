"""Renderización secuencial y reanudable de todos los PDFs de una carpeta."""

import argparse
from contextlib import closing
from importlib.metadata import version
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image
from pypdfium2 import PdfDocument, PdfiumError

from ingestion.export_pdf import write_jsonl
from ingestion.render_page import PROJECT_ROOT, render_document_page
from ingestion.file_utils import file_sha256 as _file_hash


logger = logging.getLogger(__name__)
RENDER_SCHEMA_VERSION = 1


def _can_skip(record: dict, image_path: Path) -> bool:
    if (
        record.get("status") != "ready"
        or record.get("file") != image_path.name
        or not image_path.is_file()
    ):
        return False
    return (
        image_path.stat().st_size == record.get("bytes")
        and _file_hash(image_path) == record.get("image_sha256")
    )


def _render_atomic(document, page: int, output: Path, dpi: int) -> dict:
    """Publica únicamente PNGs completos y verificados, cerrando antes de mover."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".png", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        render_document_page(document, page, temporary_path, dpi)
        with Image.open(temporary_path) as image:
            if image.format != "PNG":
                raise ValueError("El renderer no produjo un PNG válido.")
            width, height = image.size
            image.verify()
        record = {
            "status": "ready", "file": output.name,
            "bytes": temporary_path.stat().st_size,
            "image_sha256": _file_hash(temporary_path),
            "width": width, "height": height,
        }
        temporary_path.replace(output)
        return record
    finally:
        temporary_path.unlink(missing_ok=True)


def render_reports(
    input_dir: Path,
    output_dir: Path,
    dpi: int = 144,
    *,
    pattern: str = "*.pdf",
    max_new_pages: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Renderiza por página con un solo PDF abierto cada vez, sin llamadas IA.

    La reanudación exige manifiesto compatible y hash de imagen correcto. Un
    PNG previo sin manifiesto se regenera una vez. Se registra y continúa ante
    errores de página/documento; los errores al guardar el manifiesto se propagan.
    No usar simultáneamente dos procesos escribiendo en la misma carpeta.
    """
    if type(dpi) is not int or not 72 <= dpi <= 300:
        raise ValueError("dpi debe ser un entero entre 72 y 300.")
    if max_new_pages is not None and (type(max_new_pages) is not int or max_new_pages < 1):
        raise ValueError("max_new_pages debe ser un entero positivo o None.")
    if type(dry_run) is not bool:
        raise ValueError("dry_run debe ser booleano.")
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    pdfs = sorted(p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        raise ValueError("No se encontraron PDFs con el patrón indicado.")
    if len({p.stem.casefold() for p in pdfs}) != len(pdfs):
        raise ValueError("Los nombres de los PDFs producirían imágenes con rutas duplicadas.")

    report = {"documents_found": len(pdfs), "documents": [], "planned": 0,
              "rendered": 0, "skipped": 0, "errors": 0, "limited": False}
    engine_versions = {"renderer_version": version("pypdfium2"), "pillow_version": version("Pillow")}
    for pdf_path in pdfs:
        document_report = {"source_file": pdf_path.name, "page_count": None,
                           "planned": 0, "rendered": 0, "skipped": 0, "errors": 0}
        report["documents"].append(document_report)
        document = None
        try:
            source_hash = _file_hash(pdf_path)
            document = PdfDocument(str(pdf_path))
        except (OSError, PdfiumError) as error:
            document_report.update(errors=1, error_type=type(error).__name__)
            report["errors"] += 1
            logger.error("%s: error de lectura (%s)", pdf_path.name, type(error).__name__)
            continue

        with closing(document):
            count = len(document)
            document_report["page_count"] = count
            header = {"source_file": pdf_path.name, "source_sha256": source_hash,
                      "dpi": dpi, "page_count": count, "render_schema_version": RENDER_SCHEMA_VERSION,
                      **engine_versions}
            manifest_path = output_dir / f"{pdf_path.stem}.{dpi}dpi.manifest.json"
            previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            if not isinstance(previous, dict) or not isinstance(previous.get("pages", {}), dict):
                raise ValueError(f"Manifiesto no válido: {manifest_path.name}")
            compatible = all(previous.get(key) == value for key, value in header.items())
            manifest = {**header, "pages": previous.get("pages", {}) if compatible else {}}
            logger.info("%s: %s páginas; manifiesto compatible=%s", pdf_path.name, count, compatible)
            for page in range(1, count + 1):
                output = output_dir / f"{pdf_path.stem}.p{page}.{dpi}dpi.png"
                record = manifest["pages"].get(str(page), {})
                if not isinstance(record, dict):
                    raise ValueError(f"Registro de página inválido: {manifest_path.name}, {page}")
                if _can_skip(record, output):
                    report["skipped"] += 1
                    document_report["skipped"] += 1
                    continue
                if max_new_pages is not None and report["planned"] >= max_new_pages:
                    report["limited"] = True
                    break
                report["planned"] += 1
                document_report["planned"] += 1
                if dry_run:
                    continue
                try:
                    manifest["pages"][str(page)] = _render_atomic(document, page, output, dpi)
                    report["rendered"] += 1
                    document_report["rendered"] += 1
                except (OSError, ValueError, PdfiumError) as error:
                    manifest["pages"][str(page)] = {"status": "error", "error_type": type(error).__name__}
                    report["errors"] += 1
                    document_report["errors"] += 1
                    logger.error("%s página %s: %s", pdf_path.name, page, type(error).__name__)
                # Un único registro JSON reutiliza la escritura atómica existente.
                write_jsonl([manifest], manifest_path)
                if page % 25 == 0 or page == count:
                    logger.info("%s: página %s/%s; nuevas=%s; errores=%s", pdf_path.name,
                                page, count, document_report["rendered"], document_report["errors"])
        if report["limited"]:
            break
    report["complete"] = (
        not report["limited"] and report["errors"] == 0
        and (not dry_run or report["planned"] == 0)
    )
    return report


def main() -> None:
    """Lista o renderiza los PDFs y muestra un resumen con salida no cero si falla."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/raw/nvidia")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed/page_images")
    parser.add_argument("--pattern", default="*.pdf")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = render_reports(args.input_dir, args.output_dir, args.dpi,
                            pattern=args.pattern, max_new_pages=args.limit, dry_run=args.dry_run)
    print("Modo:", "simulación sin escrituras" if args.dry_run else "renderización")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
