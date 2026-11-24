"""Renderiza una página PDF como PNG para evaluar evidencia visual."""

import argparse
from contextlib import closing
from importlib import import_module
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _validate_options(pdf_page: int, output_path: Path, dpi: int) -> None:
    if type(pdf_page) is not int or pdf_page < 1:
        raise ValueError("pdf_page debe ser un entero positivo, contando desde 1.")
    if type(dpi) is not int or not 72 <= dpi <= 300:
        raise ValueError("dpi debe ser un entero entre 72 y 300.")
    if output_path.suffix.lower() != ".png":
        raise ValueError("La salida debe tener extensión .png.")


def render_document_page(
    document: Any,
    pdf_page: int,
    output_path: Path,
    dpi: int = 144,
) -> Path:
    """Renderiza una página y cierra sus recursos, conservando abierto el PDF."""
    _validate_options(pdf_page, output_path, dpi)
    if pdf_page > len(document):
        raise ValueError(f"La página {pdf_page} supera las {len(document)} páginas del PDF.")
    with closing(document[pdf_page - 1]) as page:
        with closing(page.render(scale=dpi / 72)) as bitmap:
            with closing(bitmap.to_pil()) as image:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path, format="PNG")
    return output_path


def render_pdf_page(
    pdf_path: Path,
    pdf_page: int,
    output_path: Path,
    dpi: int = 144,
) -> Path:
    """Convierte una página desde 1 a PNG; no realiza OCR ni usa IA.

    Ejecutar secuencialmente: PDFium no admite llamadas simultáneas desde
    distintos threads. El documento y sus recursos se cierran incluso al fallar.
    """
    _validate_options(pdf_page, output_path, dpi)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    pdfium = import_module("pypdfium2")
    with closing(pdfium.PdfDocument(str(pdf_path))) as document:
        return render_document_page(document, pdf_page, output_path, dpi)


def main() -> None:
    """Genera una imagen local de una página seleccionada del corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf", type=Path,
        default=PROJECT_ROOT / "data/raw/nvidia/NVIDIA-2025-Annual-Report.pdf",
    )
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        PROJECT_ROOT / "data/processed/page_images"
        / f"{args.pdf.stem}.p{args.page}.{args.dpi}dpi.png"
    )
    path = render_pdf_page(args.pdf, args.page, output, args.dpi)
    print(f"Documento: {args.pdf.name}")
    print(f"Página PDF: {args.page}; resolución: {args.dpi} DPI")
    print(f"Imagen: {path}")


if __name__ == "__main__":
    main()
