from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Inspecciona una página conocida del informe de 2025."""
    pdf_path = (
        PROJECT_ROOT / "data/raw/nvidia/NVIDIA-2025-Annual-Report.pdf"
    )
    reader = PdfReader(pdf_path)
    pdf_page = 18

    print(f"Documento: {pdf_path.name}")
    print(f"Páginas totales: {len(reader.pages)}")
    print(f"Página PDF: {pdf_page}")

    page = reader.pages[pdf_page - 1]

    print("\n--- TEXTO NORMAL ---")
    print(page.extract_text())

    print("\n--- TEXTO CON DISPOSICIÓN ---")
    print(page.extract_text(extraction_mode="layout"))


if __name__ == "__main__":
    main()