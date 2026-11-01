"""Offline checks for page extraction and source traceability."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from pypdf import PdfWriter
from pypdf.errors import PdfReadError

from ingestion.extract_pdf import extract_pages


class TestExtractPages(unittest.TestCase):
    """Preserve page identity even when text is absent or extraction fails."""

    def test_preserves_origin_and_numbering_across_missing_text(self) -> None:
        pages = [MagicMock(), MagicMock(), MagicMock()]
        for page, text in zip(pages, ["First page", None, "Third page"]):
            page.extract_text.return_value = text

        with patch("ingestion.extract_pdf.PdfReader") as reader:
            reader.return_value.pages = pages
            records = list(extract_pages(Path("reports/annual.pdf")))

        self.assertEqual(records, [
            {"source_file": "annual.pdf", "pdf_page": 1, "text": "First page"},
            {"source_file": "annual.pdf", "pdf_page": 2, "text": ""},
            {"source_file": "annual.pdf", "pdf_page": 3, "text": "Third page"},
        ])

    def test_reads_real_pdf_and_keeps_blank_pages(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "blank_pages.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            writer.add_blank_page(width=200, height=200)
            with pdf_path.open("wb") as output:
                writer.write(output)

            self.assertEqual(list(extract_pages(pdf_path)), [
                {"source_file": "blank_pages.pdf", "pdf_page": 1, "text": ""},
                {"source_file": "blank_pages.pdf", "pdf_page": 2, "text": ""},
            ])

    def test_missing_pdf_raises(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaises(FileNotFoundError):
                list(extract_pages(Path(temporary_directory) / "missing.pdf"))

    def test_corrupt_pdf_raises(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "corrupt.pdf"
            pdf_path.write_bytes(b"%PDF-1.7\nIncomplete PDF document")
            with self.assertRaises(PdfReadError):
                list(extract_pages(pdf_path))

    def test_extraction_error_is_not_hidden(self) -> None:
        first_page = MagicMock()
        first_page.extract_text.return_value = "Readable page"
        broken_page = MagicMock()
        error = RuntimeError("Page extraction failed")
        broken_page.extract_text.side_effect = error

        with patch("ingestion.extract_pdf.PdfReader") as reader:
            reader.return_value.pages = [first_page, broken_page]
            records = extract_pages(Path("annual.pdf"))
            self.assertEqual(next(records)["pdf_page"], 1)
            with self.assertRaises(RuntimeError) as captured:
                next(records)

        self.assertIs(captured.exception, error)


if __name__ == "__main__":
    unittest.main()
