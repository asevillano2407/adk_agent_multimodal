"""Offline checks for JSONL page exports."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ingestion.export_pdf import export_pdf


class TestExportPdf(unittest.TestCase):
    """Preserve page content and surface extraction failures."""

    def test_round_trip_preserves_unicode_newlines_and_empty_page(self) -> None:
        records = [
            {
                "source_file": "informe.pdf",
                "pdf_page": 1,
                "text": "España: ingresos\nSegunda línea con \"comillas\".",
            },
            {"source_file": "informe.pdf", "pdf_page": 2, "text": ""},
            {"source_file": "informe.pdf", "pdf_page": 3, "text": "Última página"},
        ]

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "nested/pages.jsonl"
            with patch(
                "ingestion.export_pdf.extract_pages", return_value=iter(records)
            ):
                count = export_pdf(Path("informe.pdf"), output_path)

            contents = output_path.read_text(encoding="utf-8")
            lines = contents.splitlines()
            self.assertEqual(count, 3)
            self.assertEqual(len(lines), 3)
            self.assertEqual([json.loads(line) for line in lines], records)
            self.assertIn("España", contents)
            self.assertTrue(contents.endswith("\n"))
            self.assertEqual(list(output_path.parent.iterdir()), [output_path])

    def test_empty_iterator_exports_zero_records(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "empty.jsonl"
            with patch("ingestion.export_pdf.extract_pages", return_value=iter(())):
                count = export_pdf(Path("empty.pdf"), output_path)

            self.assertEqual(count, 0)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "")

    def test_extraction_failure_is_propagated(self) -> None:
        error = RuntimeError("Extraction interrupted")

        def interrupted_pages(pdf_path):
            yield {"source_file": pdf_path.name, "pdf_page": 1, "text": "First"}
            raise error

        for destination_exists in (False, True):
            with self.subTest(destination_exists=destination_exists):
                with TemporaryDirectory() as temporary_directory:
                    output_path = Path(temporary_directory) / "pages.jsonl"
                    original = b'{"previous": "complete export"}\n'
                    if destination_exists:
                        output_path.write_bytes(original)
                    with patch(
                        "ingestion.export_pdf.extract_pages",
                        side_effect=interrupted_pages,
                    ):
                        with self.assertRaises(RuntimeError) as captured:
                            export_pdf(Path("interrupted.pdf"), output_path)

                    self.assertIs(captured.exception, error)
                    if destination_exists:
                        self.assertEqual(output_path.read_bytes(), original)
                        self.assertEqual(list(output_path.parent.iterdir()), [output_path])
                    else:
                        self.assertFalse(output_path.exists())
                        self.assertEqual(list(output_path.parent.iterdir()), [])

    def test_success_replaces_existing_destination(self) -> None:
        record = {"source_file": "annual.pdf", "pdf_page": 1, "text": "New export"}
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "pages.jsonl"
            output_path.write_text("Previous export\n", encoding="utf-8")
            with patch(
                "ingestion.export_pdf.extract_pages", return_value=iter([record])
            ):
                count = export_pdf(Path("annual.pdf"), output_path)

            self.assertEqual(count, 1)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), record)
            self.assertEqual(list(output_path.parent.iterdir()), [output_path])

    def test_replace_failure_preserves_destination_and_cleans_temporary(self) -> None:
        record = {"source_file": "annual.pdf", "pdf_page": 1, "text": "New export"}
        error = PermissionError("Destination locked")
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "pages.jsonl"
            original = b'{"previous": "complete export"}\n'
            output_path.write_bytes(original)
            with (
                patch("ingestion.export_pdf.extract_pages", return_value=iter([record])),
                patch("ingestion.export_pdf.Path.replace", side_effect=error),
            ):
                with self.assertRaises(PermissionError) as captured:
                    export_pdf(Path("annual.pdf"), output_path)

            self.assertIs(captured.exception, error)
            self.assertEqual(output_path.read_bytes(), original)
            self.assertEqual(list(output_path.parent.iterdir()), [output_path])

    def test_temporary_creation_failure_preserves_destination(self) -> None:
        error = OSError("Cannot create temporary file")
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "pages.jsonl"
            original = b'{"previous": "complete export"}\n'
            output_path.write_bytes(original)
            with patch("ingestion.export_pdf.NamedTemporaryFile", side_effect=error):
                with self.assertRaises(OSError) as captured:
                    export_pdf(Path("annual.pdf"), output_path)

            self.assertIs(captured.exception, error)
            self.assertEqual(output_path.read_bytes(), original)
            self.assertEqual(list(output_path.parent.iterdir()), [output_path])


if __name__ == "__main__":
    unittest.main()
