"""Renderización real de PDFs sintéticos pequeños, sin usar el corpus del usuario."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
from pypdf import PdfWriter
from pypdfium2 import PdfDocument, PdfiumError

from ingestion.render_reports import _render_atomic, render_reports


class TestRenderReports(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.inputs = self.root / "pdfs"
        self.inputs.mkdir()
        self.outputs = self.root / "images"
        self.pdf = self.inputs / "annual.pdf"
        self.write_pdf(self.pdf, 3)

    @staticmethod
    def write_pdf(path: Path, pages: int) -> None:
        with PdfWriter() as writer:
            for _ in range(pages):
                writer.add_blank_page(width=72, height=72)
            writer.write(path)

    def run_batch(self, **kwargs) -> dict:
        return render_reports(self.inputs, self.outputs, **kwargs)

    def manifest(self, dpi=144) -> dict:
        return json.loads((self.outputs / f"annual.{dpi}dpi.manifest.json").read_text(encoding="utf-8"))

    def test_dry_run_counts_pages_without_writes_or_rendering(self) -> None:
        self.write_pdf(self.inputs / "second.pdf", 2)
        original = self.pdf.read_bytes()
        with patch("ingestion.render_reports.render_document_page") as render:
            result = self.run_batch(dry_run=True)
        self.assertEqual(result["documents_found"], 2)
        self.assertEqual(result["planned"], 5)
        self.assertEqual(result["rendered"], 0)
        self.assertFalse(result["complete"])
        self.assertFalse(self.outputs.exists())
        self.assertEqual(self.pdf.read_bytes(), original)
        render.assert_not_called()

    def test_limit_resume_and_identical_second_run(self) -> None:
        with patch("ingestion.render_reports.PdfDocument", wraps=PdfDocument) as opener:
            first = self.run_batch(max_new_pages=2)
        self.assertEqual(opener.call_count, 1)
        self.assertEqual(first["rendered"], 2)
        self.assertTrue(first["limited"])
        self.assertEqual(set(self.manifest()["pages"]), {"1", "2"})
        second = self.run_batch()
        self.assertEqual(second["skipped"], 2)
        self.assertEqual(second["rendered"], 1)
        self.assertTrue(second["complete"])
        with Image.open(self.outputs / "annual.p3.144dpi.png") as image:
            self.assertEqual(image.size, (144, 144))
        with patch("ingestion.render_reports.render_document_page") as render:
            third = self.run_batch()
        self.assertEqual(third["skipped"], 3)
        self.assertEqual(third["planned"], 0)
        self.assertTrue(third["complete"])
        render.assert_not_called()

    def test_corrupted_or_missing_png_is_regenerated(self) -> None:
        self.run_batch()
        (self.outputs / "annual.p1.144dpi.png").write_bytes(b"broken")
        (self.outputs / "annual.p2.144dpi.png").unlink()
        result = self.run_batch()
        self.assertEqual(result["rendered"], 2)
        self.assertEqual(result["skipped"], 1)
        self.assertTrue(result["complete"])

    def test_untracked_png_is_not_assumed_to_match_source(self) -> None:
        self.outputs.mkdir()
        with Image.new("RGB", (2, 2), "red") as image:
            image.save(self.outputs / "annual.p1.144dpi.png")
        result = self.run_batch()
        self.assertEqual(result["skipped"], 0)
        self.assertEqual(result["rendered"], 3)

    def test_source_change_invalidates_manifest_without_deleting_orphans(self) -> None:
        self.run_batch()
        previous_hash = self.manifest()["source_sha256"]
        self.write_pdf(self.pdf, 2)
        result = self.run_batch()
        self.assertEqual(result["rendered"], 2)
        self.assertEqual(result["skipped"], 0)
        self.assertNotEqual(previous_hash, self.manifest()["source_sha256"])
        self.assertEqual(set(self.manifest()["pages"]), {"1", "2"})
        self.assertTrue((self.outputs / "annual.p3.144dpi.png").exists())

    def test_resolution_and_engine_versions_are_part_of_resume_contract(self) -> None:
        self.run_batch()
        changed_dpi = self.run_batch(dpi=72)
        self.assertEqual(changed_dpi["rendered"], 3)
        self.assertTrue((self.outputs / "annual.p1.144dpi.png").exists())
        with patch("ingestion.render_reports.version", return_value="changed"):
            changed_engine = self.run_batch(dpi=72)
        self.assertEqual(changed_engine["rendered"], 3)
        self.assertEqual(changed_engine["skipped"], 0)

    def test_page_failure_is_recorded_and_other_pages_continue_then_resume(self) -> None:
        def render(document, page, output, dpi):
            if page == 2:
                raise PdfiumError("simulated page failure")
            return _render_atomic(document, page, output, dpi)

        with (
            patch("ingestion.render_reports._render_atomic", side_effect=render),
            self.assertLogs("ingestion.render_reports", level="ERROR"),
        ):
            first = self.run_batch()
        self.assertEqual(first["rendered"], 2)
        self.assertEqual(first["errors"], 1)
        self.assertFalse(first["complete"])
        self.assertEqual(self.manifest()["pages"]["2"]["status"], "error")
        second = self.run_batch()
        self.assertEqual(second["skipped"], 2)
        self.assertEqual(second["rendered"], 1)
        self.assertTrue(second["complete"])

    def test_interruption_keeps_previous_checkpoint(self) -> None:
        def render(document, page, output, dpi):
            if page == 2:
                raise KeyboardInterrupt()
            return _render_atomic(document, page, output, dpi)

        with patch("ingestion.render_reports._render_atomic", side_effect=render):
            with self.assertRaises(KeyboardInterrupt):
                self.run_batch()
        self.assertEqual(set(self.manifest()["pages"]), {"1"})
        result = self.run_batch()
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["rendered"], 2)

    def test_partial_render_never_replaces_previous_png(self) -> None:
        self.outputs.mkdir()
        output = self.outputs / "previous.png"
        output.write_bytes(b"previous complete file")

        def fail(document, page, temporary, dpi):
            temporary.write_bytes(b"partial")
            raise OSError("simulated disk failure")

        with patch("ingestion.render_reports.render_document_page", side_effect=fail):
            with self.assertRaises(OSError):
                _render_atomic(None, 1, output, 144)
        self.assertEqual(output.read_bytes(), b"previous complete file")
        self.assertEqual(list(self.outputs.iterdir()), [output])

    def test_bad_document_is_reported_and_other_documents_continue(self) -> None:
        (self.inputs / "broken.pdf").write_bytes(b"not a PDF")
        with self.assertLogs("ingestion.render_reports", level="ERROR"):
            result = self.run_batch()
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["rendered"], 3)
        self.assertFalse(result["complete"])

    def test_invalid_options_missing_input_and_empty_selection(self) -> None:
        for options in ({"dpi": True}, {"dpi": 301}, {"max_new_pages": 0}, {"max_new_pages": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_batch(**options)
        with self.assertRaises(NotADirectoryError):
            render_reports(self.root / "missing", self.outputs)
        with self.assertRaises(ValueError):
            self.run_batch(pattern="no-match*.pdf")

    def test_pattern_selects_only_requested_reports(self) -> None:
        self.write_pdf(self.inputs / "other.pdf", 2)
        result = self.run_batch(pattern="annual*.pdf", dry_run=True)
        self.assertEqual(result["documents_found"], 1)
        self.assertEqual(result["planned"], 3)
