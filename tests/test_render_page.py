"""Tests del contrato y cierre de recursos con un renderer simulado."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ingestion.render_page import render_pdf_page


class TestRenderPage(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pdf = self.root / "annual.pdf"
        self.pdf.write_bytes(b"simulated input")
        self.output = self.root / "images/page.png"
        patcher = patch("ingestion.render_page.import_module")
        self.importer = patcher.start()
        self.addCleanup(patcher.stop)
        self.document = self.importer.return_value.PdfDocument.return_value
        self.document.__len__.return_value = 181
        self.page = self.document.__getitem__.return_value
        self.bitmap = self.page.render.return_value
        self.image = self.bitmap.to_pil.return_value

    def test_converts_page_number_to_zero_based_and_closes_all_handles(self) -> None:
        result = render_pdf_page(self.pdf, 5, self.output)
        self.assertEqual(result, self.output)
        self.document.__getitem__.assert_called_once_with(4)
        self.page.render.assert_called_once_with(scale=2.0)
        self.image.save.assert_called_once_with(self.output, format="PNG")
        for resource in (self.image, self.bitmap, self.page, self.document):
            resource.close.assert_called_once()
        self.assertEqual(self.pdf.read_bytes(), b"simulated input")

    def test_invalid_inputs_fail_before_loading_renderer(self) -> None:
        for page in (0, -1, True, 1.5):
            with self.subTest(page=page), self.assertRaises(ValueError):
                render_pdf_page(self.pdf, page, self.output)
        for dpi in (0, 301, True, 72.5):
            with self.subTest(dpi=dpi), self.assertRaises(ValueError):
                render_pdf_page(self.pdf, 5, self.output, dpi)
        with self.assertRaises(ValueError):
            render_pdf_page(self.pdf, 5, self.pdf)
        with self.assertRaises(FileNotFoundError):
            render_pdf_page(self.root / "missing.pdf", 5, self.output)
        self.importer.assert_not_called()

    def test_out_of_range_page_closes_document_without_rendering(self) -> None:
        with self.assertRaises(ValueError):
            render_pdf_page(self.pdf, 182, self.output)
        self.document.close.assert_called_once()
        self.document.__getitem__.assert_not_called()

    def test_save_failure_propagates_and_closes_resources(self) -> None:
        error = OSError("write failed")
        self.image.save.side_effect = error
        with self.assertRaises(OSError) as captured:
            render_pdf_page(self.pdf, 5, self.output)
        self.assertIs(captured.exception, error)
        for resource in (self.image, self.bitmap, self.page, self.document):
            resource.close.assert_called_once()
