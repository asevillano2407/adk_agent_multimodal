"""Validación de identidad, rutas y registros de los manifiestos del catálogo."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from retrieval.report_catalog import catalog_entries, load_report_catalog


class TestReportCatalog(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "annual.144dpi.manifest.json"
        self.manifest = {
            "source_file": "annual.pdf", "source_sha256": "a" * 64,
            "dpi": 144, "page_count": 2, "render_schema_version": 1,
            "pages": {
                "1": {"status": "ready", "file": "annual.p1.144dpi.png",
                      "bytes": 100, "image_sha256": "b" * 64, "width": 2, "height": 2},
                "2": {"status": "error", "error_type": "PdfiumError"},
            },
        }

    def write(self, manifest) -> None:
        self.path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_partial_manifest_lists_prepared_count_without_inventing_year(self) -> None:
        self.write(self.manifest)
        entries = catalog_entries(load_report_catalog(self.root))
        self.assertEqual(entries, [{
            "source_file": "annual.pdf", "document_id": "sha256:" + "a" * 64,
            "page_count": 2, "ready_pages_in_manifest": 1, "dpi": 144,
        }])

    def test_missing_manifest_is_not_an_empty_success(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_report_catalog(self.root)

    def test_invalid_headers_are_rejected(self) -> None:
        for field, value in (
            ("source_file", "../annual.pdf"), ("source_file", "C:\\annual.pdf"),
            ("source_sha256", "bad"), ("page_count", True), ("page_count", 0),
            ("dpi", 72), ("render_schema_version", True), ("pages", []),
        ):
            with self.subTest(field=field, value=value):
                self.write({**self.manifest, field: value})
                with self.assertRaises(ValueError):
                    load_report_catalog(self.root)

    def test_invalid_page_identity_and_image_paths_are_rejected(self) -> None:
        for number in ("0", "3", "01", "../1"):
            with self.subTest(number=number):
                changed = deepcopy(self.manifest)
                changed["pages"] = {number: self.manifest["pages"]["1"]}
                self.write(changed)
                with self.assertRaises(ValueError):
                    load_report_catalog(self.root)
        for field, value in (
            ("file", "../secret.png"), ("file", "other.p1.144dpi.png"),
            ("image_sha256", "bad"), ("bytes", True), ("width", 0),
        ):
            with self.subTest(field=field):
                changed = deepcopy(self.manifest)
                changed["pages"]["1"][field] = value
                self.write(changed)
                with self.assertRaises(ValueError):
                    load_report_catalog(self.root)

    def test_manifest_filename_must_match_its_document(self) -> None:
        self.write({**self.manifest, "source_file": "other.pdf"})
        with self.assertRaisesRegex(ValueError, "nombre"):
            load_report_catalog(self.root)
