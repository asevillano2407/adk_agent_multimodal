"""Ingesta textual con PDFs sintéticos y versiones de contenido verificadas."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from ingestion.extract_pdf import extract_pages
from ingestion.file_utils import file_sha256
from ingestion.prepare_reports import prepare_reports
from importlib.metadata import PackageNotFoundError


def write_pdf(path: Path, texts: list[str]) -> None:
    with PdfWriter() as writer:
        for text in texts:
            page = writer.add_blank_page(width=200, height=200)
            if text:
                font = DictionaryObject({
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                })
                page[NameObject("/Resources")] = DictionaryObject({
                    NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
                })
                stream = DecodedStreamObject()
                stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode("ascii"))
                page[NameObject("/Contents")] = writer._add_object(stream)
        writer.write(path)


class TestPrepareReports(unittest.TestCase):
    def test_fonttools_install_changes_snapshot_and_forces_new_extraction(self) -> None:
        with patch("ingestion.prepare_reports._fonttools_version", return_value="not-installed"):
            previous = self.prepare()
        old_doc = previous["documents"][0]
        old_bytes = (self.output / old_doc["chunks_file"]).read_bytes()
        with patch("ingestion.prepare_reports._fonttools_version", return_value="4.68.0"):
            simulated = self.prepare(dry_run=True)
            self.assertEqual(simulated["planned"], 2)
            self.assertEqual(simulated["skipped"], 0)
            updated = self.prepare()
        self.assertEqual(updated["prepared"], 2)
        self.assertEqual(updated["fonttools_version"], "4.68.0")
        self.assertNotEqual(old_doc["chunks_file"], updated["documents"][0]["chunks_file"])
        self.assertEqual((self.output / old_doc["chunks_file"]).read_bytes(), old_bytes)

    def test_missing_optional_fonttools_is_recorded_explicitly(self) -> None:
        def get_version(name):
            if name == "fonttools":
                raise PackageNotFoundError(name)
            return "6.21.4"

        with patch("ingestion.prepare_reports.version", side_effect=get_version):
            report = self.prepare(dry_run=True)
        self.assertEqual(report["fonttools_version"], "not-installed")

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "raw"
        self.images = self.root / "images"
        self.output = self.root / "corpus"
        self.inputs.mkdir()
        self.images.mkdir()
        self.first = self.inputs / "a.pdf"
        self.second = self.inputs / "b.pdf"
        write_pdf(self.first, ["Revenue 123 " * 220, ""])
        write_pdf(self.second, ["Other report revenue 456"])
        self.write_visual_manifest(self.first, 2)
        self.write_visual_manifest(self.second, 1)

    def write_visual_manifest(self, source: Path, pages: int) -> None:
        manifest = {
            "source_file": source.name, "source_sha256": file_sha256(source),
            "page_count": pages, "dpi": 144, "render_schema_version": 1, "pages": {},
        }
        (self.images / f"{source.stem}.144dpi.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def prepare(self, **kwargs) -> dict:
        return prepare_reports(self.inputs, self.images, self.output, **kwargs)

    def records(self, relative_path: str) -> list[dict]:
        return [json.loads(line) for line in (self.output / relative_path).read_text(encoding="utf-8").splitlines()]

    def test_dry_run_plans_without_extraction_or_writes(self) -> None:
        with patch("ingestion.prepare_reports.extract_pages") as extractor:
            report = self.prepare(dry_run=True)
        self.assertEqual(report["planned"], 2)
        self.assertEqual(report["page_count"], 3)
        self.assertIsNone(report["chunk_count"])
        self.assertFalse(report["complete"])
        self.assertFalse(self.output.exists())
        extractor.assert_not_called()

    def test_real_text_identity_offsets_empty_pages_and_baseline_preserved(self) -> None:
        baseline = self.root / "nvidia_2025.pages.jsonl"
        baseline.write_bytes(b"original baseline")
        report = self.prepare()
        self.assertTrue(report["complete"])
        self.assertEqual(report["prepared"], 2)
        self.assertEqual(report["page_count"], 3)
        ids = set()
        for document in report["documents"]:
            pages = self.records(document["pages_file"])
            chunks = self.records(document["chunks_file"])
            self.assertEqual(len(pages), document["page_count"])
            self.assertEqual(len(chunks), document["chunk_count"])
            for chunk in chunks:
                self.assertEqual(chunk["source_file"], document["source_file"])
                self.assertEqual(chunk["document_id"], document["document_id"])
                self.assertEqual(chunk["source_sha256"], document["source_sha256"])
                page = pages[chunk["pdf_page"] - 1]
                self.assertEqual(chunk["text"], page["text"][chunk["start_char"]:chunk["end_char"]])
                self.assertNotIn(chunk["chunk_id"], ids)
                ids.add(chunk["chunk_id"])
        first = report["documents"][0]
        self.assertEqual(first["empty_page_count"], 1)
        self.assertEqual(self.records(first["pages_file"])[1]["text"], "")
        self.assertEqual(baseline.read_bytes(), b"original baseline")
        self.assertTrue(Path(report["manifest_file"]).exists())

    def test_resume_skips_only_unchanged_verified_outputs(self) -> None:
        initial = self.prepare()
        with patch("ingestion.prepare_reports.extract_pages") as extractor:
            second = self.prepare()
        self.assertEqual(second["skipped"], 2)
        self.assertEqual(second["prepared"], 0)
        extractor.assert_not_called()
        bad = self.output / initial["documents"][0]["chunks_file"]
        bad.write_bytes(b"corrupted")
        third = self.prepare()
        self.assertEqual(third["prepared"], 1)
        self.assertEqual(third["skipped"], 1)

    def test_pdf_change_requires_updated_visual_manifest_and_keeps_old_snapshot(self) -> None:
        initial = self.prepare()
        old = initial["documents"][0]
        old_bytes = (self.output / old["chunks_file"]).read_bytes()
        write_pdf(self.first, ["New revenue 789", ""])
        with self.assertRaisesRegex(ValueError, "manifiesto visual"):
            self.prepare()
        self.write_visual_manifest(self.first, 2)
        updated = self.prepare()
        new = updated["documents"][0]
        self.assertNotEqual(new["document_id"], old["document_id"])
        self.assertNotEqual(new["chunks_file"], old["chunks_file"])
        self.assertEqual((self.output / old["chunks_file"]).read_bytes(), old_bytes)

    def test_chunk_parameters_create_distinct_files_and_ids(self) -> None:
        first = self.prepare()
        second = self.prepare(chunk_size=600, overlap=100)
        old_doc, new_doc = first["documents"][0], second["documents"][0]
        self.assertNotEqual(old_doc["chunks_file"], new_doc["chunks_file"])
        self.assertNotEqual(self.records(old_doc["chunks_file"])[0]["chunk_id"],
                            self.records(new_doc["chunks_file"])[0]["chunk_id"])

    def test_partial_document_failure_resumes_previously_completed_reports(self) -> None:
        def extract(path):
            if path.name == "b.pdf":
                raise RuntimeError("simulated extraction failure")
            yield from extract_pages(path)

        with patch("ingestion.prepare_reports.extract_pages", side_effect=extract):
            with self.assertRaises(RuntimeError):
                self.prepare()
        self.assertFalse((self.output / "corpus.fixed.c1200.o200.manifest.json").exists())
        with patch("ingestion.prepare_reports.extract_pages", wraps=extract_pages) as extractor:
            report = self.prepare()
        self.assertEqual(report["skipped"], 1)
        self.assertEqual(report["prepared"], 1)
        self.assertEqual(extractor.call_count, 1)

    def test_page_count_mismatch_does_not_publish_incomplete_pages(self) -> None:
        self.write_visual_manifest(self.first, 3)
        with self.assertRaisesRegex(ValueError, "número de páginas"):
            self.prepare()
        self.assertEqual(list(self.output.rglob("pages.jsonl")), [])
        self.assertEqual(list(self.output.rglob("*.tmp")), [])

    def test_missing_source_and_invalid_configuration_are_explicit(self) -> None:
        for options in ({"chunk_size": 0}, {"overlap": 1200}, {"strategy": "unknown"}, {"dry_run": "yes"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.prepare(**options)
        self.second.unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.output.exists())
