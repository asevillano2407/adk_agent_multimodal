"""Tests for chunk export without modifying the page baseline."""

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ingestion.export_chunks import export_chunks, main


class TestExportChunks(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.pages_path = Path(self.directory.name) / "pages.jsonl"
        self.output_path = Path(self.directory.name) / "chunks.jsonl"

    def test_exports_chunks_preserving_offsets_and_page_baseline(self) -> None:
        pages = [
            {"source_file": "annual.pdf", "pdf_page": 1, "text": "España"},
            {"source_file": "annual.pdf", "pdf_page": 2, "text": ""},
            {"source_file": "annual.pdf", "pdf_page": 3, "text": "abcdef"},
        ]
        self.pages_path.write_text(
            "".join(json.dumps(page, ensure_ascii=False) + "\n" for page in pages),
            encoding="utf-8",
        )
        original = self.pages_path.read_bytes()
        count = export_chunks(self.pages_path, self.output_path, 4, 1)
        chunks = [
            json.loads(line)
            for line in self.output_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(count, 4)
        self.assertEqual([chunk["pdf_page"] for chunk in chunks], [1, 1, 3, 3])
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), count)
        for chunk in chunks:
            self.assertEqual(
                chunk["text"],
                pages[chunk["pdf_page"] - 1]["text"][
                    chunk["start_char"]:chunk["end_char"]
                ],
            )
        self.assertEqual(self.pages_path.read_bytes(), original)
        self.assertEqual(len(list(self.pages_path.parent.iterdir())), 2)

    def test_same_input_and_output_are_rejected_without_changes(self) -> None:
        self.pages_path.write_text("Original index", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "distinto"):
            export_chunks(self.pages_path, self.pages_path)
        self.assertEqual(self.pages_path.read_text(encoding="utf-8"), "Original index")

    def test_malformed_later_record_keeps_previous_destination(self) -> None:
        self.pages_path.write_text(
            json.dumps({"source_file": "a.pdf", "pdf_page": 1, "text": "Readable"})
            + "\n{invalid}\n", encoding="utf-8",
        )
        self.output_path.write_bytes(b"Previous complete export\n")
        with self.assertRaisesRegex(ValueError, "línea 2"):
            export_chunks(self.pages_path, self.output_path)
        self.assertEqual(self.output_path.read_bytes(), b"Previous complete export\n")
        self.assertEqual(len(list(self.pages_path.parent.iterdir())), 2)

    def test_invalid_page_schema_is_rejected_without_partial_export(self) -> None:
        self.pages_path.write_text('{"text": null}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            export_chunks(self.pages_path, self.output_path)
        self.assertFalse(self.output_path.exists())
        self.assertEqual(list(self.pages_path.parent.iterdir()), [self.pages_path])

    def test_empty_index_and_blank_lines_produce_zero_chunks(self) -> None:
        self.pages_path.write_text("\n\n", encoding="utf-8")
        self.assertEqual(export_chunks(self.pages_path, self.output_path), 0)
        self.assertEqual(self.output_path.read_text(encoding="utf-8"), "")

    def test_invalid_parameters_are_rejected_even_for_empty_index(self) -> None:
        self.pages_path.write_text("", encoding="utf-8")
        for size, overlap in ((0, 0), (True, 0), (4, 4), (4, -1), (4, None)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                export_chunks(self.pages_path, self.output_path, size, overlap)
        self.assertFalse(self.output_path.exists())

    def test_missing_index_is_propagated_and_temporary_is_cleaned(self) -> None:
        with self.assertRaises(FileNotFoundError):
            export_chunks(self.pages_path, self.output_path)
        self.assertEqual(list(self.pages_path.parent.iterdir()), [])

    def test_exports_natural_strategy_without_changing_source(self) -> None:
        record = {"source_file": "annual.pdf", "pdf_page": 18,
                  "text": "x" * 80 + "\n\n" + "x" * 98}
        self.pages_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        original = self.pages_path.read_bytes()
        count = export_chunks(self.pages_path, self.output_path, 100, 20, strategy="natural")
        chunks = [json.loads(line) for line in self.output_path.read_text().splitlines()]
        self.assertEqual(count, 3)
        self.assertEqual(chunks[0]["end_char"], 82)
        self.assertIn(":natural:", chunks[0]["chunk_id"])
        self.assertEqual(self.pages_path.read_bytes(), original)

    def test_invalid_strategy_is_rejected_for_empty_index(self) -> None:
        self.pages_path.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            export_chunks(self.pages_path, self.output_path, strategy="unknown")
        self.assertFalse(self.output_path.exists())

    def test_cli_natural_strategy_uses_distinct_default_filename(self) -> None:
        with (
            patch("sys.argv", ["export_chunks", "--pages", str(self.pages_path),
                               "--strategy", "natural"]),
            patch("ingestion.export_chunks.export_chunks", return_value=3) as export,
            redirect_stdout(StringIO()),
        ):
            main()
        export.assert_called_once_with(
            self.pages_path,
            self.pages_path.with_name("pages.jsonl.chunks.natural.c1200.o200.jsonl"),
            1200, 200, strategy="natural",
        )


if __name__ == "__main__":
    unittest.main()
