"""Offline checks for lexical page retrieval."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from retrieval.search_pages import search_pages


class TestSearchPages(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.index_path = Path(self.directory.name) / "pages.jsonl"

    def write_index(self, records: list[dict]) -> None:
        self.index_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_ranks_unique_terms_and_preserves_source(self) -> None:
        self.write_index([
            {"source_file": "annual.pdf", "pdf_page": 1, "text": "Revenue revenue"},
            {
                "source_file": "annual.pdf",
                "pdf_page": 18,
                "text": "Compute & Networking revenue",
            },
            {"source_file": "annual.pdf", "pdf_page": 19, "text": "Compute"},
            {"source_file": "annual.pdf", "pdf_page": 20, "text": ""},
        ])
        results = search_pages("COMPUTE, Networking revenue revenue!", self.index_path, 2)
        self.assertEqual([page["pdf_page"] for page in results], [18, 1])
        self.assertEqual([page["score"] for page in results], [3, 1])
        self.assertEqual(results[0]["source_file"], "annual.pdf")
        self.assertEqual(results[0]["text"], "Compute & Networking revenue")

    def test_ties_are_ordered_by_source_then_numeric_page(self) -> None:
        self.write_index([
            {"source_file": "b.pdf", "pdf_page": 1, "text": "Revenue"},
            {"source_file": "a.pdf", "pdf_page": 10, "text": "Revenue"},
            {"source_file": "a.pdf", "pdf_page": 2, "text": "Revenue"},
        ])
        results = search_pages("revenue", self.index_path)
        self.assertEqual(
            [(page["source_file"], page["pdf_page"]) for page in results],
            [("a.pdf", 2), ("a.pdf", 10), ("b.pdf", 1)],
        )

    def test_no_matches_and_empty_index_return_empty_list(self) -> None:
        self.write_index([])
        self.assertEqual(search_pages("revenue", self.index_path), [])
        self.write_index([
            {"source_file": "annual.pdf", "pdf_page": 1, "text": "Robotics"}
        ])
        self.assertEqual(search_pages("revenue", self.index_path), [])

    def test_unicode_words_and_blank_lines(self) -> None:
        self.write_index([
            {"source_file": "informe.pdf", "pdf_page": 1, "text": "ESPAÑA: análisis"}
        ])
        with self.index_path.open("a", encoding="utf-8") as output:
            output.write("\n")
        self.assertEqual(search_pages("España análisis", self.index_path)[0]["score"], 2)

    def test_does_not_translate_query(self) -> None:
        self.write_index([
            {"source_file": "annual.pdf", "pdf_page": 1, "text": "Revenue"}
        ])
        self.assertEqual(search_pages("ingresos", self.index_path), [])

    def test_invalid_queries_are_rejected(self) -> None:
        for query in ("", "  ", "?!"):
            with self.subTest(query=query), self.assertRaises(ValueError):
                search_pages(query, self.index_path)
        with self.assertRaises(TypeError):
            search_pages(None, self.index_path)

    def test_invalid_top_k_is_rejected(self) -> None:
        for top_k in (0, -1, True, 1.5, None):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                search_pages("revenue", self.index_path, top_k)

    def test_missing_index_is_not_hidden(self) -> None:
        with self.assertRaises(FileNotFoundError):
            search_pages("revenue", self.index_path)

    def test_malformed_json_includes_line_number(self) -> None:
        self.index_path.write_text("\n{broken json}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "línea 2"):
            search_pages("revenue", self.index_path)

    def test_invalid_record_schema_is_rejected(self) -> None:
        for record in (
            [],
            {},
            {"source_file": "a.pdf", "pdf_page": 0, "text": "Revenue"},
            {"source_file": "a.pdf", "pdf_page": True, "text": "Revenue"},
            {"source_file": "a.pdf", "pdf_page": 1, "text": None},
            {"source_file": "", "pdf_page": 1, "text": "Revenue"},
        ):
            with self.subTest(record=record):
                self.write_index([record])
                with self.assertRaisesRegex(ValueError, "línea 1"):
                    search_pages("revenue", self.index_path)


if __name__ == "__main__":
    unittest.main()
