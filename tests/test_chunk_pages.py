"""Tests for character windows, coverage and source traceability."""

import unittest

from ingestion.chunk_pages import choose_chunk_end, chunk_page


def page(text: str) -> dict:
    return {"source_file": "annual.pdf", "pdf_page": 18, "text": text}


class TestChunkPage(unittest.TestCase):
    def test_versioned_identity_is_preserved_and_changes_ids(self) -> None:
        record = {**page("abcdefgh"), "source_sha256": "a" * 64,
                  "document_id": "sha256:" + "a" * 64}
        chunk = next(chunk_page(record, 5, 1))
        self.assertEqual(chunk["document_id"], record["document_id"])
        self.assertEqual(chunk["source_sha256"], record["source_sha256"])
        self.assertEqual(chunk["text"], "abcde")
        self.assertEqual(chunk["chunk_strategy"], "fixed")
        self.assertEqual(chunk["chunk_size"], 5)
        self.assertEqual(chunk["chunk_overlap"], 1)
        changed = {**record, "source_sha256": "b" * 64, "document_id": "sha256:" + "b" * 64}
        self.assertNotEqual(chunk["chunk_id"], next(chunk_page(changed, 5, 1))["chunk_id"])
        self.assertNotEqual(chunk["chunk_id"], next(chunk_page(record, 6, 1))["chunk_id"])
        self.assertNotEqual(chunk["chunk_id"], next(chunk_page(record, 5, 1, "natural"))["chunk_id"])

    def test_incomplete_or_conflicting_version_identity_is_rejected(self) -> None:
        for fields in (
            {"document_id": "sha256:" + "a" * 64},
            {"source_sha256": "a" * 64},
            {"source_sha256": "a" * 64, "document_id": "sha256:" + "b" * 64},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                list(chunk_page({**page(""), **fields}))

    def test_empty_text_has_no_chunks(self) -> None:
        self.assertEqual(list(chunk_page(page(""))), [])

    def test_short_and_exact_window_text_produce_one_chunk(self) -> None:
        for length in (1, 1199, 1200):
            with self.subTest(length=length):
                chunks = list(chunk_page(page("a" * length)))
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0]["start_char"], 0)
                self.assertEqual(chunks[0]["end_char"], length)
                self.assertEqual(chunks[0]["text"], "a" * length)

    def test_positions_overlap_and_full_text_coverage(self) -> None:
        text = ("España\nInformación 🐍 " * 150)[:2500]
        chunks = list(chunk_page(page(text)))
        self.assertEqual(
            [(chunk["start_char"], chunk["end_char"]) for chunk in chunks],
            [(0, 1200), (1000, 2200), (2000, 2500)],
        )
        for chunk in chunks:
            self.assertEqual(
                chunk["text"], text[chunk["start_char"]:chunk["end_char"]]
            )
            self.assertLessEqual(len(chunk["text"]), 1200)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous["end_char"] - current["start_char"], 200)
            self.assertEqual(previous["text"][-200:], current["text"][:200])
        reconstructed = chunks[0]["text"] + "".join(
            chunk["text"][200:] for chunk in chunks[1:]
        )
        self.assertEqual(reconstructed, text)

    def test_exact_last_window_does_not_repeat_tail(self) -> None:
        chunks = list(chunk_page(page("a" * 2200)))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[-1]["end_char"], 2200)

    def test_zero_overlap_and_final_partial_chunk(self) -> None:
        chunks = list(chunk_page(page("abcdefgh"), chunk_size=3, overlap=0))
        self.assertEqual([chunk["text"] for chunk in chunks], ["abc", "def", "gh"])
        self.assertEqual("".join(chunk["text"] for chunk in chunks), "abcdefgh")

    def test_maximum_overlap_terminates_without_redundant_tail(self) -> None:
        chunks = list(chunk_page(page("abcdefghijkl"), chunk_size=10, overlap=9))
        self.assertEqual(
            [(chunk["start_char"], chunk["end_char"]) for chunk in chunks],
            [(0, 10), (1, 11), (2, 12)],
        )

    def test_provenance_and_identifiers(self) -> None:
        original = page("abcdefgh")
        chunks = list(chunk_page(original, chunk_size=5, overlap=1))
        self.assertEqual([chunk["chunk_id"] for chunk in chunks], [
            "annual.pdf:p18:start0", "annual.pdf:p18:start4"
        ])
        for chunk in chunks:
            self.assertEqual(chunk["source_file"], "annual.pdf")
            self.assertEqual(chunk["pdf_page"], 18)
        self.assertEqual(original, page("abcdefgh"))
        other_page = {**original, "pdf_page": 19}
        self.assertNotEqual(
            chunks[0]["chunk_id"], next(chunk_page(other_page))["chunk_id"]
        )

    def test_invalid_sizes_and_overlaps(self) -> None:
        for size in (0, -1, True, 1.5, None, "1200"):
            with self.subTest(size=size), self.assertRaises(ValueError):
                list(chunk_page(page("abc"), chunk_size=size))
        for overlap in (-1, 1200, 1201, True, 0.5, None, "200"):
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                list(chunk_page(page("abc"), overlap=overlap))

    def test_missing_or_invalid_page_fields(self) -> None:
        for invalid in (
            {},
            {**page("abc"), "source_file": " "},
            {**page("abc"), "pdf_page": 0},
            {**page("abc"), "pdf_page": True},
            {**page("abc"), "pdf_page": "18"},
            {**page("abc"), "text": None},
        ):
            with self.subTest(page=invalid), self.assertRaises(ValueError):
                list(chunk_page(invalid))
        with self.assertRaises(TypeError):
            list(chunk_page(None))


class TestNaturalChunkPage(unittest.TestCase):
    def test_default_fixed_strategy_is_unchanged(self) -> None:
        record = page("x" * 80 + "\n\n" + "x" * 98)
        default = list(chunk_page(record, 100, 20))
        self.assertEqual(default, list(chunk_page(record, 100, 20, strategy="fixed")))
        self.assertEqual(default[0]["end_char"], 100)
        self.assertEqual(default[0]["chunk_id"], "annual.pdf:p18:start0")

    def test_natural_windows_keep_offsets_overlap_and_coverage(self) -> None:
        text = "x" * 80 + "\n\n" + "x" * 98
        chunks = list(chunk_page(page(text), 100, 20, strategy="natural"))
        self.assertEqual(
            [(chunk["start_char"], chunk["end_char"]) for chunk in chunks],
            [(0, 82), (62, 162), (142, 180)],
        )
        for chunk in chunks:
            self.assertEqual(chunk["text"], text[chunk["start_char"]:chunk["end_char"]])
            self.assertEqual(chunk["pdf_page"], 18)
            self.assertEqual(chunk["source_file"], "annual.pdf")
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous["end_char"] - current["start_char"], 20)
        self.assertEqual(
            chunks[0]["text"] + "".join(chunk["text"][20:] for chunk in chunks[1:]),
            text,
        )

    def test_large_overlap_falls_back_to_fixed_cut_and_terminates(self) -> None:
        text = "x" * 80 + "\n\n" + "x" * 98
        chunks = list(chunk_page(page(text), 100, 90, strategy="natural"))
        self.assertEqual(chunks[0]["end_char"], 100)
        self.assertEqual(len(chunks), 9)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertGreater(current["start_char"], previous["start_char"])
            self.assertEqual(previous["end_char"] - current["start_char"], 90)
        self.assertEqual(chunks[-1]["end_char"], len(text))

    def test_zero_overlap_reconstructs_exact_original(self) -> None:
        text = "x" * 80 + ". " + "x" * 98
        chunks = list(chunk_page(page(text), 100, 0, strategy="natural"))
        self.assertEqual(chunks[0]["end_char"], 81)
        self.assertEqual("".join(chunk["text"] for chunk in chunks), text)

    def test_natural_ids_include_strategy_and_configuration(self) -> None:
        record = page("Sample text")
        fixed = next(chunk_page(record, 100, 20))
        natural = next(chunk_page(record, 100, 20, strategy="natural"))
        alternate = next(chunk_page(record, 100, 10, strategy="natural"))
        self.assertEqual(natural["chunk_id"], "annual.pdf:p18:natural:c100:o20:start0")
        self.assertNotEqual(natural["chunk_id"], fixed["chunk_id"])
        self.assertNotEqual(natural["chunk_id"], alternate["chunk_id"])

    def test_invalid_strategy_and_empty_text(self) -> None:
        self.assertEqual(list(chunk_page(page(""), strategy="natural")), [])
        for strategy in ("semantic", None, True, []):
            with self.subTest(strategy=strategy), self.assertRaises(ValueError):
                list(chunk_page(page(""), strategy=strategy))


class TestChooseChunkEnd(unittest.TestCase):
    def make_text(self, replacements: dict[int, str]) -> str:
        characters = list("x" * 180)
        for position, replacement in replacements.items():
            characters[position:position + len(replacement)] = replacement
        return "".join(characters)

    def test_paragraph_precedes_later_sentence_and_space(self) -> None:
        text = self.make_text({80: "\n\n", 94: ". ", 98: " "})
        self.assertEqual(choose_chunk_end(text, 0, 100), 82)

    def test_last_paragraph_within_window_wins(self) -> None:
        text = self.make_text({70: "\n\n", 90: "\n\n", 120: "\n\n"})
        self.assertEqual(choose_chunk_end(text, 0, 100), 92)

    def test_sentence_precedes_later_whitespace(self) -> None:
        text = self.make_text({70: ". ", 85: "? ", 97: " "})
        self.assertEqual(choose_chunk_end(text, 0, 100), 86)

    def test_early_boundaries_are_ignored(self) -> None:
        text = self.make_text({20: "\n\n", 40: ". ", 80: " "})
        self.assertEqual(choose_chunk_end(text, 0, 100), 81)

    def test_no_boundary_falls_back_to_maximum(self) -> None:
        self.assertEqual(choose_chunk_end("x" * 180, 0, 100), 100)

    def test_offsets_are_absolute_with_nonzero_start(self) -> None:
        text = self.make_text({130: "\n\n", 145: ". "})
        self.assertEqual(choose_chunk_end(text, 50, 100), 132)

    def test_remaining_text_is_kept_even_with_earlier_boundaries(self) -> None:
        text = "Texto\n\nFrase. Final"
        for start in (0, 5, len(text)):
            with self.subTest(start=start):
                self.assertEqual(choose_chunk_end(text, start, 100), len(text))
        self.assertEqual(choose_chunk_end("", 0, 100), 0)

    def test_paragraph_handles_crlf_and_blank_line_spaces(self) -> None:
        for separator in ("\r\n\r\n", "\n \t\n"):
            with self.subTest(separator=separator):
                text = self.make_text({75: separator})
                end = choose_chunk_end(text, 0, 100)
                self.assertEqual(end, 75 + len(separator))
                self.assertTrue(text[:end].endswith(separator))

    def test_decimal_point_is_not_a_sentence_boundary(self) -> None:
        text = self.make_text({70: "116.2", 90: " "})
        self.assertEqual(choose_chunk_end(text, 0, 100), 91)

    def test_boundaries_at_minimum_and_maximum_are_allowed(self) -> None:
        self.assertEqual(choose_chunk_end(self.make_text({58: "\n\n"}), 0, 100), 60)
        self.assertEqual(choose_chunk_end(self.make_text({99: ". "}), 0, 100), 100)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            choose_chunk_end(None, 0, 100)
        for start in (-1, True, 0.5, None, 4):
            with self.subTest(start=start), self.assertRaises(ValueError):
                choose_chunk_end("abc", start, 100)
        for size in (0, -1, True, 1.5, None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                choose_chunk_end("abc", 0, size)


if __name__ == "__main__":
    unittest.main()
