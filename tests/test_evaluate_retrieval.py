"""Checks for retrieval metrics, cutoff and dataset validity."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.evaluate_retrieval import evaluate_retrieval, retrieval_metrics


def page(number: int, source: str = "annual.pdf") -> dict:
    return {"source_file": source, "pdf_page": number}


class TestRetrievalMetrics(unittest.TestCase):
    def test_first_and_second_rank(self) -> None:
        for rank in (1, 2):
            with self.subTest(rank=rank):
                retrieved = [page(8)] * (rank - 1) + [page(18)]
                metrics = retrieval_metrics(retrieved, [page(18)])
                self.assertEqual(metrics["hit_at_k"], 1)
                self.assertEqual(metrics["reciprocal_rank_at_k"], 1 / rank)
                self.assertEqual(metrics["first_relevant_rank"], rank)

    def test_missing_empty_and_beyond_cutoff(self) -> None:
        for retrieved in ([], [page(1)], [page(1), page(2), page(3), page(18)]):
            with self.subTest(retrieved=retrieved):
                self.assertEqual(retrieval_metrics(retrieved, [page(18)], 3), {
                    "hit_at_k": 0,
                    "reciprocal_rank_at_k": 0.0,
                    "first_relevant_rank": None,
                })

    def test_first_match_with_multiple_references(self) -> None:
        metrics = retrieval_metrics([page(2), page(18)], [page(18), page(2)])
        self.assertEqual(metrics["first_relevant_rank"], 1)

    def test_identity_includes_document(self) -> None:
        metrics = retrieval_metrics([page(18, "other.pdf")], [page(18)])
        self.assertEqual(metrics["hit_at_k"], 0)

    def test_invalid_cutoff_or_empty_labels(self) -> None:
        for top_k in (0, -1, True, 1.5):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                retrieval_metrics([], [page(18)], top_k)
        with self.assertRaises(ValueError):
            retrieval_metrics([], [])


class TestEvaluateRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cases_path = Path(self.directory.name) / "cases.json"
        self.index_path = Path(self.directory.name) / "index.jsonl"
        self.index_path.write_text(
            json.dumps({**page(1), "text": "alpha"}) + "\n"
            + json.dumps({**page(2), "text": "alpha gamma"}) + "\n",
            encoding="utf-8",
        )

    def write_cases(self, cases) -> None:
        self.cases_path.write_text(json.dumps(cases), encoding="utf-8")

    def test_end_to_end_means_and_no_document_text_in_report(self) -> None:
        self.write_cases([
            {"id": "hit", "query": "alpha", "relevant_pages": [page(2)]},
            {"id": "miss", "query": "delta", "relevant_pages": [page(2)]},
        ])
        report = evaluate_retrieval(self.cases_path, self.index_path)
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["mean_hit_at_k"], 0.5)
        self.assertEqual(report["mrr_at_k"], 0.25)
        self.assertEqual(report["cases"][0]["first_relevant_rank"], 2)
        self.assertNotIn("text", report["cases"][0]["retrieved_pages"][0])

    def test_absent_reference_is_configuration_error(self) -> None:
        self.write_cases([
            {"id": "bad_label", "query": "alpha", "relevant_pages": [page(99)]}
        ])
        with self.assertRaisesRegex(ValueError, "ausente"):
            evaluate_retrieval(self.cases_path, self.index_path)

    def test_duplicate_ids_and_empty_dataset_are_rejected(self) -> None:
        case = {"id": "duplicate", "query": "alpha", "relevant_pages": [page(1)]}
        for cases in ([], [case, case]):
            with self.subTest(cases=cases):
                self.write_cases(cases)
                with self.assertRaises(ValueError):
                    evaluate_retrieval(self.cases_path, self.index_path)


if __name__ == "__main__":
    unittest.main()
