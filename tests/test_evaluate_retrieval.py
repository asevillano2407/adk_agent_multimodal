"""Checks for retrieval metrics, cutoff and dataset validity."""

import json
from datetime import datetime, timezone
from io import StringIO
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.evaluate_retrieval import evaluate_retrieval, retrieval_metrics, evidence_metrics
from retrieval.prepared_corpus import PreparedCorpus


def page(number: int, source: str = "annual.pdf") -> dict:
    return {"source_file": source, "pdf_page": number}


class TestRetrievalMetrics(unittest.TestCase):
    def test_2021_alternative_segment_table_accepts_multiline_heading(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = json.loads((root / "evals/retrieval_multireport.json").read_text(encoding="utf-8"))
        case = next(c for c in cases if c["id"] == "report_2021_segment_revenue")
        hit = {**page(157, case["source_file"]),
               "text": "Compute & \nNetworking Consolidated (In millions) 2021 Revenue 6,841 16,675"}
        self.assertEqual(retrieval_metrics([hit], case["relevant_pages"])["first_relevant_rank"], 1)
        self.assertEqual(evidence_metrics([hit], case["evidence"], 3)["first_evidence_rank"], 1)
        hit["text"] = "Compute & Networking 2021 Revenue increased by 109%"
        self.assertEqual(evidence_metrics([hit], case["evidence"], 3)["evidence_hit_at_k"], 0)

    def test_multireport_summary_tables_are_valid_alternative_evidence(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = json.loads((root / "evals/retrieval_multireport.json").read_text(encoding="utf-8"))
        for year, pdf_page, segment, total in (
            (2023, 15, "15.1", "27.0"), (2024, 20, "47.4", "60.9"),
            (2025, 18, "116.2", "130.5"), (2026, 23, "193.5", "215.9"),
        ):
            case = next(c for c in cases if c["id"] == f"report_{year}_segment_revenue")
            hit = {**page(pdf_page, case["source_file"]),
                   "text": f"{year} Compute & Networking Revenue {segment} {total} billion"}
            with self.subTest(year=year):
                self.assertEqual(retrieval_metrics([hit], case["relevant_pages"])["hit_at_k"], 1)
                self.assertEqual(evidence_metrics([hit], case["evidence"], 3)["evidence_hit_at_k"], 1)
                hit["text"] = f"{year} Compute & Networking Revenue {total} billion"
                self.assertEqual(evidence_metrics([hit], case["evidence"], 3)["evidence_hit_at_k"], 0)

    def test_multireport_document_filters_match_reference_filenames_exactly(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = json.loads((root / "evals/retrieval_multireport.json").read_text(encoding="utf-8"))
        positives = [case for case in cases if case.get("answerable", True)]
        self.assertEqual(len(positives), 7)
        for case in positives:
            with self.subTest(case=case["id"]):
                self.assertEqual({reference["source_file"] for reference in case["relevant_pages"]},
                                 {case["source_file"]})
                self.assertEqual({reference["source_file"] for reference in case["evidence"]},
                                 {case["source_file"]})

    def test_net_income_alternative_table_counts_at_evidence_rank_three(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = json.loads((root / "evals/retrieval_extended.json").read_text(encoding="utf-8"))
        case = next(c for c in cases if c["id"] == "net_income_es")
        source = "NVIDIA-2025-Annual-Report.pdf"
        hits = [
            {**page(130, source), "text": "Operating expenses for fiscal 2025"},
            {**page(18, source), "text": "Revenue 130.5 billion"},
            {**page(144, source), "text": "In millions; 2025; Net income $ 72,880"},
        ]
        self.assertEqual(retrieval_metrics(hits, case["relevant_pages"])["first_relevant_rank"], 1)
        self.assertEqual(evidence_metrics(hits, case["evidence"], 3)["first_evidence_rank"], 3)

    def test_evidence_requires_all_anchors_in_same_chunk_and_page(self) -> None:
        evidence = [{**page(18), "all_of": ["Revenue", "116.2"]}]
        hits = [{**page(18), "text": "Revenue"}, {**page(18), "text": "116.2"}]
        self.assertEqual(evidence_metrics(hits, evidence, 3)["evidence_hit_at_k"], 0)
        hits.append({**page(18), "text": "REVENUE\n  116.2"})
        self.assertEqual(evidence_metrics(hits, evidence, 3)["first_evidence_rank"], 3)
        self.assertEqual(evidence_metrics(hits, evidence, 2)["evidence_hit_at_k"], 0)

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

    def test_memory_corpus_and_case_search_preserve_document_request(self) -> None:
        requested = "requested.pdf"
        record = {**page(2, requested), "text": "Revenue 2024", "chunk_id": "versioned-id"}
        case = {"id": "filtered", "query": "report 2025, revenue FY2024",
                "source_file": requested, "relevant_pages": [page(2, requested)]}
        self.write_cases([case])
        search = Mock(return_value=[record])
        report = evaluate_retrieval(
            self.cases_path, Path("absent-manifest.json"), index_records=[record],
            search_case_fn=search,
        )
        search.assert_called_once_with(case, 3)
        self.assertEqual(report["mean_hit_at_k"], 1)
        self.assertEqual(report["cases"][0]["requested_source_file"], requested)

    def test_document_filter_must_match_labels_before_search(self) -> None:
        self.write_cases([{"id": "bad", "query": "alpha", "source_file": "other.pdf",
                           "relevant_pages": [page(1)]}])
        search = Mock()
        with self.assertRaisesRegex(ValueError, "Filtro"):
            evaluate_retrieval(self.cases_path, self.index_path, search_case_fn=search)
        search.assert_not_called()

    def test_memory_corpus_requires_explicit_search_and_one_callback(self) -> None:
        self.write_cases([{"id": "valid", "query": "alpha", "relevant_pages": [page(1)]}])
        with self.assertRaisesRegex(ValueError, "explícita"):
            evaluate_retrieval(self.cases_path, self.index_path, index_records=[])
        with self.assertRaisesRegex(ValueError, "única"):
            evaluate_retrieval(self.cases_path, self.index_path,
                               search_fn=Mock(), search_case_fn=Mock())

    def test_negative_cases_are_excluded_from_hit_and_mrr(self) -> None:
        self.write_cases([
            {"id": "positive", "query": "gamma", "relevant_pages": [page(2)],
             "evidence": [{**page(2), "all_of": ["gamma"]}], "split": "holdout"},
            {"id": "negative", "query": "alpha", "relevant_pages": [], "answerable": False},
        ])
        report = evaluate_retrieval(self.cases_path, self.index_path)
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["answerable_count"], 1)
        self.assertEqual(report["unanswerable_count"], 1)
        self.assertEqual(report["mean_hit_at_k"], 1)
        self.assertEqual(report["mean_evidence_hit_at_k"], 1)
        self.assertIsNone(report["cases"][1]["hit_at_k"])
        self.assertGreater(report["cases"][1]["returned_units"], 0)
        self.assertEqual(report["splits"]["holdout"]["mrr_at_k"], 1)

    def test_missing_evidence_anchor_is_configuration_error(self) -> None:
        self.write_cases([{
            "id": "bad_anchor", "query": "alpha", "relevant_pages": [page(1)],
            "evidence": [{**page(1), "all_of": ["invented"]}],
        }])
        search = Mock()
        with self.assertRaisesRegex(ValueError, "Anclas"):
            evaluate_retrieval(self.cases_path, self.index_path, search_fn=search)
        search.assert_not_called()

    def test_extended_dataset_has_twelve_cases_and_verified_anchors(self) -> None:
        root = Path(__file__).resolve().parents[1]
        index = root / "data/processed/nvidia_2025.chunks.c1200.o200.jsonl"
        if not index.exists():
            self.skipTest("El corpus local no se versiona.")
        report = evaluate_retrieval(
            root / "evals/retrieval_extended.json", index,
            search_fn=Mock(return_value=[]), retriever_name="offline_no_search",
        )
        self.assertEqual(report["case_count"], 12)
        self.assertEqual(report["answerable_count"], 10)
        self.assertEqual(report["unanswerable_count"], 2)

    def test_alternative_evidence_is_a_hit_for_either_retriever(self) -> None:
        self.write_cases([
            {"id": "alternative", "query": "alpha", "relevant_pages": [page(1), page(2)]}
        ])
        lexical = evaluate_retrieval(self.cases_path, self.index_path)
        search = Mock(return_value=[{**page(2), "text": "alpha gamma", "distance": 0.2}])
        vector = evaluate_retrieval(
            self.cases_path, self.index_path,
            search_fn=search, retriever_name="simulated_vector",
        )
        search.assert_called_once_with("alpha", 3)
        self.assertEqual(lexical["mrr_at_k"], 1.0)
        self.assertEqual(vector["mrr_at_k"], 1.0)
        self.assertEqual(vector["retriever"], "simulated_vector")
        self.assertEqual(lexical["cases_sha256"], vector["cases_sha256"])
        self.assertEqual(vector["cases"][0]["retrieved_pages"][0]["distance"], 0.2)
        self.assertNotIn("text", vector["cases"][0]["retrieved_pages"][0])

    def test_all_labels_are_validated_before_any_external_search(self) -> None:
        self.write_cases([
            {"id": "valid", "query": "alpha", "relevant_pages": [page(1)]},
            {"id": "invalid", "query": "gamma", "relevant_pages": [page(99)]},
        ])
        search = Mock()
        with self.assertRaisesRegex(ValueError, "ausente"):
            evaluate_retrieval(self.cases_path, self.index_path, search_fn=search)
        search.assert_not_called()

    def test_end_to_end_means_and_no_document_text_in_report(self) -> None:
        self.write_cases([
            {"id": "hit", "query": "alpha", "relevant_pages": [page(2)]},
            {"id": "miss", "query": "delta", "relevant_pages": [page(2)]},
        ])
        report = evaluate_retrieval(self.cases_path, self.index_path)
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["mean_hit_at_k"], 0.5)
        self.assertEqual(report["mrr_at_k"], 0.25)
        self.assertEqual(report["retrieval_unit"], "page")
        self.assertEqual(report["cases"][0]["first_relevant_rank"], 2)
        self.assertNotIn("text", report["cases"][0]["retrieved_pages"][0])

    def test_chunk_ranks_preserve_offsets_and_count_distinct_pages(self) -> None:
        records = [
            {
                **page(number), "text": "alpha",
                "chunk_id": f"annual.pdf:p{number}:start{start}",
                "start_char": start, "end_char": start + 5,
            }
            for number, start in ((1, 0), (1, 4), (2, 0))
        ]
        self.index_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        self.write_cases([
            {"id": "chunk_test", "query": "alpha", "relevant_pages": [page(2)]}
        ])
        report = evaluate_retrieval(self.cases_path, self.index_path, top_k=3)
        case = report["cases"][0]
        self.assertEqual(report["retrieval_unit"], "chunk")
        self.assertEqual(report["relevance_granularity"], "page")
        self.assertEqual(case["returned_units"], 3)
        self.assertEqual(case["unique_pages"], 2)
        self.assertEqual(case["first_relevant_rank"], 3)
        self.assertAlmostEqual(case["reciprocal_rank_at_k"], 1 / 3)
        self.assertEqual(case["retrieved_text_characters"], 15)
        self.assertEqual(report["mean_retrieved_text_characters"], 15)
        self.assertEqual(case["retrieved_pages"][1]["chunk_id"], records[1]["chunk_id"])
        self.assertEqual(case["retrieved_pages"][1]["start_char"], 4)
        self.assertEqual(case["retrieved_pages"][1]["end_char"], 9)

    def test_mixed_page_and_chunk_index_is_rejected(self) -> None:
        self.index_path.write_text(
            json.dumps({**page(1), "text": "alpha"}) + "\n"
            + json.dumps({**page(2), "chunk_id": "child", "text": "alpha"}) + "\n",
            encoding="utf-8",
        )
        self.write_cases([
            {"id": "mixed", "query": "alpha", "relevant_pages": [page(2)]}
        ])
        with self.assertRaisesRegex(ValueError, "mezcla"):
            evaluate_retrieval(self.cases_path, self.index_path)

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


class TestMultireportCLI(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cases = self.root / "cases.json"
        self.case = {"id": "prior", "query": "FY2024 in report 2025",
                     "source_file": "annual2025.pdf",
                     "relevant_pages": [page(1, "annual2025.pdf")]}
        self.cases.write_text(json.dumps([self.case]), encoding="utf-8")
        self.record = {**page(1, "annual2025.pdf"), "text": "Revenue FY2024",
                       "chunk_id": "id"}
        self.corpus = PreparedCorpus([self.record], {}, "a" * 64)
        self.argv = ["evaluate_retrieval", "--cases", str(self.cases),
                     "--corpus-manifest", str(self.root / "manifest.json")]

    def test_validate_only_does_not_open_chroma_or_model(self) -> None:
        from evals.evaluate_retrieval import main

        with (
            patch("sys.argv", self.argv + ["--validate-only"]),
            patch("retrieval.prepared_corpus.load_prepared_corpus", return_value=self.corpus),
            patch("chromadb.PersistentClient") as store,
            patch("google.genai.Client") as client,
            patch("sys.stdout", new_callable=StringIO) as output,
        ):
            main()
        store.assert_not_called()
        client.assert_not_called()
        self.assertIn("PASS: 1 casos", output.getvalue())
        self.assertFalse((self.root / "evals/runs").exists())

    def test_lexical_corpus_evaluation_never_opens_chroma_or_model(self) -> None:
        from evals.evaluate_retrieval import main

        for filtered in (False, True):
            with (
                self.subTest(filtered=filtered),
                patch("sys.argv", self.argv + (["--filter-by-document"] if filtered else [])),
                patch("evals.evaluate_retrieval.PROJECT_ROOT", self.root),
                patch("retrieval.prepared_corpus.load_prepared_corpus", return_value=self.corpus),
                patch("chromadb.PersistentClient") as store,
                patch("google.genai.Client") as client,
                patch("sys.stdout", new_callable=StringIO),
                patch("evals.evaluate_retrieval.datetime") as clock,
            ):
                clock.now.return_value = datetime(2026, 1, 1, 0, 0, int(filtered), tzinfo=timezone.utc)
                main()
            store.assert_not_called()
            client.assert_not_called()
        saved = [json.loads(path.read_text()) for path in (self.root / "evals/runs").glob("*.json")]
        self.assertEqual(len(saved), 2)
        self.assertEqual({report["filter_by_document"] for report in saved}, {False, True})
        self.assertTrue(all(report["mean_hit_at_k"] == 1 for report in saved))

    def test_wrong_corpus_is_refused_before_model_client(self) -> None:
        from evals.evaluate_retrieval import main

        with (
            patch("sys.argv", self.argv + ["--retriever", "vector"]),
            patch("retrieval.prepared_corpus.load_prepared_corpus", return_value=self.corpus),
            patch("chromadb.PersistentClient") as store,
            patch("google.genai.Client") as client,
        ):
            store.return_value.get_collection.return_value.metadata = {
                "index_complete": True, "text_corpus_sha256": "b" * 64,
            }
            with self.assertRaisesRegex(ValueError, "corpus"):
                main()
        client.assert_not_called()

    def test_filtered_execution_uses_requested_document_and_saves_identity(self) -> None:
        from evals.evaluate_retrieval import main

        with (
            patch("sys.argv", self.argv + ["--retriever", "vector", "--filter-by-document"]),
            patch("evals.evaluate_retrieval.PROJECT_ROOT", self.root),
            patch("retrieval.prepared_corpus.load_prepared_corpus", return_value=self.corpus),
            patch("chromadb.PersistentClient") as store,
            patch("google.genai.Client") as client,
            patch("dotenv.load_dotenv"),
            patch("retrieval.search_vectors.search_chunks", return_value=[self.record]) as search,
            patch("sys.stdout", new_callable=StringIO),
        ):
            collection = store.return_value.get_collection.return_value
            collection.metadata = {"index_complete": True, "text_corpus_sha256": "a" * 64}
            collection.count.return_value = 1
            main()
        self.assertEqual(search.call_args.kwargs["source_file"], "annual2025.pdf")
        saved = json.loads(next((self.root / "evals/runs").glob("*.json")).read_text())
        self.assertTrue(saved["filter_by_document"])
        self.assertEqual(saved["text_corpus_sha256"], self.corpus.corpus_sha256)
        self.assertEqual(saved["mean_hit_at_k"], 1)


if __name__ == "__main__":
    unittest.main()
