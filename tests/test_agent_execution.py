"""Controles del evaluador del agente sin modelos ni almacenes reales."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from google.genai import types

from evals.agent_execution import check_agent_index, collect_tool_events, validate_agent_execution


class TestAgentExecution(unittest.TestCase):
    def test_unpromoted_agent_never_opens_store(self) -> None:
        with patch("evals.agent_execution.chromadb.PersistentClient") as store:
            with self.assertRaisesRegex(ValueError, "activada"):
                check_agent_index(Path("unused"), "nvidia_fixed_gemini001_768")
        store.assert_not_called()

    def test_preflight_checks_corpus_completion_hash_and_nonempty_count(self) -> None:
        corpus = SimpleNamespace(corpus_sha256="a" * 64, chunks=[{"text": "one"}, {"text": " "}])
        with (
            patch("evals.agent_execution.load_prepared_corpus", return_value=corpus),
            patch("evals.agent_execution.chromadb.PersistentClient") as store,
        ):
            collection = store.return_value.get_collection.return_value
            for complete, digest, count in ((False, "a" * 64, 1), (True, "b" * 64, 1), (True, "a" * 64, 0)):
                collection.metadata = {"index_complete": complete, "text_corpus_sha256": digest}
                collection.count.return_value = count
                with self.subTest(complete=complete, count=count, digest=digest), self.assertRaises(ValueError):
                    check_agent_index(Path("unused"), "nvidia_reports_fixed_gemini001_768")
            collection.metadata = {"index_complete": True, "text_corpus_sha256": "a" * 64}
            collection.count.return_value = 1
            result = check_agent_index(Path("unused"), "nvidia_reports_fixed_gemini001_768")
            self.assertEqual(result["text_corpus_sha256"], "a" * 64)

    def test_trace_omits_query_document_text_images_and_final_answer(self) -> None:
        event = SimpleNamespace(content=types.Content(parts=[
            types.Part(function_call=types.FunctionCall(name="search_corporate_reports",
                args={"query": "secret-query", "source_file": "annual.pdf"})),
            types.Part(function_response=types.FunctionResponse(name="search_corporate_reports",
                response={"status": "ok", "evidence": [{"source_file": "annual.pdf",
                    "pdf_page": 1, "text": "secret-text"}], "image": "secret-image"})),
            types.Part(function_response=types.FunctionResponse(name="set_model_response",
                response={"answer": "secret-answer"})),
        ]))
        trace = collect_tool_events(event)
        self.assertNotIn("secret-", str(trace))
        self.assertEqual(trace[1]["evidence"], [{"source_file": "annual.pdf", "pdf_page": 1}])
        self.assertEqual(collect_tool_events(SimpleNamespace(content=None)), [])

    def setUp(self) -> None:
        self.case = {"source_file": "annual.pdf"}
        self.response = SimpleNamespace(status="answered", sources=["annual.pdf, página 1"])
        self.trace = [
            {"kind": "call", "tool_name": "search_corporate_reports", "source_file": "annual.pdf"},
            {"kind": "response", "tool_name": "search_corporate_reports", "status": "ok",
             "evidence": [{"source_file": "annual.pdf", "pdf_page": 1}]},
        ]

    def test_grounded_execution_and_evidence_scoped_abstention_pass(self) -> None:
        validate_agent_execution(self.response, self.case, self.trace)
        self.response.status = "insufficient_evidence"
        self.response.sources = []
        self.trace[1].update(status="no_results", evidence=[])
        validate_agent_execution(self.response, self.case, self.trace)

    def test_no_search_wrong_filter_backend_failure_or_wrong_evidence_fail(self) -> None:
        variants = [[], deepcopy(self.trace), deepcopy(self.trace), deepcopy(self.trace)]
        variants[1][0]["source_file"] = "other.pdf"
        variants[2][1]["status"] = "retrieval_error"
        variants[3][1]["evidence"][0]["source_file"] = "other.pdf"
        for trace in variants:
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                validate_agent_execution(self.response, self.case, trace)

    def test_answer_without_citations_or_without_evidence_fails(self) -> None:
        self.response.sources = []
        with self.assertRaises(ValueError):
            validate_agent_execution(self.response, self.case, self.trace)
        self.response.sources = ["other.pdf, página 1"]
        with self.assertRaises(ValueError):
            validate_agent_execution(self.response, self.case, self.trace)
        self.response.sources = ["annual.pdf, página 1"]
        self.trace[1]["evidence"] = []
        with self.assertRaises(ValueError):
            validate_agent_execution(self.response, self.case, self.trace)


if __name__ == "__main__":
    unittest.main()
