"""Offline checks for evaluation failures and resource cleanup."""

import json
import importlib.util
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from google.genai import types

from evals.run_case import main, record_failure


class TestRunCase(unittest.IsolatedAsyncioTestCase):
    """Check that failures remain visible and resources are closed."""

    async def check_failure(self, *, during_generation: bool) -> None:
        error = RuntimeError("Simulated execution failure")
        runner = MagicMock()
        runner.app_name = "corporate_reports"
        runner.close = AsyncMock()
        runner.session_service.create_session = AsyncMock(
            return_value=SimpleNamespace(user_id="test_user", id="test_session")
        )

        async def failed_events(**kwargs):
            raise error
            yield  # Make this an async iterator, like Runner.run_async.

        if during_generation:
            runner.run_async.side_effect = failed_events
        else:
            runner.session_service.create_session.side_effect = error

        case = {
            "id": "missing_document",
            "question": "Question without a document",
            "input_files": [],
            "expected_status": "insufficient_evidence",
        }
        # Keep agent initialization and configuration outside this control-flow test.
        package = ModuleType("corporate_reports")
        package.__path__ = []
        isolated_modules = {
            "corporate_reports": package,
            "corporate_reports.agent": SimpleNamespace(root_agent=MagicMock()),
            "corporate_reports.schemas": SimpleNamespace(ReportAnswer=MagicMock()),
            "evals.check_calculation": SimpleNamespace(validate_calculation=MagicMock()),
        }
        with (
            patch.dict("sys.modules", isolated_modules),
            patch("evals.run_case.load_dotenv"),
            patch("evals.run_case.load_case", return_value=case) as load_case,
            patch("evals.run_case.InMemoryRunner", return_value=runner),
            patch(
                "evals.run_case.record_failure",
                return_value=Path("simulated_error.json"),
            ) as save_failure,
            redirect_stdout(StringIO()),
        ):
            with self.assertRaises(RuntimeError) as captured:
                await main("missing_document")

        self.assertIs(captured.exception, error)
        load_case.assert_called_once_with("missing_document")
        save_failure.assert_called_once_with("missing_document", error)
        runner.close.assert_awaited_once()
        if during_generation:
            runner.run_async.assert_called_once_with(
                user_id="test_user",
                session_id="test_session",
                new_message=unittest.mock.ANY,
            )
        else:
            runner.run_async.assert_not_called()

    async def test_session_failure_is_recorded_and_propagated(self) -> None:
        await self.check_failure(during_generation=False)

    async def test_generation_failure_is_recorded_and_propagated(self) -> None:
        await self.check_failure(during_generation=True)

    async def check_success(
        self, *, use_retrieval: bool, chunk_index: bool = False,
        vector_backend: bool = False,
        tool_mode: bool = False,
    ) -> None:
        """Use the real schema and simulated events, never a live model."""
        schema_path = (
            Path(__file__).resolve().parents[1] / "corporate_reports/schemas.py"
        )
        spec = importlib.util.spec_from_file_location(
            "offline_report_schemas", schema_path
        )
        schema_module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {spec.name: schema_module}):
            spec.loader.exec_module(schema_module)

        payload = {
            "status": "insufficient_evidence",
            "answer": "La imagen no incluye ingresos de España.",
            "sources": ["Imagen adjunta"],
            "limitations": ["Falta el desglose geográfico."],
            "revenue_share_calculation": None,
        }
        if use_retrieval:
            payload.update({
                "status": "answered",
                "answer": "La participación es aproximadamente 89,04 %.",
                "sources": ["annual.pdf, página PDF 18"],
                "limitations": ["Cifras de origen redondeadas."],
                "revenue_share_calculation": {
                    "segment_revenue": 116.2,
                    "consolidated_revenue": 130.5,
                    "unit": "billion_USD",
                    "percentage": 89.04,
                },
            })
        event = MagicMock()
        event.is_final_response.return_value = True
        event.content.parts = [
            SimpleNamespace(text=json.dumps(payload), thought=False)
        ]

        async def successful_events(**kwargs):
            if tool_mode:
                yield SimpleNamespace(
                    is_final_response=lambda: False,
                    content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
                        name="search_corporate_reports", args={"query": "private query", "source_file": "annual.pdf"},
                    ))]),
                )
                yield SimpleNamespace(
                    is_final_response=lambda: False,
                    content=types.Content(parts=[types.Part(function_response=types.FunctionResponse(
                        name="search_corporate_reports", response={"status": "ok", "evidence": [
                            {"source_file": "annual.pdf", "pdf_page": 18, "text": "private document"},
                        ]},
                    ))]),
                )
            yield event

        runner = MagicMock()
        runner.app_name = "corporate_reports"
        runner.close = AsyncMock()
        runner.session_service.create_session = AsyncMock(
            return_value=SimpleNamespace(user_id="test_user", id="test_session")
        )
        runner.run_async.side_effect = successful_events
        case = {
            "id": "spain_revenue_missing",
            "question": "¿Qué ingresos corresponden a España?",
            "input_files": ["fixture.png"],
            "expected_status": "insufficient_evidence",
        }
        retrieved = [{
            "source_file": "annual.pdf",
            "pdf_page": 18,
            "text": "Compute & Networking: 116.2; Consolidated: 130.5",
            "score": 3,
        }]
        if use_retrieval:
            case.update({
                "id": "rag_compute_revenue_share",
                "question": "¿Qué porcentaje corresponde al segmento? Cita la página.",
                "input_files": [],
                "retrieval_query": "Compute Networking revenue",
                "expected_status": "answered",
                "expected_calculation": {
                    "segment_revenue": 116.2,
                    "consolidated_revenue": 130.5,
                    "unit": "billion_USD",
                },
            })
        if chunk_index:
            case["retrieval_index"] = "data/processed/test.chunks.jsonl"
            retrieved[0].update({
                "chunk_id": "annual.pdf:p18:start0",
                "start_char": 0,
                "end_char": len(retrieved[0]["text"]),
            })
        if vector_backend:
            case["retriever"] = "vector"
        if tool_mode:
            case.update({"execution_mode": "agent_tools", "source_file": "annual.pdf", "input_files": []})
        package = ModuleType("corporate_reports")
        package.__path__ = []
        validate_calculation = MagicMock()
        isolated_modules = {
            "corporate_reports": package,
            "corporate_reports.agent": SimpleNamespace(
                root_agent=SimpleNamespace(
                    model="offline-model",
                    model_copy=lambda **kwargs: SimpleNamespace(model="offline-model", tools=[]),
                )
            ),
            "corporate_reports.schemas": schema_module,
            "corporate_reports.tools": SimpleNamespace(COLLECTION_NAME="nvidia_reports_fixed_gemini001_768"),
            "evals.check_calculation": SimpleNamespace(
                validate_calculation=validate_calculation
            ),
        }
        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            image_bytes = b"offline image fixture"
            (project_root / "fixture.png").write_bytes(image_bytes)
            with (
                patch.dict("sys.modules", isolated_modules),
                patch("evals.run_case.PROJECT_ROOT", project_root),
                patch("evals.run_case.load_dotenv"),
                patch("evals.run_case.load_case", return_value=case),
                patch("evals.run_case.InMemoryRunner", return_value=runner),
                patch("evals.run_case.perf_counter", side_effect=[10.0, 10.125]),
                patch("evals.run_case.version", return_value="2.13.0"),
                patch("evals.run_case.search_pages", return_value=retrieved) as search,
                patch("evals.run_case.search_chunks", return_value=retrieved) as vector_search,
                patch("evals.run_case.chromadb.PersistentClient") as store,
                patch("evals.run_case.genai.Client") as embedding_client,
                patch("evals.run_case.record_failure") as save_failure,
                patch("evals.run_case.check_agent_index", return_value={"collection": "offline-index"}) as preflight,
                redirect_stdout(StringIO()),
            ):
                await main(case["id"])

            files = list((project_root / "evals/runs").iterdir())
            self.assertEqual(len(files), 2)
            metadata_path = next(
                path for path in files if path.name.endswith(".metadata.json")
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            response_path = metadata_path.parent / metadata["response_file"]
            self.assertEqual(
                json.loads(response_path.read_text(encoding="utf-8")), payload
            )
            self.assertEqual(metadata["case_id"], case["id"])
            self.assertEqual(metadata["model"], "offline-model")
            self.assertEqual(metadata["adk_version"], "2.13.0")
            self.assertEqual(metadata["question"], case["question"])
            self.assertEqual(metadata["input_files"], case["input_files"])
            self.assertEqual(metadata["generation_seconds"], 0.125)
            self.assertEqual(metadata["automated_checks"], "passed")
            if tool_mode:
                preflight.assert_called_once()
                self.assertEqual(metadata["execution_mode"], "agent_tools")
                self.assertEqual(metadata["agent_index"], {"collection": "offline-index"})
                self.assertEqual(len(metadata["tool_trace"]), 2)
                self.assertNotIn("private query", json.dumps(metadata))
                self.assertNotIn("private document", json.dumps(metadata))
                self.assertEqual(runner.run_async.call_args.kwargs["run_config"].max_llm_calls, 8)
            else:
                preflight.assert_not_called()
            self.assertEqual(metadata["retrieval_query"], case.get("retrieval_query"))
            expected_refs = [
                {
                    key: value for key, value in record.items()
                    if key not in ("text", "score")
                }
                for record in retrieved
            ] if use_retrieval else []
            self.assertEqual(metadata["retrieved_pages"], expected_refs)
            expected_index = case.get(
                "retrieval_index", "data/processed/nvidia_2025.pages.jsonl"
            ) if use_retrieval else None
            self.assertEqual(metadata["retrieval_index"], expected_index)
            self.assertEqual(
                metadata["retrieved_text_characters"],
                len(retrieved[0]["text"]) if use_retrieval else 0,
            )
            self.assertEqual(
                response_path.stem,
                f"{case['id']}_{metadata['timestamp_utc']}",
            )

        message = runner.run_async.call_args.kwargs["new_message"]
        self.assertEqual(message.parts[0].text, case["question"])
        if use_retrieval:
            if vector_backend:
                search.assert_not_called()
                vector_search.assert_called_once_with(
                    embedding_client.return_value.__enter__.return_value,
                    store.return_value.get_collection.return_value,
                    case["retrieval_query"], top_k=3,
                )
                embedding_client.return_value.__exit__.assert_called_once()
                self.assertEqual(metadata["retriever"], "vector")
            else:
                search.assert_called_once_with(
                    case["retrieval_query"],
                    project_root / expected_index,
                    top_k=3,
                )
            self.assertEqual(len(message.parts), 2)
            self.assertIsNone(message.parts[1].inline_data)
            context = json.loads(message.parts[1].text.split("\n", 1)[1])
            self.assertEqual(context, [{
                key: value for key, value in retrieved[0].items() if key != "score"
            }])
            validate_calculation.assert_called_once()
            checked_response, checked_reference = validate_calculation.call_args.args
            self.assertEqual(checked_reference, case["expected_calculation"])
            self.assertEqual(checked_response.revenue_share_calculation.percentage, 89.04)
        elif tool_mode:
            self.assertEqual(len(message.parts), 1)
            search.assert_not_called()
            vector_search.assert_not_called()
            validate_calculation.assert_not_called()
        else:
            search.assert_not_called()
            self.assertEqual(message.parts[1].inline_data.data, image_bytes)
            self.assertEqual(message.parts[1].inline_data.mime_type, "image/png")
            validate_calculation.assert_not_called()
        save_failure.assert_not_called()
        runner.close.assert_awaited_once()

    async def test_success_saves_response_and_matching_metadata(self) -> None:
        await self.check_success(use_retrieval=False)

    async def test_tool_execution_starts_without_context_and_saves_sanitized_trace(self) -> None:
        await self.check_success(use_retrieval=False, tool_mode=True)

    async def test_rag_sends_retrieved_context_and_records_provenance(self) -> None:
        await self.check_success(use_retrieval=True)

    async def test_rag_selects_chunk_index_and_preserves_chunk_provenance(self) -> None:
        await self.check_success(use_retrieval=True, chunk_index=True)

    async def test_vector_rag_sends_evidence_and_closes_embedding_client(self) -> None:
        await self.check_success(use_retrieval=True, chunk_index=True, vector_backend=True)


class TestCalculation(unittest.TestCase):
    """Check arithmetic independently of schema ranges and source values."""

    def test_rejects_in_range_but_incorrect_percentage(self) -> None:
        from corporate_reports.schemas import ReportAnswer
        from evals.check_calculation import validate_calculation

        response = ReportAnswer.model_validate({
            "status": "answered",
            "answer": "Simulated answer with an incorrect percentage.",
            "sources": ["Test fixture"],
            "limitations": [],
            "revenue_share_calculation": {
                "segment_revenue": 116.2,
                "consolidated_revenue": 130.5,
                "unit": "billion_USD",
                "percentage": 80,
            },
        })
        reference = {
            "segment_revenue": 116.2,
            "consolidated_revenue": 130.5,
            "unit": "billion_USD",
        }
        with self.assertRaisesRegex(ValueError, "Porcentaje incorrecto"):
            validate_calculation(response, reference)


class TestRecordFailure(unittest.TestCase):
    """Check diagnostics without writing into the real evaluation history."""

    def test_records_code_without_exception_message(self) -> None:
        error = RuntimeError("Sensitive message must not be persisted")
        error.code = 429

        with TemporaryDirectory() as temporary_directory:
            with patch("evals.run_case.PROJECT_ROOT", Path(temporary_directory)):
                output_path = record_failure("offline_test", error)

            contents = output_path.read_text(encoding="utf-8")
            diagnostic = json.loads(contents)
            self.assertEqual(output_path.parent, Path(temporary_directory) / "evals/runs")
            self.assertEqual(diagnostic["case_id"], "offline_test")
            self.assertEqual(diagnostic["status"], "execution_error")
            self.assertEqual(diagnostic["error_type"], "RuntimeError")
            self.assertEqual(diagnostic["error_code"], 429)
            self.assertNotIn(str(error), contents)


if __name__ == "__main__":
    unittest.main()
