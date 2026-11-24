"""Presupuesto por turno y regresión de serialización con el ADK instalado."""

import asyncio
import json
import unittest
from types import SimpleNamespace

from google.adk.events import Event, EventActions
from google.adk.flows.llm_flows.prompt._schema import get_structured_model_response
from google.adk.sessions.state import State
from google.adk.tools.set_model_response_tool import SetModelResponseTool
from google.genai import types

from corporate_reports.callbacks import limit_document_searches, preserve_response_nulls
from corporate_reports.schemas import ReportAnswer


def context(invocation_id="turn-1", shared=None):
    return SimpleNamespace(
        invocation_id=invocation_id,
        state=State(shared if shared is not None else {}, {}),
        actions=EventActions(),
    )


class TestReportCallbacks(unittest.IsolatedAsyncioTestCase):
    async def test_visual_budget_is_independent_and_resets_on_next_turn(self) -> None:
        ctx = context()
        visual = SimpleNamespace(name="read_report_page")
        search = SimpleNamespace(name="search_corporate_reports")
        self.assertIsNone(await limit_document_searches(visual, {"pdf_page": 5}, ctx))
        self.assertIsNone(await limit_document_searches(visual, {"pdf_page": 18}, ctx))
        result = await limit_document_searches(visual, {"pdf_page": 133}, ctx)
        self.assertEqual(result["status"], "visual_limit_reached")
        self.assertIsNone(await limit_document_searches(search, {}, ctx))
        ctx.invocation_id = "turn-2"
        self.assertIsNone(await limit_document_searches(visual, {"pdf_page": 5}, ctx))

    async def test_only_two_searches_are_authorized_per_turn(self) -> None:
        ctx = context()
        tool = SimpleNamespace(name="search_corporate_reports")
        self.assertIsNone(await limit_document_searches(tool, {}, ctx))
        self.assertIsNone(await limit_document_searches(tool, {}, ctx))
        for _ in range(2):
            result = await limit_document_searches(tool, {}, ctx)
            self.assertEqual(result["status"], "search_limit_reached")
            self.assertEqual(result["evidence"], [])

    async def test_new_turn_and_other_sessions_have_their_own_budget(self) -> None:
        shared = {}
        tool = SimpleNamespace(name="search_corporate_reports")
        first = context(shared=shared)
        await limit_document_searches(tool, {}, first)
        await limit_document_searches(tool, {}, first)
        self.assertIsNone(await limit_document_searches(tool, {}, context("turn-2", shared)))
        self.assertIsNone(await limit_document_searches(tool, {}, context("turn-1")))

    async def test_parallel_callbacks_share_the_same_reservation(self) -> None:
        shared = {}
        tool = SimpleNamespace(name="search_corporate_reports")
        results = await asyncio.gather(*[
            limit_document_searches(tool, {}, context(shared=shared)) for _ in range(3)
        ])
        self.assertEqual(results.count(None), 2)
        self.assertEqual(results[2]["status"], "search_limit_reached")

    async def test_other_tools_are_not_blocked(self) -> None:
        ctx = context()
        tool = SimpleNamespace(name="set_model_response")
        for _ in range(3):
            self.assertIsNone(await limit_document_searches(tool, {}, ctx))
        self.assertEqual(ctx.state.to_dict(), {})

    async def test_real_adk_final_event_preserves_required_null(self) -> None:
        tool = SetModelResponseTool(ReportAnswer)
        ctx = context()
        payload = {
            "status": "insufficient_evidence", "answer": "Falta el desglose.",
            "sources": [], "limitations": ["No hay evidencia suficiente."],
            "revenue_share_calculation": None,
        }
        result = await tool.run_async(args=payload, tool_context=ctx)
        self.assertNotIn("revenue_share_calculation", result)
        corrected = preserve_response_nulls(tool, payload, ctx, result)
        self.assertIn("revenue_share_calculation", corrected)
        self.assertIsNone(corrected["revenue_share_calculation"])
        event = Event(
            author="corporate_reports_agent", actions=ctx.actions,
            content=types.Content(role="user", parts=[types.Part(
                function_response=types.FunctionResponse(
                    name="set_model_response", response=corrected,
                )
            )]),
        )
        final_json = get_structured_model_response(event)
        self.assertEqual(json.loads(final_json), payload)
        ReportAnswer.model_validate_json(final_json)

    async def test_rejected_args_are_not_repaired_or_accepted(self) -> None:
        tool = SetModelResponseTool(ReportAnswer)
        ctx = context()
        result = await tool.run_async(args={"status": "answered"}, tool_context=ctx)
        self.assertIn("error", result)
        self.assertIsNone(preserve_response_nulls(tool, {}, ctx, result))
        self.assertIsNone(ctx.actions.set_model_response)

    async def test_normalizer_ignores_search_results(self) -> None:
        tool = SimpleNamespace(name="search_corporate_reports")
        self.assertIsNone(preserve_response_nulls(tool, {}, context(), {"status": "ok"}))
