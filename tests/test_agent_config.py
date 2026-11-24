"""Coherencia entre configuración de retrieval e instrucciones del agente."""

import unittest

from corporate_reports.agent import root_agent
from corporate_reports.config import (
    COLLECTION_NAME, LEGACY_COLLECTION_NAME, LEGACY_SOURCE_FILE,
    MULTIREPORT_COLLECTION_NAME, text_search_scope,
)
from corporate_reports import tools


class TestAgentConfig(unittest.TestCase):
    def test_prompt_describes_the_same_collection_used_by_tool(self) -> None:
        self.assertEqual(tools.COLLECTION_NAME, COLLECTION_NAME)
        self.assertIn(text_search_scope(tools.COLLECTION_NAME), root_agent.instruction)

    def test_active_multireport_agent_does_not_claim_legacy_text_scope(self) -> None:
        self.assertEqual(COLLECTION_NAME, MULTIREPORT_COLLECTION_NAME)
        self.assertNotIn("limitada al informe NVIDIA FY2025", root_agent.instruction)

    def test_legacy_scope_does_not_infer_text_availability_from_visual_catalog(self) -> None:
        scope = text_search_scope(LEGACY_COLLECTION_NAME)
        self.assertIn(LEGACY_SOURCE_FILE, scope)
        self.assertIn("eso no significa", scope)

    def test_multireport_scope_requires_provenance_and_explicit_document_filter(self) -> None:
        scope = text_search_scope(MULTIREPORT_COLLECTION_NAME)
        self.assertIn("list_report_documents", scope)
        self.assertIn("source_file", scope)
        self.assertIn("index_not_ready", scope)
        self.assertIn("ejercicio fiscal", scope)
        self.assertNotIn("limitada al informe NVIDIA FY2025", scope)

    def test_unknown_collection_does_not_get_a_misleading_scope(self) -> None:
        with self.assertRaises(ValueError):
            text_search_scope("unrecognized_collection")


if __name__ == "__main__":
    unittest.main()
