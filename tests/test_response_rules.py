"""Regresión sobre una respuesta real con abstención e inferencia no respaldada."""

import json
import unittest
from pathlib import Path

from corporate_reports.schemas import ReportAnswer
from evals.response_rules import validate_response_rules


class TestResponseRules(unittest.TestCase):
    def agent_case(self, case_id: str) -> dict:
        cases = json.loads((self.root / "evals/cases.json").read_text(encoding="utf-8"))
        return next(case for case in cases if case["id"] == case_id)

    def test_note_numbers_are_rejected_in_answer_sources_and_limitations(self) -> None:
        case = self.agent_case("agent_spain_missing_multireport")
        for field in ("answer", "sources", "limitations"):
            for note in ("Nota 17", "Note 20", "NOTA\n16"):
                response = ReportAnswer(
                    status="insufficient_evidence", answer="Las evidencias no permiten responder.",
                    sources=[], limitations=[], revenue_share_calculation=None,
                )
                setattr(response, field, note if field == "answer" else [note])
                with self.subTest(field=field, note=note), self.assertRaisesRegex(ValueError, "nota"):
                    validate_response_rules(response, case)

    def test_whole_report_absence_claims_from_observed_response_are_rejected(self) -> None:
        case = self.agent_case("agent_spain_missing_multireport")
        for claim in (
            "Para poder calcularlo faltan los siguientes datos en el informe.",
            "El informe no menciona ni desglosa de manera individual la cifra de España.",
            "El informe NVIDIA-2025-Annual-Report.pdf no presenta información sobre España.",
        ):
            response = ReportAnswer(
                status="insufficient_evidence", answer="Falta evidencia.", sources=[],
                limitations=[claim], revenue_share_calculation=None,
            )
            with self.subTest(claim=claim), self.assertRaisesRegex(ValueError, "regresión"):
                validate_response_rules(response, case)

    def test_concise_evidence_scoped_agent_abstention_passes(self) -> None:
        response = ReportAnswer(
            status="insufficient_evidence",
            answer="Las evidencias recuperadas no permiten determinar el porcentaje solicitado.",
            sources=["NVIDIA-2025-Annual-Report.pdf, página 171"],
            limitations=["No se recuperó un desglose de España para Compute & Networking FY2025."],
            revenue_share_calculation=None,
        )
        validate_response_rules(response, self.agent_case("agent_spain_missing_multireport"))

    def test_exact_result_phrase_is_rejected_for_rounded_financial_case(self) -> None:
        response = ReportAnswer(
            status="answered", answer="Resultado exacto: 77,8126%.",
            sources=[], limitations=[], revenue_share_calculation=None,
        )
        with self.assertRaisesRegex(ValueError, "regresión"):
            validate_response_rules(response, self.agent_case("agent_fy2024_from_report2025"))

    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        cases = json.loads((self.root / "evals/cases.json").read_text(encoding="utf-8"))
        self.case = next(c for c in cases if c["id"] == "rag_spain_revenue_missing_vector")

    def test_historical_response_is_schema_valid_but_fails_regression(self) -> None:
        response = ReportAnswer.model_validate_json(
            (self.root / "evals/responses/spain_other_unsupported.json").read_text(encoding="utf-8")
        )
        self.assertEqual(response.status, "insufficient_evidence")
        with self.assertRaisesRegex(ValueError, "regresión"):
            validate_response_rules(response, self.case)

    def test_evidence_scoped_abstention_passes(self) -> None:
        response = ReportAnswer(
            status="insufficient_evidence",
            answer="Las evidencias recuperadas no permiten determinar el porcentaje de España.",
            sources=[], limitations=["Falta el desglose España por segmento."],
            revenue_share_calculation=None,
        )
        validate_response_rules(response, self.case)

    def test_known_phrase_is_detected_in_limitations_with_spacing_variations(self) -> None:
        response = ReportAnswer(
            status="insufficient_evidence", answer="Falta el dato.", sources=[],
            limitations=["Sus DATOS se\nencuentran agregados en Other."],
            revenue_share_calculation=None,
        )
        with self.assertRaisesRegex(ValueError, "regresión"):
            validate_response_rules(response, self.case)

    def test_no_calculation_rule_is_enforced(self) -> None:
        response = ReportAnswer(
            status="insufficient_evidence", answer="Falta el dato.", sources=[], limitations=[],
            revenue_share_calculation={
                "segment_revenue": 1, "consolidated_revenue": 2,
                "unit": "billion_USD", "percentage": 50,
            },
        )
        with self.assertRaisesRegex(ValueError, "null"):
            validate_response_rules(response, self.case)
