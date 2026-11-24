"""Contrato del experimento texto/imagen; sin llamadas a modelos."""

import unittest

from evals.run_case import load_case


class TestVisualCases(unittest.TestCase):
    def test_only_image_input_changes_between_conditions(self) -> None:
        text = load_case("visual_racks_text_only")
        image = load_case("visual_racks_image")
        for key in ("question", "retrieval_query", "retrieval_index"):
            self.assertEqual(text[key], image[key])
        self.assertEqual(text["input_files"], [])
        self.assertEqual(image["input_files"], [
            "data/processed/page_images/NVIDIA-2025-Annual-Report.p5.144dpi.png"
        ])
        self.assertEqual(text["expected_status"], "insufficient_evidence")
        self.assertEqual(image["expected_status"], "answered")
        self.assertTrue(text["expect_no_calculation"])
        self.assertTrue(image["expect_no_calculation"])
