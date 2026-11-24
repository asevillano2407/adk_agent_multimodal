"""Offline checks for provider configuration and embedding validation."""

import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from google.genai import errors

from retrieval.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    embed_text,
)


class TestEmbedText(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()

    def set_vector(self, values) -> None:
        self.client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=values)]
        )

    def test_supported_tasks_use_expected_model_and_dimension(self) -> None:
        for task in ("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"):
            with self.subTest(task=task):
                self.client.reset_mock()
                original = [3.0, 4.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)
                self.set_vector(original)
                result = embed_text(self.client, "  Evidence text  ", task)

                self.client.models.embed_content.assert_called_once()
                arguments = self.client.models.embed_content.call_args.kwargs
                self.assertEqual(arguments["model"], EMBEDDING_MODEL)
                self.assertEqual(arguments["contents"], "  Evidence text  ")
                self.assertEqual(arguments["config"].task_type, task)
                self.assertEqual(arguments["config"].output_dimensionality, 768)
                self.assertEqual(len(result), 768)
                self.assertAlmostEqual(result[0], 0.6)
                self.assertAlmostEqual(result[1], 0.8)
                self.assertAlmostEqual(math.hypot(*result), 1.0)
                self.assertEqual(original[:2], [3.0, 4.0])
                self.client.close.assert_not_called()

    def test_empty_or_invalid_text_does_not_call_provider(self) -> None:
        for text in ("", " \n\t"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                embed_text(self.client, text, "RETRIEVAL_DOCUMENT")
        for text in (None, 12, []):
            with self.subTest(text=text), self.assertRaises(TypeError):
                embed_text(self.client, text, "RETRIEVAL_DOCUMENT")
        self.client.models.embed_content.assert_not_called()

    def test_invalid_task_does_not_call_provider(self) -> None:
        for task in ("SEMANTIC_SIMILARITY", "", None, []):
            with self.subTest(task=task), self.assertRaises(ValueError):
                embed_text(self.client, "Evidence", task)
        self.client.models.embed_content.assert_not_called()

    def test_missing_or_multiple_embeddings_are_rejected(self) -> None:
        for embeddings in (None, [], [SimpleNamespace(), SimpleNamespace()]):
            with self.subTest(embeddings=embeddings):
                self.client.models.embed_content.return_value = SimpleNamespace(
                    embeddings=embeddings
                )
                with self.assertRaisesRegex(ValueError, "exactamente un embedding"):
                    embed_text(self.client, "Evidence", "RETRIEVAL_DOCUMENT")

    def test_missing_values_or_incorrect_dimension_are_rejected(self) -> None:
        for values in (None, [], [1.0] * 767, [1.0] * 769, "invalid"):
            with self.subTest(length=len(values) if values is not None else None):
                self.set_vector(values)
                with self.assertRaisesRegex(ValueError, "768 dimensiones"):
                    embed_text(self.client, "Evidence", "RETRIEVAL_DOCUMENT")

    def test_non_numeric_and_non_finite_values_are_rejected(self) -> None:
        for invalid in (True, "0.2", None, 1j, math.nan, math.inf, -math.inf):
            with self.subTest(value=invalid):
                self.set_vector([invalid] + [1.0] * 767)
                with self.assertRaisesRegex(ValueError, "números finitos"):
                    embed_text(self.client, "Evidence", "RETRIEVAL_QUERY")

    def test_zero_vector_and_unrepresentable_norm_are_rejected(self) -> None:
        for values in ([0.0] * 768, [sys.float_info.max] * 768):
            with self.subTest(first_value=values[0]):
                self.set_vector(values)
                with self.assertRaisesRegex(ValueError, "norma"):
                    embed_text(self.client, "Evidence", "RETRIEVAL_DOCUMENT")

    def test_integer_overflow_is_reported_as_validation_error(self) -> None:
        self.set_vector([10**1000] + [1.0] * 767)
        with self.assertRaisesRegex(ValueError, "rango numérico"):
            embed_text(self.client, "Evidence", "RETRIEVAL_DOCUMENT")

    def test_very_small_finite_values_can_be_normalized(self) -> None:
        self.set_vector([3e-300, 4e-300] + [0.0] * 766)
        result = embed_text(self.client, "Evidence", "RETRIEVAL_DOCUMENT")
        self.assertAlmostEqual(result[0], 0.6)
        self.assertAlmostEqual(result[1], 0.8)
        self.assertAlmostEqual(math.hypot(*result), 1.0)

    def test_provider_errors_are_propagated_without_extra_retries(self) -> None:
        for error in (
            errors.ClientError(429, {"error": {"message": "Quota exceeded"}}),
            errors.ServerError(503, {"error": {"message": "Unavailable"}}),
        ):
            with self.subTest(code=error.code):
                self.client.reset_mock()
                self.client.models.embed_content.side_effect = error
                with self.assertRaises(type(error)) as captured:
                    embed_text(self.client, "Evidence", "RETRIEVAL_QUERY")
                self.assertIs(captured.exception, error)
                self.client.models.embed_content.assert_called_once()


if __name__ == "__main__":
    unittest.main()
