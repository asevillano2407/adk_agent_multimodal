"""Tests de recuperación vectorial sin llamadas a modelos externos."""

import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

import chromadb

from retrieval.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from retrieval.search_vectors import search_chunks


VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


class TestSearchChunks(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()
        self.collection = MagicMock()
        self.collection.metadata = {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
        }
        self.collection.configuration = {"hnsw": {"space": "cosine"}}
        self.collection.count.return_value = 1
        self.origin = {
            "source_file": "annual.pdf", "pdf_page": 18,
            "start_char": 0, "end_char": 7,
        }
        self.collection.query.return_value = {
            "ids": [["chunk-18"]], "documents": [["Revenue"]],
            "metadatas": [[self.origin]], "distances": [[0.12]],
        }
        patcher = patch("retrieval.search_vectors.embed_text", return_value=VECTOR)
        self.embed = patcher.start()
        self.addCleanup(patcher.stop)

    def test_query_task_explicit_vector_and_preserved_provenance(self) -> None:
        hits = search_chunks(self.client, self.collection, "ingresos", top_k=3)
        self.embed.assert_called_once_with(self.client, "ingresos", "RETRIEVAL_QUERY")
        self.collection.query.assert_called_once_with(
            query_embeddings=[VECTOR], n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        self.assertEqual(hits, [{
            **self.origin, "chunk_id": "chunk-18", "text": "Revenue", "distance": 0.12,
        }])

    def test_invalid_inputs_fail_before_embedding(self) -> None:
        for query in ("", " \n "):
            with self.subTest(query=query), self.assertRaises(ValueError):
                search_chunks(self.client, self.collection, query)
        with self.assertRaises(TypeError):
            search_chunks(self.client, self.collection, None)
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                search_chunks(self.client, self.collection, "ingresos", k)
        self.embed.assert_not_called()

    def test_incompatible_configuration_fails_before_embedding(self) -> None:
        for field, value in (("embedding_model", "other"), ("embedding_dimensions", 3)):
            original = self.collection.metadata[field]
            self.collection.metadata[field] = value
            with self.assertRaises(ValueError):
                search_chunks(self.client, self.collection, "ingresos")
            self.collection.metadata[field] = original
        self.collection.configuration = {"hnsw": {"space": "l2"}}
        with self.assertRaises(ValueError):
            search_chunks(self.client, self.collection, "ingresos")
        self.embed.assert_not_called()

    def test_empty_collection_does_not_call_provider(self) -> None:
        self.collection.count.return_value = 0
        self.assertEqual(search_chunks(self.client, self.collection, "ingresos"), [])
        self.embed.assert_not_called()
        self.collection.query.assert_not_called()

    def test_invalid_returned_fields_are_not_silently_truncated(self) -> None:
        for field, value in (
            ("documents", None), ("documents", [[]]), ("ids", []),
            ("metadatas", [[None]]), ("distances", [[float("nan")]]),
        ):
            with self.subTest(field=field, value=value):
                result = self.collection.query.return_value
                original = result[field]
                result[field] = value
                with self.assertRaises(ValueError):
                    search_chunks(self.client, self.collection, "ingresos")
                result[field] = original

    def test_failures_propagate_without_extra_retries(self) -> None:
        error = RuntimeError("provider failure")
        self.embed.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            search_chunks(self.client, self.collection, "ingresos")
        self.assertIs(caught.exception, error)
        self.embed.assert_called_once()
        self.collection.query.assert_not_called()
        self.embed.side_effect = None
        self.collection.query.side_effect = error
        with self.assertRaises(RuntimeError):
            search_chunks(self.client, self.collection, "ingresos")

    def test_real_chroma_orders_by_cosine_distance_without_api(self) -> None:
        store = chromadb.EphemeralClient()
        name = f"test-search-{uuid4().hex}"
        collection = store.create_collection(
            name, embedding_function=None, metadata=self.collection.metadata,
            configuration={"hnsw": {"space": "cosine"}},
        )
        try:
            orthogonal = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)
            collection.upsert(
                ids=["near", "far"], embeddings=[VECTOR, orthogonal],
                documents=["Revenue", "Revenue"], metadatas=[self.origin, self.origin],
            )
            hits = search_chunks(self.client, collection, "ingresos", 2)
            self.assertEqual([hit["chunk_id"] for hit in hits], ["near", "far"])
            self.assertLess(hits[0]["distance"], hits[1]["distance"])
            self.assertEqual(collection.count(), 2)
        finally:
            store.delete_collection(name)


if __name__ == "__main__":
    unittest.main()
