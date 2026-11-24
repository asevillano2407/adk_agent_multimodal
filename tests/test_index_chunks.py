"""Offline tests for incremental indexing, limits and partial progress."""

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from uuid import uuid4

import chromadb
from google.genai import errors

from ingestion.index_chunks import index_chunks
from retrieval.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL


VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


def chunk(number: int, text: str = "Revenue") -> dict:
    return {
        "chunk_id": f"annual.pdf:p{number}:start0",
        "source_file": "annual.pdf", "pdf_page": number,
        "start_char": 0, "end_char": len(text), "text": text,
    }


def metadata(record: dict) -> dict:
    return {key: record[key] for key in ("source_file", "pdf_page", "start_char", "end_char")}


class FakeCollection:
    """Small stateful substitute; no embedding generation or network access."""

    def __init__(self) -> None:
        self.metadata = {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "chunk_strategy": "fixed",
        }
        self.configuration = {"hnsw": {"space": "cosine"}}
        self.records = {}

    def get(self, *, ids, include):
        if ids[0] not in self.records:
            return {"ids": [], "documents": [], "metadatas": []}
        stored = self.records[ids[0]]
        return {
            "ids": ids, "documents": [stored["text"]],
            "metadatas": [stored["metadata"]],
        }

    def upsert(self, *, ids, embeddings, documents, metadatas):
        self.records[ids[0]] = {
            "text": documents[0], "metadata": deepcopy(metadatas[0]),
            "vector": list(embeddings[0]),
        }

    def update(self, *, ids, metadatas):
        self.records[ids[0]]["metadata"].update(metadatas[0])


class TestIndexChunks(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "chunks.jsonl"
        self.collection = FakeCollection()
        self.client = MagicMock()
        sleeper = patch("ingestion.index_chunks.time.sleep")
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)

    def test_interval_only_applies_between_required_embeddings(self) -> None:
        self.store(chunk(1))
        self.write([chunk(1), chunk(2), chunk(3), chunk(4)])
        with patch("ingestion.index_chunks.embed_text", return_value=VECTOR):
            result = index_chunks(
                self.client, self.collection, self.path, request_interval=1.5
            )
        self.assertEqual(result["saved"], 3)
        self.assertEqual(self.sleep.call_count, 2)
        self.sleep.assert_called_with(1.5)

    def test_dry_run_and_zero_interval_never_sleep(self) -> None:
        self.write([chunk(1), chunk(2)])
        index_chunks(None, self.collection, self.path, dry_run=True)
        with patch("ingestion.index_chunks.embed_text", return_value=VECTOR):
            index_chunks(self.client, self.collection, self.path, request_interval=0)
        self.sleep.assert_not_called()

    def test_invalid_interval_fails_before_embedding(self) -> None:
        with patch("ingestion.index_chunks.embed_text") as embed:
            for interval in (-1, float("nan"), float("inf"), True, "1", None):
                with self.subTest(interval=interval), self.assertRaises(ValueError):
                    index_chunks(
                        self.client, self.collection, self.path,
                        request_interval=interval,
                    )
            embed.assert_not_called()

    def write(self, records) -> None:
        self.path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    def store(self, record: dict) -> None:
        self.collection.upsert(
            ids=[record["chunk_id"]], embeddings=[VECTOR],
            documents=[record["text"]], metadatas=[metadata(record)],
        )

    def test_identical_record_is_skipped_without_embedding(self) -> None:
        record = chunk(1)
        self.write([record])
        self.store(record)
        with patch("ingestion.index_chunks.embed_text") as embed:
            result = index_chunks(self.client, self.collection, self.path)
        embed.assert_not_called()
        self.assertEqual(result, {
            "reviewed": 1, "skipped": 1, "planned": 0, "saved": 0,
            "metadata_updated": 0, "empty_skipped": 0,
        })

    def test_whitespace_chunk_is_counted_without_embedding_or_write(self) -> None:
        self.write([chunk(1, " \t\n "), chunk(2)])
        with patch("ingestion.index_chunks.embed_text", return_value=VECTOR) as embed:
            result = index_chunks(self.client, self.collection, self.path, 1)
        embed.assert_called_once_with(self.client, "Revenue", "RETRIEVAL_DOCUMENT")
        self.assertEqual(result["empty_skipped"], 1)
        self.assertEqual(result["reviewed"], 2)
        self.assertEqual(result["saved"], 1)
        self.assertNotIn(chunk(1)["chunk_id"], self.collection.records)
        self.sleep.assert_not_called()

    def test_whitespace_chunk_still_requires_valid_offsets(self) -> None:
        self.write([{**chunk(1, "   "), "end_char": 99}])
        with patch("ingestion.index_chunks.embed_text") as embed:
            with self.assertRaises(ValueError):
                index_chunks(self.client, self.collection, self.path)
        embed.assert_not_called()

    def test_whitespace_chunk_is_also_omitted_in_dry_run(self) -> None:
        self.write([chunk(1, " \n ")])
        with patch("ingestion.index_chunks.embed_text") as embed:
            result = index_chunks(None, self.collection, self.path, dry_run=True)
        self.assertEqual(result["empty_skipped"], 1)
        self.assertEqual(result["planned"], 0)
        self.assertFalse(self.collection.records)
        embed.assert_not_called()

    def test_limit_and_resume_do_not_repeat_saved_embeddings(self) -> None:
        self.write([chunk(1), chunk(2), chunk(3)])
        with patch("ingestion.index_chunks.embed_text", return_value=VECTOR) as embed:
            first = index_chunks(self.client, self.collection, self.path, 2)
            self.assertEqual(embed.call_count, 2)
            embed.assert_called_with(self.client, "Revenue", "RETRIEVAL_DOCUMENT")
            second = index_chunks(self.client, self.collection, self.path, 2)
        self.assertEqual(first["saved"], 2)
        self.assertEqual(second["skipped"], 2)
        self.assertEqual(second["saved"], 1)
        self.assertEqual(embed.call_count, 3)
        self.assertEqual(len(self.collection.records), 3)

    def test_changed_text_is_reembedded_under_same_id(self) -> None:
        self.store(chunk(1, "Old"))
        new = chunk(1, "Changed content")
        self.write([new])
        with patch("ingestion.index_chunks.embed_text", return_value=VECTOR) as embed:
            result = index_chunks(self.client, self.collection, self.path)
        embed.assert_called_once_with(self.client, new["text"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(result["saved"], 1)
        self.assertEqual(len(self.collection.records), 1)
        self.assertEqual(self.collection.records[new["chunk_id"]]["text"], new["text"])
        self.assertEqual(self.collection.records[new["chunk_id"]]["metadata"], metadata(new))

    def test_metadata_is_refreshed_without_reembedding_identical_text(self) -> None:
        record = chunk(1)
        self.store(record)
        self.collection.records[record["chunk_id"]]["metadata"]["end_char"] = 999
        self.write([record])
        with patch("ingestion.index_chunks.embed_text") as embed:
            result = index_chunks(self.client, self.collection, self.path)
        embed.assert_not_called()
        self.assertEqual(result["metadata_updated"], 1)
        self.assertEqual(self.collection.records[record["chunk_id"]]["metadata"], metadata(record))

    def test_dry_run_neither_embeds_nor_writes(self) -> None:
        record = chunk(1)
        self.store(record)
        self.collection.records[record["chunk_id"]]["metadata"] = {}
        self.write([record, chunk(2), chunk(3)])
        before = deepcopy(self.collection.records)
        with patch("ingestion.index_chunks.embed_text") as embed:
            result = index_chunks(None, self.collection, self.path, 1, dry_run=True)
        embed.assert_not_called()
        self.assertEqual(self.collection.records, before)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["planned"], 1)
        self.assertEqual(result["saved"], 0)

    def test_provider_failure_preserves_progress_and_allows_resume(self) -> None:
        self.write([chunk(1), chunk(2)])
        error = errors.ClientError(429, {"error": {"message": "Quota exceeded"}})
        with patch("ingestion.index_chunks.embed_text", side_effect=[VECTOR, error]) as embed:
            with self.assertRaises(errors.ClientError) as captured:
                index_chunks(self.client, self.collection, self.path)
        self.assertIs(captured.exception, error)
        self.assertEqual(embed.call_count, 2)
        self.assertEqual(list(self.collection.records), [chunk(1)["chunk_id"]])
        with patch("ingestion.index_chunks.embed_text", return_value=VECTOR) as embed:
            result = index_chunks(self.client, self.collection, self.path)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["saved"], 1)
        embed.assert_called_once()

    def test_collection_mismatch_prevents_provider_calls(self) -> None:
        self.write([chunk(1)])
        for field, value in (
            ("embedding_model", "different-model"),
            ("embedding_dimensions", 3072),
            ("chunk_strategy", "natural"),
        ):
            with self.subTest(field=field):
                original = self.collection.metadata[field]
                self.collection.metadata[field] = value
                with patch("ingestion.index_chunks.embed_text") as embed:
                    with self.assertRaises(ValueError):
                        index_chunks(self.client, self.collection, self.path)
                    embed.assert_not_called()
                self.collection.metadata[field] = original
        self.collection.configuration["hnsw"]["space"] = "l2"
        with self.assertRaisesRegex(ValueError, "cosine"):
            index_chunks(self.client, self.collection, self.path)

    def test_invalid_limits_and_missing_client(self) -> None:
        for limit in (0, -1, True, 1.5, None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                index_chunks(self.client, self.collection, self.path, limit)
        with self.assertRaisesRegex(ValueError, "cliente"):
            index_chunks(None, self.collection, self.path)

    def test_invalid_chunk_or_duplicate_id_is_not_silently_accepted(self) -> None:
        for invalid in (
            {}, None, {**chunk(1), "text": ""},
            {**chunk(1), "end_char": 999},
            {**chunk(1), "pdf_page": True},
            {**chunk(1), "chunk_id": "annual.pdf:p1:natural:start0"},
        ):
            with self.subTest(record=invalid):
                self.write([invalid])
                with self.assertRaises(ValueError):
                    index_chunks(None, self.collection, self.path, dry_run=True)
        self.write([chunk(1), chunk(1)])
        with self.assertRaisesRegex(ValueError, "duplicado"):
            index_chunks(None, self.collection, self.path, dry_run=True)

    def test_empty_file_missing_file_and_malformed_json(self) -> None:
        with self.assertRaises(FileNotFoundError):
            index_chunks(None, self.collection, self.path, dry_run=True)
        self.path.write_text("\n", encoding="utf-8")
        self.assertEqual(index_chunks(None, self.collection, self.path, dry_run=True)["reviewed"], 0)
        self.path.write_text("\n{broken}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "línea 2"):
            index_chunks(None, self.collection, self.path, dry_run=True)

    def test_works_with_real_chroma_and_explicit_simulated_vector(self) -> None:
        store = chromadb.EphemeralClient()
        name = f"test-index-{uuid4().hex}"
        collection = store.create_collection(
            name, embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata=self.collection.metadata,
        )
        try:
            record = chunk(1)
            self.write([record])
            with patch("ingestion.index_chunks.embed_text", return_value=VECTOR) as embed:
                first = index_chunks(self.client, collection, self.path)
                second = index_chunks(self.client, collection, self.path)
            embed.assert_called_once()
            stored = collection.get(ids=[record["chunk_id"]], include=["documents", "metadatas", "embeddings"])
            self.assertEqual(first["saved"], 1)
            self.assertEqual(second["skipped"], 1)
            self.assertEqual(collection.count(), 1)
            self.assertEqual(stored["documents"], [record["text"]])
            self.assertEqual(stored["metadatas"], [metadata(record)])
            self.assertEqual(len(stored["embeddings"][0]), 768)
        finally:
            store.delete_collection(name)


if __name__ == "__main__":
    unittest.main()
