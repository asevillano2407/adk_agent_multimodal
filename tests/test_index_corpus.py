"""Corpus sintético y Chroma real; embeddings simulados sin llamadas externas."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

import chromadb
from chromadb.errors import NotFoundError
from google.genai import errors

from ingestion.chunk_pages import chunk_page
from ingestion.export_pdf import write_jsonl
from ingestion.file_utils import file_sha256
from ingestion.index_corpus import _reuse_vectors, index_corpus
from retrieval.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from retrieval.prepared_corpus import load_prepared_corpus
from retrieval.search_vectors import search_chunks


VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


class TestIndexCorpus(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = self.root / "raw"
        self.sources.mkdir()
        self.corpus_dir = self.root / "corpus"
        self.corpus_dir.mkdir()
        config = {
            "prepare_schema_version": 2, "extractor": "pypdf", "extractor_version": "6.21.4",
            "fonttools_version": "4.68.0", "chunk_strategy": "fixed", "chunk_size": 1200, "overlap": 200,
        }
        documents = []
        for filename, texts in (
            ("NVIDIA-2025-Annual-Report.pdf", ["Revenue 116.2", "   "]),
            ("annual2024.pdf", ["Other revenue", "More evidence\u2028in the same JSON line"]),
        ):
            source = self.sources / filename
            source.write_bytes(f"synthetic source {filename}".encode())
            digest = file_sha256(source)
            pages = [{"source_file": filename, "pdf_page": i, "text": text,
                      "source_sha256": digest, "document_id": f"sha256:{digest}"}
                     for i, text in enumerate(texts, 1)]
            chunks = [chunk for page in pages for chunk in chunk_page(page)]
            pages_path = self.corpus_dir / f"{source.stem}.pages.jsonl"
            chunks_path = self.corpus_dir / f"{source.stem}.chunks.jsonl"
            write_jsonl(pages, pages_path)
            write_jsonl(chunks, chunks_path)
            documents.append({
                **config, "source_file": filename, "source_sha256": digest,
                "document_id": f"sha256:{digest}", "page_count": len(pages),
                "chunk_count": len(chunks), "empty_chunk_count": sum(not c["text"].strip() for c in chunks),
                "chunks_with_text": sum(bool(c["text"].strip()) for c in chunks),
                "pages_file": pages_path.name, "chunks_file": chunks_path.name,
                "pages_sha256": file_sha256(pages_path), "chunks_sha256": file_sha256(chunks_path),
                "status": "prepared",
            })
        self.manifest = {**config, "complete": True, "documents": documents,
                         "page_count": 4, "chunk_count": 4}
        self.manifest_path = self.corpus_dir / "corpus.manifest.json"
        self.save_manifest()
        self.corpus = load_prepared_corpus(self.manifest_path, self.sources)
        self.legacy_name = f"test-legacy-{uuid4().hex}"
        legacy_name_patch = patch("ingestion.index_corpus.LEGACY_COLLECTION", self.legacy_name)
        legacy_name_patch.start()
        self.addCleanup(legacy_name_patch.stop)
        self.store = chromadb.PersistentClient(path=str(self.root / "chroma"))
        self.target = f"test-corpus-{uuid4().hex}"
        self.legacy = self.store.create_collection(
            self.legacy_name, embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": EMBEDDING_MODEL, "embedding_dimensions": EMBEDDING_DIMENSIONS,
                      "chunk_strategy": "fixed"},
        )
        self.addCleanup(self.cleanup_collections)
        chunk = self.corpus.chunks[0]
        self.legacy.upsert(
            ids=[f"{chunk['source_file']}:p1:start0"], documents=[chunk["text"]], embeddings=[VECTOR],
            metadatas=[{key: chunk[key] for key in ("source_file", "pdf_page", "start_char", "end_char")}],
        )
        self.client = MagicMock()
        patcher = patch("ingestion.index_corpus.embed_text", return_value=VECTOR)
        self.embed = patcher.start()
        self.addCleanup(patcher.stop)
        sleeper = patch("ingestion.index_corpus.time.sleep")
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)

    def cleanup_collections(self):
        try:
            for name in (self.legacy_name, self.target):
                try:
                    self.store.delete_collection(name)
                except NotFoundError:
                    pass
        finally:
            # Liberar los archivos de la instancia aislada antes del cleanup
            # de TemporaryDirectory, especialmente en Windows.
            self.store._system.stop()

    def save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def index(self, client=None, **kwargs):
        return index_corpus(client, self.store, self.corpus, collection_name=self.target, **kwargs)

    def target_collection(self):
        return self.store.get_collection(self.target, embedding_function=None)

    def test_dry_run_plans_reuse_and_api_without_creating_collection(self) -> None:
        report = self.index(dry_run=True)
        self.assertEqual(report["reviewed"], 4)
        self.assertEqual(report["empty_skipped"], 1)
        self.assertEqual(report["reusable_embeddings"], 1)
        self.assertEqual(report["new_embeddings_needed"], 2)
        self.assertEqual(report["saved"], 0)
        self.assertEqual(self.legacy.count(), 1)
        with self.assertRaises(NotFoundError):
            self.target_collection()
        self.embed.assert_not_called()

    def test_empty_legacy_collection_requires_all_nonempty_chunks(self) -> None:
        self.legacy.delete(ids=self.legacy.get()["ids"])
        report = self.index(dry_run=True)
        self.assertEqual(report["reusable_embeddings"], 0)
        self.assertEqual(report["new_embeddings_needed"], 3)
        self.embed.assert_not_called()

    def test_zero_limit_copies_then_resumes_without_duplicate_embeddings(self) -> None:
        first = self.index(max_new_embeddings=0)
        self.assertEqual(first["reused"], 1)
        self.assertEqual(first["embedded"], 0)
        self.assertEqual(first["pending"], 2)
        self.assertFalse(self.target_collection().metadata["index_complete"])
        self.embed.assert_not_called()
        second = self.index(self.client, max_new_embeddings=1)
        self.assertEqual(second["already_indexed"], 1)
        self.assertEqual(second["embedded"], 1)
        third = self.index(self.client, max_new_embeddings=10)
        self.assertEqual(third["already_indexed"], 2)
        self.assertTrue(third["complete"])
        self.assertEqual(self.target_collection().count(), 3)
        self.assertTrue(self.target_collection().metadata["index_complete"])
        self.assertEqual(self.embed.call_count, 2)
        self.assertEqual(self.legacy.count(), 1)
        self.assertNotIn("index_complete", self.legacy.metadata)
        self.embed.reset_mock()
        final = self.index(self.client)
        self.assertEqual(final["saved"], 0)
        self.embed.assert_not_called()

    def test_changed_cached_text_is_not_reused(self) -> None:
        legacy_id = self.legacy.get()["ids"][0]
        self.legacy.update(ids=[legacy_id], documents=["Different text"], embeddings=[VECTOR])
        # Metadatos coherentes con el texto cambiado: ya no es una coincidencia.
        self.legacy.update(ids=[legacy_id], metadatas=[{"end_char": len("Different text")}])
        report = self.index(dry_run=True)
        self.assertEqual(report["reusable_embeddings"], 0)
        self.assertEqual(report["new_embeddings_needed"], 3)

    def test_failed_provider_preserves_saved_vectors_and_incomplete_status(self) -> None:
        error = errors.ClientError(429, {"error": {"message": "quota"}})
        self.embed.side_effect = [VECTOR, error]
        with self.assertRaises(errors.ClientError):
            self.index(self.client, max_new_embeddings=5)
        self.assertEqual(self.target_collection().count(), 2)
        self.assertFalse(self.target_collection().metadata["index_complete"])
        self.embed.side_effect = None
        report = self.index(self.client, max_new_embeddings=5)
        self.assertEqual(report["embedded"], 1)
        self.assertTrue(report["complete"])

    def test_collection_mismatch_or_same_destination_fails_before_api(self) -> None:
        self.index(max_new_embeddings=0)
        self.target_collection().modify(metadata={**self.target_collection().metadata, "text_corpus_sha256": "different"})
        with self.assertRaises(ValueError):
            self.index(self.client)
        with self.assertRaises(ValueError):
            index_corpus(None, self.store, self.corpus, collection_name=self.legacy_name, dry_run=True)
        self.embed.assert_not_called()

    def test_wrong_metadata_in_existing_target_is_not_silently_accepted(self) -> None:
        self.index(max_new_embeddings=0)
        target = self.target_collection()
        target.update(ids=target.get()["ids"], metadatas=[{"document_id": "wrong"}])
        with self.assertRaisesRegex(ValueError, "registro"):
            self.index(self.client)
        self.embed.assert_not_called()

    def test_invalid_reuse_vectors_fail_before_writes(self) -> None:
        source = MagicMock()
        source.name = self.legacy_name
        source.metadata = self.legacy.metadata
        source.configuration = self.legacy.configuration
        rows = self.legacy.get(include=["documents", "metadatas", "embeddings"])
        for vector in ([0.0] * 768, [float("nan")] * 768, [1.0], [2.0] + [0.0] * 767):
            with self.subTest(vector_start=vector[:1]):
                source.get.return_value = {**rows, "embeddings": [vector]}
                with self.assertRaises(ValueError):
                    _reuse_vectors(source)

    def test_incomplete_collection_is_not_searchable(self) -> None:
        self.index(max_new_embeddings=0)
        with patch("retrieval.search_vectors.embed_text") as query:
            with self.assertRaisesRegex(ValueError, "incompleta"):
                search_chunks(self.client, self.target_collection(), "revenue")
        query.assert_not_called()

    def test_document_and_version_filters_apply_inside_chroma(self) -> None:
        self.index(self.client, max_new_embeddings=5)
        target = self.target_collection()
        other = self.manifest["documents"][1]
        with patch("retrieval.search_vectors.embed_text", return_value=VECTOR):
            hits = search_chunks(self.client, target, "revenue", source_file="annual2024.pdf")
            self.assertEqual(len(hits), 2)
            self.assertTrue(all(h["source_file"] == "annual2024.pdf" for h in hits))
            self.assertTrue(all(h["document_id"] == other["document_id"] for h in hits))
            hits = search_chunks(self.client, target, "revenue", document_id=other["document_id"])
            self.assertEqual(len(hits), 2)
            empty = search_chunks(self.client, target, "revenue", source_file="annual2024.pdf",
                                  document_id=self.manifest["documents"][0]["document_id"])
            self.assertEqual(empty, [])

    def test_manifest_checks_detect_tampering_and_unsafe_paths(self) -> None:
        original = deepcopy(self.manifest)
        for field, value in (("complete", False), ("page_count", 99), ("chunk_count", 1)):
            with self.subTest(field=field):
                self.manifest = {**original, field: value}
                self.save_manifest()
                with self.assertRaises(ValueError):
                    load_prepared_corpus(self.manifest_path, self.sources)
        self.manifest = deepcopy(original)
        self.manifest["documents"][0]["chunks_file"] = "../escape.jsonl"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "fuera"):
            load_prepared_corpus(self.manifest_path, self.sources)

    def test_semantic_chunk_tampering_is_rejected_even_with_updated_file_hash(self) -> None:
        document = self.manifest["documents"][1]
        path = self.corpus_dir / document["chunks_file"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
        rows[0]["start_char"] = 1
        write_jsonl(rows, path)
        document["chunks_sha256"] = file_sha256(path)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "offsets"):
            load_prepared_corpus(self.manifest_path, self.sources)

    def test_source_change_or_file_hash_change_is_rejected(self) -> None:
        source = self.sources / self.manifest["documents"][0]["source_file"]
        source.write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "cambió"):
            load_prepared_corpus(self.manifest_path, self.sources)

    def test_fingerprint_ignores_run_status_but_not_extraction_config(self) -> None:
        for doc in self.manifest["documents"]:
            doc["status"] = "skipped"
        self.save_manifest()
        same = load_prepared_corpus(self.manifest_path, self.sources)
        self.assertEqual(same.corpus_sha256, self.corpus.corpus_sha256)
        self.manifest["fonttools_version"] = "different"
        for doc in self.manifest["documents"]:
            doc["fonttools_version"] = "different"
        self.save_manifest()
        different = load_prepared_corpus(self.manifest_path, self.sources)
        self.assertNotEqual(different.corpus_sha256, self.corpus.corpus_sha256)

    def test_invalid_limits_do_not_call_provider_or_create_target(self) -> None:
        for options in ({"max_new_embeddings": -1}, {"max_new_embeddings": True},
                        {"request_interval": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.index(self.client, **options)
        with self.assertRaises(NotFoundError):
            self.target_collection()
        self.embed.assert_not_called()
