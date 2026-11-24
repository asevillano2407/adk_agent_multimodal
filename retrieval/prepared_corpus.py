"""Carga íntegra y validada del corpus antes de generar embeddings."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from ingestion.chunk_pages import chunk_page
from ingestion.file_utils import file_sha256


@dataclass(frozen=True)
class PreparedCorpus:
    """Instantánea de chunks verificados e identidad estable de su preparación."""

    chunks: list[dict]
    config: dict
    corpus_sha256: str


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _count(record: dict, key: str, expected: int) -> None:
    if type(record.get(key)) is not int or record[key] != expected:
        raise ValueError(f"El recuento {key} no coincide con el contenido del corpus.")


def _records(root: Path, relative_path: str, expected_hash: str) -> list[dict]:
    if not isinstance(relative_path, str) or not _digest(expected_hash):
        raise ValueError("Referencia de archivo o hash no válidos.")
    path = (root / relative_path).resolve()
    if Path(relative_path).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Un archivo del corpus apunta fuera de su carpeta.")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"El hash no coincide: {path.name}")
    records = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Los registros JSONL deben ser objetos.")
    return records


def load_prepared_corpus(manifest_path: Path, source_dir: Path) -> PreparedCorpus:
    """Verifica todos los PDFs, hashes, identidades, offsets e IDs antes de usar IA.

    El corpus de este proyecto cabe en memoria. Se conserva una instantánea
    verificada para que cambios posteriores de archivos no alteren el lote.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise ValueError("Se requiere un manifiesto de corpus completo.")
    config = {key: manifest.get(key) for key in (
        "prepare_schema_version", "extractor", "extractor_version", "fonttools_version",
        "chunk_strategy", "chunk_size", "overlap",
    )}
    if (
        type(config["prepare_schema_version"]) is not int
        or config["prepare_schema_version"] != 2 or config["extractor"] != "pypdf"
        or any(not isinstance(config[key], str) or not config[key].strip()
               for key in ("extractor_version", "fonttools_version"))
        or config["chunk_strategy"] not in ("fixed", "natural")
        or type(config["chunk_size"]) is not int or config["chunk_size"] < 1
        or type(config["overlap"]) is not int
        or not 0 <= config["overlap"] < config["chunk_size"]
    ):
        raise ValueError("Configuración de preparación incompatible o inválida.")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("El corpus debe contener documentos preparados.")
    chunks, identities, seen_names, seen_ids = [], [], set(), set()
    total_pages = 0
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Cada documento debe ser un objeto.")
        source, source_hash = document.get("source_file"), document.get("source_sha256")
        if (
            not isinstance(source, str) or not source.strip()
            or any(char in source for char in ("/", "\\", ":", "\x00"))
            or Path(source).suffix.lower() != ".pdf" or not _digest(source_hash)
            or document.get("document_id") != f"sha256:{source_hash}"
            or source.casefold() in seen_names
            or any(document.get(key) != value for key, value in config.items())
        ):
            raise ValueError("Identidad o configuración del documento no válida.")
        seen_names.add(source.casefold())
        source_path = (source_dir / source).resolve()
        if not source_path.is_relative_to(source_dir.resolve()):
            raise ValueError("El PDF apunta fuera de la carpeta de entrada.")
        if file_sha256(source_path) != source_hash:
            raise ValueError(f"El PDF {source} cambió desde la preparación.")
        pages = _records(manifest_path.parent, document.get("pages_file"), document.get("pages_sha256"))
        stored = _records(manifest_path.parent, document.get("chunks_file"), document.get("chunks_sha256"))
        _count(document, "page_count", len(pages))
        if not pages:
            raise ValueError("Un documento debe conservar al menos una página.")
        expected = []
        for number, page in enumerate(pages, 1):
            if (
                page.get("source_file") != source or type(page.get("pdf_page")) is not int
                or page["pdf_page"] != number or page.get("source_sha256") != source_hash
                or page.get("document_id") != document["document_id"]
            ):
                raise ValueError("Una página no coincide con la identidad de su documento.")
            expected.extend(chunk_page(page, config["chunk_size"], config["overlap"], config["chunk_strategy"]))
        if stored != expected:
            raise ValueError("Los chunks no coinciden con páginas, offsets o configuración.")
        _count(document, "chunk_count", len(stored))
        _count(document, "empty_chunk_count", sum(not chunk["text"].strip() for chunk in stored))
        _count(document, "chunks_with_text", sum(bool(chunk["text"].strip()) for chunk in stored))
        for chunk in stored:
            if chunk["chunk_id"] in seen_ids:
                raise ValueError("El corpus contiene IDs de chunk duplicados.")
            seen_ids.add(chunk["chunk_id"])
        total_pages += len(pages)
        chunks.extend(stored)
        identities.append({key: document[key] for key in (
            "source_file", "document_id", "pages_sha256", "chunks_sha256",
        )})
    _count(manifest, "page_count", total_pages)
    _count(manifest, "chunk_count", len(chunks))
    stable = {"config": config, "documents": sorted(identities, key=lambda doc: doc["source_file"])}
    fingerprint = hashlib.sha256(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest()
    return PreparedCorpus(chunks=chunks, config=config, corpus_sha256=fingerprint)
