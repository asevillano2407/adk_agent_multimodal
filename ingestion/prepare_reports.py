"""Prepara texto y chunks versionados del corpus con manifiestos visuales."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
from typing import Literal

from ingestion.export_chunks import export_chunks
from ingestion.export_pdf import write_jsonl
from ingestion.extract_pdf import extract_pages
from ingestion.file_utils import file_sha256
from retrieval.report_catalog import PROJECT_ROOT, load_report_catalog


logger = logging.getLogger(__name__)


def _fonttools_version() -> str:
    try:
        return version("fonttools")
    except PackageNotFoundError:
        return "not-installed"


def _prepare_document(
    pdf_path: Path,
    output_dir: Path,
    visual_manifest: dict,
    config: dict,
    dry_run: bool,
) -> dict:
    source_hash = file_sha256(pdf_path)
    if source_hash != visual_manifest["source_sha256"]:
        raise ValueError(f"El PDF {pdf_path.name} no coincide con el manifiesto visual.")
    header = {
        "source_file": pdf_path.name, "source_sha256": source_hash,
        "document_id": f"sha256:{source_hash}",
        "page_count": visual_manifest["page_count"], **config,
    }
    snapshot_dir = (
        output_dir / pdf_path.stem / source_hash
        / f"pypdf-{config['extractor_version']}"
        / f"fonttools-{config['fonttools_version']}"
    )
    tag = f"{config['chunk_strategy']}.c{config['chunk_size']}.o{config['overlap']}"
    pages_path = snapshot_dir / "pages.jsonl"
    chunks_path = snapshot_dir / f"chunks.{tag}.jsonl"
    manifest_path = snapshot_dir / f"prepare.{tag}.manifest.json"
    cached = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if not isinstance(cached, dict):
        raise ValueError(f"Manifiesto de preparación inválido: {pdf_path.name}")
    paths = {
        "pages_file": pages_path.relative_to(output_dir).as_posix(),
        "chunks_file": chunks_path.relative_to(output_dir).as_posix(),
    }
    compatible = (
        all(cached.get(key) == value for key, value in {**header, **paths}.items())
        and pages_path.is_file() and chunks_path.is_file()
        and file_sha256(pages_path) == cached.get("pages_sha256")
        and file_sha256(chunks_path) == cached.get("chunks_sha256")
    )
    if compatible:
        return {**cached, "status": "skipped"}
    if dry_run:
        return {**header, **paths, "status": "planned", "chunk_count": None}

    counts = {"page_count": 0, "empty_page_count": 0, "text_characters": 0}

    def pages():
        for page in extract_pages(pdf_path):
            counts["page_count"] += 1
            if page["source_file"] != pdf_path.name or page["pdf_page"] != counts["page_count"]:
                raise ValueError("La extracción cambió la identidad o el orden de las páginas.")
            counts["empty_page_count"] += not page["text"].strip()
            counts["text_characters"] += len(page["text"])
            yield {
                **page, "document_id": header["document_id"], "source_sha256": source_hash,
            }
            if counts["page_count"] % 25 == 0:
                logger.info("%s: extraídas %s páginas", pdf_path.name, counts["page_count"])
        if counts["page_count"] != header["page_count"]:
            raise ValueError("La extracción no coincide con el número de páginas del catálogo.")
        if file_sha256(pdf_path) != source_hash:
            raise ValueError("El PDF cambió durante la extracción; no se publica la preparación.")

    write_jsonl(pages(), pages_path)
    chunk_count = export_chunks(
        pages_path, chunks_path, config["chunk_size"], config["overlap"],
        strategy=config["chunk_strategy"],
    )
    with chunks_path.open(encoding="utf-8") as chunks:
        empty_chunks = sum(not json.loads(line)["text"].strip() for line in chunks if line.strip())
    prepared = {
        **header, **counts, **paths, "chunk_count": chunk_count,
        "empty_chunk_count": empty_chunks, "chunks_with_text": chunk_count - empty_chunks,
        "pages_sha256": file_sha256(pages_path), "chunks_sha256": file_sha256(chunks_path),
    }
    write_jsonl([prepared], manifest_path)
    return {**prepared, "status": "prepared"}


def prepare_reports(
    input_dir: Path,
    images_dir: Path,
    output_dir: Path,
    chunk_size: int = 1200,
    overlap: int = 200,
    strategy: Literal["fixed", "natural"] = "fixed",
    *,
    dry_run: bool = False,
) -> dict:
    """Prepara documentos del catálogo, reanudando por informe completo.

    Conserva páginas vacías y texto original. Un fallo se propaga; los informes
    previamente preparados se conservan. El manifiesto del corpus se publica
    solo al completar todos los informes y referencia rutas relativas y hashes.
    No usar dos procesos que escriban simultáneamente en el mismo destino.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size debe ser un entero positivo.")
    if type(overlap) is not int or not 0 <= overlap < chunk_size:
        raise ValueError("overlap debe ser un entero entre cero y chunk_size - 1.")
    if strategy not in ("fixed", "natural"):
        raise ValueError("Estrategia de chunking no reconocida.")
    if type(dry_run) is not bool:
        raise ValueError("dry_run debe ser booleano.")
    catalog = load_report_catalog(images_dir)
    sources = [input_dir / source for source in sorted(catalog)]
    for source in sources:
        if not source.resolve().is_relative_to(input_dir.resolve()):
            raise ValueError("Un documento apunta fuera de la carpeta de entrada.")
        if not source.is_file():
            raise FileNotFoundError(source)
    config = {
        "prepare_schema_version": 2, "extractor": "pypdf", "extractor_version": version("pypdf"),
        "fonttools_version": _fonttools_version(),
        "chunk_strategy": strategy, "chunk_size": chunk_size, "overlap": overlap,
    }
    documents = []
    for source in sources:
        record = _prepare_document(source, output_dir, catalog[source.name], config, dry_run)
        documents.append(record)
        logger.info("%s: %s; páginas=%s; chunks=%s", source.name, record["status"],
                    record["page_count"], record["chunk_count"])
    complete = not any(doc["status"] == "planned" for doc in documents)
    tag = f"{strategy}.c{chunk_size}.o{overlap}"
    manifest_path = output_dir / f"corpus.{tag}.manifest.json"
    report = {
        **config, "documents": documents,
        "prepared": sum(doc["status"] == "prepared" for doc in documents),
        "skipped": sum(doc["status"] == "skipped" for doc in documents),
        "planned": sum(doc["status"] == "planned" for doc in documents),
        "page_count": sum(doc["page_count"] for doc in documents),
        "chunk_count": sum(doc["chunk_count"] for doc in documents) if complete else None,
        "complete": complete, "manifest_file": str(manifest_path),
    }
    if not dry_run:
        write_jsonl([report], manifest_path)
    return report


def main() -> None:
    """Muestra el plan o prepara el corpus sin embeddings ni llamadas a Gemini."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/raw/nvidia")
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "data/processed/page_images")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed/report_corpus")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--strategy", choices=("fixed", "natural"), default="fixed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = prepare_reports(args.input_dir, args.images_dir, args.output_dir,
                             args.chunk_size, args.overlap, args.strategy, dry_run=args.dry_run)
    print("Modo:", "simulación sin escrituras" if args.dry_run else "preparación de texto y chunks")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
