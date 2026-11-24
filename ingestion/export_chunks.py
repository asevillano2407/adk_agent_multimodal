"""Exporta ventanas de caracteres desde el índice por páginas, sin modelos."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from ingestion.chunk_pages import chunk_page
from ingestion.export_pdf import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def export_chunks(
    pages_path: Path,
    output_path: Path,
    chunk_size: int = 1200,
    overlap: int = 200,
    strategy: Literal["fixed", "natural"] = "fixed",
) -> int:
    """Publica chunks en un JSONL distinto del índice por páginas."""
    if pages_path.resolve() == output_path.resolve():
        raise ValueError("El destino debe ser distinto del índice por páginas.")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size debe ser un entero mayor que cero.")
    if type(overlap) is not int or not 0 <= overlap < chunk_size:
        raise ValueError("overlap debe ser un entero entre cero y chunk_size - 1.")
    if strategy not in ("fixed", "natural"):
        raise ValueError("strategy debe ser 'fixed' o 'natural'.")

    def records() -> Iterator[dict]:
        with pages_path.open(encoding="utf-8") as pages:
            for line_number, line in enumerate(pages, start=1):
                if not line.strip():
                    continue
                try:
                    page = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSON no válido en la línea {line_number} del índice."
                    ) from error
                yield from chunk_page(page, chunk_size, overlap, strategy=strategy)

    return write_jsonl(records(), output_path)


def main() -> None:
    """Exporta el índice local con tamaño y solapamiento explícitos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pages", type=Path,
        default=PROJECT_ROOT / "data/processed/nvidia_2025.pages.jsonl",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--strategy", choices=("fixed", "natural"), default="fixed")
    args = parser.parse_args()
    output_path = args.output
    if output_path is None:
        name = args.pages.name.removesuffix(".pages.jsonl")
        strategy_tag = ".natural" if args.strategy == "natural" else ""
        output_path = args.pages.with_name(
            f"{name}.chunks{strategy_tag}.c{args.chunk_size}.o{args.overlap}.jsonl"
        )

    count = export_chunks(
        args.pages, output_path, args.chunk_size, args.overlap, strategy=args.strategy
    )
    print(f"Exportados {count} chunks a {output_path}")
    print(f"Caracteres por chunk: {args.chunk_size}; solapamiento: {args.overlap}")
    print(f"Estrategia: {args.strategy}")


if __name__ == "__main__":
    main()
