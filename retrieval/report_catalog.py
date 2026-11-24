"""Catálogo de informes derivado de los manifiestos de renderización locales."""

import argparse
import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_report_catalog(images_dir: Path, dpi: int = 144) -> dict[str, dict]:
    """Lee y valida manifiestos; no renderiza ni comprueba todos los PNGs.

    Los recuentos de páginas listas proceden del manifiesto. La herramienta
    de lectura verifica el PDF y el PNG concretos antes de entregar la imagen.
    No infiere empresa, ejercicio fiscal ni contenido a partir del nombre.
    """
    if type(dpi) is not int or not 72 <= dpi <= 300:
        raise ValueError("dpi debe ser un entero entre 72 y 300.")
    paths = sorted(images_dir.glob(f"*.{dpi}dpi.manifest.json"))
    if not paths:
        raise FileNotFoundError("No hay manifiestos para la resolución solicitada.")
    catalog = {}
    seen_names = set()
    for path in paths:
        if not path.resolve().is_relative_to(images_dir.resolve()):
            raise ValueError("Un manifiesto apunta fuera de la carpeta de imágenes.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Cada manifiesto debe contener un objeto JSON.")
        source = manifest.get("source_file")
        count = manifest.get("page_count")
        pages = manifest.get("pages")
        if (
            not isinstance(source, str) or not source.strip()
            or any(char in source for char in ("/", "\\", ":", "\x00"))
            or Path(source).suffix.lower() != ".pdf"
            or not _valid_hash(manifest.get("source_sha256"))
            or type(count) is not int or count < 1
            or type(manifest.get("dpi")) is not int or manifest["dpi"] != dpi
            or type(manifest.get("render_schema_version")) is not int
            or manifest["render_schema_version"] != 1
            or not isinstance(pages, dict)
        ):
            raise ValueError(f"Cabecera de manifiesto no válida: {path.name}")
        if path.name != f"{Path(source).stem}.{dpi}dpi.manifest.json":
            raise ValueError("El manifiesto no corresponde al nombre de su documento.")
        if source.casefold() in seen_names:
            raise ValueError("El catálogo contiene documentos con nombres duplicados.")
        seen_names.add(source.casefold())
        for number, record in pages.items():
            if (
                not number.isascii() or not number.isdecimal()
                or str(int(number)) != number or not 1 <= int(number) <= count
                or not isinstance(record, dict)
                or record.get("status") not in ("ready", "error")
            ):
                raise ValueError("Registro de página no válido en el manifiesto.")
            if record["status"] == "error":
                continue
            if (
                record.get("file") != f"{Path(source).stem}.p{number}.{dpi}dpi.png"
                or not _valid_hash(record.get("image_sha256"))
                or any(type(record.get(key)) is not int or record[key] < 1
                       for key in ("bytes", "width", "height"))
            ):
                raise ValueError("Imagen con metadatos no válidos en el manifiesto.")
        catalog[source] = manifest
    return catalog


def catalog_entries(catalog: dict[str, dict]) -> list[dict]:
    """Expone identidad y disponibilidad, sin rutas locales ni contenido del PDF."""
    return [
        {
            "source_file": source,
            "document_id": f"sha256:{manifest['source_sha256']}",
            "page_count": manifest["page_count"],
            "ready_pages_in_manifest": sum(
                page["status"] == "ready" for page in manifest["pages"].values()
            ),
            "dpi": manifest["dpi"],
        }
        for source, manifest in sorted(catalog.items())
    ]


def main() -> None:
    """Lista el catálogo sin convertir documentos ni llamar a modelos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", type=Path,
        default=PROJECT_ROOT / "data/processed/page_images",
    )
    args = parser.parse_args()
    entries = catalog_entries(load_report_catalog(args.images_dir))
    print(json.dumps({"documents": entries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
