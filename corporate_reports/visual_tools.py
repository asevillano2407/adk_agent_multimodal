"""Acceso acotado a páginas PNG preparadas por la ingesta local."""

from io import BytesIO
import hashlib
import logging
from pathlib import Path

from google.genai import types
from PIL import Image
from retrieval.report_catalog import catalog_entries, load_report_catalog


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = "NVIDIA-2025-Annual-Report.pdf"
PAGE_DPI = 144
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 8_000_000
logger = logging.getLogger(__name__)


def list_report_documents() -> dict:
    """Lista nombres exactos, hashes y páginas de los informes preparados.

    El nombre orienta sobre el informe; el ejercicio fiscal debe verificarse
    en el contenido. Las páginas listas son las registradas en el manifiesto.
    """
    try:
        catalog = load_report_catalog(PROJECT_ROOT / "data/processed/page_images", PAGE_DPI)
        return {"status": "ok", "documents": catalog_entries(catalog)}
    except (OSError, ValueError) as error:
        logger.warning("Fallo del catálogo: %s", type(error).__name__)
        return {"status": "catalog_error", "documents": [],
                "message": "No se pudo leer un catálogo válido de documentos preparados."}


def read_report_page(pdf_page: int, source_file: str = SOURCE_FILE) -> dict:
    """Entrega una página visual de un documento listado por list_report_documents.

    Args:
        pdf_page: Posición en el PDF desde 1; no numeración impresa.
        source_file: Nombre exacto del catálogo. Por defecto el informe NVIDIA FY2025.

    Returns:
        Estado, documento, página y PNG como contenido multimodal no confiable.
        Usar para logos, posiciones, gráficos o tablas cuyo texto sea insuficiente.
        Requiere un PNG preparado localmente a 144 DPI. No renderiza el PDF.
        page_not_prepared significa que falta la imagen, no el dato solicitado.
    """
    if type(pdf_page) is not int or pdf_page < 1:
        return {
            "status": "invalid_page", "message": "La página PDF debe ser un entero positivo.",
        }
    if not isinstance(source_file, str) or not source_file.strip():
        return {"status": "unknown_document", "message": "Selecciona un nombre exacto del catálogo."}
    images_dir = PROJECT_ROOT / "data/processed/page_images"
    try:
        catalog = load_report_catalog(images_dir, PAGE_DPI)
    except (OSError, ValueError) as error:
        logger.warning("Fallo del catálogo: %s", type(error).__name__)
        return {"status": "catalog_error", "message": "No se pudo leer un catálogo válido."}
    if source_file not in catalog:
        return {"status": "unknown_document", "message": "El documento no está en el catálogo. Usa list_report_documents."}
    manifest = catalog[source_file]
    origin = {"source_file": source_file, "pdf_page": pdf_page, "dpi": PAGE_DPI,
              "document_id": f"sha256:{manifest['source_sha256']}"}
    if pdf_page > manifest["page_count"]:
        return {**origin, "status": "invalid_page", "message": "La página supera el tamaño de este documento."}
    record = manifest["pages"].get(str(pdf_page), {})
    if record.get("status") != "ready":
        return {**origin, "status": "page_not_prepared", "message": "La imagen no consta como preparada en el manifiesto."}
    image_path = images_dir / record["file"]
    source_path = PROJECT_ROOT / "data/raw/nvidia" / source_file
    try:
        if not source_path.resolve().is_relative_to((PROJECT_ROOT / "data/raw/nvidia").resolve()):
            raise ValueError("El documento apunta fuera del corpus.")
        if not image_path.resolve().is_relative_to(images_dir.resolve()):
            raise ValueError("La imagen apunta fuera de su carpeta.")
        if not source_path.is_file():
            return {**origin, "status": "source_unavailable", "message": "El PDF original no está disponible para verificar la imagen."}
        with source_path.open("rb") as source:
            source_hash = hashlib.file_digest(source, "sha256").hexdigest()
        if source_hash != manifest["source_sha256"]:
            return {**origin, "status": "source_changed", "message": "El PDF ha cambiado respecto a las imágenes preparadas. Requiere actualizar la ingesta."}
        with image_path.open("rb") as image_file:
            data = image_file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            return {**origin, "status": "image_too_large", "message": "El PNG supera el límite de 8 MiB."}
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["image_sha256"]:
            return {**origin, "status": "image_outdated", "message": "El PNG no coincide con el hash de su manifiesto."}
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("El archivo debe contener un PNG real.")
            width, height = image.size
            if width != record["width"] or height != record["height"]:
                raise ValueError("Las dimensiones no coinciden con el manifiesto.")
            if width * height > MAX_IMAGE_PIXELS:
                return {
                    **origin, "status": "image_too_large",
                    "message": "La imagen supera el límite de 8 millones de píxeles.",
                }
            image.verify()
        return {
            **origin, "status": "ok", "content_is_untrusted": True,
            "width": width, "height": height,
            "image": types.Part.from_bytes(data=data, mime_type="image/png"),
        }
    except FileNotFoundError:
        return {
            **origin, "status": "page_not_prepared",
            "message": "La imagen de esta página no está preparada. No deduzcas que el PDF carece del dato.",
        }
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        logger.warning("Fallo al cargar página visual: %s", type(error).__name__)
        return {
            **origin, "status": "visual_error",
            "message": "No se pudo validar o leer la imagen de la página.",
        }
