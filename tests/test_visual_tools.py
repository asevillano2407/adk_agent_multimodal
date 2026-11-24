"""PNG sintéticos y transporte multimodal real de ADK, sin llamadas a modelos."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from google.adk.flows.llm_flows.tools._caller import _build_function_response_content
from google.adk.tools import FunctionTool
from PIL import Image

from corporate_reports.agent import root_agent
from corporate_reports.visual_tools import list_report_documents, read_report_page


class TestVisualTools(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        patcher = patch("corporate_reports.visual_tools.PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.path = self.root / "data/processed/page_images/NVIDIA-2025-Annual-Report.p5.144dpi.png"
        self.path.parent.mkdir(parents=True)
        with Image.new("RGB", (2, 2), "white") as image:
            image.save(self.path, format="PNG")
        self.source = self.root / "data/raw/nvidia/NVIDIA-2025-Annual-Report.pdf"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"original PDF fixture")
        self.write_manifest()

    def write_manifest(self) -> None:
        record = {
            "status": "ready", "file": self.path.name,
            "bytes": self.path.stat().st_size,
            "image_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "width": 2, "height": 2,
        }
        manifest = {
            "source_file": self.source.name,
            "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "page_count": 181, "dpi": 144, "render_schema_version": 1,
            "pages": {"5": record},
        }
        self.manifest_path = self.path.parent / "NVIDIA-2025-Annual-Report.144dpi.manifest.json"
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_schema_adds_optional_document_and_preserves_old_page_calls(self) -> None:
        self.assertIn(read_report_page, root_agent.tools)
        declaration = FunctionTool(read_report_page)._get_declaration()
        schema = declaration.parameters_json_schema or declaration.parameters.model_dump()
        self.assertEqual(set(schema["properties"]), {"pdf_page", "source_file"})
        self.assertEqual(schema["required"], ["pdf_page"])

    def test_actual_adk_response_contains_image_bytes_and_provenance(self) -> None:
        result = read_report_page(5)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["content_is_untrusted"])
        content = _build_function_response_content(FunctionTool(read_report_page), result, "call-image")
        response = content.parts[0].function_response
        self.assertEqual(response.id, "call-image")
        self.assertEqual(response.response["pdf_page"], 5)
        self.assertEqual(response.response["source_file"], "NVIDIA-2025-Annual-Report.pdf")
        self.assertNotIn("image", response.response)
        self.assertEqual(len(response.parts), 1)
        self.assertEqual(response.parts[0].inline_data.mime_type, "image/png")
        self.assertEqual(response.parts[0].inline_data.data, self.path.read_bytes())

    def test_invalid_pages_cannot_select_arbitrary_files(self) -> None:
        for page in (0, -1, 182, True, 5.0, "../../secret", None):
            with self.subTest(page=page):
                self.assertEqual(read_report_page(page)["status"], "invalid_page")

    def test_missing_page_is_distinct_from_absent_document_evidence(self) -> None:
        result = read_report_page(18)
        self.assertEqual(result["status"], "page_not_prepared")
        self.assertNotIn("image", result)

    def test_byte_and_pixel_limits_exclude_image_from_response(self) -> None:
        with patch("corporate_reports.visual_tools.MAX_IMAGE_BYTES", 1):
            self.assertEqual(read_report_page(5)["status"], "image_too_large")
        with patch("corporate_reports.visual_tools.MAX_IMAGE_PIXELS", 1):
            self.assertEqual(read_report_page(5)["status"], "image_too_large")

    def test_corrupt_or_non_png_file_is_an_explicit_error(self) -> None:
        for payload in (b"broken image", self.jpeg_bytes()):
            with self.subTest(payload=payload[:10]):
                self.path.write_bytes(payload)
                self.write_manifest()
                with self.assertLogs("corporate_reports.visual_tools", level="WARNING"):
                    result = read_report_page(5)
                self.assertEqual(result["status"], "visual_error")
                self.assertNotIn("image", result)

    def test_modified_image_is_rejected_before_being_sent(self) -> None:
        self.path.write_bytes(b"modified image")
        result = read_report_page(5)
        self.assertEqual(result["status"], "image_outdated")
        self.assertNotIn("image", result)

    def test_changed_or_missing_source_pdf_blocks_image(self) -> None:
        self.source.write_bytes(b"different PDF version")
        self.assertEqual(read_report_page(5)["status"], "source_changed")
        self.source.unlink()
        self.assertEqual(read_report_page(5)["status"], "source_unavailable")

    def test_unknown_document_and_path_traversal_never_return_an_image(self) -> None:
        for name in ("missing.pdf", "../secret.pdf", "C:/secret.pdf", None):
            with self.subTest(name=name):
                result = read_report_page(5, name)
                self.assertEqual(result["status"], "unknown_document")
                self.assertNotIn("image", result)

    def test_catalog_lists_only_identity_and_manifest_availability(self) -> None:
        self.assertIn(list_report_documents, root_agent.tools)
        result = list_report_documents()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["documents"][0]["ready_pages_in_manifest"], 1)
        self.assertEqual(result["documents"][0]["page_count"], 181)
        self.assertTrue(result["documents"][0]["document_id"].startswith("sha256:"))

    def test_second_document_is_selected_explicitly_without_crossing_pages(self) -> None:
        name = "2024-annual-report-insert-web-IR-update.pdf"
        other_source = self.source.parent / name
        other_source.write_bytes(b"other PDF")
        other_image = self.path.parent / f"{Path(name).stem}.p1.144dpi.png"
        with Image.new("RGB", (3, 3), "blue") as image:
            image.save(other_image, format="PNG")
        manifest = {
            "source_file": name,
            "source_sha256": hashlib.sha256(other_source.read_bytes()).hexdigest(),
            "page_count": 174, "dpi": 144, "render_schema_version": 1,
            "pages": {"1": {
                "status": "ready", "file": other_image.name,
                "bytes": other_image.stat().st_size,
                "image_sha256": hashlib.sha256(other_image.read_bytes()).hexdigest(),
                "width": 3, "height": 3,
            }},
        }
        path = other_image.parent / f"{Path(name).stem}.144dpi.manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        result = read_report_page(1, name)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["source_file"], name)
        self.assertEqual(result["image"].inline_data.data, other_image.read_bytes())
        self.assertEqual(read_report_page(175, name)["status"], "invalid_page")
        self.assertEqual(read_report_page(5, name)["status"], "page_not_prepared")
        self.assertEqual(read_report_page(5)["status"], "ok")

    def test_malformed_catalog_is_an_explicit_error(self) -> None:
        self.manifest_path.write_text("{broken", encoding="utf-8")
        with self.assertLogs("corporate_reports.visual_tools", level="WARNING"):
            self.assertEqual(list_report_documents()["status"], "catalog_error")
            self.assertEqual(read_report_page(5)["status"], "catalog_error")

    @staticmethod
    def jpeg_bytes() -> bytes:
        with BytesIO() as buffer, Image.new("RGB", (2, 2)) as image:
            image.save(buffer, format="JPEG")
            return buffer.getvalue()
