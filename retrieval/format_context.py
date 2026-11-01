import json


def format_context(pages: list[dict]) -> str:
    """Serializa evidencias recuperadas conservando su procedencia."""
    evidence = [
        {
            "source_file": page["source_file"],
            "pdf_page": page["pdf_page"],
            "text": page["text"],
        }
        for page in pages
    ]
    return json.dumps(evidence, ensure_ascii=False, indent=2)