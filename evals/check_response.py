import json
from pathlib import Path

from corporate_reports.schemas import ReportAnswer


def main() -> None:
    """Comprueba el contrato de la respuesta sin documentos."""
    cases = json.loads(
        Path("evals/cases.json").read_text(encoding="utf-8")
    )
    case = next(item for item in cases if item["id"] == "missing_document")

    response = ReportAnswer.model_validate_json(
        Path("evals/responses/missing_document.json").read_text(
            encoding="utf-8"
        )
    )

    if response.status != case["expected_status"]:
        raise ValueError(f"Estado inesperado: {response.status}")

    print("PASS: estructura válida y estado esperado.")


if __name__ == "__main__":
    main()