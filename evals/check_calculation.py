import json

from math import isclose
from pathlib import Path

from corporate_reports.schemas import ReportAnswer

def validate_calculation(
    response: ReportAnswer,
    reference: dict,
) -> None:
    """Contrasta los datos de origen y la operación con la referencia."""

    calculation = response.revenue_share_calculation
    if calculation is None:
        raise ValueError("Falta el cálculo esperado.")

    if not isclose(
    calculation.segment_revenue, reference["segment_revenue"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("Los ingresos del segmento no coinciden con la imagen.")

    if not isclose(
    calculation.consolidated_revenue, reference["consolidated_revenue"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("Los ingresos consolidados no coinciden con la imagen.")

    if calculation.unit != reference["unit"]:
        raise ValueError("La unidad no coincide con la referencia.")

    expected = (
        calculation.segment_revenue
        / calculation.consolidated_revenue
        * 100
    )
    if not isclose(
        calculation.percentage, expected, rel_tol=0, abs_tol=0.005
    ):
        raise ValueError(
            f"Porcentaje incorrecto: {calculation.percentage}; "
            f"esperado: {expected:.4f}"
        )

    print("PASS: estructura válida y cálculo coherente.")

def main() -> None:
    """Valida la estructura y la operación del caso de participación."""

    cases = json.loads(
        Path("evals/cases.json").read_text(encoding="utf-8")
    )
    case = next(
        item for item in cases if item["id"] == "compute_revenue_share"
    )
    reference = case["expected_calculation"]

    response = ReportAnswer.model_validate_json(
        Path("evals/responses/compute_revenue_share.json").read_text(
            encoding="utf-8"
        )
    )

    if response.status != case["expected_status"]:
            raise ValueError("Estado de respuesta inesperado.")

    validate_calculation(response, reference)


if __name__ == "__main__":
    main()