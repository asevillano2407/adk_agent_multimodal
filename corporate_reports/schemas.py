from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RevenueShareCalculation(BaseModel):
    """Datos y resultado para calcular una participación en ingresos."""

    segment_revenue: float = Field(ge=0)
    consolidated_revenue: float = Field(ge=0)
    unit: Literal["billion_USD"]
    percentage: float = Field(ge=0, le=100)

    @field_validator("consolidated_revenue")
    @classmethod
    def validate_positive_revenue(cls, value: float) -> float:
        """Rechaza un denominador nulo o negativo."""
        if value <= 0:
            raise ValueError("consolidated_revenue debe ser mayor que cero")
        return value

class ReportAnswer(BaseModel):
    """Respuesta estructurada del asistente documental."""

    status: Literal["answered", "insufficient_evidence"]
    answer: str
    sources: list[str] = Field(
        description="Referencias disponibles. Lista vacía si no existen."
    )
    limitations: list[str] = Field(
        description="Limitaciones de la respuesta. Lista vacía si no existen."
    )
    revenue_share_calculation: RevenueShareCalculation | None = Field(
    description=(
        "Datos del cálculo de participación en ingresos. "
        "Usa null si no corresponde o falta evidencia."
    )
)