"""Controles locales de consumo y contrato de salida para ADK."""

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from .schemas import ReportAnswer


MAX_SEARCHES_PER_TURN = 2
SEARCH_BUDGET_KEY = "temp:corporate_search_budget"
MAX_VISUAL_PAGES_PER_TURN = 2
VISUAL_BUDGET_KEY = "temp:corporate_visual_budget"


async def limit_document_searches(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
) -> dict | None:
    """Limita búsqueda y lectura visual por invocación, incluso si fallan.

    El estado pertenece a la sesión y se identifica por invocación, sin
    contadores globales. No hay awaits entre leer y reservar el intento.
    Hay presupuestos separados para búsqueda textual y lectura visual.
    No limita todos los turnos internos del LLM.
    """
    if tool.name == "search_corporate_reports":
        key, limit, status = SEARCH_BUDGET_KEY, MAX_SEARCHES_PER_TURN, "search_limit_reached"
    elif tool.name == "read_report_page":
        key, limit, status = VISUAL_BUDGET_KEY, MAX_VISUAL_PAGES_PER_TURN, "visual_limit_reached"
    else:
        return None
    budget = tool_context.state.get(key, {})
    used = (
        budget.get("used", 0)
        if budget.get("invocation_id") == tool_context.invocation_id else 0
    )
    if used >= limit:
        return {
            "status": status,
            "evidence": [],
            "message": "Límite de esta herramienta alcanzado. Responde con la evidencia ya disponible y explica sus límites.",
        }
    tool_context.state[key] = {
        "invocation_id": tool_context.invocation_id, "used": used + 1,
    }
    return None


def preserve_response_nulls(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict,
) -> dict | None:
    """Conserva null en la salida validada de set_model_response de ADK 2.13.

    ADK 2.13.0 serializa esa respuesta con exclude_none=True. Reutilizamos
    sus argumentos ya validados y actualizamos también la acción con la que
    ADK construye el evento final. No repara argumentos rechazados por ADK.
    """
    if (
        tool.name != "set_model_response"
        or tool_context.actions.set_model_response is None
    ):
        return None
    response = ReportAnswer.model_validate(args).model_dump(mode="json")
    tool_context.actions.set_model_response = response
    return response
