"""Configuración local del índice textual y alcance comunicado al agente."""


LEGACY_COLLECTION_NAME = "nvidia_fixed_gemini001_768"
MULTIREPORT_COLLECTION_NAME = "nvidia_reports_fixed_gemini001_768"
LEGACY_SOURCE_FILE = "NVIDIA-2025-Annual-Report.pdf"

# Colección multiinforme completada y evaluada en desarrollo.
# Reiniciar ADK tras cambiar esta configuración: se carga al importar el agente.
COLLECTION_NAME = MULTIREPORT_COLLECTION_NAME


def text_search_scope(collection_name: str) -> str:
    """Describe el alcance del índice seleccionado sin declarar su disponibilidad."""
    if collection_name == LEGACY_COLLECTION_NAME:
        return (
            "La búsqueda textual activa está limitada al informe NVIDIA FY2025 "
            f"({LEGACY_SOURCE_FILE}). No presentes sus resultados como evidencia "
            "procedente de otro informe. El catálogo visual puede incluir otros "
            "documentos; eso no significa que estén disponibles para búsqueda textual. "
        )
    if collection_name == MULTIREPORT_COLLECTION_NAME:
        return (
            "La búsqueda textual está configurada para la colección multiinforme. "
            "Usa list_report_documents para conocer los nombres exactos. "
            "Cuando el usuario solicite un informe concreto, pasa source_file "
            "explícitamente a search_corporate_reports; si no hay un documento "
            "concreto solicitado, puedes buscar sin filtro. Conserva la procedencia "
            "de cada fragmento y verifica el ejercicio fiscal en su contenido. "
            "La herramienta comprueba la disponibilidad del índice; si devuelve "
            "index_not_ready, no lo trates como ausencia de información. "
        )
    raise ValueError("Colección textual no reconocida para configurar el agente.")
