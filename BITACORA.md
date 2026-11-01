# Bitácora — Asistente multimodal de informes corporativos

## Punto actual

- **Última actualización:** 2026-10-29 (Europe/Madrid).
- **Fase activa:** 3 de 11 — RAG multimodal, preparación de la ingesta documental.
- **Último hito:** fase 2 completada con tres ejecuciones reales revisadas, metadatos y cinco tests offline pasando.
- **Próxima acción:** preparar extracción local por página conservando documento y número de página; comparar texto extraído con la página visual antes de diseñar la recuperación.
- **No avanzar de fase sin confirmación del usuario.**

## Objetivo y forma de aprender

Construir un asistente que consulte informes corporativos con texto, tablas e imágenes, responda con evidencia y pueda integrarse y operarse como servicio seguro en GCP.

El usuario implementa y ejecuta para aprender a reproducir el trabajo. El mentor explica concepto, alternativas y riesgos antes de cada paso; ofrece fragmentos pequeños y realiza cambios completos cuando el usuario lo pide. Cada fase termina con preguntas, ejercicio y checklist.

## Hoja de ruta

| Fase | Estado | Objetivo y evidencia de finalización |
| --- | --- | --- |
| 1. Primer agente local | Completada | ADK y Gemini funcionando en CLI y web; consulta con imagen e inspección de eventos, latencia y tokens. |
| 2. Respuestas estructuradas y evaluación | Completada (base inicial) | Contrato de salida validado, casos positivos y de abstención, ejecución repetible de casos contra el agente y resultados registrados. |
| 3. RAG multimodal | En curso | Ingesta de PDFs con páginas y metadatos, tratamiento de texto/tablas/imágenes, recuperación de evidencias y citas; evaluar retrieval y generación por separado. |
| 4. Tools, estado y memoria | Pendiente | Herramientas propias de consulta y cálculo, validación de entradas y errores; distinguir historial, estado de sesión y memoria persistente. |
| 5. Multi-agente y protocolos | Pendiente | Sub-agentes, flujos secuencial/paralelo/bucle con límites; servidor y cliente A2A, Agent Card y decisión A2A vs sub-agentes vs MCP. |
| 6. GCP e identidad | Pendiente | Proyecto y control de costes, APIs, ADC, impersonación, service accounts, IAM de mínimo privilegio y Secret Manager, sin claves de service account persistentes. |
| 7. Selección de modelos | Pendiente | Comparar Gemini, Claude y un modelo abierto disponible mediante Model Garden con las mismas tareas: calidad, latencia, coste y restricciones. |
| 8. Agentes gestionados de datos | Pendiente | Datos verificados en BigQuery; probar la oferta vigente de análisis conversacional y justificar solución gestionada vs construcción propia. |
| 9. API e integración en producto | Pendiente | Comparar servidor ADK y FastAPI; POST, autenticación/autorización OAuth/OIDC e ID tokens cuando correspondan, aislamiento, control de consumo, citas y feedback. |
| 10. Despliegue en GCP | Pendiente | Docker multi-stage/no root/healthcheck, Artifact Registry, Cloud Run y runtime gestionado vigente; comparar sesiones, memoria, versiones, escalado y costes. |
| 11. Operación y mantenimiento | Pendiente | CI/CD con evals, trazas/logs/métricas/alertas, cuotas, cambios de modelo, entornos, rollback y documentación para operar o asumir soluciones de terceros. |

Seguridad, privacidad, tests y observabilidad se aplican durante todo el recorrido. Verificar nombres de servicios, comandos, modelos, precios y regiones en documentación oficial antes de usarlos. Los bloques de comparación no tienen por qué convertirse todos en componentes de la aplicación final.

## Fase 1 — Hitos realizados

- [x] Caso elegido: asistente de informes corporativos de NVIDIA.
- [x] Seis PDFs cuyos nombres corresponden a 2021–2026 en `data/raw/nvidia/`.
- [x] Git inicializado y remoto `origin` configurado hacia `https://github.com/asevillano2407/adk_agent_multimodal.git`.
- [x] `.gitignore` excluye PDFs originales, entorno, secretos y archivos locales de ADK.
- [x] Python 3.12 y entorno virtual `adk_env`.
- [x] ADK 2.13.0 registrado en `requirements.txt`.
- [x] Agente `corporate_reports_agent` en `corporate_reports/`.
- [x] Gemini Developer API configurada en nivel gratuito; modelo utilizado: `gemini-3.5-flash`.
- [x] Ejecución mediante `adk run corporate_reports` y `adk web .`.
- [x] Prueba sin documentos: reconoce falta de evidencia.
- [x] Tabla adjunta: extrae ingresos, calcula participación y se abstiene ante una pregunta sobre España.
- [x] Inspección de trazas: modelo, agente, latencia y consumo de tokens.

**Límites:** adjuntar una imagen no da acceso autónomo a `data/raw/`. Una página aportada en el prompt no es una cita localizada automáticamente. Los tokens de razonamiento y de salida no deben sumarse sin verificar su definición en la integración.

## Fase 2 — Hitos y pendientes

### Realizados

- [x] `ReportAnswer`: estado, respuesta, fuentes, limitaciones y cálculo opcional en valor (`null`), obligatorio en presencia.
- [x] `RevenueShareCalculation`: importes, unidad y porcentaje; denominador positivo validado localmente.
- [x] Tres casos en `evals/cases.json`: sin documento, ausencia de desglose de España y participación de ingresos.
- [x] Imagen de prueba conservada en `evals/fixtures/nvidia_fy2025_results.png`.
- [x] Respuestas históricas guardadas para los casos sin documento y de participación.
- [x] `check_response.py`: estructura y estado esperado del caso sin documento.
- [x] `check_calculation.py`: estructura, estado, unidad, cifras contra referencia y coherencia matemática.
- [x] Valores de referencia separados del código mediante `expected_calculation` en `cases.json`.
- [x] Pruebas negativas observadas: porcentaje 145 rechazado, denominador cero rechazado y cifra de segmento alterada rechazada.
- [x] Valores originales del cálculo restaurados y comprobador de cálculo pasando.

### Pendientes antes de cerrar

- [x] Adaptar `evals/responses/missing_document.json` con `revenue_share_calculation: null` (confirmado por el usuario).
- [x] Ejecutar de nuevo ambos comprobadores: PASS confirmado por el usuario.
- [x] Porcentaje 80, permitido por el esquema pero incorrecto, rechazado en test específico.
- [x] Tres casos ejecutados contra el agente con sesiones nuevas y respuestas guardadas sin sobrescribir anteriores.
- [x] Metadatos básicos: fecha UTC de guardado, modelo configurado, ADK, pregunta, rutas de entrada, duración de generación y vínculo a respuesta.
- [x] Contenido y limitaciones de las tres ejecuciones revisados manualmente.
- [x] Preguntas y mini-ejercicio de cierre planteados; el usuario autoriza continuar. No se registraron respuestas a las preguntas de comprensión.

**Límites del evaluador actual:** no lee la imagen ni demuestra fidelidad de toda la prosa. Compara campos numéricos con referencias revisadas por una persona. Un script sobre respuestas guardadas no equivale a ejecutar una evaluación nueva contra el modelo. La lista `must_include`/`must_not_include` todavía se revisa manualmente.

## Fase 3 — Plan de trabajo

- [ ] Inspeccionar un PDF y extraer texto por página, conservando origen y numeración PDF desde 1.
- [ ] Comparar extracción textual y representación visual de páginas seleccionadas; detectar tablas, imágenes y límites de orden de lectura.
- [ ] Definir unidades de recuperación y metadatos; conservar relación entre fragmentos, páginas y elementos visuales.
- [ ] Implementar una recuperación básica y evaluar si encuentra las evidencias necesarias antes de añadir complejidad.
- [ ] Conectar recuperación y generación; citas trazables y abstención si falta evidencia.
- [ ] Incorporar evidencia visual cuando el texto sea insuficiente y comparar calidad, latencia y consumo.
- [ ] Evaluar recuperación y generación por separado; cierre de comprensión y confirmación para avanzar.

No elegir una base vectorial o arquitectura multiagente antes de entender los documentos y comprobar qué necesita el caso de uso.

## Decisiones e incidencias

| Tema | Decisión o aprendizaje |
| --- | --- |
| Credenciales | `.env` local excluido de Git; nunca copiar claves a la bitácora, código o chat. La gratuidad está sujeta a cuotas y condiciones verificables. |
| Numeración | Distinguir página del PDF contando desde 1 y página impresa. La tabla utilizada se identificó como PDF 18 / impresa 4. |
| Cálculo | `116.2 / 130.5 × 100 ≈ 89.04 %`; son miles de millones de USD. Crecimiento del 145 % no significa participación del 145 %. |
| Precisión | Las cifras publicadas están redondeadas; cualquier intervalo de redondeo requiere explicitar sus supuestos. |
| Error 403 en web | La llamada a Gemini devolvió 403 mientras el endpoint local devolvía 200. CLI funcionó y web volvió a funcionar tras reinicio/sesión nueva. Causa no confirmada; no atribuirla automáticamente a facturación. |
| Esquema incompatible | `Field(gt=0)` produjo `exclusiveMinimum`, rechazado al convertir el esquema en el SDK instalado. Se usa `ge=0` más `field_validator` local para conservar la restricción positiva. |
| Validación local | Los validadores Python personalizados no se ejecutan en Gemini. Validar explícitamente con Pydantic antes de consumir datos. |
| Evolución del contrato | `Tipo | None` sin valor por defecto acepta `null` pero exige el campo. Adaptar una respuesta antigua no constituye una nueva salida del modelo. |

## Comandos de trabajo

Desde la raíz del repositorio en PowerShell:

```powershell
.\adk_env\Scripts\Activate.ps1
adk run corporate_reports
# Salir de CLI con exit antes de iniciar la web en esa terminal.
adk web .
```

Comprobaciones locales, sin llamadas a Gemini:

```powershell
python -m json.tool --no-ensure-ascii evals/cases.json
python -m evals.check_response
python -m evals.check_calculation
```

## Registro de avances

### 2026-10-28 — Consolidación de la bitácora

- Registro retrospectivo de hitos del aprendizaje; no se asignan fechas exactas no verificadas a los pasos anteriores.
- Revisados el esquema y los dos comprobadores existentes.
- Ejecutado `check_calculation`: **PASS**.
- Ejecutado `check_response`: **FAIL esperado por fixture desactualizado**, falta `revenue_share_calculation`. Pendiente de adaptar, sin ocultar el fallo.
- No se realizaron llamadas a Gemini en esta revisión.
- Próximo paso acordado: terminar validaciones locales de fase 2 y preparar ejecución reproducible contra el agente.

### 2026-10-28 — Validaciones locales actualizadas

- El usuario confirma que `check_response` y `check_calculation` devuelven PASS tras adaptar la respuesta histórica.
- Se mantiene pendiente la evaluación nueva contra Gemini; estos PASS corresponden a respuestas guardadas.
- Próximo paso: preparar un runner Python con sesión nueva por caso y posterior validación y guardado de la respuesta final.

### 2026-10-29 — Cierre de fase 2 e inicio autorizado de fase 3

- `evals/run_case.py` recibe el caso por argumento, carga configuración local, crea sesión nueva, adjunta PNG cuando procede, valida y guarda resultados con identificador temporal.
- Se reutiliza `validate_calculation` tanto para respuestas históricas como para las nuevas ejecuciones.
- Diagnósticos de error separados: tipo y código cuando está disponible, sin persistir el mensaje de excepción. El fallo se propaga y el runner se cierra.
- Se experimentó un 503 de alta demanda y un 429 por cuota diaria. No son fallos de calidad de respuesta; no se implementaron reintentos automáticos.
- Revisados los pares de respuesta/metadatos: `missing_document_20261029T101315608167Z`, `spain_revenue_missing_20261029T101514713958Z` y `compute_revenue_share_20261029T101712120441Z`.
- Los tres pasan las reglas automatizadas aplicables y la revisión manual inicial. Duraciones registradas: 7.799 s, 7.596 s y 10.198 s, respectivamente; no son estadísticas representativas.
- Cinco tests en `tests/test_run_case.py` pasando: fallos de sesión y generación, diagnóstico sin mensaje sensible, respuesta/metadatos/imagen y rechazo de porcentaje matemáticamente incorrecto dentro del rango válido. Tests ejecutados sin llamadas a Gemini.
- Limitaciones que permanecen: tres casos no constituyen un benchmark amplio; faltan evaluación semántica automática, hashes/versionado completo de prompts/inputs y captura completa de salidas inválidas. El registro actual no garantiza capturar fallos anteriores al `try` o los del propio guardado/cierre.
- El usuario delega creación y ejecución de tests al mentor; la implementación principal sigue siendo guiada.
- El usuario confirma continuar a fase 3. Los PDFs todavía no son consultados automáticamente por el agente.

## Cómo mantener esta bitácora

Actualizar este documento **solo al cierre de cada fase**, conforme a la preferencia del usuario, registrando sus hitos y la transición autorizada. Mantener el resumen de “Punto actual” sincronizado con el checklist y añadir una entrada fechada con cambios, evidencia, decisiones y pendientes. No marcar como realizado un paso solo por haberlo propuesto. Conservar el historial de incidencias y registrar sus causas como desconocidas cuando no se hayan demostrado.

Al cerrar cada fase registrar:

1. Hitos y archivos relevantes.
2. Comandos/pruebas realizadas y resultados.
3. Decisiones, riesgos y limitaciones pendientes.
4. Preguntas de comprensión y ejercicio de cierre.
5. Confirmación del usuario antes de iniciar la siguiente.
