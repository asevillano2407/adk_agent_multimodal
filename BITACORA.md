# Bitácora — Asistente multimodal de informes corporativos

## Punto actual

- **Última actualización:** 2026-11-24 (Europe/Madrid).
- **Fase activa:** 3 de 11 — base técnica local completada; cierre de aprendizaje pendiente.
- **Último hito:** seis informes, 4143 chunks no vacíos en Chroma, herramientas textuales y visuales activas, y evaluación real de cálculo y abstención revisada.
- **Próxima acción:** preguntas y ejercicio de cierre; confirmar disponibilidad para empezar la fase 4. La fase 4 no se ha iniciado.
- **No avanzar de fase sin confirmación del usuario.**

## Objetivo y forma de aprender

Construir un asistente que consulte informes corporativos con texto, tablas e imágenes, responda con evidencia y pueda integrarse y operarse como servicio seguro en GCP.

El usuario implementa y ejecuta para aprender a reproducir el trabajo. El mentor explica concepto, alternativas y riesgos antes de cada paso; ofrece fragmentos pequeños y realiza cambios completos cuando el usuario lo pide. Cada fase termina con preguntas, ejercicio y checklist.

## Hoja de ruta

| Fase | Estado | Objetivo y evidencia de finalización |
| --- | --- | --- |
| 1. Primer agente local | Completada | ADK y Gemini funcionando en CLI y web; consulta con imagen e inspección de eventos, latencia y tokens. |
| 2. Respuestas estructuradas y evaluación | Completada (base inicial) | Contrato de salida validado, casos positivos y de abstención, ejecución repetible de casos contra el agente y resultados registrados. |
| 3. RAG multimodal | Base técnica completada; cierre de aprendizaje pendiente | Ingesta de PDFs con páginas y metadatos, recuperación textual y lectura visual, citas y evaluación separada de retrieval y generación. Quedan limitaciones de calidad documentadas. |
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

## Fase 3 — Hitos realizados y cierre

- [x] Extraer texto por página conservando archivo y numeración PDF desde 1; ampliar a los seis informes.
- [x] Comparar texto y representación visual: tablas financieras y posiciones de logos que el texto no permite justificar.
- [x] Implementar chunks fijos de 1200 caracteres con overlap de 200, dentro de cada página, con offsets y procedencia. Alternativa `natural` implementada; la colección activa conserva `fixed` como baseline.
- [x] Mantener un baseline léxico de términos distintos compartidos; evaluar búsqueda global y filtrada por documento.
- [x] Generar embeddings con `gemini-embedding-001`, 768 dimensiones, tareas DOCUMENT/QUERY y normalización L2; guardar en Chroma con distancia coseno y recuperar top 3.
- [x] Preparar 1057 páginas y 4146 chunks; omitir tres chunks vacíos de la indexación: 4143 registros finales.
- [x] Reutilizar 732 embeddings de la colección anterior y generar 3411 nuevos, con lotes reanudables, hashes e identidad de corpus. La colección anterior se conserva para comparar.
- [x] Renderizar las 1057 páginas como PNG a 144 DPI; manifiestos, hashes, límites de tamaño y validación antes de entregar imágenes al modelo.
- [x] Activar `nvidia_reports_fixed_gemini001_768`; centralizar selección y alcance del prompt en `corporate_reports/config.py`.
- [x] Registrar tres herramientas: búsqueda textual, catálogo de documentos y lectura visual de una página. Máximo dos búsquedas y dos lecturas visuales por turno, con presupuestos separados.
- [x] Separar evaluación de retrieval, generación con contexto controlado y agente con tools; sesiones nuevas y trazas mínimas sin texto de documentos ni consultas de herramientas.
- [x] Revisar ejecuciones reales de cálculo FY2024 desde el informe de 2025 y abstenciones sobre España/FY2035. Un PASS automático no equivale a fidelidad completa de la prosa.
- [ ] Responder preguntas de comprensión y completar el mini-ejercicio indicados abajo.
- [ ] Confirmar el inicio de la fase 4.

### Archivos principales

- `ingestion/`: extracción/exportación, chunking, renderización, preparación versionada e indexación reanudable.
- `retrieval/`: baseline léxico, embeddings, consulta vectorial, preparación del contexto y validación de catálogos/corpus.
- `corporate_reports/agent.py`, `config.py`, `tools.py`, `visual_tools.py`, `callbacks.py`: orquestación, alcance, herramientas y límites.
- `evals/evaluate_retrieval.py`, `retrieval_multireport.json`: referencias conocidas y métricas Hit/MRR/evidencia. `evals/README.md` explica comandos y límites.
- `evals/run_case.py`, `agent_execution.py`, `response_rules.py`, `cases.json`: ejecución, trazas y controles deterministas. `tests/`: pruebas offline con datos sintéticos y mocks de APIs.

### Resultados de recuperación y cómo interpretarlos

En el conjunto de ocho casos de desarrollo (siete positivos y uno negativo), el baseline léxico filtrado obtuvo Hit@3=0,7143, MRR@3=0,6190 y Hit de evidencia@3=0,5714. Los negativos se excluyen de estas medias: recuperar fragmentos no prueba que exista respuesta.

La ejecución vectorial filtrada `retrieval_20261114T184537425173Z.json` publicó inicialmente Hit@3=1, MRR@3=0,8571 y evidencia=0,8571. La revisión mostró que la página 157 del informe de 2021 contenía una tabla válida ausente de las etiquetas. Tras añadirla, la relectura offline de **esos mismos resultados guardados** obtuvo Hit@3=1, MRR@3=0,9286 y evidencia=1. No fue una nueva búsqueda y no se reescribió el informe histórico.

También se añadieron tablas resumen válidas con cifras redondeadas en 2023–2026. Estas revisiones son correcciones de etiquetas de desarrollo, no mejoras del recuperador. Las referencias siguen siendo parciales, el conjunto es pequeño y no constituye un benchmark independiente de producción. Presencia de anclas en un chunk no demuestra interpretación correcta de filas, columnas o unidades.

### Límites y trabajo que permanece

- No hay combinación híbrida léxico/vectorial, BM25, reranking ni umbral de relevancia. Se conserva el baseline para justificar cambios con evidencia.
- La ruta visual lee páginas concretas; no hay un índice de embeddings de imágenes ni OCR en la ingesta actual. Renderizar previamente ahorra conversiones repetidas, pero no elimina el consumo del modelo al interpretar cada imagen.
- Los porcentajes se calculan en el LLM y se verifican en los evaluadores. Una herramienta de cálculo determinista pertenece al siguiente bloque.
- Persisten defectos ocasionales de prosa: afirmaciones sobre la ausencia de datos en todo un PDF, detalles innecesarios y formato numérico mixto. Las reglas de frases/notas son detectores acotados, no un juez semántico ni una garantía contra prompt injection.
- Las citas son cadenas libres. Los controles verifican procedencia y casos conocidos; no validan semánticamente todas las páginas citadas.
- Latencia y consumo aún necesitan medición sistemática. El último caso positivo tardó 100,88 s; es una observación, no una media ni un objetivo de servicio aceptado. No se ha establecido un benchmark de coste o tokens por modalidad.
- El despliegue, IAM, persistencia gestionada, autenticación, CI/CD y operación siguen pendientes en sus fases. Esta es una base local de aprendizaje.

### Cierre de aprendizaje pendiente

1. ¿Por qué Hit@3 puede valer 1 mientras Hit de evidencia@3 vale 0?
2. ¿Qué diferencia hay entre buscar dentro del informe de 2025 y seleccionar cifras del ejercicio FY2024?
3. ¿Cuándo usarías `read_report_page` y qué limitación tiene localizar una imagen sin texto identificable?

**Mini-ejercicio:** abrir la página PDF 133 del informe de 2025, localizar la columna FY2024 y contrastar 47405 y 60922 millones de USD con los campos 47.405 y 60.922 `billion_USD` de la última respuesta. Comprobar en su metadata el filtro `source_file` y las páginas devueltas por la herramienta. No requiere nuevas llamadas al modelo.

**Checklist de cierre técnico:** los seis documentos están preparados; la colección completa coincide con su corpus; retrieval y generación se evalúan por separado; el agente usa el documento solicitado; cálculo y abstención pasan los controles aplicables; los límites conocidos están descritos. La comprensión y la transición a fase 4 continúan pendientes de respuesta del usuario.

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

### 2026-11-24 — Cierre técnico de fase 3; transición pendiente

- Registro retrospectivo de hitos de la fase, respetando la actualización de la bitácora solo al cierre y sin inventar fechas de implementación intermedias.
- Corpus activo: `nvidia_reports_fixed_gemini001_768`, 4143 registros, hash `814b5da3f57a6ae12c5990fdb52a333d9032eb3833b1b2c1484948a0ee405ab4`. El usuario ejecutó las ingestas y llamadas reales; el mentor implementó lo delegado y ejecutó tests offline.
- Dependencias incorporadas: pypdf, fonttools, pypdfium2, Pillow y Chroma; se resolvieron incompatibilidades de OpenTelemetry y el usuario confirmó `pip check` limpio.
- Incidencias: cuota diaria de embeddings 429 (límite 1000 según respuestas recibidas) y 503 de alta demanda durante generación. Reanudar no exige regenerar vectores ya guardados. Esperar entre solicitudes no evita agotar la cuota diaria.
- Corregido un fallo de rutas largas en Windows: el prefijo del temporal incluía el nombre completo del manifiesto; se sustituyó por un prefijo corto conservando la escritura atómica.
- Corregido el filtro de 2023 por diferencia entre `ir` e `IR`. Los filtros de metadatos exigen nombres exactos.
- `agent_fy2024_from_report2025_20261124T002041117809Z`: una búsqueda filtrada a 2025, cifras FY2024 de 47.405/60.922 `billion_USD`, 77,81 %, fuentes PDF 170 y 133; controles pasando y revisión principal satisfactoria. Permanecen mejoras de formato numérico y de la expresión «cálculo exacto» sobre datos redondeados.
- `agent_spain_missing_multireport_20261124T001115751793Z`: dos búsquedas, abstención y cálculo null; sin nota inventada ni atribución de España a Other. Continúa una afirmación global sobre el documento en limitations; no se considera prosa plenamente validada.
- `agent_future_missing_multireport_20261124T001323356565Z`: catálogo y una búsqueda en el informe de 2026; abstención FY2035, cálculo null, referencias recuperadas 125 y 139. Las limitaciones acotan la búsqueda; la primera afirmación todavía puede formularse con mayor cautela.
- Respuestas anteriores revelaron notas inventadas y expresiones de precisión excesiva. Se ajustaron instrucciones y reglas de regresión, sin modificar artefactos históricos. El PASS de una ejecución corresponde a las reglas vigentes en ese momento.
- Antes de preparar los commits de cierre se ejecutó la suite completa del estado final: **236 tests offline pasando**. No se realizaron llamadas a Gemini ni se modificó el índice real durante esas pruebas. También se revisaron patrones de credenciales en los archivos candidatos a Git y `git diff --check` no detectó errores de espacios.
- Se registran preguntas, mini-ejercicio y límites. No se ha autorizado ni iniciado la fase 4.

## Cómo mantener esta bitácora

Actualizar este documento **solo al cierre de cada fase**, conforme a la preferencia del usuario, registrando sus hitos y la transición autorizada. Mantener el resumen de “Punto actual” sincronizado con el checklist y añadir una entrada fechada con cambios, evidencia, decisiones y pendientes. No marcar como realizado un paso solo por haberlo propuesto. Conservar el historial de incidencias y registrar sus causas como desconocidas cuando no se hayan demostrado.

Al cerrar cada fase registrar:

1. Hitos y archivos relevantes.
2. Comandos/pruebas realizadas y resultados.
3. Decisiones, riesgos y limitaciones pendientes.
4. Preguntas de comprensión y ejercicio de cierre.
5. Confirmación del usuario antes de iniciar la siguiente.
