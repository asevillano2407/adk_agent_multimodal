# Evaluación del corpus de seis informes

Desde la raíz del proyecto, con `adk_env` activo, valida las referencias sin
consultar Chroma ni llamar a Gemini:

```powershell
python -m evals.evaluate_retrieval --cases evals/retrieval_multireport.json --corpus-manifest data/processed/report_corpus/corpus.fixed.c1200.o200.manifest.json --validate-only
```

Esta comprobación verifica identidades y hashes del corpus, páginas y anclas
textuales. No mide calidad de recuperación ni de generación. No requiere que
la indexación vectorial esté completa y no guarda un informe de métricas.

Cuando la colección esté completa, ejecutar la evaluación vectorial consume
embeddings de consulta. Se rechaza una colección incompleta o cuyo hash no
coincida con el manifiesto verificado:

```powershell
python -m evals.evaluate_retrieval --cases evals/retrieval_multireport.json --corpus-manifest data/processed/report_corpus/corpus.fixed.c1200.o200.manifest.json --retriever vector --collection nvidia_reports_fixed_gemini001_768
```

Mientras se completa la indexación, se puede medir el baseline léxico sobre
los mismos chunks sin llamadas a Gemini ni acceso a Chroma:

```powershell
python -m evals.evaluate_retrieval --cases evals/retrieval_multireport.json --corpus-manifest data/processed/report_corpus/corpus.fixed.c1200.o200.manifest.json --retriever lexical
```

El algoritmo sigue contando términos distintos compartidos. No traduce ni
reformula preguntas. Repite con `--filter-by-document` para comparar la
búsqueda global y la búsqueda dentro del informe solicitado, con el mismo K.

En ambos backends, el filtro usa `source_file` explícito en el caso, que representa
una selección del usuario. No obtiene filtros de las páginas o cifras de
respuesta. No evalúa la capacidad del agente de elegir el documento.

El corpus incluye cifras de ejercicios anteriores dentro de informes más
recientes. El año del documento no equivale al ejercicio del dato. Las anclas
solo comprueban presencia textual en un mismo chunk: no demuestran que el
modelo seleccione la columna correcta ni efectúe bien el cálculo.

Las páginas etiquetadas son ejemplos conocidos, no una lista exhaustiva de
evidencias válidas. Otras páginas pueden responder correctamente. Los casos
negativos no se puntúan con Hit/MRR: requieren revisar la abstención en la
respuesta del agente. Estos ocho casos son de desarrollo, no un benchmark
independiente para afirmar calidad general en producción.

Los resultados reales se guardan en `evals/runs`, con la identidad del corpus,
la colección y el modo de filtro. No se guarda el texto de los fragmentos.

## Evaluación del agente con herramientas

Los casos anteriores de `run_case` siguen usando contexto controlado, con las
herramientas deshabilitadas. Los casos con `execution_mode: agent_tools`
empiezan únicamente con la pregunta y permiten que el agente consulte tools.
No pasan al modelo las respuestas esperadas ni las referencias numéricas.

Preparados para ejecutar **después** de completar, evaluar y activar la nueva
colección en el agente:

```powershell
python -m evals.run_case agent_fy2024_from_report2025
python -m evals.run_case agent_spain_missing_multireport
python -m evals.run_case agent_future_missing_multireport
```

Estas ejecuciones consumen llamadas a Gemini y embeddings de consulta. El
runner rechaza una colección no activada, incompleta o de otro corpus antes
de crear la sesión. Conserva los límites de herramientas del agente y añade
un máximo de ocho llamadas al LLM por ejecución; eso no es un límite de tokens.

Los controles comprueban estado, cálculo cuando corresponde, filtro por
documento, uso de búsqueda sin fallos del backend y procedencia de las citas.
Las trazas guardan nombres de herramientas y referencias; omiten consultas,
textos recuperados e imágenes. El archivo de respuesta sí conserva la salida
del modelo, como los casos anteriores, y debe seguir excluido de Git.

Un PASS no demuestra fidelidad de toda la prosa ni que una página citada sea
la correcta: las citas son cadenas libres. Revisar contenido, ejercicio fiscal,
unidades, página y limitaciones sigue siendo necesario. La recuperación se
evalúa por separado antes de atribuir un fallo al razonamiento del agente.
