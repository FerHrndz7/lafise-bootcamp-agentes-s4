# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · MLflow Tracing: la "caja negra" del Agente de Alertas
# MAGIC
# MAGIC **Objetivo:** ver **cada paso** que da el agente para dictaminar una alerta (qué herramienta llamó, con qué
# MAGIC argumentos, qué le devolvió, cuánto tardó y cuántos tokens consumió) y dejar esa evidencia lista para auditar y
# MAGIC evaluar. ⏱️ ~20 min.
# MAGIC
# MAGIC ### Qué es una traza
# MAGIC
# MAGIC     traza (una ejecución del agente, con tags: alerta_id, version_agente)
# MAGIC     └── agente_alertas                      AGENT      ← nuestra función, decorada con @mlflow.trace
# MAGIC         └── LangGraph                       CHAIN      ← autolog de LangChain/LangGraph
# MAGIC             ├── modelo → ChatDatabricks     CHAT_MODEL ← tokens, latencia, mensajes
# MAGIC             ├── herramientas → obtener_alerta          TOOL
# MAGIC             ├── herramientas → buscar_normativa_pld    TOOL
# MAGIC             │   └── recuperar_normativa     RETRIEVER  ← documentos recuperados (para groundedness)
# MAGIC             └── modelo → ChatDatabricks     CHAT_MODEL ← dictamen final
# MAGIC
# MAGIC - Cada caja es un **span**: tiene entradas, salidas, duración, tipo y atributos.
# MAGIC - **Autolog** crea los spans de LangChain/LangGraph sin cambiar el código del agente.
# MAGIC - **`@mlflow.trace`** añade spans propios donde autolog no llega (nuestra función y el retrieval).
# MAGIC - Las trazas viven en el **experimento** de MLflow y se pueden buscar, comparar, anotar y evaluar.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.17" "databricks-langchain>=0.20.0" "langgraph>=1.2,<1.3" "unitycatalog-langchain[databricks]>=0.3.0" "databricks-sdk>=0.94.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re

dbutils.widgets.text("catalog", "", "Catálogo asignado (REQUERIDO)")
dbutils.widgets.text("schema", "pld_aml", "Schema base")
dbutils.widgets.text("llm", "", "Endpoint LLM del agente (opcional)")
user_email = spark.sql("SELECT current_user()").collect()[0][0]
user_prefix = re.sub(r"[^a-z0-9]", "_", user_email.split("@")[0].lower())
CATALOG = dbutils.widgets.get("catalog").strip()
assert CATALOG, "Escribe en el widget 'catalog' el mismo catálogo que usaste en 00_setup."
SCHEMA = f"{(dbutils.widgets.get('schema').strip() or 'pld_aml')}_{user_prefix}"
FQ = f"`{CATALOG}`.{SCHEMA}"
EXPERIMENTO = f"/Users/{user_email}/lafise_s4_tracing_evaluacion"

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %run ./utils/agente_eval_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 — Activar el tracing
# MAGIC Dos líneas: el **experimento** donde se guardan las trazas y **`mlflow.langchain.autolog()`**, que instrumenta
# MAGIC LangChain y LangGraph (el framework del agente de la Sesión 2). MLflow tiene autolog para más de 30 frameworks
# MAGIC (OpenAI, LangGraph, DSPy, LlamaIndex, CrewAI, Anthropic…): el patrón es el mismo.

# COMMAND ----------

import mlflow

mlflow.set_experiment(EXPERIMENTO)
mlflow.langchain.autolog()

LLM = elegir_llm(w, dbutils.widgets.get("llm").strip())
agente, info = construir_agente(w, spark, CATALOG, SCHEMA, LLM)
dictaminar = crear_predict_fn(agente, version="v1")
print(f"Modelo: {LLM} · herramientas: {info['herramientas']} · retrieval: {info['modo_retrieval']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 — Dictaminar una alerta… y mirar la traza
# MAGIC Ejecuta la celda. Debajo del resultado aparece la **traza**: ábrela y recorre el árbol de spans. Fíjate en:
# MAGIC - el **orden** en que el agente llamó las herramientas (¿siguió el procedimiento de sus instrucciones?);
# MAGIC - los **argumentos** de cada herramienta (¿qué nombre le pasó a `verificar_sanciones`?);
# MAGIC - los **documentos** que trajo `recuperar_normativa` y si el dictamen cita esos mismos documentos.

# COMMAND ----------

dictamen = dictaminar(pregunta="Analiza la alerta AT-001 y emite tu dictamen.")
print(json.dumps(dictamen, ensure_ascii=False, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 — La traza como datos: spans, latencia y tokens
# MAGIC Lo que ves en la interfaz también es accesible por código: `mlflow.get_trace(...)` devuelve el objeto `Trace`
# MAGIC con `info` (metadatos: duración, tags, tokens) y `data.spans` (cada paso).

# COMMAND ----------

import pandas as pd

trace_id = mlflow.get_last_active_trace_id()
traza = mlflow.get_trace(trace_id)

spans = pd.DataFrame([{
    "span": s.name, "tipo": s.span_type,
    "ms": round((s.end_time_ns - s.start_time_ns) / 1e6),
} for s in traza.data.spans])
display(spans)

print(f"Trace ID: {trace_id}")
print(f"Duración total: {traza.info.execution_duration} ms")
print(f"Tokens: {traza.info.token_usage}")
print(f"Tags: { {k: v for k, v in traza.info.tags.items() if not k.startswith('mlflow.')} }")

# COMMAND ----------

# MAGIC %md
# MAGIC **Para comentar:** ¿qué tipo de span consume más tiempo, el modelo o las herramientas? En un agente típico, más
# MAGIC del 80 % de la latencia es el LLM. Si una herramienta tarda mucho (por ejemplo, una consulta SQL pesada), la
# MAGIC traza lo muestra de inmediato.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 — El span RETRIEVER: qué contexto tenía el agente
# MAGIC Los jueces anti-alucinación (notebook `03`) comparan la respuesta con **lo que el agente recuperó**. Para eso el
# MAGIC retrieval debe quedar registrado como un span de tipo **RETRIEVER** que devuelve documentos. Así lo hicimos en
# MAGIC `utils/agente_eval_utils` (función `herramienta_normativa`):
# MAGIC
# MAGIC     @mlflow.trace(name="recuperar_normativa", span_type=SpanType.RETRIEVER)
# MAGIC     def recuperar_normativa(consulta: str):
# MAGIC         return [Document(page_content=..., id=chunk_id, metadata={"doc_uri": source_doc, ...}) ...]

# COMMAND ----------

retrievers = traza.search_spans(span_type=SpanType.RETRIEVER)
print(f"Spans RETRIEVER en la traza: {len(retrievers)}")
for s in retrievers:
    print(f"\n🔎 consulta: {s.inputs}")
    for doc in s.outputs or []:
        meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        print(f"   · {meta.get('doc_uri')} — {meta.get('seccion')}")

citados = [c.get("documento") for c in dictamen.get("citas", [])]
print(f"\n📌 Documentos citados en el dictamen: {citados}")

# COMMAND ----------

assert retrievers, "No hay spans RETRIEVER: revisa que el agente haya llamado a buscar_normativa_pld"
print("✅ El retrieval quedó trazado: los jueces podrán verificar las citas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 — Spans manuales: instrumentar tu propio código
# MAGIC Cuando autolog no cubre algo (un post-proceso, una regla de negocio, una llamada a otro sistema), abre un span
# MAGIC con `mlflow.start_span`. Ejemplo: después del dictamen, preparamos el registro para el **sistema de gestión de
# MAGIC casos**. Todo queda en **la misma traza** porque ocurre dentro de la función trazada.

# COMMAND ----------

@mlflow.trace(name="caso_para_gestion", span_type=SpanType.CHAIN)
def dictaminar_y_registrar(alerta_id: str) -> dict:
    d = dictaminar(pregunta=f"Analiza la alerta {alerta_id} y emite tu dictamen.")
    with mlflow.start_span(name="formatear_caso", span_type=SpanType.PARSER) as span:
        span.set_inputs({"dictamen": d})
        caso = {"alerta_id": alerta_id, "disposicion_recomendada": d.get("disposicion"),
                "requiere_revision": d.get("disposicion") in (E, R), "analista_asignado": None}
        span.set_outputs(caso)
        span.set_attribute("regla_asignacion", "escalados y ROS → analista senior")
    mlflow.update_current_trace(tags={"caso_registrado": "si"})
    return caso

caso = dictaminar_y_registrar("AT-003")
print(json.dumps(caso, ensure_ascii=False, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6 — Buscar trazas
# MAGIC Con muchas trazas, se buscan como en una base de datos. Sintaxis: prefijo `tags.` o `attributes.`, comillas
# MAGIC **simples** y `AND` (no `OR`). Corremos un par de alertas más para tener de dónde buscar.

# COMMAND ----------

for a in ["AT-002", "AT-009"]:
    print(a, "→", dictaminar(pregunta=f"Analiza la alerta {a} y emite tu dictamen.").get("disposicion"))

# COMMAND ----------

# Todas las trazas de AT-003 (incluida la del paso 5)
t_at003 = mlflow.search_traces(filter_string="tags.alerta_id = 'AT-003'", max_results=10)
print(f"Trazas de AT-003: {len(t_at003)}")

# Las trazas lentas (más de 15 s) de la versión v1
lentas = mlflow.search_traces(filter_string="tags.version_agente = 'v1' AND attributes.execution_time_ms > 15000",
                              max_results=20)
print(f"Trazas v1 de más de 15 s: {len(lentas)}")

cols = [c for c in ["trace_id", "request", "response", "execution_duration", "tags"] if c in t_at003.columns]
display(t_at003[cols])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 7 — La opinión del analista también es parte de la traza
# MAGIC La traza no solo guarda lo que hizo el agente: también lo que opina un **humano**. Un analista PLD revisa el
# MAGIC dictamen de `AT-003` y registra:
# MAGIC - un **feedback** (¿el dictamen es correcto?) y
# MAGIC - una **expectativa** (cuál era la disposición correcta: verdad de terreno).
# MAGIC
# MAGIC Estas anotaciones se ven en la interfaz (pestaña *Assessments* de la traza) y sirven para construir sets de
# MAGIC evaluación a partir de casos reales (notebook `02`) y para **alinear jueces** con el criterio del equipo.

# COMMAND ----------

from mlflow.entities import AssessmentSource, AssessmentSourceType

traza_at003 = mlflow.search_traces(filter_string="tags.alerta_id = 'AT-003' AND tags.caso_registrado = 'si'",
                                   max_results=1, return_type="list")[0]
analista = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=user_email)

mlflow.log_feedback(trace_id=traza_at003.info.trace_id, name="revision_analista", value=True, source=analista,
                    rationale="Correcto: el cliente coincide con la lista OFAC, se escala de inmediato (Art. 9).")
mlflow.log_expectation(trace_id=traza_at003.info.trace_id, name="disposicion_esperada",
                       value="ESCALAR_INVESTIGACION", source=analista)

anotada = mlflow.get_trace(traza_at003.info.trace_id)
for a in anotada.info.assessments:
    print(f"  {a.name}: {_valor(a)}  (fuente: {a.source.source_type})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 8 — Explorar en la interfaz (5 min)
# MAGIC Abre **Experiments** → `lafise_s4_tracing_evaluacion` → pestaña **Traces**:
# MAGIC 1. Agrega las columnas `alerta_id` y `version_agente` (son tags) y filtra por `AT-003`.
# MAGIC 2. Abre una traza y cambia a la vista de **línea de tiempo**: ¿qué partes ocurren en paralelo?
# MAGIC 3. Revisa la pestaña de **Assessments** de la traza de `AT-003`: ahí está la revisión del analista.
# MAGIC 4. *(Si tu workspace tiene Genie Code)* pregúntale en lenguaje natural sobre las trazas, por ejemplo:
# MAGIC    *"¿Qué herramienta tarda más en promedio?"*.
# MAGIC
# MAGIC > **Para producción:** Databricks recomienda guardar las trazas en **tablas de Unity Catalog** (formato
# MAGIC > OpenTelemetry): sin límite de trazas, gobernadas con permisos de UC y consultables con SQL y dashboards.
# MAGIC > Lo vemos en el notebook opcional `06`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen
# MAGIC - Activaste **MLflow Tracing** con una línea (`autolog`) y añadiste spans propios con `@mlflow.trace` y
# MAGIC   `mlflow.start_span`.
# MAGIC - Leíste la traza como datos: spans, duración, **tokens** y el contexto del span **RETRIEVER**.
# MAGIC - Buscaste trazas por tags y latencia, y registraste la **opinión de un analista** en la traza.
# MAGIC
# MAGIC ➡️ En `02_dataset_de_evaluacion` convertimos el criterio de los analistas en un **set de evaluación**.
