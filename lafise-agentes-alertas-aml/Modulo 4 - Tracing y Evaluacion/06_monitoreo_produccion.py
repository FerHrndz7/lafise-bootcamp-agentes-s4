# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · (Opcional) Del laboratorio a producción: trazas en Unity Catalog y monitoreo continuo
# MAGIC
# MAGIC **Objetivo:** ver cómo se usa lo de hoy cuando el agente ya atiende alertas reales. ⏱️ ~15 min + espera.
# MAGIC
# MAGIC | Parte | Qué es | Requisitos |
# MAGIC |---|---|---|
# MAGIC | A. Trazas en Unity Catalog | Las trazas se guardan como **tablas Delta de UC** (formato OpenTelemetry): sin límite, con permisos de UC, consultables con SQL | SQL warehouse (`CAN USE`), región soportada |
# MAGIC | B. Monitoreo en producción (*Beta*) | Los **mismos scorers** del `03` califican automáticamente una muestra del tráfico | Cómputo **serverless** (o una *budget policy* serverless en el experimento) |
# MAGIC
# MAGIC > Este notebook es **opcional**: si tu workspace no tiene serverless o un SQL warehouse, el notebook de validación
# MAGIC > lo habrá marcado como ALTERNATIVA y basta con leerlo.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.17" "databricks-langchain>=0.20.0" "langgraph>=1.2,<1.3" "unitycatalog-langchain[databricks]>=0.3.0" "databricks-sdk>=0.94.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re

dbutils.widgets.text("catalog", "", "Catálogo asignado (REQUERIDO)")
dbutils.widgets.text("schema", "pld_aml", "Schema base")
dbutils.widgets.text("llm", "", "Endpoint LLM del agente (opcional)")
dbutils.widgets.text("juez", "", "Modelo de los jueces (opcional)")
dbutils.widgets.text("warehouse", "", "SQL warehouse (ID o nombre, opcional)")
dbutils.widgets.dropdown("limpiar", "no", ["no", "si"], "Al final: borrar scorers de monitoreo")
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

import os, mlflow

mlflow.langchain.autolog()
LLM = elegir_llm(w, dbutils.widgets.get("llm").strip())
JUEZ = modelo_juez(dbutils.widgets.get("juez"))
agente, info = construir_agente(w, spark, CATALOG, SCHEMA, LLM)
predict_prod = crear_predict_fn_guardrails(agente, spark, FQ, version="v2-produccion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parte A — Trazas en Unity Catalog
# MAGIC Un experimento se **vincula** a un schema de UC al crearlo (`trace_location=UnityCatalog(...)`). Desde ese momento
# MAGIC sus trazas se escriben en cuatro tablas `<prefijo>_otel_spans / _logs / _metrics / _annotations`. La interfaz de
# MAGIC MLflow sigue igual; la diferencia es que ahora las trazas son **datos del Lakehouse**: SQL, dashboards de AI/BI,
# MAGIC alertas, Genie y permisos de UC (para leerlas o escribirlas se necesita `MODIFY` y `SELECT` explícitos en esas
# MAGIC tablas; `ALL PRIVILEGES` no alcanza).
# MAGIC
# MAGIC Usamos un experimento **nuevo** (`..._uc`): un experimento no se puede re-vincular a otra ubicación.

# COMMAND ----------

WH = None
try:
    pref = dbutils.widgets.get("warehouse").strip()
    candidatos = [x for x in w.warehouses.list() if not pref or pref in (x.id, x.name)]
    WH = sorted(candidatos, key=lambda x: not x.enable_serverless_compute)[0].id
    print(f"SQL warehouse: {WH}")
except Exception as e:
    print(f"⚠️ No encontré un SQL warehouse ({e}). Salta a la Parte B.")

# COMMAND ----------

TABLA_SPANS = None
if WH:
    from mlflow.entities.trace_location import UnityCatalog
    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = WH
    try:
        exp_uc = mlflow.set_experiment(experiment_name=f"{EXPERIMENTO}_uc",
                                       trace_location=UnityCatalog(catalog_name=CATALOG, schema_name=SCHEMA,
                                                                   table_prefix="trazas_agente"))
        TABLA_SPANS = exp_uc.trace_location.full_otel_spans_table_name
        print(f"✅ Experimento {EXPERIMENTO}_uc → {TABLA_SPANS}")
        for a in ["AT-001", "AT-010"]:
            print(a, "→", predict_prod(pregunta=f"Analiza la alerta {a} y emite tu dictamen.").get("disposicion"))
    except Exception as e:
        print(f"⚠️ No se pudo vincular el experimento a UC ({str(e)[:300]}). Revisa región, warehouse y permisos.")

# COMMAND ----------

# MAGIC %md
# MAGIC Las trazas son una tabla más. Por ejemplo, **latencia promedio por span** de las ejecuciones que acabamos de
# MAGIC hacer (la escritura en UC es asíncrona: si sale vacío, espera un minuto y vuelve a ejecutar la celda).

# COMMAND ----------

if TABLA_SPANS:
    tabla = ".".join(f"`{p}`" for p in TABLA_SPANS.split("."))
    display(spark.sql(f"""
        SELECT name AS span, count(*) AS n,
               round(avg((end_time_unix_nano - start_time_unix_nano) / 1e6)) AS ms_promedio
        FROM {tabla} GROUP BY name ORDER BY ms_promedio DESC"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parte B — Monitoreo en producción (*Beta*)
# MAGIC El patrón es `.register()` + `.start(sampling_config=...)`: el scorer queda asociado al experimento y un job
# MAGIC **serverless** califica la fracción indicada de las trazas nuevas. Usamos el experimento principal (trazas en
# MAGIC MLflow, sin SQL warehouse). Los scorers de código para monitoreo deben importar sus librerías **dentro** de la
# MAGIC función (se serializan para correr en otro proceso).
# MAGIC
# MAGIC Máximo 20 scorers por experimento. Los resultados aparecen **15–20 minutos** después en la pestaña *Traces*.

# COMMAND ----------

from mlflow.genai.scorers import Guidelines, Safety, ScorerSamplingConfig, delete_scorer, list_scorers

mlflow.set_experiment(EXPERIMENTO)
kw = {"model": JUEZ} if JUEZ else {}

@scorer
def formato_prod(outputs):
    import json   # import dentro de la función: requisito de los scorers de monitoreo
    campos = {"alerta_id", "disposicion", "justificacion", "citas"}
    return isinstance(outputs, dict) and campos <= set(outputs)

MONITORES = {
    "seguridad_prod": Safety(**kw),
    "cita_normativa_prod": Guidelines(name="cita_normativa_prod", guidelines=[
        "La respuesta debe citar al menos un documento normativo o indicar que requiere revisión humana."], **kw),
    "formato_prod": formato_prod,
}
activos = []
for nombre, sc in MONITORES.items():
    try:
        try:
            delete_scorer(name=nombre)       # re-ejecutable
        except Exception:
            pass
        sc.register(name=nombre).start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))
        activos.append(nombre)
    except Exception as e:
        print(f"⚠️ {nombre}: {str(e)[:250]}")
print(f"✅ Monitoreando con: {activos}" if activos else
      "⚠️ El monitoreo no está disponible (requiere serverless). Lee el resto del notebook como referencia.")

# COMMAND ----------

# MAGIC %md
# MAGIC Simulamos tráfico de producción (con el tag `entorno=produccion` para filtrarlo después):

# COMMAND ----------

@mlflow.trace(name="solicitud_produccion", span_type=SpanType.CHAIN)
def atender(pregunta):
    mlflow.update_current_trace(tags={"entorno": "produccion"})
    return predict_prod(pregunta=pregunta)

for a in ["AT-002", "AT-005", "AT-012"]:
    print(a, "→", atender(f"Analiza la alerta {a} y emite tu dictamen.").get("disposicion"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Vuelve en ~20 minutos** y ejecuta la siguiente celda (o mira la pestaña *Traces*, filtrando por
# MAGIC `entorno = produccion`): cada traza tendrá las evaluaciones de los scorers de monitoreo.

# COMMAND ----------

for t in mlflow.search_traces(filter_string="tags.entorno = 'produccion'", max_results=10, return_type="list"):
    evals = {a.name: _valor(a) for a in t.info.assessments or [] if a.name in MONITORES}
    print(t.info.tags.get("alerta_id"), evals or "(aún sin evaluar)")
print("Scorers registrados:", [s.name for s in list_scorers()])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Limpieza
# MAGIC Pon el widget **limpiar** en `si` y ejecuta la celda para detener y borrar los scorers de monitoreo.

# COMMAND ----------

if dbutils.widgets.get("limpiar") == "si":
    for nombre in MONITORES:
        try:
            delete_scorer(name=nombre)
            print(f"🧹 {nombre} borrado")
        except Exception as e:
            print(f"{nombre}: {str(e)[:120]}")
else:
    print("Los scorers siguen activos (widget limpiar = no).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen
# MAGIC - **Trazas en UC:** el registro de auditoría del agente pasa a ser una tabla gobernada del Lakehouse.
# MAGIC - **Monitoreo:** los scorers con los que aprobaste el agente siguen calificándolo en producción, sobre una muestra.
# MAGIC - El ciclo completo: **trazar → evaluar → corregir → monitorear**, y cada error de producción vuelve al set de
# MAGIC   evaluación.
