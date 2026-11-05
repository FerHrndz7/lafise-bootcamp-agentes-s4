# Databricks notebook source
# MAGIC %md
# MAGIC # Validación de entorno — Workshop de Agentes AML (Sesión 4: MLflow Tracing & Evaluation)
# MAGIC
# MAGIC Ejecuta este notebook en el **workspace de desarrollo de LAFISE**, con un **clúster clásico** adjunto
# MAGIC (DBR **17.3 LTS**, modo *Standard*), para confirmar que el entorno soporta el laboratorio de la Sesión 4.
# MAGIC
# MAGIC Prueba cada pieza (un modelo con *tool calling*, una traza de MLflow, los jueces LLM, un juez personalizado, un
# MAGIC dataset de evaluación en Unity Catalog y, opcionales, trazas en UC y monitoreo en producción) y reporta
# MAGIC **PASA / FALLA / ALTERNATIVA** por requisito. Al final borra lo que creó.
# MAGIC
# MAGIC ## Instrucciones
# MAGIC 1. Adjunta un **clúster clásico** DBR 17.3 LTS.
# MAGIC 2. En el widget **catalog** escribe el catálogo de los asistentes (el de las Sesiones 1 a 3).
# MAGIC 3. *(Opcional)* en **schema_s1** escribe el schema de un asistente de la Sesión 1 (por ejemplo
# MAGIC    `pld_aml_juan_perez`) para probar su índice de AI Search.
# MAGIC 4. **Run all** (~6 min). Comparte la **tabla resumen final** con el facilitador.
# MAGIC
# MAGIC | Estado | Significado |
# MAGIC |---|---|
# MAGIC | **PASA** | Funciona tal cual en la sesión |
# MAGIC | **ALTERNATIVA** | No está disponible, pero el laboratorio tiene un camino alternativo (se indica cuál) |
# MAGIC | **FALLA** | Bloqueante: hay que resolverlo con el administrador antes de la sesión |

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.17" "databricks-langchain>=0.20.0" "unitycatalog-langchain[databricks]>=0.3.0" "databricks-sdk>=0.94.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catálogo (requerido)")
dbutils.widgets.text("schema", "validacion_s4", "Schema de prueba")
dbutils.widgets.text("schema_s1", "", "Schema de la Sesión 1 (opcional)")
dbutils.widgets.text("warehouse", "", "SQL warehouse (opcional)")

import json, os, re, time, uuid
resultados = {}

def marcar(req, estado, detalle=""):
    resultados[req] = (estado, str(detalle)[:300])
    print(f"[{estado}] {req} — {detalle}")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip() or "validacion_s4"
assert CATALOG, "Escribe el catálogo en el widget 'catalog' antes de ejecutar."
FQ = f"`{CATALOG}`.{SCHEMA}"
usuario = spark.sql("SELECT current_user()").collect()[0][0]
EXP = f"/Users/{usuario}/validacion_s4_{uuid.uuid4().hex[:6]}"

import mlflow
from importlib.metadata import version
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
marcar("0. Librerías instaladas desde PyPI", "PASA", f"mlflow {version('mlflow')}, databricks-langchain {version('databricks-langchain')}")

# COMMAND ----------

try:
    dbr = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
    marcar("1. Clúster clásico y runtime", "PASA" if dbr.startswith("17.") else "ALTERNATIVA", dbr)
except Exception as e:
    marcar("1. Clúster clásico y runtime", "ALTERNATIVA", f"No legible: {e}")

# COMMAND ----------

LLM = None
try:
    from databricks_langchain import ChatDatabricks
    from langchain_core.tools import tool
    servidos = {e.name for e in w.serving_endpoints.list()}
    LLM = next(c for c in ["databricks-claude-sonnet-4-6", "databricks-claude-sonnet-4-5", "databricks-claude-sonnet-5-5",
                           "databricks-gpt-5-4-mini", "databricks-meta-llama-3-3-70b-instruct",
                           "databricks-llama-4-maverick"] if c in servidos)

    @tool
    def score_alerta(alerta_id: str) -> str:
        """Devuelve el score de riesgo de una alerta."""
        return "91"

    r = ChatDatabricks(endpoint=LLM, temperature=0).bind_tools([score_alerta]).invoke(
        "¿Cuál es el score de la alerta AT-001? Usa la herramienta.")
    marcar("2. Modelo con tool calling (Foundation Model APIs)", "PASA" if r.tool_calls else "FALLA",
           f"{LLM} · tool_calls={bool(r.tool_calls)}")
except Exception as e:
    marcar("2. Modelo con tool calling (Foundation Model APIs)", "FALLA", e)

# COMMAND ----------

try:
    mlflow.set_experiment(EXP)
    mlflow.langchain.autolog()
    ChatDatabricks(endpoint=LLM, temperature=0).invoke("Responde solo: ok")
    tid = mlflow.get_last_active_trace_id()
    traza = mlflow.get_trace(tid)
    assert traza and traza.data.spans, "no se registró la traza"
    from mlflow.entities import AssessmentSource, AssessmentSourceType
    mlflow.log_feedback(trace_id=tid, name="prueba", value=True,
                        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=usuario))
    time.sleep(3)
    encontradas = mlflow.search_traces(max_results=5, return_type="list")
    marcar("3. MLflow Tracing (autolog, búsqueda y feedback)", "PASA",
           f"{len(traza.data.spans)} spans · {len(encontradas)} traza(s) en el experimento")
except Exception as e:
    marcar("3. MLflow Tracing (autolog, búsqueda y feedback)", "FALLA", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Jueces LLM integrados
# MAGIC Los jueces usan por defecto un modelo gestionado por Databricks. Si un admin desactivó las
# MAGIC *Partner-powered AI features*, ese modelo no está disponible, pero los jueces funcionan pasando un modelo propio
# MAGIC (`model="databricks:/<endpoint>"`). Si sale **ALTERNATIVA**, el detalle dice qué escribir en el widget **juez** de
# MAGIC los notebooks.

# COMMAND ----------

from mlflow.genai.judges import is_correct, is_grounded

def probar_juez(model=None):
    kw = {"model": model} if model else {}
    fb = is_correct(request="¿Cuál es el umbral del Reporte de Transacción en Efectivo?",
                    response="El umbral es de USD 10,000.", expected_facts=["El umbral es USD 10,000"], **kw)
    if getattr(fb, "error", None):
        raise RuntimeError(getattr(fb.error, "error_message", fb.error))
    return str(fb.value)

JUEZ = None
try:
    v = probar_juez()
    marcar("4. Jueces LLM integrados (modelo predeterminado)", "PASA" if v.lower() == "yes" else "ALTERNATIVA",
           f"Correctness = {v}")
except Exception as e:
    error_default = str(e)[:150]
    for cand in ["databricks-gpt-oss-120b", "databricks-meta-llama-3-3-70b-instruct", "databricks-gpt-5-mini",
                 "databricks-claude-sonnet-4-6"]:
        if cand in servidos:
            try:
                probar_juez(f"databricks:/{cand}")
                JUEZ = f"databricks:/{cand}"
                break
            except Exception:
                pass
    if JUEZ:
        marcar("4. Jueces LLM integrados (modelo predeterminado)", "ALTERNATIVA",
               f"El juez predeterminado falló ({error_default}). Funciona con un modelo propio: en la sesión escriban "
               f"'{JUEZ.split(':/')[1]}' en el widget 'juez'")
    else:
        marcar("4. Jueces LLM integrados (modelo predeterminado)", "FALLA", error_default)

# COMMAND ----------

try:
    from typing import Literal
    from mlflow.entities import Feedback
    from mlflow.genai.judges import make_judge
    from mlflow.genai.scorers import Guidelines, scorer
    kw = {"model": JUEZ} if JUEZ else {}

    @scorer
    def disposicion_valida(outputs):
        return outputs.get("disposicion") in ("CERRAR_SIN_ACCION", "ESCALAR_INVESTIGACION", "REPORTAR_ROS")

    juez = make_judge(name="calidad", instructions="Califica el dictamen {{ outputs }} para la solicitud {{ inputs }}: "
                      "responde 'bueno' si recomienda una disposición justificada, si no 'malo'.",
                      feedback_value_type=Literal["bueno", "malo"], **kw)
    datos = [{"inputs": {"pregunta": "Analiza la alerta AT-001"},
              "outputs": {"disposicion": "REPORTAR_ROS", "justificacion": "9 depósitos en efectivo justo bajo el "
                          "umbral de USD 10,000: estructuración (Tipología 1)."}}]
    res = mlflow.genai.evaluate(data=datos, scorers=[disposicion_valida, juez,
                                Guidelines(name="recomienda", guidelines=["La respuesta debe recomendar una disposición para la alerta."], **kw)])
    t = mlflow.search_traces(run_id=res.run_id, return_type="list")[0]
    vals = {a.name: getattr(a, "value", None) for a in t.info.assessments}
    errores = [a.name for a in t.info.assessments if getattr(a, "error", None)]
    ok = not errores and {"disposicion_valida", "calidad", "recomienda"} <= set(vals)
    marcar("5. mlflow.genai.evaluate + make_judge + Guidelines + scorer de código", "PASA" if ok else "FALLA",
           f"{vals}" + (f" · errores en {errores}" if errores else ""))
except Exception as e:
    marcar("5. mlflow.genai.evaluate + make_judge + Guidelines + scorer de código", "FALLA", e)

# COMMAND ----------

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
    import mlflow.genai.datasets as eval_ds
    nombre = f"{CATALOG}.{SCHEMA}.eval_prueba"
    ds = eval_ds.create_dataset(name=nombre)
    ds.merge_records([{"inputs": {"pregunta": "Analiza la alerta AT-001"},
                       "expectations": {"expected_facts": ["Estructuración"]}}])
    n = len(eval_ds.get_dataset(name=nombre).to_df())
    eval_ds.delete_dataset(name=nombre)
    marcar("6. Dataset de evaluación de MLflow en Unity Catalog", "PASA" if n == 1 else "ALTERNATIVA", f"{n} registro(s)")
except Exception as e:
    marcar("6. Dataset de evaluación de MLflow en Unity Catalog", "ALTERNATIVA",
           f"{str(e)[:200]} · El notebook 02 guarda el set en una tabla Delta y la evaluación funciona igual")

# COMMAND ----------

S1 = dbutils.widgets.get("schema_s1").strip()
if S1:
    try:
        r = w.vector_search_indexes.query_index(index_name=f"{CATALOG}.{S1}.chunks_index",
                                                columns=["chunk_id", "source_doc"], query_text="umbral de efectivo",
                                                num_results=2, query_type="HYBRID")
        marcar("7. Índice de AI Search de la Sesión 1", "PASA", f"{len(r.result.data_array or [])} resultados")
    except Exception as e:
        marcar("7. Índice de AI Search de la Sesión 1", "ALTERNATIVA",
               f"{str(e)[:150]} · El agente busca la normativa por palabras clave en la tabla chunks")
else:
    marcar("7. Índice de AI Search de la Sesión 1", "ALTERNATIVA",
           "No se indicó schema_s1 (sin índice, el agente busca por palabras clave en la tabla chunks)")

# COMMAND ----------

try:
    spark.sql(f"""CREATE OR REPLACE FUNCTION {FQ}.score_alerta(p_id STRING COMMENT 'ID de alerta')
                  RETURNS TABLE (alerta_id STRING, score INT) COMMENT 'Score de una alerta'
                  RETURN SELECT p_id, 91""")
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient
    t0 = time.time()
    r = DatabricksFunctionClient().execute_function(f"{CATALOG}.{SCHEMA}.score_alerta", {"p_id": "AT-001"})
    assert not r.error, r.error
    marcar("8. Funciones de UC ejecutadas en serverless", "PASA", f"{time.time() - t0:.1f}s")
except Exception as e:
    marcar("8. Funciones de UC ejecutadas en serverless", "ALTERNATIVA",
           f"{str(e)[:150]} · El agente las ejecuta con spark.sql en el clúster clásico (igual que en la S2)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Opcionales (notebook 06)

# COMMAND ----------

EXP_UC = None
try:
    pref = dbutils.widgets.get("warehouse").strip()
    whs = [x for x in w.warehouses.list() if not pref or pref in (x.id, x.name)]
    assert whs, "No hay SQL warehouses visibles para este usuario"
    WH = sorted(whs, key=lambda x: not x.enable_serverless_compute)[0]
    from mlflow.entities.trace_location import UnityCatalog
    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = WH.id
    EXP_UC = f"{EXP}_uc"
    exp = mlflow.set_experiment(experiment_name=EXP_UC, trace_location=UnityCatalog(
        catalog_name=CATALOG, schema_name=SCHEMA, table_prefix="trazas_prueba"))
    ChatDatabricks(endpoint=LLM, temperature=0).invoke("Responde solo: ok")
    marcar("9. (Opcional) Trazas en tablas de Unity Catalog", "PASA", exp.trace_location.full_otel_spans_table_name)
except Exception as e:
    marcar("9. (Opcional) Trazas en tablas de Unity Catalog", "ALTERNATIVA",
           f"{str(e)[:200]} · Las trazas quedan en el experimento de MLflow (suficiente para la sesión)")
finally:
    os.environ.pop("MLFLOW_TRACING_SQL_WAREHOUSE_ID", None)
    mlflow.set_experiment(EXP)

# COMMAND ----------

try:
    from mlflow.genai.scorers import Safety, ScorerSamplingConfig, delete_scorer
    kw = {"model": JUEZ} if JUEZ else {}
    Safety(**kw).register(name="validacion_safety").start(sampling_config=ScorerSamplingConfig(sample_rate=0.1))
    delete_scorer(name="validacion_safety")
    marcar("10. (Opcional) Monitoreo en producción (Beta, serverless)", "PASA", "register + start + delete")
except Exception as e:
    marcar("10. (Opcional) Monitoreo en producción (Beta, serverless)", "ALTERNATIVA",
           f"{str(e)[:200]} · El notebook 06 se presenta como demostración")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Limpieza y resumen

# COMMAND ----------

try:
    for nombre in [EXP, EXP_UC]:
        if nombre and (e := mlflow.get_experiment_by_name(nombre)):
            mlflow.delete_experiment(e.experiment_id)
    spark.sql(f"DROP SCHEMA IF EXISTS {FQ} CASCADE")
    print("🧹 Limpieza completa")
except Exception as e:
    print("⚠️ Limpieza parcial:", e)

print(f"\n{'Requisito':72s} {'Estado':12s} Detalle")
for req, (estado, det) in resultados.items():
    print(f"{req:72s} {estado:12s} {det[:110]}")
