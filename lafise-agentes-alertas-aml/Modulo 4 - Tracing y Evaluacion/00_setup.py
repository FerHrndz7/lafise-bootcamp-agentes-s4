# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo 4 · Preparación — Tracing y Evaluación del Agente de Alertas AML
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Duración** | ~95 min de práctica (esta preparación: ~8 min) |
# MAGIC | **Nivel** | Intermedio |
# MAGIC | **Lenguaje** | Python |
# MAGIC | **Cómputo** | Clúster **clásico** (DBR **17.3 LTS** recomendado, modo de acceso *Standard*) |
# MAGIC | **Catalog.Schema** | `<catálogo_asignado>.pld_aml_<tu_usuario>` (el mismo de las Sesiones 1 a 3) |
# MAGIC
# MAGIC ## Qué haremos hoy 🔎
# MAGIC Ya tenemos el **Agente de Alertas** por dos caminos (código en la S2, Genie Agent en la S3). La pregunta de hoy
# MAGIC es la que hará cualquier auditor o regulador: **¿podemos confiar en sus dictámenes?** Para responderla:
# MAGIC
# MAGIC 1. `00_setup` — dependencias, datos, herramientas y **verdad de terreno** (**estás aquí**).
# MAGIC 2. `01_tracing_del_agente` — **MLflow Tracing**: cada llamada al modelo y a cada herramienta, con latencia y tokens.
# MAGIC 3. `02_dataset_de_evaluacion` — un **set de evaluación** en Unity Catalog: 12 alertas con la disposición de un analista.
# MAGIC 4. `03_evaluar_con_jueces` — `mlflow.genai.evaluate()` con **jueces LLM** y scorers de código: el *scorecard*.
# MAGIC 5. `04_alucinaciones_y_guardrails` — provocamos una **alucinación** (quitamos el contexto) y añadimos **guardrails**.
# MAGIC 6. `05_reto` — **tarea**: la regla PEP como scorer + guardrail.
# MAGIC 7. `06_monitoreo_produccion` — *opcional* (serverless, Beta): los mismos scorers sobre tráfico de producción.
# MAGIC
# MAGIC ### Objetivos de aprendizaje
# MAGIC - Leer la traza de un agente y explicar *por qué* recomendó una disposición.
# MAGIC - Construir un set de evaluación con verdad de terreno de analistas.
# MAGIC - Medir correctitud y **groundedness** (anti-alucinación) con jueces LLM y scorers propios.
# MAGIC - Poner **guardrails** de entrada y salida y medir su efecto.
# MAGIC
# MAGIC > ⚠️ **Datos sintéticos.** Todo es inventado (Faker + hechos regulatorios plausibles pero ficticios).
# MAGIC > Nunca uses PII real de clientes en un workspace de laboratorio.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 1 — Dependencias
# MAGIC Lo ideal es que el clúster ya tenga instaladas las librerías de `requirements.txt` (librerías de clúster).
# MAGIC Esta celda las asegura para **este** notebook (si ya están, termina rápido). Cada notebook trae la misma celda.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.17" "databricks-langchain>=0.20.0" "langgraph>=1.2,<1.3" "unitycatalog-langchain[databricks]>=0.3.0" "databricks-sdk>=0.94.0" "faker>=25.0.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 2 — Configuración (aislamiento por participante)
# MAGIC
# MAGIC - **catalog** (requerido): el **mismo catálogo de las Sesiones 1 a 3**. Tu schema lleva tu usuario automáticamente.
# MAGIC - **llm** (opcional): el modelo del agente. Vacío = elegimos el mejor modelo con *tool calling* disponible.
# MAGIC - **juez** (opcional): el modelo de los **jueces LLM**. Vacío = el juez gestionado por Databricks. Escríbelo
# MAGIC   solo si el facilitador te lo indica (por ejemplo, si el workspace tiene desactivadas las
# MAGIC   *Partner-powered AI features*).

# COMMAND ----------

import re

dbutils.widgets.text("catalog", "", "Catálogo asignado (REQUERIDO)")
dbutils.widgets.text("schema", "pld_aml", "Schema base")
dbutils.widgets.text("llm", "", "Endpoint LLM del agente (opcional)")
dbutils.widgets.text("juez", "", "Modelo de los jueces (opcional)")

user_email = spark.sql("SELECT current_user()").collect()[0][0]
user_prefix = re.sub(r"[^a-z0-9]", "_", user_email.split("@")[0].lower())
CATALOG = dbutils.widgets.get("catalog").strip()
assert CATALOG, ("Escribe en el widget 'catalog' el catálogo que te asignó el facilitador "
                 "(el mismo de las Sesiones 1 a 3).")
SCHEMA = f"{(dbutils.widgets.get('schema').strip() or 'pld_aml')}_{user_prefix}"
FQ = f"`{CATALOG}`.{SCHEMA}"
EXPERIMENTO = f"/Users/{user_email}/lafise_s4_tracing_evaluacion"

print(f"Usuario:     {user_email}")
print(f"Destino:     {CATALOG}.{SCHEMA}")
print(f"Experimento: {EXPERIMENTO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 3 — Verificación del entorno

# COMMAND ----------

import sys
from importlib.metadata import version

print(f"Python: {sys.version.split()[0]}")
assert sys.version_info >= (3, 10), "Se requiere Python 3.10+"
for p in ["mlflow", "databricks-sdk", "databricks-langchain", "langgraph", "unitycatalog-ai"]:
    try:
        print(f"  {p:22s} {version(p)}")
    except Exception:
        print(f"  {p:22s} ⚠️ no instalado")
assert tuple(int(x) for x in version("mlflow").split(".")[:2]) >= (3, 17), "Se requiere MLflow 3.17 o superior"

try:
    print(f"DBR del clúster: {spark.conf.get('spark.databricks.clusterUsageTags.sparkVersion')}")
except Exception:
    print("DBR del clúster: no legible aquí (confirma 17.3 LTS en la configuración del clúster).")

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %run ./utils/agente_eval_utils

# COMMAND ----------

LLM = elegir_llm(w, dbutils.widgets.get("llm").strip())
JUEZ = modelo_juez(dbutils.widgets.get("juez"))
print(f"Modelo del agente: {LLM}")
print(f"Modelo de los jueces: {JUEZ or 'juez gestionado por Databricks (predeterminado)'}")

from databricks_langchain import ChatDatabricks
print("Respuesta del modelo:", ChatDatabricks(endpoint=LLM, temperature=0).invoke("Responde solo: listo").content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 4 — Permisos
# MAGIC Hoy crearás **tablas** (la verdad de terreno y el set de evaluación) y **funciones** en tu schema.

# COMMAND ----------

try:
    spark.sql(f"USE CATALOG `{CATALOG}`")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
    spark.sql(f"CREATE OR REPLACE TABLE {FQ}._prueba_permiso AS SELECT 1 AS ok")
    spark.sql(f"CREATE OR REPLACE FUNCTION {FQ}._prueba_permiso_fn() RETURNS STRING RETURN 'ok'")
    spark.sql(f"DROP TABLE IF EXISTS {FQ}._prueba_permiso")
    spark.sql(f"DROP FUNCTION IF EXISTS {FQ}._prueba_permiso_fn")
    print(f"✅ Puedes crear tablas y funciones en {CATALOG}.{SCHEMA}")
except Exception as e:
    raise PermissionError(
        f"No se pueden crear objetos en {CATALOG}.{SCHEMA}. Pide al administrador USE CATALOG + USE SCHEMA + "
        f"CREATE TABLE + CREATE FUNCTION. Error subyacente: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 5 — Datos, herramientas y verdad de terreno
# MAGIC El bootstrap **reutiliza** lo que construiste en las sesiones anteriores y crea las **13 alertas testigo** con
# MAGIC su disposición esperada (la "verdad de terreno" de un analista). Si tu índice de AI Search ya no existe, lo
# MAGIC recrea **sin esperar**: mientras sincroniza, el agente busca la normativa por palabras clave.

# COMMAND ----------

# MAGIC %run ./data/bootstrap_sesion4 $catalog=$catalog $schema=$schema

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 6 — Experimento de MLflow y prueba del agente
# MAGIC Todas las trazas y evaluaciones de hoy quedan en **un experimento** de MLflow. Hacemos una primera corrida del
# MAGIC agente con `AT-001` (estructuración clara) para confirmar que todo funciona.

# COMMAND ----------

import mlflow

mlflow.set_experiment(EXPERIMENTO)
agente, info = construir_agente(w, spark, CATALOG, SCHEMA, LLM)
print(json.dumps(info, ensure_ascii=False, indent=2))

dictaminar = crear_predict_fn(agente, version="v1-setup")
dictamen = dictaminar(pregunta="Analiza la alerta AT-001 y emite tu dictamen.")
print(json.dumps(dictamen, ensure_ascii=False, indent=2))

# COMMAND ----------

# Resultado esperado: REPORTAR_ROS (o ESCALAR_INVESTIGACION), con al menos una cita.
assert dictamen.get("disposicion") in (R, E), f"AT-001 es estructuración clara; obtuvimos {dictamen}"
assert dictamen.get("citas"), "El dictamen debe citar al menos un fundamento"
print(f"✅ AT-001 → {dictamen['disposicion']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 7 — Resumen

# COMMAND ----------

print("=" * 64)
print("PREPARACIÓN COMPLETA ✅")
print("=" * 64)
print(f"Catalog.Schema   : {CATALOG}.{SCHEMA}")
print(f"Experimento      : {EXPERIMENTO}")
print(f"Modelo del agente: {LLM}  ·  jueces: {JUEZ or 'gestionado por Databricks'}")
print(f"Funciones UC     : {info['modo_funciones_uc']}  ·  retrieval: {info['modo_retrieval']}")
display(spark.sql(f"SELECT alerta_id, disposicion_esperada, disposiciones_aceptables, hit_sanciones, es_pep "
                  f"FROM {FQ}.verdad_terreno ORDER BY alerta_id"))
print("➡️  Continúa con 01_tracing_del_agente")
