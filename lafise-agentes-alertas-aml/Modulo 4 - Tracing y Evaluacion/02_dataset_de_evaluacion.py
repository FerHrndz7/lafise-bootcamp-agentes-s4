# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · El set de evaluación: el criterio del analista, en datos
# MAGIC
# MAGIC **Objetivo:** construir un **set de evaluación** (alertas + lo que haría un analista experimentado) y guardarlo
# MAGIC en **Unity Catalog**, versionado y gobernado. Es la "prueba de regresión" del agente: cada cambio de prompt, de
# MAGIC modelo o de herramientas se mide contra este mismo set. ⏱️ ~10 min.
# MAGIC
# MAGIC ### Anatomía de un registro
# MAGIC
# MAGIC     {
# MAGIC       "inputs":       {"pregunta": "Analiza la alerta AT-001 y emite tu dictamen."},   ← lo que recibe el agente
# MAGIC       "expectations": {                                                                 ← la verdad de terreno
# MAGIC          "expected_facts": ["Hubo 9 depósitos en efectivo justo por debajo de USD 10,000", ...],
# MAGIC          "disposicion_esperada": "REPORTAR_ROS",
# MAGIC          "disposiciones_aceptables": ["REPORTAR_ROS", "ESCALAR_INVESTIGACION"],
# MAGIC          "hit_sanciones": false, "es_pep": false
# MAGIC       }
# MAGIC     }
# MAGIC
# MAGIC - `inputs` se pasa **desempacado** a la función del agente: `predict_fn(pregunta="...")`.
# MAGIC - `expected_facts` es una clave **reservada**: la usa el juez integrado `Correctness`.
# MAGIC - El resto de claves son nuestras: las leen los **scorers de código** del notebook `03`.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.17" "databricks-sdk>=0.94.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re

dbutils.widgets.text("catalog", "", "Catálogo asignado (REQUERIDO)")
dbutils.widgets.text("schema", "pld_aml", "Schema base")
user_email = spark.sql("SELECT current_user()").collect()[0][0]
user_prefix = re.sub(r"[^a-z0-9]", "_", user_email.split("@")[0].lower())
CATALOG = dbutils.widgets.get("catalog").strip()
assert CATALOG, "Escribe en el widget 'catalog' el mismo catálogo que usaste en 00_setup."
SCHEMA = f"{(dbutils.widgets.get('schema').strip() or 'pld_aml')}_{user_prefix}"
FQ = f"`{CATALOG}`.{SCHEMA}"
EXPERIMENTO = f"/Users/{user_email}/lafise_s4_tracing_evaluacion"
DATASET = f"{CATALOG}.{SCHEMA}.eval_dictamenes_aml"

import json, mlflow
mlflow.set_experiment(EXPERIMENTO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 — La verdad de terreno
# MAGIC El bootstrap creó la tabla `verdad_terreno` con 13 alertas testigo. En un proyecto real esta tabla sale de
# MAGIC **casos ya dispuestos por analistas** (y de las anotaciones en trazas que vimos en el `01`). Usamos las 12 del
# MAGIC set base: `AT-013` la reservamos para el reto.
# MAGIC
# MAGIC Fíjate en la columna `disposiciones_aceptables`: en PLD hay casos donde dos analistas razonables difieren
# MAGIC (¿escalar o reportar?). Evaluar contra una lista de aceptables es más justo que exigir una sola respuesta.

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {FQ}.verdad_terreno WHERE en_eval_base ORDER BY alerta_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 — De filas a registros de evaluación

# COMMAND ----------

def a_registro(f) -> dict:
    return {
        "inputs": {"pregunta": f["pregunta"]},
        "expectations": {
            "expected_facts": list(f["hechos_esperados"]),
            "disposicion_esperada": f["disposicion_esperada"],
            "disposiciones_aceptables": list(f["disposiciones_aceptables"]),
            "hit_sanciones": bool(f["hit_sanciones"]),
            "es_pep": bool(f["es_pep"]),
        },
    }

filas = spark.sql(f"SELECT * FROM {FQ}.verdad_terreno WHERE en_eval_base ORDER BY alerta_id").collect()
registros = [a_registro(f) for f in filas]
print(f"{len(registros)} registros. Ejemplo:")
print(json.dumps(registros[0], ensure_ascii=False, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 — Guardarlo como **dataset de evaluación de MLflow** en Unity Catalog
# MAGIC `mlflow.genai.datasets.create_dataset(name="catalogo.schema.tabla")` crea una tabla de UC con el esquema de
# MAGIC evaluación (inputs, expectations, fuente, tags, quién lo creó y cuándo) y la asocia al experimento. Ventajas:
# MAGIC permisos y linaje de UC, versionado, y se puede ampliar desde la interfaz con trazas reales
# MAGIC (*Traces → Actions → Add to evaluation dataset*).
# MAGIC
# MAGIC Para que el notebook sea re-ejecutable, si el dataset ya existe lo borramos y lo volvemos a crear.
# MAGIC
# MAGIC > **Si falla** (por ejemplo, por permisos), no pasa nada: guardamos los mismos registros en una tabla Delta
# MAGIC > normal y el `03` evalúa a partir de ella. `mlflow.genai.evaluate()` acepta tanto un dataset de MLflow como una
# MAGIC > lista de diccionarios o un DataFrame.

# COMMAND ----------

import mlflow.genai.datasets as eval_ds

MODO_DATASET = "mlflow_uc"
try:
    try:
        eval_ds.delete_dataset(name=DATASET)
    except Exception:
        pass                                        # no existía
    dataset = eval_ds.create_dataset(name=DATASET)
    dataset.merge_records(registros)
    print(f"✅ Dataset de evaluación de MLflow en UC: {DATASET}")
except Exception as e:
    MODO_DATASET = "tabla_delta"
    print(f"⚠️ No se pudo crear el dataset de MLflow ({str(e)[:200]}). Uso una tabla Delta.")
    (spark.createDataFrame([(json.dumps(r["inputs"], ensure_ascii=False), json.dumps(r["expectations"], ensure_ascii=False))
                            for r in registros], "inputs string, expectations string")
     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.eval_dictamenes_aml_tabla"))
    print(f"✅ Tabla Delta: {CATALOG}.{SCHEMA}.eval_dictamenes_aml_tabla")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 — Revisar el dataset

# COMMAND ----------

if MODO_DATASET == "mlflow_uc":
    df = eval_ds.get_dataset(name=DATASET).to_df()
    print(f"Registros: {len(df)} · columnas: {list(df.columns)}")
    display(df[[c for c in ["inputs", "expectations", "created_by", "create_time"] if c in df.columns]])
else:
    display(spark.table(f"{FQ}.eval_dictamenes_aml_tabla"))

# COMMAND ----------

assert (len(eval_ds.get_dataset(name=DATASET).to_df()) if MODO_DATASET == "mlflow_uc"
        else spark.table(f"{FQ}.eval_dictamenes_aml_tabla").count()) == 12
print("✅ 12 registros listos para evaluar")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 — ¿Cómo crece un set de evaluación en la vida real?
# MAGIC 1. **Casos dorados** (lo que hicimos): ejemplos curados por el equipo PLD, uno por tipología y regla dura.
# MAGIC 2. **Desde producción**: filtras trazas (por ejemplo, las que un analista marcó como incorrectas con un feedback,
# MAGIC    como en el paso 7 del `01`) y las agregas con `dataset.merge_records(mlflow.search_traces(...))` o desde la
# MAGIC    interfaz. Las expectativas registradas en la traza viajan con ella.
# MAGIC 3. **Sintéticos**: Databricks puede generar preguntas a partir de los documentos (útil para la parte normativa).
# MAGIC
# MAGIC Regla práctica: **cada error que encuentren en producción se convierte en un registro del set**. Así el mismo
# MAGIC error no vuelve a pasar sin que la evaluación lo detecte.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen
# MAGIC - Convertiste la **verdad de terreno** de los analistas en registros `inputs` + `expectations`.
# MAGIC - Guardaste el set como **dataset de evaluación de MLflow en Unity Catalog** (o como tabla Delta de respaldo).
# MAGIC
# MAGIC ➡️ En `03_evaluar_con_jueces` corremos el agente contra el set y lo calificamos con **jueces LLM**.
