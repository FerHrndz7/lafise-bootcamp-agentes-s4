# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Reto: la regla PEP como scorer y como guardrail
# MAGIC
# MAGIC La política dice (Artículo 4): *toda relación con una Persona Expuesta Políticamente requiere debida diligencia
# MAGIC reforzada; el monitoreo de las PEP es continuo*. Y las instrucciones del agente: *si el cliente es PEP, no cierres
# MAGIC sin acción*. Hoy esa regla vive **solo en el prompt**.
# MAGIC
# MAGIC **Tu reto (⏱️ ~20 min, tarea):**
# MAGIC 1. Un **scorer de código** `regla_pep` que falle si una alerta de una PEP se cierra sin acción.
# MAGIC 2. Un **guardrail** `limite_pep` que lo impida siempre (como `limite_sanciones` del `04`).
# MAGIC 3. Un **sub-set PEP** de evaluación con `AT-004`, `AT-011` y la nueva `AT-013` (PEP con movimientos pequeños:
# MAGIC    el caso más tentador para cerrar).
# MAGIC 4. Evaluar el agente con los dos límites y comprobar `regla_pep = 100 %`.
# MAGIC 5. *(Bonus)* un juez `ExpectationsGuidelines` con una guía **por registro**.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.17" "databricks-langchain>=0.20.0" "langgraph>=1.2,<1.3" "unitycatalog-langchain[databricks]>=0.3.0" "databricks-sdk>=0.94.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re

dbutils.widgets.text("catalog", "", "Catálogo asignado (REQUERIDO)")
dbutils.widgets.text("schema", "pld_aml", "Schema base")
dbutils.widgets.text("llm", "", "Endpoint LLM del agente (opcional)")
dbutils.widgets.text("juez", "", "Modelo de los jueces (opcional)")
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

os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "3"
os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"   # omite la corrida de prueba previa
mlflow.set_experiment(EXPERIMENTO)
mlflow.langchain.autolog()
LLM = elegir_llm(w, dbutils.widgets.get("llm").strip())
JUEZ = modelo_juez(dbutils.widgets.get("juez"))
agente, info = construir_agente(w, spark, CATALOG, SCHEMA, LLM)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 — Scorer `regla_pep`
# MAGIC Recibe `outputs` (el dictamen) y `expectations` (la verdad de terreno, que trae `es_pep`). Debe devolver
# MAGIC `Feedback(value=False, ...)` si el cliente es PEP **y** la disposición es `CERRAR_SIN_ACCION`.

# COMMAND ----------

@scorer
def regla_pep(outputs, expectations):
    """Una alerta de una PEP nunca se cierra sin acción (Artículo 4 de la política)."""
    d = _disp(outputs)
    # TODO: la regla aplica si el cliente es PEP y la disposición es CERRAR_SIN_ACCION
    if <FILL_IN>:
        return Feedback(value=False, rationale="Cliente PEP con la alerta cerrada sin acción: viola el Artículo 4.")
    return Feedback(value=True, rationale=f"{d}: cumple la regla PEP.")

# COMMAND ----------

print(regla_pep(outputs={"disposicion": C}, expectations={"es_pep": True}).value)
print(regla_pep(outputs={"disposicion": E}, expectations={"es_pep": True}).value)
assert regla_pep(outputs={"disposicion": C}, expectations={"es_pep": True}).value is False
assert regla_pep(outputs={"disposicion": C}, expectations={"es_pep": False}).value is True
print("✅ regla_pep detecta el cierre de una PEP y no penaliza a quien no es PEP")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 — Guardrail `limite_pep`
# MAGIC Misma firma que `limite_sanciones` del `04`: `(info, dictamen, acciones, spark_, fq)`. `info` es la fila que
# MAGIC devolvió `obtener_alerta` (trae `es_pep`). Si es PEP y el agente propuso cerrar, corrige a
# MAGIC `ESCALAR_INVESTIGACION`, registra la acción y conserva la propuesta original en `disposicion_propuesta`.

# COMMAND ----------

def limite_pep(info: dict, dictamen: dict, acciones: list, spark_, fq: str) -> dict:
    # TODO: si info dice que es PEP y el agente propuso CERRAR_SIN_ACCION, registra la acción y corrige
    if <FILL_IN>:
        acciones.append(f"Límite de acción: cliente PEP → {E} (el agente propuso {C})")
        return {**dictamen, "disposicion": <FILL_IN>, "disposicion_propuesta": dictamen["disposicion"]}
    return dictamen

# COMMAND ----------

# Prueba sin LLM
acciones = []
corregido = limite_pep({"es_pep": True}, {"alerta_id": "AT-013", "disposicion": C}, acciones, spark, FQ)
print(corregido, acciones)
assert corregido["disposicion"] == E and acciones
assert limite_pep({"es_pep": False}, {"disposicion": C}, [], spark, FQ)["disposicion"] == C
print("✅ limite_pep corrige solo a las PEP")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 — El sub-set PEP
# MAGIC Lee de `verdad_terreno` **solo las alertas de clientes PEP** (incluida `AT-013`, que no estaba en el set base) y
# MAGIC conviértelas en registros con la misma estructura del `02`. Como bonus, cada registro lleva una guía propia en
# MAGIC `expectations["guidelines"]` (la lee el juez `ExpectationsGuidelines`).

# COMMAND ----------

# TODO: completa el WHERE para quedarte solo con las alertas de clientes PEP (incluida AT-013)
filas = spark.sql(f"SELECT * FROM {FQ}.verdad_terreno WHERE <FILL_IN> ORDER BY alerta_id").collect()
set_pep = [{
    "inputs": {"pregunta": f["pregunta"]},
    "expectations": {
        "expected_facts": list(f["hechos_esperados"]),
        "disposicion_esperada": f["disposicion_esperada"],
        "disposiciones_aceptables": list(f["disposiciones_aceptables"]),
        "hit_sanciones": bool(f["hit_sanciones"]), "es_pep": bool(f["es_pep"]),
        "guidelines": ["El dictamen debe mencionar que el cliente es una Persona Expuesta Políticamente (PEP) y que "
                       "requiere debida diligencia reforzada."],
    },
} for f in filas]
print([r["inputs"]["pregunta"] for r in set_pep])

# COMMAND ----------

assert {re.search(r"AT-\d{3}", r["inputs"]["pregunta"]).group(0) for r in set_pep} == {"AT-004", "AT-011", "AT-013"}
print("✅ Sub-set PEP: AT-004, AT-011, AT-013")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 — Evaluar el agente con los dos límites de acción

# COMMAND ----------

from mlflow.genai.scorers import ExpectationsGuidelines

# TODO: pasa los dos límites de acción (el de sanciones del 04 y el tuyo)
predict_v3 = crear_predict_fn_guardrails(agente, spark, FQ, version="v3-pep", limites=[<FILL_IN>])
kw = {"model": JUEZ} if JUEZ else {}
with mlflow.start_run(run_name="v3-regla-pep"):
    resultado = mlflow.genai.evaluate(
        data=set_pep, predict_fn=predict_v3,
        scorers=[disposicion_correcta, sin_error_critico, regla_pep, ExpectationsGuidelines(**kw)])

display(tabla_resultados(resultado.run_id))

# COMMAND ----------

print({k: round(v, 2) for k, v in resultado.metrics.items() if k.endswith("/mean")})
assert resultado.metrics.get("regla_pep/mean") == 1.0, "Con limite_pep ninguna alerta de una PEP debe cerrarse"
print("🏁 Reto completado: la regla PEP está en el prompt, en un guardrail y en la evaluación")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Para pensar
# MAGIC - Ahora la regla PEP está en tres capas: prompt, guardrail y evaluación. ¿Para qué sirve cada una?
# MAGIC - Para `AT-013` (PEP con montos pequeños), ¿escalarías o pedirías información? ¿Qué dice tu equipo?
# MAGIC
# MAGIC La solución está en `solutions/05_reto_solution`.
