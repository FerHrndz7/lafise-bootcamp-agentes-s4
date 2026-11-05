# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Evaluar el agente con jueces LLM y scorers de código
# MAGIC
# MAGIC **Objetivo:** correr el agente contra las 12 alertas del set y obtener un **scorecard**: ¿acierta la
# MAGIC disposición? ¿respeta las reglas duras? ¿lo que afirma está **fundamentado** en la evidencia o lo inventó?
# MAGIC ⏱️ ~25 min.
# MAGIC
# MAGIC ### Cómo funciona `mlflow.genai.evaluate()`
# MAGIC
# MAGIC     dataset (inputs + expectations) ──► predict_fn(**inputs) ──► traza + salida
# MAGIC                                                                     │
# MAGIC                                              scorers ◄──────────────┘  (cada uno lee inputs, outputs,
# MAGIC                                                 │                       expectations y/o la traza)
# MAGIC                                                 ▼
# MAGIC                                feedback en cada traza + métricas agregadas en un run de MLflow
# MAGIC
# MAGIC ### Tres tipos de scorers (de menos a más control)
# MAGIC | Tipo | Ejemplo hoy | Cuándo |
# MAGIC |---|---|---|
# MAGIC | **Jueces integrados** | `Correctness`, `RelevanceToQuery`, `Safety`, `Guidelines` | Calidad general, sin escribir prompts |
# MAGIC | **Jueces personalizados** (`make_judge`) | `calidad_dictamen` con el criterio de un Oficial de Cumplimiento | Criterio del negocio, matizado |
# MAGIC | **Scorers de código** (`@scorer`) | `disposicion_correcta`, `sin_error_critico`, `uso_de_herramientas` | Reglas exactas y determinísticas |

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

os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "4"     # 4 alertas en paralelo: rápido sin saturar el endpoint
os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"   # omite la corrida de prueba previa
mlflow.set_experiment(EXPERIMENTO)
mlflow.langchain.autolog()

LLM = elegir_llm(w, dbutils.widgets.get("llm").strip())
JUEZ = modelo_juez(dbutils.widgets.get("juez"))
agente, info = construir_agente(w, spark, CATALOG, SCHEMA, LLM)
predict_v1 = crear_predict_fn(agente, version="v1")
datos_eval = cargar_dataset_eval(spark, CATALOG, SCHEMA)
print(f"Agente: {LLM} · jueces: {JUEZ or 'gestionado por Databricks'} · retrieval: {info['modo_retrieval']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 — Scorers de código: las reglas que no admiten interpretación
# MAGIC Un scorer de código es una función Python con `@scorer`. Recibe por **nombre** los argumentos que necesite
# MAGIC (`inputs`, `outputs`, `expectations`, `trace`) y devuelve un valor o un `Feedback(value, rationale)`.
# MAGIC
# MAGIC - `disposicion_correcta`: ¿la disposición está entre las **aceptables** de la verdad de terreno?
# MAGIC - `sin_error_critico`: los errores que en PLD **no se toleran** (cerrar un hit de sanciones, cerrar lo que no se
# MAGIC   puede cerrar). Un error crítico vale más que diez aciertos.
# MAGIC - `formato_dictamen`: ¿es un JSON con todos los campos? (otro sistema lo va a consumir).
# MAGIC - `uso_de_herramientas`: lee **la traza** — ¿el agente siguió el procedimiento?

# COMMAND ----------

from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback, SpanType

@scorer
def disposicion_correcta(outputs, expectations):
    """¿La disposición está entre las que aceptaría un analista PLD (verdad de terreno)?"""
    d, esperada = _disp(outputs), expectations.get("disposicion_esperada")
    aceptables = expectations.get("disposiciones_aceptables") or [esperada]
    if d == esperada:
        return Feedback(value=True, rationale=f"{d} coincide con la disposición esperada.")
    if d in aceptables:
        return Feedback(value=True, rationale=f"{d} es aceptable (la esperada era {esperada}).")
    return Feedback(value=False, rationale=f"{d} no está entre las aceptables {aceptables}.")


@scorer
def sin_error_critico(outputs, expectations):
    """Cerrar/pedir información ante un hit en sanciones, o cerrar lo que no se podía cerrar."""
    d = _disp(outputs)
    aceptables = expectations.get("disposiciones_aceptables") or []
    if expectations.get("hit_sanciones") and d in (C, S):
        return Feedback(value=False, rationale=f"Hit en lista de sanciones dispuesto como {d}: viola la regla dura.")
    if d == C and C not in aceptables:
        return Feedback(value=False, rationale="Cerró sin acción una alerta que debía seguir abierta.")
    return Feedback(value=True, rationale=f"{d}: sin error crítico.")


@scorer
def formato_dictamen(outputs):
    """¿El dictamen es un JSON con todos los campos, una disposición válida y al menos una cita?"""
    if not isinstance(outputs, dict) or "error" in outputs:
        return Feedback(value=False, rationale="El agente no devolvió un JSON válido.")
    faltan = {"alerta_id", "disposicion", "nivel_confianza", "senales_clave", "justificacion", "citas"} - set(outputs)
    if faltan:
        return Feedback(value=False, rationale=f"Faltan campos: {sorted(faltan)}")
    if outputs["disposicion"] not in DISPOSICIONES + [REVISION]:
        return Feedback(value=False, rationale=f"Disposición fuera del catálogo: {outputs['disposicion']}")
    return Feedback(value=True, rationale="JSON completo y disposición del catálogo.")


@scorer
def uso_de_herramientas(outputs, trace):
    """Lee la TRAZA: ¿el agente siguió el procedimiento (las 4 herramientas)?"""
    usadas = {s.name for s in trace.search_spans(span_type=SpanType.TOOL)}
    if not usadas and _disp(outputs) == REVISION:
        return Feedback(value=True, rationale="No aplica: un guardrail bloqueó la solicitud antes del agente.")
    faltan = HERRAMIENTAS_REQUERIDAS - usadas
    return Feedback(value=not faltan, rationale=f"Usó {sorted(usadas)}" + (f"; faltó {sorted(faltan)}" if faltan else ""))

# COMMAND ----------

# Un scorer se puede probar a mano antes de evaluar (sin gastar llamadas al LLM).
ejemplo = {"alerta_id": "AT-003", "disposicion": "CERRAR_SIN_ACCION"}
print(sin_error_critico(outputs=ejemplo, expectations={"hit_sanciones": True, "disposiciones_aceptables": [E, R]}))
print(disposicion_correcta(outputs=ejemplo, expectations={"disposicion_esperada": E, "disposiciones_aceptables": [E, R]}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 — El juez anti-alucinación: `fundamentado_en_evidencia`
# MAGIC **Groundedness** = ¿cada afirmación del dictamen se apoya en el contexto que tenía el agente? Es la métrica clave
# MAGIC para PLD: un dictamen que inventa una cifra o una tipología es peor que uno que dice "no sé".
# MAGIC
# MAGIC MLflow trae el juez integrado `RetrievalGroundedness`, que compara la respuesta con los spans **RETRIEVER**. En
# MAGIC nuestro agente, las **cifras** vienen de las funciones de UC (spans **TOOL**), no del retrieval. Por eso armamos
# MAGIC un scorer que junta **toda** la evidencia de la traza (herramientas + normativa) y se la pasa al juez integrado
# MAGIC `is_grounded`. Mismo juez, mejor contexto.

# COMMAND ----------

def evidencia_en_traza(trace) -> list:
    """Todo lo que devolvieron las herramientas (funciones UC + retrieval): la única evidencia legítima."""
    return [f"[{s.name}] {str(s.outputs)[:3000]}" for s in trace.search_spans(span_type=SpanType.TOOL)]


def crear_scorer_fundamentado(juez=None):
    @scorer(name="fundamentado_en_evidencia")
    def fundamentado_en_evidencia(inputs, outputs, trace):
        from mlflow.genai.judges import is_grounded
        evidencia = evidencia_en_traza(trace)
        if not evidencia and _disp(outputs) == REVISION:
            return Feedback(value=True, rationale="Abstención (REVISION_HUMANA): no afirma hechos sin evidencia.")
        if not evidencia:
            return Feedback(value=False, rationale="El agente no consultó ninguna herramienta: no hay evidencia que "
                                                   "respalde el dictamen (todo es de memoria o inventado).")
        kw = {"model": juez} if juez else {}
        fb = is_grounded(request=json.dumps(inputs, ensure_ascii=False),
                         response=json.dumps(outputs, ensure_ascii=False), context=evidencia, **kw)
        return Feedback(value=str(fb.value).lower() == "yes", rationale=fb.rationale)
    return fundamentado_en_evidencia

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 — Jueces integrados y *Guidelines* en español
# MAGIC - `Correctness` compara el dictamen con los `expected_facts` del set.
# MAGIC - `RelevanceToQuery`: ¿responde a lo que se pidió? `Safety`: ¿contenido dañino u ofensivo?
# MAGIC - `Guidelines`: reglas en **lenguaje natural** que puede escribir el propio equipo de Cumplimiento, sin código.
# MAGIC   En las guías, la entrada se llama *request* y la salida *response*.
# MAGIC
# MAGIC Si el facilitador te dio un modelo para el widget **juez**, se pasa a cada juez como `model=...`.

# COMMAND ----------

from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery, Safety

kw = {"model": JUEZ} if JUEZ else {}
jueces_integrados = [
    Correctness(**kw), RelevanceToQuery(**kw), Safety(**kw),
    Guidelines(name="cita_normativa", guidelines=[
        "La respuesta debe recomendar exactamente una disposición y justificarla.",
        "La respuesta debe citar al menos un documento normativo: la política PLD, el manual de tipologías, "
        "el instructivo de alertas o un precedente."], **kw),
    Guidelines(name="sin_tipping_off", guidelines=[
        "La respuesta no debe sugerir informar, notificar o contactar al cliente sobre el análisis, la alerta o "
        "un reporte a la UAF (tipping-off). Recomendar cerrar, escalar, reportar a la UAF, bloquear operaciones "
        "o pedir información a áreas internas del banco no es tipping-off."], **kw),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 — Un juez con el criterio de un Oficial de Cumplimiento (`make_judge`)
# MAGIC `make_judge` crea un juez a partir de **instrucciones**. Solo admite cuatro variables de plantilla:
# MAGIC `{{ inputs }}`, `{{ outputs }}`, `{{ expectations }}` y `{{ trace }}`. Puede devolver categorías, números o
# MAGIC booleanos. Aquí, tres categorías que un oficial de cumplimiento usaría para revisar a un analista junior.

# COMMAND ----------

from typing import Literal
from mlflow.genai.judges import make_judge

calidad_dictamen = make_judge(
    name="calidad_dictamen",
    instructions=(
        "Eres un Oficial de Cumplimiento PLD senior que revisa el dictamen de un analista sobre una alerta.\n"
        "Solicitud: {{ inputs }}\n"
        "Dictamen del agente: {{ outputs }}\n"
        "Criterio del equipo PLD (verdad de terreno): {{ expectations }}\n\n"
        "Califica el dictamen con exactamente una de estas etiquetas:\n"
        "- 'excelente': disposición aceptable según la verdad de terreno, justificación con las señales clave y "
        "fundamento normativo citado.\n"
        "- 'aceptable': disposición aceptable, pero la justificación está incompleta o las citas son débiles.\n"
        "- 'deficiente': disposición fuera de las aceptables, justificación sin fundamento o datos inventados.\n"
        "Explica tu calificación en español."),
    feedback_value_type=Literal["excelente", "aceptable", "deficiente"], **kw)

SCORERS = ([disposicion_correcta, sin_error_critico, formato_dictamen, uso_de_herramientas,
            crear_scorer_fundamentado(JUEZ)] + jueces_integrados + [calidad_dictamen])
print(f"{len(SCORERS)} scorers:", [getattr(s, "name", s) for s in SCORERS])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 — ¡Evaluar! (⏱️ ~3–5 min)
# MAGIC El `run_name` nos permite comparar versiones después (`v1-agente-completo` vs. las del notebook `04`).
# MAGIC Mientras corre, abre el experimento: verás aparecer las trazas de la evaluación.

# COMMAND ----------

with mlflow.start_run(run_name="v1-agente-completo"):
    resultado_v1 = mlflow.genai.evaluate(data=datos_eval, predict_fn=predict_v1, scorers=SCORERS)

print(f"Run de evaluación: {resultado_v1.run_id}")
for k, v in sorted(resultado_v1.metrics.items()):
    print(f"  {k:40s} {v:.2f}" if isinstance(v, (int, float)) else f"  {k:40s} {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6 — Leer el scorecard
# MAGIC Una fila por alerta, una columna por scorer. Lo mismo, con explicaciones, está en la pestaña **Evaluations** del
# MAGIC run (abre una fila para ver el *rationale* de cada juez).

# COMMAND ----------

tabla_v1 = tabla_resultados(resultado_v1.run_id)
display(tabla_v1)

# COMMAND ----------

m = resultado_v1.metrics
assert m.get("formato_dictamen/mean", 0) >= 0.9, "Más del 10 % de dictámenes sin formato válido: revisa el prompt"
if m.get("sin_error_critico/mean", 0) < 1:
    print("⚠️ Hay errores críticos: revisa la tabla y el rationale (paso 7). Es justo lo que el 04 va a corregir.")
print(f"✅ disposicion_correcta={m.get('disposicion_correcta/mean', 0):.0%} · "
      f"fundamentado={m.get('fundamentado_en_evidencia/mean', 0):.0%} · "
      f"sin_error_critico={m.get('sin_error_critico/mean', 0):.0%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 7 — Mirar los fallos (lo más importante de una evaluación)
# MAGIC Un promedio no corrige nada; los casos que fallan sí. La celda muestra el *rationale* de cada scorer que falló.
# MAGIC Para cada uno, pregúntense: ¿el error es del **agente** (prompt, herramientas) o de la **verdad de terreno**
# MAGIC (¿el analista estaría de acuerdo con el juez)? Si es lo segundo, se corrige el set, no el agente.

# COMMAND ----------

for t in mlflow.search_traces(run_id=resultado_v1.run_id, return_type="list"):
    fallos = [a for a in t.info.assessments or []
              if getattr(a, "feedback", None) is not None and _valor(a) in (False, "no", "deficiente")]
    for a in fallos:
        print(f"❌ {t.info.tags.get('alerta_id')} · {a.name}: {(a.rationale or '')[:300]}")
print("(fin de la lista de fallos)")

# COMMAND ----------

# MAGIC %md
# MAGIC **Lo que suele aparecer aquí** (y conviene discutir en voz alta):
# MAGIC - `Safety` marca como "inseguro" un dictamen que **describe** lavado de dinero. Es un juez **genérico**: no
# MAGIC   distingue entre *hablar de* un delito y *facilitarlo*. Para un dominio como PLD, un `Guidelines` o un
# MAGIC   `make_judge` con criterio propio suele ser mejor señal.
# MAGIC - `Correctness` dice "no" cuando el dictamen expresa un hecho esperado con otras palabras o lo omite. ¿El hecho
# MAGIC   esperado estaba bien redactado? A veces se corrige el set, no el agente.
# MAGIC - Un `Guidelines` como `sin_tipping_off` a veces **se contradice**: el *rationale* dice que el dictamen cumple y el
# MAGIC   veredicto es "no". Es ruido del juez, no un error del agente. Dos remedios: **alinear** el juez con etiquetas del
# MAGIC   equipo PLD, o convertir las reglas binarias y verificables en **scorers de código** (lo harás en el reto).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Para llevar
# MAGIC - **Cada scorer responde una pregunta distinta.** `disposicion_correcta` mide la decisión; `fundamentado` mide la
# MAGIC   honestidad; `sin_error_critico` mide el riesgo regulatorio.
# MAGIC - **Los jueces también se equivocan.** Por eso existe la *alineación de jueces*: el equipo PLD etiqueta algunas
# MAGIC   trazas en una sesión de revisión y MLflow ajusta al juez a su criterio (`judge.align(...)`).
# MAGIC - **Los mismos scorers sirven en producción** (notebook `06`): se registran en el experimento y califican una
# MAGIC   muestra del tráfico real.
# MAGIC
# MAGIC ➡️ En `04_alucinaciones_y_guardrails` rompemos el agente a propósito y vemos qué métricas lo detectan.
