# Databricks notebook source
# MAGIC %md
# MAGIC # LAFISE — Bootcamp de Agentes AML · Sesión 4
# MAGIC ## MLflow Tracing & Evaluation: ¿podemos confiar en los dictámenes del agente?
# MAGIC
# MAGIC En la **Sesión 1** construimos la base de conocimiento, en la **Sesión 2** programamos el **Agente de Alertas** y en
# MAGIC la **Sesión 3** lo construimos sin código. Hoy lo hacemos **confiable**: **trazamos** cada paso que da, lo
# MAGIC **evaluamos** contra alucinaciones con jueces LLM y la verdad de terreno de los analistas, y le ponemos
# MAGIC **guardrails** para los errores que en PLD no se pueden tolerar.
# MAGIC
# MAGIC > **Sobre los nombres (noviembre 2026).** La evaluación se hace con **MLflow 3** (`mlflow.genai.evaluate()`,
# MAGIC > *scorers* y jueces LLM). Lo que antes se llamaba *Mosaic AI Agent Evaluation* hoy es parte de MLflow en
# MAGIC > Databricks. Los guardrails de plataforma son las **service policies** de **Unity Gateway** (Beta), que veremos en
# MAGIC > la Sesión 5.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## A. Aviso
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 14px 18px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="display:block; color:#c62828; margin-bottom:6px; font-size: 1.1em;">Solo laboratorio · datos sintéticos</strong>
# MAGIC <div style="color:#333;">
# MAGIC <ul>
# MAGIC <li>Ejecuta esta práctica únicamente en el <strong>workspace de laboratorio</strong> proporcionado.</li>
# MAGIC <li>Clientes, transacciones, alertas, listas de sanciones y la verdad de terreno son <strong>sintéticos</strong>. <strong>Nunca</strong> uses PII real de clientes de LAFISE.</li>
# MAGIC <li>El agente <strong>recomienda</strong>; la decisión final siempre es del analista humano. Los umbrales y plazos son <strong>didácticos</strong>, no asesoría de cumplimiento.</li>
# MAGIC </ul>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Recorrido de la sesión (2 h)
# MAGIC
# MAGIC <div style="max-width: 1200px; margin: 0 auto; font-family: sans-serif;">
# MAGIC <div style="background: #F9F7F4; border-radius: 10px; padding: 20px 24px; box-shadow: 0 2px 8px rgba(27,49,57,0.06);">
# MAGIC <style>.agenda-table td, .agenda-table th { font-size: 14pt !important; }</style>
# MAGIC <table class="agenda-table" style="width: 100%; border-collapse: collapse; line-height: 1.5;">
# MAGIC   <thead>
# MAGIC      <tr style="background: #1B5162; color: white;">
# MAGIC       <th style="padding: 10px 14px; text-align: center; border: 1px solid #EEEDE9; width: 60px;">#</th>
# MAGIC       <th style="padding: 10px 14px; text-align: center; border: 1px solid #EEEDE9; width: 90px;">Tiempo</th>
# MAGIC       <th style="padding: 10px 14px; text-align: left; border: 1px solid #EEEDE9;">Notebook</th>
# MAGIC      </tr>
# MAGIC   </thead>
# MAGIC   <tbody>
# MAGIC      <tr style="background: white;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">—</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">15 min</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">Introducción del facilitador: por qué evaluar un agente, trazas, jueces y guardrails</td></tr>
# MAGIC      <tr style="background: #FBFAF8;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">00</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">8 min</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">Preparación: datos de las sesiones anteriores, herramientas y verdad de terreno (13 alertas testigo)</td></tr>
# MAGIC      <tr style="background: white;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">01</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">20 min</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">MLflow Tracing: spans, latencia, tokens, el span RETRIEVER y la opinión del analista en la traza</td></tr>
# MAGIC      <tr style="background: #FBFAF8;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">02</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">10 min</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">Set de evaluación en Unity Catalog: 12 alertas con la disposición de un analista</td></tr>
# MAGIC      <tr style="background: white;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">03</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">25 min</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;"><code>mlflow.genai.evaluate()</code> con jueces LLM, <code>make_judge</code> y scorers de código: el scorecard</td></tr>
# MAGIC      <tr style="background: #FBFAF8;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">04</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">25 min</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">Provocar una alucinación (quitar contexto) y añadir guardrails: PII, inyección, límites de acción y citas</td></tr>
# MAGIC      <tr style="background: white;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">05</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">Tarea</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">Reto: la regla PEP como scorer y como guardrail</td></tr>
# MAGIC      <tr style="background: #FBFAF8;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">06</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">Opcional</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">Trazas en Unity Catalog y monitoreo en producción (serverless, Beta)</td></tr>
# MAGIC   </tbody>
# MAGIC </table>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Empieza en `Modulo 4 - Tracing y Evaluacion/00_setup`. Todos los notebooks traen el **código completo**; solo el
# MAGIC `05_reto` tiene espacios para completar (su solución está en `solutions/`).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Requisitos
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 14px 18px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="color:#333;">
# MAGIC
# MAGIC - Conecta un **clúster clásico** (DBR **17.3 LTS**, modo de acceso *Standard*) para ejecutar los notebooks.
# MAGIC - En el widget **catalog** escribe el **mismo catálogo de las Sesiones 1 a 3**.
# MAGIC - Necesitas acceso a **Foundation Model APIs** (el agente y los jueces LLM) y permiso para crear tablas y
# MAGIC   funciones en tu schema.
# MAGIC - Si un admin desactivó las *Partner-powered AI features*, el facilitador te dará un modelo para el widget **juez**.
# MAGIC - El notebook `06` es opcional: necesita un **SQL warehouse** (trazas en UC) y **serverless** (monitoreo, Beta).
# MAGIC - Usa **Chrome o Firefox** (no Safari), sin modo Incógnito.
# MAGIC - Todo se valida antes de la sesión con el notebook de validación de entorno.
# MAGIC
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. Todos los derechos reservados.
