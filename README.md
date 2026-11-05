# Bootcamp de Agentes AML · Sesión 4: MLflow Tracing & Evaluation

Contenido práctico de la Sesión 4 del bootcamp de Agentes AML para el equipo de PLD de Grupo LAFISE: hacer
**confiable** el **Agente de Alertas AML** de la Sesión 2. Vamos a **trazar** cada paso que da el agente con MLflow
Tracing, a **evaluarlo** con jueces LLM y scorers de código contra la disposición que daría un analista (incluida una
métrica anti-alucinación) y a ponerle **guardrails** para los errores que en PLD no se pueden tolerar.

> **Sobre los nombres (noviembre 2026).** La evaluación se hace con **MLflow 3** (`mlflow.genai.evaluate()`, scorers
> y jueces LLM). Los guardrails de plataforma son las **service policies** de **Unity Gateway** (Beta), que veremos en
> la Sesión 5; en esta sesión los guardrails se escriben en código alrededor del agente.

> Todos los datos del laboratorio son **100% sintéticos**. Nunca uses PII real en un workspace de laboratorio.

## Contenido

| Ruta | Qué es |
|---|---|
| `lafise-agentes-alertas-aml/00 - Course Overview` | Agenda y requisitos de la sesión |
| `lafise-agentes-alertas-aml/Modulo 4 - Tracing y Evaluacion/` | Notebooks `00_setup` a `04` (código completo), `05_reto` (tarea) y `06_monitoreo_produccion` (opcional), con `solutions/`, `data/` y `utils/` |
| `lafise-agentes-alertas-aml/requirements.txt` | Librerías (para instalarlas como librerías de clúster si no hay salida a PyPI) |
| `validacion/validacion_entorno_lafise_s4.py` | Notebook de validación de entorno |
| `build/lafise-agentes-alertas-aml-s4.dbc` | El mismo árbol de notebooks en un solo archivo, para importar en un clic |

## Requisitos

- Clúster clásico **DBR 17.3 LTS**, modo de acceso **Standard**, para ejecutar los notebooks.
- Acceso a **Foundation Model APIs**: las usan el agente y los jueces LLM.
- Permiso para crear tablas y funciones en el schema `pld_aml_<usuario>` del catálogo de las Sesiones 1 a 3.
- Si un administrador desactivó las *Partner-powered AI features*, escribir en el widget `juez` el modelo que indique
  el notebook de validación (punto 4).
- Opcional, solo para el notebook `06`: un **SQL warehouse** con `CAN USE` (trazas en Unity Catalog) y cómputo
  **serverless** (monitoreo en producción, Beta).

## Cómo cargarlo en Databricks

**Opción A (recomendada): archivo `.dbc`.** Importa todo el árbol de notebooks en un clic.
1. Descargar el archivo desde este repositorio: abrir `build/lafise-agentes-alertas-aml-s4.dbc` y pulsar **Download raw file**
   (o usar el enlace directo
   `https://github.com/FerHrndz7/lafise-bootcamp-agentes-s4/raw/main/build/lafise-agentes-alertas-aml-s4.dbc`).
2. En el workspace: **Workspace → (carpeta destino) → Import → File** y subir el `.dbc`.
3. Abrir `00 - Course Overview`.

**Opción B: Git folder.** No requiere descargar archivos.
1. En el workspace: **Workspace → Create → Git folder**.
2. URL: `https://github.com/FerHrndz7/lafise-bootcamp-agentes-s4`, proveedor **GitHub** (repositorio público, no
   requiere credenciales).

**Opción C: descargar el ZIP.** En GitHub: **Code → Download ZIP**, descomprimir e importar la carpeta
`lafise-agentes-alertas-aml` al workspace.

## Prueba completa antes de la sesión (30 a 40 min)

Con una **cuenta de asistente (no administrador)**, usando en todos los notebooks el widget `catalog` = el catálogo de
los asistentes y dejando `schema` = `pld_aml`, `llm` y `juez` vacíos (salvo que la validación indique un modelo):

1. `validacion/validacion_entorno_lafise_s4`: **Run all** (~6 min).
2. `Modulo 4 - Tracing y Evaluacion/00_setup`: datos, funciones de UC y las 13 alertas testigo con su verdad de terreno.
3. `01_tracing_del_agente`: dictamina AT-001 y muestra la traza (spans, tokens y documentos recuperados).
4. `02_dataset_de_evaluacion`: crea el set de evaluación de 12 alertas.
5. `03_evaluar_con_jueces`: evaluación completa (3 a 5 min).
6. `04_alucinaciones_y_guardrails`: versión sin contexto, guardrails y casos adversariales (5 a 7 min).

Resultado esperado: celdas de validación en ✅ (por ejemplo, `AT-001 → REPORTAR_ROS` en `00_setup`, y en el `04`
"La evaluación detectó la alucinación" y "v2: cero errores críticos y cero PII"). Los resultados de un LLM varían un
poco entre corridas: que uno o dos jueces marquen "no" en el paso 7 del `03` es normal y se discute en la sesión.
`05_reto` es la tarea para después de la sesión (su solución está en `solutions/`) y `06` es opcional.
