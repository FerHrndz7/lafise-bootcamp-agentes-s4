# Databricks notebook source
# MAGIC %md
# MAGIC # Bootstrap — Datos, herramientas y verdad de terreno para la Sesión 4
# MAGIC
# MAGIC Lo llama `00_setup`. Deja tu schema `pld_aml_<usuario>` listo para **trazar y evaluar** el Agente de Alertas,
# MAGIC **reutilizando lo que ya construiste en las Sesiones 1 a 3** y regenerando solo lo que falte:
# MAGIC
# MAGIC | Pieza | Si ya existe | Si falta |
# MAGIC |---|---|---|
# MAGIC | `clientes`, `transacciones`, `alertas`, `narrativas_casos` | se reutilizan | se generan con Faker (mismo generador de la S1) |
# MAGIC | `chunks` (base de conocimiento) | se reutiliza | se genera desde el corpus PLD (ID determinístico) |
# MAGIC | Índice `chunks_index` (AI Search) | se reutiliza | se crea **sin esperar**; mientras tanto el agente busca por palabras clave |
# MAGIC | **Alertas testigo** `AT-001..013` + `lista_sanciones` | se refrescan (idempotente) | se crean |
# MAGIC | Funciones `obtener_alerta`, `features_transaccionales`, `verificar_sanciones` | se recrean (idempotente) | se crean |
# MAGIC | Tabla `verdad_terreno` (disposición esperada por alerta) | se recrea | se crea |
# MAGIC
# MAGIC > ⚠️ **Todo es SINTÉTICO** (Faker + hechos regulatorios plausibles pero ficticios). Nunca PII real.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "pld_aml", "Schema base")
dbutils.widgets.text("num_clientes", "500", "Num clientes")

# COMMAND ----------

import re, json, random, time
from datetime import date, timedelta

CATALOG = dbutils.widgets.get("catalog").strip()
NUM_CLIENTES = int(dbutils.widgets.get("num_clientes"))
user_email = spark.sql("SELECT current_user()").collect()[0][0]
user_prefix = re.sub(r"[^a-z0-9]", "_", user_email.split("@")[0].lower())
assert CATALOG, "Falta el catálogo. Ejecuta este bootstrap desde 00_setup (o pásale el widget 'catalog')."
SCHEMA = f"{(dbutils.widgets.get('schema').strip() or 'pld_aml')}_{user_prefix}"
FQ = f"`{CATALOG}`.{SCHEMA}"

VS_ENDPOINT = "lafise_vs_endpoint"                  # mismo endpoint compartido de la Sesión 1
INDEX_NAME = f"{CATALOG}.{SCHEMA}.chunks_index"

spark.sql(f"USE CATALOG `{CATALOG}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
random.seed(42)

def existe_tabla(t):
    return spark.catalog.tableExists(f"{CATALOG}.{SCHEMA}.{t}")

print(f"Destino: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Corpus PLD (idéntico al de la Sesión 1, para que las citas coincidan)

# COMMAND ----------

POLITICA = (
    "politica_pld_lafise", "pdf",
    "Política de Prevención de Lavado de Dinero y Financiamiento al Terrorismo (PLD/FT) — Grupo LAFISE",
    [
        ("Artículo 1. Objeto y marco",
         ["La presente política establece el marco de prevención de lavado de dinero y financiamiento al "
          "terrorismo (PLD/FT) de la institución, alineado con las 40 Recomendaciones del GAFI y la normativa "
          "de la Unidad de Análisis Financiero (UAF). Su cumplimiento es obligatorio para todo colaborador."]),
        ("Artículo 2. Enfoque basado en riesgo",
         ["La institución adopta un enfoque basado en riesgo (EBR): asigna a cada cliente un nivel de riesgo "
          "(bajo, medio, alto) según su perfil, actividad económica, geografía y canal, y calibra la intensidad "
          "de la debida diligencia en función de ese nivel."]),
        ("Artículo 3. Debida diligencia del cliente (DDC)",
         ["Se aplica debida diligencia estándar a clientes de riesgo bajo o medio, y debida diligencia "
          "reforzada (EDD) a clientes de alto riesgo. La identificación del beneficiario final es obligatoria "
          "en todos los casos."]),
        ("Artículo 4. Personas Expuestas Políticamente (PEP)",
         ["Toda relación con una Persona Expuesta Políticamente (PEP) requiere debida diligencia reforzada y "
          "la aprobación expresa de la alta gerencia antes de abrir o mantener la cuenta. El monitoreo de las "
          "PEP es continuo."]),
        ("Artículo 5. Reporte de Transacciones en Efectivo (RTE)",
         ["Toda transacción en efectivo igual o superior a USD 10,000 —individual o acumulada por un mismo "
          "cliente durante un mes calendario— genera un Reporte de Transacción en Efectivo (RTE) que se remite "
          "a la UAF. El fraccionamiento deliberado para evadir este umbral constituye una señal de alerta."]),
        ("Artículo 6. Reporte de Operación Sospechosa (ROS)",
         ["Cuando el análisis confirma indicios de una operación sospechosa, el Oficial de Cumplimiento remite "
          "un Reporte de Operación Sospechosa (ROS) a la UAF dentro de un plazo máximo de 3 días hábiles "
          "contados desde la confirmación. El ROS es confidencial: está prohibido informar al cliente (tipping-off)."]),
        ("Artículo 7. Conservación de registros",
         ["Los registros de identificación de clientes, transacciones y análisis de alertas se conservan por un "
          "plazo mínimo de 5 años contados desde el cierre de la relación o la fecha de la operación."]),
        ("Artículo 8. Monitoreo transaccional",
         ["El monitoreo transaccional es automatizado y basado en reglas y tipologías. Genera alertas que el "
          "área de cumplimiento analiza y dispone según el Instructivo de Escalamiento y Disposición de Alertas."]),
        ("Artículo 9. Listas de sanciones",
         ["La institución coteja clientes y contrapartes contra listas de sanciones (OFAC, Consejo de Seguridad "
          "de la ONU y listas locales). Una coincidencia (hit) confirmada bloquea la operación y se escala de "
          "inmediato; nunca se cierra sin acción."]),
    ],
)

TIPOLOGIAS = (
    "manual_tipologias", "pdf",
    "Manual de Tipologías de Lavado de Activos — Cumplimiento LAFISE",
    [
        ("Tipología 1. Estructuración (pitufeo)",
         ["Descripción: fraccionar una suma grande en múltiples depósitos o transferencias menores al umbral de "
          "USD 10,000 para evadir el Reporte de Transacción en Efectivo.",
          "Señales de alerta: múltiples depósitos en efectivo por montos apenas por debajo de USD 10,000; uso de "
          "varias sucursales el mismo día; depósitos por terceros en la cuenta de una misma persona."]),
        ("Tipología 2. Movimiento rápido de fondos (pass-through)",
         ["Descripción: fondos que ingresan y salen de una cuenta en un lapso muy corto (menos de 48 a 72 horas), "
          "dejando un saldo cercano a cero, sin una razón económica aparente.",
          "Señales de alerta: la cuenta funciona como puente; entradas y salidas del mismo monto; beneficiario "
          "final distinto del titular."]),
        ("Tipología 3. Uso de terceros y testaferros",
         ["Descripción: uso de terceros (familiares, empleados, sociedades pantalla) para ocultar al verdadero "
          "beneficiario final de los fondos.",
          "Señales de alerta: cuentas de personas sin capacidad económica que reciben montos altos; poderes "
          "amplios a terceros; sociedades recién constituidas sin actividad real."]),
        ("Tipología 4. Operaciones inconsistentes con el perfil",
         ["Descripción: transacciones cuyo monto, frecuencia o naturaleza no corresponden con el perfil "
          "transaccional declarado ni con la actividad económica del cliente.",
          "Señales de alerta: un cliente de perfil bajo que de pronto mueve montos altos; giros al exterior sin "
          "relación con su negocio; cambios abruptos de comportamiento."]),
        ("Tipología 5. Geografías de alto riesgo",
         ["Descripción: operaciones con origen o destino en jurisdicciones de alto riesgo o no cooperantes según "
          "el GAFI, o en zonas de frontera asociadas a economías ilícitas.",
          "Señales de alerta: transferencias frecuentes hacia países de alto riesgo; uso de centros financieros "
          "extraterritoriales sin justificación; ruteo innecesario de fondos por múltiples jurisdicciones."]),
    ],
)

INSTRUCTIVO = (
    "instructivo_alertas", "pdf",
    "Instructivo de Escalamiento y Disposición de Alertas — Unidad PLD",
    [
        ("1. Alcance",
         ["Este instructivo define cómo el analista de la Unidad de Cumplimiento analiza y dispone las alertas "
          "generadas por el monitoreo transaccional."]),
        ("2. Plazo de análisis (SLA)",
         ["Cada alerta debe analizarse y disponerse en un plazo máximo de 15 días hábiles desde su generación. "
          "Las alertas de prioridad alta (score de riesgo elevado o hit en lista) se atienden en 24 horas."]),
        ("3. Disposiciones posibles",
         ["(a) Cerrar sin acción: cuando el análisis descarta el indicio y la operación es consistente con el "
          "perfil. Debe documentarse el fundamento.",
          "(b) Escalar a investigación: cuando persisten indicios que requieren profundización por la unidad de "
          "investigaciones.",
          "(c) Reportar: cuando se confirma la sospecha, se genera un ROS a la UAF dentro de 3 días hábiles."]),
        ("4. Reglas duras",
         ["Un hit confirmado en lista de sanciones (OFAC/ONU) nunca se cierra sin acción: se bloquea y se escala "
          "de inmediato. Toda disposición debe citar la política o la tipología que la fundamenta."]),
        ("5. Priorización",
         ["Las alertas se priorizan por score de riesgo, monto involucrado, y presencia de PEP o de geografías de "
          "alto riesgo."]),
    ],
)


DOCS = [POLITICA, TIPOLOGIAS, INSTRUCTIVO]

NARRATIVAS = [
    ("K-2024-018", "Estructuración", "Reportado",
     "El cliente realizó nueve depósitos en efectivo de entre USD 9,200 y USD 9,800 en tres sucursales "
     "distintas durante cinco días. El patrón es consistente con estructuración para evadir el umbral de RTE. "
     "Se confirmó la sospecha y se remitió ROS a la UAF. Fundamento: Tipología 1 y Artículo 5 de la política."),
    ("K-2024-042", "Pass-through", "Escalado",
     "Una cuenta empresarial recibió tres transferencias del exterior por USD 45,000 que salieron hacia una "
     "tercera cuenta en menos de 48 horas, dejando saldo cero. No hay soporte comercial. Movimiento rápido de "
     "fondos; se escaló a investigación para identificar al beneficiario final."),
    ("K-2023-233", "Inconsistente con perfil", "Cerrado",
     "Un cliente de banca de personas con perfil bajo recibió USD 30,000. Tras el análisis se documentó que "
     "correspondía a la venta de un inmueble con escritura verificable. Operación consistente; se cerró sin acción."),
    ("K-2024-107", "Geografía de alto riesgo", "Reportado",
     "Transferencias frecuentes hacia una jurisdicción de alto riesgo listada por el GAFI, sin relación con la "
     "actividad declarada del cliente. Se confirmó la sospecha y se reportó a la UAF."),
    ("K-2024-151", "Testaferros", "Escalado",
     "Depósitos altos en la cuenta de una persona sin capacidad económica aparente, con poder amplio otorgado a "
     "un tercero. Posible uso de testaferro; se escaló a investigación."),
]
# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tablas estructuradas (solo si faltan)

# COMMAND ----------

from pyspark.sql import Row

if not existe_tabla("alertas"):
    from faker import Faker
    fake = Faker("es_MX"); Faker.seed(42)
    SEGMENTOS = ["Banca de personas", "Banca de empresas", "Comercio exterior"]
    TIPOLOGIAS_CAT = ["Estructuración", "Pass-through", "Testaferros",
                      "Inconsistente con perfil", "Geografía de alto riesgo"]
    hoy = date.today()
    clientes = [Row(cliente_id=f"C{i:05d}", nombre=fake.name(), segmento=random.choice(SEGMENTOS),
                    pais=random.choice(["Nicaragua", "Costa Rica", "Honduras", "Panamá", "Guatemala"]),
                    es_pep=random.random() < 0.05,
                    nivel_riesgo=random.choice(["Alto", "Alto", "Medio", "Bajo", "Bajo", "Bajo"]),
                    fecha_alta=fake.date_between(start_date="-5y", end_date="today").isoformat())
                for i in range(NUM_CLIENTES)]
    spark.createDataFrame(clientes).write.mode("overwrite").saveAsTable(f"{FQ}.clientes")
    txns = []
    for i in range(NUM_CLIENTES * 8):
        monto = round(random.choice([random.uniform(50, 3000), random.uniform(9000, 9999),
                                     random.uniform(100, 50000)]), 2)
        txns.append(Row(txn_id=f"T{i:07d}", cliente_id=f"C{random.randint(0, NUM_CLIENTES-1):05d}",
                        fecha=(hoy - timedelta(days=random.randint(0, 180))).isoformat(), monto_usd=monto,
                        canal=random.choice(["Efectivo", "Transferencia", "ACH", "Cheque"]),
                        pais_contraparte=random.choice(["Nicaragua", "Panamá", "EE.UU.", "Islas Caimán", "Colombia"])))
    spark.createDataFrame(txns).write.mode("overwrite").saveAsTable(f"{FQ}.transacciones")
    alertas = [Row(alerta_id=f"A{i:05d}", cliente_id=f"C{random.randint(0, NUM_CLIENTES-1):05d}",
                   fecha=(hoy - timedelta(days=random.randint(0, 60))).isoformat(),
                   tipologia=random.choice(TIPOLOGIAS_CAT), score_riesgo=random.randint(30, 99),
                   monto_involucrado_usd=round(random.uniform(8000, 120000), 2),
                   estado=random.choice(["Abierta", "En análisis", "Cerrada", "Escalada", "Reportada"]))
               for i in range(120)]
    spark.createDataFrame(alertas).write.mode("overwrite").saveAsTable(f"{FQ}.alertas")
    print("✅ Tablas clientes/transacciones/alertas generadas (no venían de la Sesión 1)")
else:
    print("♻️  Reutilizando clientes/transacciones/alertas de la Sesión 1")

if not existe_tabla("narrativas_casos"):
    spark.createDataFrame([Row(caso_id=c[0], tipologia=c[1], disposicion=c[2], narrativa=c[3]) for c in NARRATIVAS]) \
        .write.mode("overwrite").saveAsTable(f"{FQ}.narrativas_casos")
    print("✅ narrativas_casos generada")
else:
    print("♻️  Reutilizando narrativas_casos de la Sesión 1")
# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Alertas testigo `AT-001..013` + lista de sanciones
# MAGIC
# MAGIC Trece alertas **diseñadas a mano**, cada una con un comportamiento esperado conocido. Las tres primeras son las
# MAGIC de la Sesión 2; las demás cubren los casos que más le importan al equipo PLD: PEP, geografías de alto riesgo,
# MAGIC falta de soporte, coincidencias aproximadas en listas y actividad benigna que **no** debe escalarse.

# COMMAND ----------

hoy = date.today()
d = lambda n: (hoy - timedelta(days=n)).isoformat()

CLIENTES_T = [
    ("CT-001", "Marvin Antonio Zelaya Ríos", "Banca de personas", "Nicaragua", False, "Medio", "2021-03-15"),
    ("CT-002", "Lucía Fernanda Castillo Mena", "Banca de personas", "Costa Rica", False, "Bajo", "2016-08-02"),
    ("CT-003", "Ramiro Quintanilla Rojas", "Banca de empresas", "Honduras", False, "Medio", "2023-11-20"),
    ("CT-004", "Gabriela Ortiz Salinas", "Banca de personas", "Panamá", True, "Alto", "2019-06-11"),
    ("CT-005", "José Daniel Membreño Paz", "Banca de personas", "Honduras", False, "Bajo", "2014-02-27"),
    ("CT-006", "Andrea Valeria Solís Cordero", "Banca de personas", "Costa Rica", False, "Bajo", "2020-09-30"),
    ("CT-007", "Óscar Mauricio Rivas León", "Banca de personas", "Nicaragua", False, "Bajo", "2018-04-04"),
    ("CT-008", "Rosa Elena Pineda Torres", "Banca de personas", "Guatemala", False, "Bajo", "2015-12-01"),
    ("CT-009", "Inversiones Costa Azul S.A.", "Banca de empresas", "Panamá", False, "Medio", "2022-07-19"),
    ("CT-010", "Distribuidora El Roble S.A.", "Banca de empresas", "Costa Rica", False, "Medio", "2012-05-14"),
    ("CT-011", "Federico Arturo Bermúdez Mairena", "Banca de personas", "Nicaragua", True, "Alto", "2017-01-23"),
    ("CT-012", "Hector Ivan Montalvo", "Banca de personas", "Guatemala", False, "Medio", "2024-03-08"),
    ("CT-013", "Mariela Isabel Duarte Vega", "Banca de personas", "Nicaragua", True, "Medio", "2013-10-10"),
]
TXNS_T = (
    # AT-001 · estructuración: 9 depósitos en efectivo de USD 9,2xx–9,8xx en 5 días
    [(f"TT-001-{i:02d}", "CT-001", d(9 - i // 2), float(9200 + 70 * i), "Efectivo", "Nicaragua") for i in range(9)]
    + [("TT-001-90", "CT-001", d(40), 850.00, "Transferencia", "Nicaragua")]
    # AT-002 · benigno: ACH pequeños y regulares
    + [(f"TT-002-{i:02d}", "CT-002", d(5 + 9 * i), float(1200 + 35 * i), "ACH", "Costa Rica") for i in range(6)]
    # AT-003 · cliente en lista OFAC, movimiento moderado
    + [("TT-003-01", "CT-003", d(12), 18500.00, "Transferencia", "Panamá"),
       ("TT-003-02", "CT-003", d(6), 7400.00, "Transferencia", "Islas Caimán")]
    # AT-004 · PEP con transferencias moderadas
    + [(f"TT-004-{i:02d}", "CT-004", d(4 + 5 * i), float(6200 + 560 * i), "Transferencia", "Panamá") for i in range(4)]
    # AT-005 · estructuración: 6 depósitos en efectivo de USD 9,1xx–9,6xx en 3 días
    + [(f"TT-005-{i:02d}", "CT-005", d(5 - i // 2), float(9150 + 100 * i), "Efectivo", "Honduras") for i in range(6)]
    # AT-006 · geografía de alto riesgo, perfil bajo
    + [(f"TT-006-{i:02d}", "CT-006", d(3 + 6 * i), float(12000 + 2000 * i), "Transferencia", "Islas Caimán")
       for i in range(4)]
    # AT-007 · un ingreso grande sin soporte
    + [("TT-007-01", "CT-007", d(4), 30000.00, "Transferencia", "Nicaragua")]
    # AT-008 · efectivo de montos bajos (comerciante)
    + [(f"TT-008-{i:02d}", "CT-008", d(3 + 3 * i), float(300 + 170 * i), "Efectivo", "Guatemala") for i in range(8)]
    # AT-009 · empresa en lista OFAC
    + [("TT-009-01", "CT-009", d(10), 14000.00, "Transferencia", "Panamá"),
       ("TT-009-02", "CT-009", d(5), 9000.00, "Transferencia", "Panamá")]
    # AT-010 · empresa con ACH recurrentes
    + [(f"TT-010-{i:02d}", "CT-010", d(3 + 6 * i), float(5000 + 750 * i), "ACH", "Costa Rica") for i in range(5)]
    # AT-011 · PEP + Islas Caimán
    + [(f"TT-011-{i:02d}", "CT-011", d(4 + 7 * i), float(22000 + 6500 * i), "Transferencia", "Islas Caimán")
       for i in range(3)]
    # AT-012 · coincidencia aproximada con lista ONU (nombre sin tildes)
    + [(f"TT-012-{i:02d}", "CT-012", d(4 + 8 * i), float(4000 + 1000 * i), "Transferencia", "Guatemala")
       for i in range(3)]
    # AT-013 · PEP con movimientos pequeños (reto)
    + [(f"TT-013-{i:02d}", "CT-013", d(3 + 4 * i), float(900 + 100 * i), "ACH", "Nicaragua") for i in range(6)]
)
ALERTAS_T = [
    ("AT-001", "CT-001", d(2), "Estructuración", 91, 85320.00, "Abierta"),
    ("AT-002", "CT-002", d(2), "Inconsistente con perfil", 38, 3705.00, "Abierta"),
    ("AT-003", "CT-003", d(2), "Geografía de alto riesgo", 64, 25900.00, "Abierta"),
    ("AT-004", "CT-004", d(2), "Inconsistente con perfil", 72, 28160.00, "Abierta"),
    ("AT-005", "CT-005", d(2), "Estructuración", 88, 56400.00, "Abierta"),
    ("AT-006", "CT-006", d(2), "Geografía de alto riesgo", 81, 60000.00, "Abierta"),
    ("AT-007", "CT-007", d(2), "Inconsistente con perfil", 55, 30000.00, "Abierta"),
    ("AT-008", "CT-008", d(2), "Estructuración", 41, 7160.00, "Abierta"),
    ("AT-009", "CT-009", d(2), "Geografía de alto riesgo", 67, 23000.00, "Abierta"),
    ("AT-010", "CT-010", d(2), "Inconsistente con perfil", 35, 32500.00, "Abierta"),
    ("AT-011", "CT-011", d(2), "Geografía de alto riesgo", 90, 85500.00, "Abierta"),
    ("AT-012", "CT-012", d(2), "Inconsistente con perfil", 58, 15000.00, "Abierta"),
    ("AT-013", "CT-013", d(2), "Inconsistente con perfil", 44, 6900.00, "Abierta"),
]

def refrescar(tabla, filas, cols, clave, prefijo):
    """Borra las filas testigo previas (por prefijo) y vuelve a insertarlas -> idempotente."""
    destino = spark.table(f"{FQ}.{tabla}")
    df = spark.createDataFrame(filas, cols)
    df = df.select([df[c].cast(destino.schema[c].dataType).alias(c) for c in destino.columns])
    spark.sql(f"DELETE FROM {FQ}.{tabla} WHERE {clave} LIKE '{prefijo}%'")
    df.write.mode("append").saveAsTable(f"{FQ}.{tabla}")

refrescar("clientes", CLIENTES_T,
          ["cliente_id", "nombre", "segmento", "pais", "es_pep", "nivel_riesgo", "fecha_alta"], "cliente_id", "CT-")
refrescar("transacciones", TXNS_T,
          ["txn_id", "cliente_id", "fecha", "monto_usd", "canal", "pais_contraparte"], "txn_id", "TT-")
refrescar("alertas", ALERTAS_T,
          ["alerta_id", "cliente_id", "fecha", "tipologia", "score_riesgo", "monto_involucrado_usd", "estado"],
          "alerta_id", "AT-")

SANCIONES = [
    ("Ramiro Quintanilla Rojas", "OFAC SDN", "Honduras", "2024-05-10", "Facilitador de redes de lavado"),
    ("Inversiones Costa Azul S.A.", "OFAC SDN", "Panamá", "2023-09-01", "Sociedad pantalla"),
    ("Héctor Iván Montalvo", "ONU CSNU", "Guatemala", "2022-02-14", "Financiamiento al terrorismo"),
    ("Grupo Comercial Tres Ríos", "Lista local UAF", "Nicaragua", "2025-01-20", "Contrabando"),
]
spark.createDataFrame(SANCIONES, ["nombre_listado", "lista", "pais", "fecha_inclusion", "motivo"]) \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.lista_sanciones")
print(f"✅ Alertas testigo AT-001..{len(ALERTAS_T):03d} + lista_sanciones listas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verdad de terreno (lo que decidiría un analista PLD experimentado)
# MAGIC
# MAGIC Para cada alerta testigo: la disposición **esperada**, las disposiciones **aceptables** (un analista podría
# MAGIC razonablemente elegir cualquiera de ellas), los **hechos** que todo buen dictamen debe mencionar y dos banderas
# MAGIC para las reglas duras. `AT-013` queda fuera del set base: es para el reto.

# COMMAND ----------

E, R, C, S = "ESCALAR_INVESTIGACION", "REPORTAR_ROS", "CERRAR_SIN_ACCION", "SOLICITAR_INFORMACION"
VERDAD = [
    # alerta, esperada, aceptables, hechos esperados, hit_sanciones, es_pep, en_eval_base
    ("AT-001", R, [R, E], ["Hubo 9 depósitos en efectivo justo por debajo del umbral de USD 10,000",
                           "El patrón corresponde a la tipología de estructuración (pitufeo)"], False, False, True),
    ("AT-002", C, [C, S], ["Los movimientos son pequeños y consistentes con el perfil de riesgo bajo de la clienta"],
     False, False, True),
    ("AT-003", E, [E, R], ["El cliente coincide con la lista de sanciones OFAC",
                           "Una coincidencia en lista de sanciones nunca se cierra sin acción"], True, False, True),
    ("AT-004", E, [E, S], ["La clienta es una Persona Expuesta Políticamente (PEP)",
                           "Una PEP requiere debida diligencia reforzada"], False, True, True),
    ("AT-005", R, [R, E], ["Hubo 6 depósitos en efectivo justo por debajo del umbral de USD 10,000",
                           "El patrón corresponde a la tipología de estructuración (pitufeo)"], False, False, True),
    ("AT-006", E, [E, R], ["Hay transferencias hacia Islas Caimán, una jurisdicción de alto riesgo",
                           "La actividad es inconsistente con el perfil de riesgo bajo de la clienta"], False, False, True),
    ("AT-007", S, [S, E], ["Se recibió una transferencia de USD 30,000 inconsistente con el perfil del cliente",
                           "Falta soporte del origen de los fondos"], False, False, True),
    ("AT-008", C, [C, S], ["Los depósitos en efectivo son de montos bajos, lejos del umbral de USD 10,000"],
     False, False, True),
    ("AT-009", E, [E, R], ["El cliente Inversiones Costa Azul S.A. coincide con la lista de sanciones OFAC"],
     True, False, True),
    ("AT-010", C, [C, S], ["Los pagos ACH son recurrentes y consistentes con la actividad de la empresa"],
     False, False, True),
    ("AT-011", E, [E, R], ["El cliente es una Persona Expuesta Políticamente (PEP)",
                           "Hay transferencias hacia Islas Caimán, una jurisdicción de alto riesgo"], False, True, True),
    ("AT-012", E, [E, R], ["El nombre del cliente coincide de forma aproximada con un listado de sanciones de la ONU"],
     True, False, True),
    ("AT-013", S, [S, E], ["La clienta es una Persona Expuesta Políticamente (PEP)",
                           "Aunque los montos son pequeños, una alerta de una PEP no se cierra sin acción"],
     False, True, False),
]
filas = [(a, f"Analiza la alerta {a} y emite tu dictamen.", esp, acep, hechos, hit, pep, base)
         for a, esp, acep, hechos, hit, pep, base in VERDAD]
(spark.createDataFrame(filas, "alerta_id string, pregunta string, disposicion_esperada string, "
                              "disposiciones_aceptables array<string>, hechos_esperados array<string>, "
                              "hit_sanciones boolean, es_pep boolean, en_eval_base boolean")
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.verdad_terreno"))
spark.sql(f"COMMENT ON TABLE {FQ}.verdad_terreno IS 'Verdad de terreno de las alertas testigo AT-001..013: "
          f"disposición esperada y aceptables según un analista PLD. Base del set de evaluación de la Sesión 4.'")
print(f"✅ verdad_terreno: {len(filas)} alertas ({sum(f[-1] for f in filas)} en el set base)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Tabla `chunks` (solo si falta)
# MAGIC Un chunk por artículo/sección con su metadata de citación. El `chunk_id` es **determinístico**.

# COMMAND ----------

from pyspark.sql import functions as F

if not existe_tabla("chunks"):
    filas = []
    for name, fmt, title, sections in DOCS:
        doc = f"{name}.{fmt}"
        for heading, paragraphs in sections:
            m = re.match(r"^(Artículo\s+\d+|Tipología\s+\d+|\d+)", heading)
            articulo = m.group(0) if m else heading[:40]
            texto = heading + "\n" + "\n".join(paragraphs)
            filas.append((doc, heading, articulo, texto, f"[{doc} · {heading}] {texto}"))
    for c in NARRATIVAS:
        txt = f"Caso {c[0]} — Tipología {c[1]} — Disposición: {c[2]}. {c[3]}"
        filas.append(("narrativas_casos", f"Caso {c[0]}", c[1], txt, f"[Precedente · {c[1]}] {txt}"))
    df = (spark.createDataFrame(filas, ["source_doc", "seccion", "articulo", "chunk_to_retrieve", "chunk_to_embed"])
          .withColumn("chunk_id", F.sha2(F.concat_ws("|", "source_doc", "seccion"), 256)))
    df.write.saveAsTable(f"{FQ}.chunks")
    spark.sql(f"ALTER TABLE {FQ}.chunks SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"✅ chunks generada ({df.count()} filas)")
else:
    print("♻️  Reutilizando chunks de la Sesión 1")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Índice de AI Search (solo si falta, **sin esperar**)
# MAGIC Si tu `chunks_index` de la Sesión 1 existe, lo reutilizamos. Si no, lo creamos y **no esperamos**: la primera
# MAGIC sincronización puede tardar 1–15 min. Mientras no esté listo, la herramienta `buscar_normativa_pld` busca por
# MAGIC palabras clave sobre la misma tabla `chunks` (mismos documentos, mismas citas).

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest, EmbeddingSourceColumn, PipelineType, VectorIndexType)
w = WorkspaceClient()

def estado_indice():
    try:
        return "listo" if w.vector_search_indexes.get_index(INDEX_NAME).status.ready else "sincronizando"
    except Exception:
        return "no existe"

estado = estado_indice()
if estado == "no existe":
    try:
        endpoints = {e.name for e in w.vector_search_endpoints.list_endpoints()}
        if VS_ENDPOINT not in endpoints:
            raise RuntimeError(f"no existe el endpoint {VS_ENDPOINT} de la Sesión 1")
        servidos = {e.name for e in w.serving_endpoints.list()}
        EMB = next(c for c in ["databricks-qwen3-embedding-0-6b", "databricks-gte-large-en",
                               "databricks-bge-large-en"] if c in servidos)
        w.vector_search_indexes.create_index(
            name=INDEX_NAME, endpoint_name=VS_ENDPOINT, primary_key="chunk_id",
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=f"{CATALOG}.{SCHEMA}.chunks", pipeline_type=PipelineType.TRIGGERED,
                embedding_source_columns=[EmbeddingSourceColumn(name="chunk_to_embed",
                                                                embedding_model_endpoint_name=EMB)]))
        estado = "creado (sincronizando)"
    except Exception as e:
        estado = f"no disponible ({str(e)[:120]}) → se usará búsqueda por palabras clave"
print(f"Índice {INDEX_NAME}: {estado}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Funciones de Unity Catalog (herramientas del agente)
# MAGIC Las mismas de la Sesión 2: `obtener_alerta`, `features_transaccionales` y `verificar_sanciones` (el reto de la S2).

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.obtener_alerta(
  alerta_id STRING COMMENT 'ID de la alerta de monitoreo, por ejemplo AT-001'
)
RETURNS TABLE (
  alerta_id STRING, fecha_alerta STRING, tipologia STRING, score_riesgo INT, monto_involucrado_usd DOUBLE,
  estado STRING, cliente_id STRING, nombre_cliente STRING, segmento STRING, pais STRING, es_pep BOOLEAN,
  nivel_riesgo STRING, cliente_desde STRING
)
COMMENT 'Devuelve la alerta de monitoreo transaccional y el perfil KYC del cliente (segmento, país, si es PEP, nivel de riesgo). Úsala primero para entender la alerta.'
RETURN
  SELECT a.alerta_id, a.fecha, a.tipologia, CAST(a.score_riesgo AS INT), a.monto_involucrado_usd, a.estado,
         c.cliente_id, c.nombre, c.segmento, c.pais, c.es_pep, c.nivel_riesgo, c.fecha_alta
  FROM {FQ}.alertas a JOIN {FQ}.clientes c ON a.cliente_id = c.cliente_id
  WHERE a.alerta_id = obtener_alerta.alerta_id
""")
spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.features_transaccionales(
  alerta_id STRING COMMENT 'ID de la alerta de monitoreo, por ejemplo AT-001'
)
RETURNS TABLE (
  alerta_id STRING, cliente_id STRING, ventana_dias INT, n_transacciones BIGINT, monto_total_usd DOUBLE,
  monto_efectivo_usd DOUBLE, pct_efectivo DOUBLE, n_efectivo_cerca_umbral BIGINT, dias_con_efectivo BIGINT,
  n_contraparte_alto_riesgo BIGINT, monto_contraparte_alto_riesgo_usd DOUBLE, txn_maxima_usd DOUBLE
)
COMMENT 'Calcula señales transaccionales del cliente en los 30 días previos a la alerta: montos, porcentaje en efectivo, depósitos en efectivo justo por debajo del umbral de USD 10,000 (posible estructuración) y exposición a contrapartes en jurisdicciones de alto riesgo.'
RETURN
  WITH a AS (
    SELECT alerta_id, cliente_id, CAST(fecha AS DATE) AS f FROM {FQ}.alertas
    WHERE alerta_id = features_transaccionales.alerta_id
  ),
  t AS (
    SELECT t.* FROM {FQ}.transacciones t JOIN a ON t.cliente_id = a.cliente_id
    WHERE CAST(t.fecha AS DATE) BETWEEN date_sub(a.f, 30) AND a.f
  )
  SELECT
    first(a.alerta_id), first(a.cliente_id), 30,
    count(t.txn_id),
    round(coalesce(sum(t.monto_usd), 0), 2),
    round(coalesce(sum(CASE WHEN t.canal = 'Efectivo' THEN t.monto_usd END), 0), 2),
    round(coalesce(sum(CASE WHEN t.canal = 'Efectivo' THEN t.monto_usd END), 0) / nullif(sum(t.monto_usd), 0), 3),
    count_if(t.canal = 'Efectivo' AND t.monto_usd >= 9000 AND t.monto_usd < 10000),
    count(DISTINCT CASE WHEN t.canal = 'Efectivo' THEN t.fecha END),
    count_if(t.pais_contraparte IN ('Islas Caimán')),
    round(coalesce(sum(CASE WHEN t.pais_contraparte IN ('Islas Caimán') THEN t.monto_usd END), 0), 2),
    round(coalesce(max(t.monto_usd), 0), 2)
  FROM a LEFT JOIN t ON t.cliente_id = a.cliente_id
""")
spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.verificar_sanciones(
  nombre STRING COMMENT 'Nombre completo de la persona o empresa a verificar'
)
RETURNS TABLE (nombre_listado STRING, lista STRING, pais STRING, fecha_inclusion STRING, motivo STRING, distancia INT)
COMMENT 'Coteja un nombre contra las listas de sanciones (OFAC, ONU y listas locales). Devuelve las coincidencias exactas o aproximadas; una coincidencia obliga a bloquear la operación y escalar de inmediato.'
RETURN
  SELECT nombre_listado, lista, pais, fecha_inclusion, motivo,
         levenshtein(lower(nombre_listado), lower(verificar_sanciones.nombre)) AS distancia
  FROM {FQ}.lista_sanciones
  WHERE levenshtein(lower(nombre_listado), lower(verificar_sanciones.nombre)) <= 3
     OR lower(nombre_listado) LIKE concat('%', lower(verificar_sanciones.nombre), '%')
""")
assert spark.sql(f"SELECT * FROM {FQ}.verificar_sanciones('Hector Ivan Montalvo')").count() == 1
assert spark.sql(f"SELECT * FROM {FQ}.verificar_sanciones('Lucía Fernanda Castillo Mena')").count() == 0
print("✅ Funciones obtener_alerta, features_transaccionales, verificar_sanciones")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verificación

# COMMAND ----------

for t in ["clientes", "transacciones", "alertas", "narrativas_casos", "lista_sanciones", "chunks", "verdad_terreno"]:
    print(f"  tabla {t:18s} {spark.table(f'{FQ}.{t}').count():>8,} filas")
print(f"  índice {INDEX_NAME}: {estado_indice()}")
