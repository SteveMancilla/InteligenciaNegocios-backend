# talita-backend/app.py
import os
import sqlite3
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from chat import process_user_message, handle_action

# --------------------------------------------------
# Configuración Flask
# --------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

DB_PATH = os.path.join(DATA_DIR, "talita_realista.db")

# Modelo general (3 KPIs en uno)
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_ohe.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.npy")
TREE_PATH = os.path.join(MODEL_DIR, "modelo_arbol.pkl")

# Modelo específico KPI2 (forecast por cliente)
KPI2_MODEL_PATH = os.path.join(MODEL_DIR, "modelo_kpi2_cliente.pkl")
KPI2_ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_kpi2.pkl")
KPI2_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns_kpi2.npy")

# --------------------------------------------------
# Cargar modelo + encoder (modelo general)
# --------------------------------------------------
if not (
    os.path.exists(ENCODER_PATH)
    and os.path.exists(FEATURES_PATH)
    and os.path.exists(TREE_PATH)
):
    raise SystemExit(
        "❌ No se encontraron encoder_ohe.pkl, feature_columns.npy o modelo_arbol.pkl"
    )

encoder = joblib.load(ENCODER_PATH)
feature_columns: np.ndarray = np.load(FEATURES_PATH, allow_pickle=True)
model = joblib.load(TREE_PATH)

# columnas numéricas y categóricas usadas en el entrenamiento (modelo general)
NUM_COLS: List[str] = [
    "edad",
    "precio_plato",
    "costo_plato",
    "cantidad",
    "subtotal",
    "total_ticket",
    "ticket_promedio_ticket",
    "num_clientes",
    "ventas_totales_dia",
    "cant_top_mes",
]

CAT_COLS: List[str] = [
    "modalidad",
    "tipo_cliente",
    "categoria_plato",
    "extra",
    "nombre_plato",
]

# --------------------------------------------------
# Cargar modelo KPI2 (forecast por cliente)
# --------------------------------------------------
if not (
    os.path.exists(KPI2_MODEL_PATH)
    and os.path.exists(KPI2_ENCODER_PATH)
    and os.path.exists(KPI2_COLS_PATH)
):
    raise SystemExit(
        "❌ Faltan archivos de modelo KPI2: ejecuta models/train_kpi2_forecast.py"
    )

kpi2_model = joblib.load(KPI2_MODEL_PATH)
kpi2_encoder = joblib.load(KPI2_ENCODER_PATH)
kpi2_feature_cols: np.ndarray = np.load(KPI2_COLS_PATH, allow_pickle=True)

# --------------------------------------------------
# Helpers generales
# --------------------------------------------------
def get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def build_feature_row(plato_row, costo: float, precio_simulado: float) -> pd.DataFrame:
    """
    Crea una fila de features aproximada para el modelo general,
    usando el plato, costo y precio que queremos evaluar.
    """
    base = {
        "edad": 30,  # aproximación
        "precio_plato": precio_simulado,
        "costo_plato": costo,
        "cantidad": 1,
        "subtotal": precio_simulado,
        "total_ticket": precio_simulado,
        "ticket_promedio_ticket": precio_simulado,
        "num_clientes": 1,
        "ventas_totales_dia": precio_simulado,
        "cant_top_mes": 10,  # valor medio
        "modalidad": "local",
        "tipo_cliente": "regular",
        "categoria_plato": plato_row["categoria"],
        "extra": "ninguno",
        "nombre_plato": plato_row["nombre"],
    }

    df = pd.DataFrame([base])
    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")

    # One-hot de categóricas
    X_cat = encoder.transform(df[CAT_COLS])
    cat_feature_names = encoder.get_feature_names_out(CAT_COLS)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names)

    X_num_df = df[NUM_COLS].reset_index(drop=True)
    X_full = pd.concat([X_num_df, X_cat_df], axis=1)

    # Alinear columnas esperadas
    for col in feature_columns:
        if col not in X_full.columns:
            X_full[col] = 0.0

    X_full = X_full[list(feature_columns)]
    return X_full


def build_kpi2_feature_row(id_cliente: int, dias_horizonte: int) -> Optional[pd.DataFrame]:
    """
    Arma la fila de features para predecir el ticket promedio
    de un cliente en un horizonte de N días, usando su ÚLTIMO
    ticket en modalidad 'local'.
    """
    conn = get_db_conn()
    cur = conn.cursor()

    # Último ticket del cliente en local
    cur.execute(
        """
        SELECT 
            t.id_ticket,
            t.total,
            t.ticket_promedio,
            c.edad,
            c.tipo_cliente
        FROM tickets t
        JOIN clientes c ON c.id_cliente = t.id_cliente
        WHERE t.modalidad = 'local'
          AND t.id_cliente = ?
        ORDER BY t.fecha DESC
        LIMIT 1
        """,
        (id_cliente,),
    )
    row = cur.fetchone()
    if row is None:
        conn.close()
        return None

    id_ticket, total, ticket_promedio, edad, tipo_cliente = row

    # Plato principal + número de platos de ese ticket
    cur.execute(
        """
        SELECT 
            p.nombre,
            SUM(d.cantidad) AS num_platos,
            SUM(d.subtotal) AS monto
        FROM detalle_ticket d
        JOIN platos p ON p.id_plato = d.id_plato
        WHERE d.id_ticket = ?
        GROUP BY p.id_plato, p.nombre
        ORDER BY monto DESC
        LIMIT 1
        """,
        (id_ticket,),
    )
    row_p = cur.fetchone()
    conn.close()

    if row_p:
        plato_principal, num_platos, _monto = row_p
    else:
        plato_principal = "desconocido"
        num_platos = 1

    base = {
        "edad": float(edad),
        "ticket_actual": float(ticket_promedio),
        "total_actual": float(total),
        "num_platos": float(num_platos),
        "dias_horizonte": float(dias_horizonte),
        "tipo_cliente": tipo_cliente,
        "plato_principal": plato_principal,
    }

    df = pd.DataFrame([base])

    num_cols = ["edad", "ticket_actual", "total_actual", "num_platos", "dias_horizonte"]
    cat_cols = ["tipo_cliente", "plato_principal"]

    X_num = df[num_cols].apply(pd.to_numeric, errors="coerce")
    X_cat = kpi2_encoder.transform(df[cat_cols])
    cat_names = kpi2_encoder.get_feature_names_out(cat_cols)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_names)

    X_full = pd.concat([X_num.reset_index(drop=True), X_cat_df], axis=1)

    # Alinear con columnas del entrenamiento KPI2
    for col in kpi2_feature_cols:
        if col not in X_full.columns:
            X_full[col] = 0.0

    X_full = X_full[list(kpi2_feature_cols)]
    return X_full


# --------------------------------------------------
# API KPI1
# --------------------------------------------------
@app.get("/api/kpi1/platos")
def api_kpi1_platos():
    """
    Lista de platos para llenar el combo.
    """
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id_plato, nombre, categoria, precio, costo
        FROM platos
        ORDER BY nombre
        """
    )
    rows = cur.fetchall()
    conn.close()

    platos = [
        {
            "id_plato": r["id_plato"],
            "nombre": r["nombre"],
            "categoria": r["categoria"],
            "precio": float(r["precio"]),
            "costo": float(r["costo"]),
        }
        for r in rows
    ]
    return jsonify(platos)


@app.post("/api/kpi1/simular")
def api_kpi1_simular():
    """
    Simulación de margen por plato (KPI1).
    """
    data = request.get_json(force=True)
    id_plato = data.get("id_plato")
    costo = float(data.get("costo", 0) or 0)
    precio_simulado = float(data.get("precio_simulado", 0) or 0)
    margen_objetivo = float(data.get("margen_objetivo", 0) or 0)

    if not id_plato or costo <= 0 or precio_simulado <= 0 or margen_objetivo <= 0:
        return jsonify({"error": "Datos incompletos o inválidos."}), 400

    # Buscar plato
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id_plato, nombre, categoria, precio, costo FROM platos WHERE id_plato = ?",
        (id_plato,),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return jsonify({"error": "Plato no encontrado."}), 404

    # Features para el modelo general
    X_row = build_feature_row(row, costo=costo, precio_simulado=precio_simulado)
    pred = model.predict(X_row)[0]  # [margen_linea, ticket_promedio_dia, rotacion_code]
    margen_model_soles = float(pred[0])

    # Teoría simple
    utilidad_teorica_soles = precio_simulado - costo
    margen_teorico_pct = (utilidad_teorica_soles / precio_simulado) * 100.0

    # Precio objetivo para el margen deseado
    margen_obj_frac = margen_objetivo / 100.0
    if margen_obj_frac >= 1.0:
        precio_objetivo = None
    else:
        precio_objetivo = costo / (1.0 - margen_obj_frac)

    # Margen según el modelo en porcentaje aprox
    margen_model_pct = (
        (margen_model_soles / precio_simulado) * 100.0 if precio_simulado > 0 else 0.0
    )

    diff_pct = margen_teorico_pct - margen_objetivo

    if margen_teorico_pct <= margen_objetivo - 5:
        clasificacion = "bajo"
        mensaje = (
            f"El margen teórico está {abs(diff_pct):.1f} puntos por debajo del "
            f"objetivo ({margen_objetivo:.1f}%). Podrías considerar subir el precio "
            "o reducir costos."
        )
    elif margen_teorico_pct >= margen_objetivo + 5:
        clasificacion = "alto"
        mensaje = (
            f"El margen teórico está {diff_pct:.1f} puntos por encima del objetivo. "
            "Es un margen cómodo; revisa que el precio siga siendo competitivo."
        )
    else:
        clasificacion = "alineado"
        mensaje = (
            "El margen teórico está dentro de un rango razonable respecto al objetivo."
        )

    result = {
        "plato": row["nombre"],
        "categoria": row["categoria"],
        "costo_usado": round(costo, 2),
        "precio_simulado": round(precio_simulado, 2),
        "utilidad_teorica_soles": round(utilidad_teorica_soles, 2),
        "margen_teorico_pct": round(margen_teorico_pct, 1),
        "margen_objetivo_pct": round(margen_objetivo, 1),
        "precio_objetivo": round(precio_objetivo, 2) if precio_objetivo else None,
        "margen_model_soles": round(margen_model_soles, 2),
        "margen_model_pct": round(margen_model_pct, 1),
        "diff_pct": round(diff_pct, 1),
        "clasificacion": clasificacion,
        "mensaje": mensaje,
    }
    return jsonify(result)


# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/api/health")
def api_health():
    return jsonify({"status": "ok"})


# ============================================================
# KPI 2 – Ticket promedio diario (clientes en local)
# ============================================================

API_PREFIX = "/api"


@app.get(f"{API_PREFIX}/kpi2/resumen")
def api_kpi2_resumen():
    """
    Resumen general de KPI2:
    - Ticket promedio último mes / semana / día (solo modalidad 'local')
    - Predicción simple para 1, 3 y 7 días (agregada)
    - Platos donde más gastan los clientes en local
    """
    conn = get_db_conn()
    cur = conn.cursor()

    # 1) Tickets en modalidad local
    cur.execute(
        """
        SELECT 
            fecha,
            id_cliente,
            modalidad,
            total,
            ticket_promedio
        FROM tickets
        WHERE modalidad = 'local'
        """
    )
    rows = cur.fetchall()
    conn.close()

    if len(rows) == 0:
        return jsonify({"error": "No hay tickets en modalidad 'local'"}), 404

    df = pd.DataFrame(
        rows, columns=["fecha", "id_cliente", "modalidad", "total", "ticket_promedio"]
    )
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Último día, semana y mes
    fecha_max = df["fecha"].max()
    ultimo_dia = fecha_max.normalize()
    inicio_semana = ultimo_dia - pd.Timedelta(days=6)
    inicio_mes = ultimo_dia.replace(day=1)

    df_ultimo_mes = df[df["fecha"] >= inicio_mes]
    df_ultima_semana = df[df["fecha"] >= inicio_semana]
    df_ultimo_dia = df[df["fecha"] == ultimo_dia]

    ticket_prom_ultimo_mes = float(df_ultimo_mes["ticket_promedio"].mean())
    ticket_prom_ultima_semana = float(df_ultima_semana["ticket_promedio"].mean())
    ticket_prom_ultimo_dia = float(df_ultimo_dia["ticket_promedio"].mean())

    # 2) Predicción agregada simple
    base_ref = (
        ticket_prom_ultima_semana
        if not pd.isna(ticket_prom_ultima_semana)
        else ticket_prom_ultimo_mes
    )

    pred_1d = base_ref * 1.05  # +5 %
    pred_3d = base_ref * 1.07  # +7 %
    pred_7d = base_ref * 1.10  # +10 %

    # 3) Platos donde más gastan en local
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            p.nombre,
            COUNT(d.id_detalle)     AS consumos,
            SUM(d.subtotal)         AS monto_total,
            AVG(t.ticket_promedio)  AS ticket_promedio
        FROM detalle_ticket d
        JOIN platos   p ON p.id_plato = d.id_plato
        JOIN tickets  t ON t.id_ticket = d.id_ticket
        WHERE t.modalidad = 'local'
        GROUP BY p.id_plato, p.nombre
        ORDER BY monto_total DESC
        LIMIT 6
        """
    )
    rows_platos = cur.fetchall()
    conn.close()

    top_platos = []
    for nombre, consumos, monto_total, ticket_promedio in rows_platos:
        top_platos.append(
            {
                "nombre_plato": nombre,
                "consumos": int(consumos),
                "monto_total": float(monto_total),
                "ticket_promedio": float(ticket_promedio),
            }
        )

    return jsonify(
        {
            "ticket_prom_ultimo_mes": round(ticket_prom_ultimo_mes, 2),
            "ticket_prom_ultima_semana": round(ticket_prom_ultima_semana, 2),
            "ticket_prom_ultimo_dia": round(ticket_prom_ultimo_dia, 2),
            "prediccion_1d": round(pred_1d, 2),
            "prediccion_3d": round(pred_3d, 2),
            "prediccion_7d": round(pred_7d, 2),
            "top_platos_local": top_platos,
        }
    )


@app.get(f"{API_PREFIX}/kpi2/clientes")
def api_kpi2_clientes():
    """
    Lista de clientes (solo modalidad local) ordenados por ticket promedio,
    con un parámetro 'limit' para decidir cuántos ver (10, 20, 50...).
    """
    try:
        limit = int(request.args.get("limit", 10))
        if limit <= 0:
            limit = 10
    except ValueError:
        limit = 10

    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT 
            c.id_cliente,
            c.nombre,
            COUNT(t.id_ticket)          AS num_tickets,
            SUM(t.total)                AS total_gastado,
            AVG(t.ticket_promedio)      AS ticket_promedio
        FROM tickets t
        JOIN clientes c ON c.id_cliente = t.id_cliente
        WHERE t.modalidad = 'local'
        GROUP BY c.id_cliente, c.nombre
        ORDER BY ticket_promedio DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()

    clientes = []
    for id_cliente, nombre, num_tickets, total_gastado, ticket_promedio in rows:
        clientes.append(
            {
                "id_cliente": int(id_cliente),
                "nombre": nombre,
                "num_tickets": int(num_tickets),
                "total_gastado": float(total_gastado),
                "ticket_promedio": float(ticket_promedio),
            }
        )

    return jsonify({"clientes": clientes})


@app.get(f"{API_PREFIX}/kpi2/cliente/<int:id_cliente>/detalle")
def api_kpi2_cliente_detalle(id_cliente: int):
    """
    Detalle de un cliente específico:
    - resumen de su historial (tickets, total gastado, ticket promedio)
    - 'ticket_esperado_proxima_semana' (proyección simple = su promedio)
    - top platos que consume ese cliente en local
    """
    conn = get_db_conn()
    cur = conn.cursor()

    # Resumen del cliente
    cur.execute(
        """
        SELECT 
            c.id_cliente,
            c.nombre,
            COUNT(t.id_ticket)          AS num_tickets,
            SUM(t.total)                AS total_gastado,
            AVG(t.ticket_promedio)      AS ticket_promedio
        FROM tickets t
        JOIN clientes c ON c.id_cliente = t.id_cliente
        WHERE t.modalidad = 'local'
          AND c.id_cliente = ?
        """,
        (id_cliente,),
    )
    row = cur.fetchone()

    if not row:
        conn.close()
        return jsonify({"error": "Cliente no encontrado o sin tickets en local"}), 404

    id_cli, nombre, num_tickets, total_gastado, ticket_promedio = row

    # Top platos del cliente
    cur.execute(
        """
        SELECT 
            p.id_plato,
            p.nombre,
            SUM(d.cantidad) AS total_porciones,
            SUM(d.subtotal) AS monto
        FROM detalle_ticket d
        JOIN platos  p ON p.id_plato = d.id_plato
        JOIN tickets t ON t.id_ticket = d.id_ticket
        WHERE t.modalidad = 'local'
          AND t.id_cliente = ?
        GROUP BY p.id_plato, p.nombre
        ORDER BY monto DESC
        LIMIT 5
        """,
        (id_cliente,),
    )

    rows_platos = cur.fetchall()
    conn.close()

    top_platos = []
    for id_plato, nombre_plato, total_porciones, monto in rows_platos:
        top_platos.append(
            {
                "id_plato": int(id_plato),
                "nombre": nombre_plato,
                "total_porciones": float(total_porciones),
                "monto": float(monto),
            }
        )

    # Proyección simple: se espera que su ticket típico la próxima semana
    # sea similar a su ticket promedio histórico en local
    ticket_esperado = float(ticket_promedio)

    return jsonify(
        {
            "cliente": {
                "id_cliente": int(id_cli),
                "nombre": nombre,
                "num_tickets": int(num_tickets),
                "total_gastado": float(total_gastado),
                "ticket_promedio": float(ticket_promedio),
            },
            "ticket_esperado_proxima_semana": round(ticket_esperado, 2),
            "top_platos": top_platos,
        }
    )


@app.get(f"{API_PREFIX}/kpi2/cliente/<int:id_cliente>/forecast")
def api_kpi2_cliente_forecast(id_cliente: int):
    """
    Predicción del ticket promedio de un cliente en modalidad local
    para un horizonte de N días (1–60), usando el modelo KPI2.
    """
    dias_str = request.args.get("dias", "7")
    try:
        dias = int(dias_str)
    except ValueError:
        dias = 7

    if dias < 1:
        dias = 1
    if dias > 60:
        dias = 60

    X = build_kpi2_feature_row(id_cliente, dias)
    if X is None:
        return jsonify({"error": "Cliente sin tickets en modalidad local"}), 404

    y_pred = float(kpi2_model.predict(X)[0])

    return jsonify(
        {
            "id_cliente": id_cliente,
            "horizonte_dias": dias,
            "ticket_predicho": round(y_pred, 2),
        }
    )


# ============================================================
# KPI 3 – Rotación de platos (resumen)
# ============================================================
def _clas_rotacion(cantidad: float) -> int:
    """
    Misma lógica que usaste para entrenar el modelo:
      0 = baja
      1 = media
      2 = alta
    """
    if cantidad <= 10:
        return 0
    if cantidad <= 20:
        return 1
    return 2

# ============================================================
# KPI 3 – Helpers de rotación
# ============================================================

def _label_rotacion(code: int) -> str:
    """Devuelve la etiqueta de rotación a partir del código 0/1/2."""
    if code == 2:
        return "Alta"
    if code == 1:
        return "Media"
    return "Baja"


def build_kpi3_feature_row_for_model(
    nombre_plato: str,
    categoria_plato: str,
    precio_plato: float,
    costo_plato: float,
    cant_prom_mes: float,
) -> pd.DataFrame:
    """
    Crea una fila de features aproximada para que el modelo
    estime la rotación (rotacion_code) de un plato.

    Usamos heurísticas razonables para las demás variables numéricas.
    """
    base = {
        # valores promedio aproximados
        "edad": 35,
        "precio_plato": float(precio_plato),
        "costo_plato": float(costo_plato),
        "cantidad": 1.0,
        "subtotal": float(precio_plato),
        "total_ticket": float(precio_plato),
        "ticket_promedio_ticket": float(precio_plato),
        "num_clientes": 20.0,
        "ventas_totales_dia": float(precio_plato) * 20.0,
        "cant_top_mes": float(cant_prom_mes),

        "modalidad": "local",
        "tipo_cliente": "regular",
        "categoria_plato": categoria_plato,
        "extra": "ninguno",
        "nombre_plato": nombre_plato,
    }

    df = pd.DataFrame([base])
    # aseguramos numéricos
    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")

    # One-Hot de categóricas usando el encoder entrenado
    X_cat = encoder.transform(df[CAT_COLS])
    cat_feature_names = encoder.get_feature_names_out(CAT_COLS)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names)

    X_num_df = df[NUM_COLS].reset_index(drop=True)
    X_full = pd.concat([X_num_df, X_cat_df], axis=1)

    # alineamos columnas esperadas por el modelo
    for col in feature_columns:
        if col not in X_full.columns:
            X_full[col] = 0.0

    X_full = X_full[list(feature_columns)]
    return X_full



@app.get("/api/kpi3/resumen")
def api_kpi3_resumen():
    """
    Analiza la tabla rotacion_insumos + platos y devuelve:

    - mes_analizado: último mes disponible (ej. '2024-08')
    - conteo_rotacion: cuántos platos hay en baja / media / alta
    - platos_por_clase (alta, media, baja)
    """
    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT 
            r.mes,           -- formato 'YYYY-MM'
            r.id_plato,
            r.cantidad_vendida_mes,
            p.nombre
        FROM rotacion_insumos r
        JOIN platos p ON p.id_plato = r.id_plato
        """
    )
    rows = cur.fetchall()

    if len(rows) == 0:
        conn.close()
        return jsonify({"error": "No hay datos en rotacion_insumos"}), 404

    df = pd.DataFrame(
        rows, columns=["mes", "id_plato", "cantidad_vendida_mes", "nombre"]
    )

    # mes -> fecha para poder ordenar
    df["mes"] = pd.to_datetime(df["mes"].astype(str) + "-01")  # '2024-08' -> 2024-08-01

    # Último mes
    ultimo_mes = df["mes"].max()
    df_ultimo = df[df["mes"] == ultimo_mes].copy()

    # Cantidades por plato en el último mes
    g_ultimo = (
        df_ultimo.groupby(["id_plato", "nombre"])["cantidad_vendida_mes"]
        .sum()
        .reset_index()
        .rename(columns={"cantidad_vendida_mes": "cantidad_mes"})
    )

    # Promedio de los últimos 3 meses
    meses_ordenados = sorted(df["mes"].unique())
    ultimos_3 = meses_ordenados[-3:] if len(meses_ordenados) >= 3 else meses_ordenados

    df_3m = df[df["mes"].isin(ultimos_3)].copy()
    g_3m = (
        df_3m.groupby(["id_plato", "nombre"])["cantidad_vendida_mes"]
        .mean()
        .reset_index()
        .rename(columns={"cantidad_vendida_mes": "promedio_3m"})
    )

    # Merge
    df_merge = pd.merge(g_ultimo, g_3m, on=["id_plato", "nombre"], how="left")

    # Demanda semanal esperada: promedio mensual / 4
    df_merge["demanda_semanal_esperada"] = (df_merge["promedio_3m"] / 4.0).round(1)

    # Clasificación de rotación
    df_merge["rotacion_code"] = df_merge["cantidad_mes"].apply(_clas_rotacion)

    def label_rot(c):
        if c == 2:
            return "Alta"
        if c == 1:
            return "Media"
        return "Baja"

    df_merge["rotacion_label"] = df_merge["rotacion_code"].apply(label_rot)

    # Conteos por clase
    conteo = (
        df_merge.groupby(["rotacion_code", "rotacion_label"])["id_plato"]
        .count()
        .reset_index()
        .rename(columns={"id_plato": "conteo"})
        .to_dict("records")
    )

    # Listas de platos por clase
    df_ordenado = df_merge.sort_values("cantidad_mes", ascending=False)

    def platos_por_clase(code):
        sub = df_ordenado[df_ordenado["rotacion_code"] == code].head(6)
        return [
            {
                "id_plato": int(row.id_plato),
                "nombre": row.nombre,
                "cantidad_mes": float(row.cantidad_mes),
                "demanda_semanal_esperada": float(row.demanda_semanal_esperada),
            }
            for _, row in sub.iterrows()
        ]

    platos_alta = platos_por_clase(2)
    platos_media = platos_por_clase(1)
    platos_baja = platos_por_clase(0)

    conn.close()

    return jsonify(
        {
            "mes_analizado": ultimo_mes.strftime("%Y-%m"),
            "conteo_rotacion": conteo,
            "platos_por_clase": {
                "alta": platos_alta,
                "media": platos_media,
                "baja": platos_baja,
            },
        }
    )

@app.get("/api/kpi3/proyeccion")
def api_kpi3_proyeccion():
    """
    Proyección de rotación de platos a N días (7, 14, 30, etc.)
    usando:
      - histórico de rotacion_insumos (real)
      - modelo de árbol para ajustar la rotación (rotacion_code)

    Parámetros (query):
      - horizonte_dias: entero > 0 (por defecto 7)
      - limit: cuántos platos devolver (por defecto 10)
    """
    # --- parámetros de la query ---
    try:
        horizonte_dias = int(request.args.get("horizonte_dias", 7))
        if horizonte_dias <= 0:
            horizonte_dias = 7
    except ValueError:
        horizonte_dias = 7

    try:
        limit = int(request.args.get("limit", 10))
        if limit <= 0:
            limit = 10
    except ValueError:
        limit = 10

    # --- 1) Traer histórico de rotación + datos del plato ---
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            r.mes,                     -- 'YYYY-MM'
            r.id_plato,
            r.cantidad_vendida_mes,
            p.nombre,
            p.categoria,
            p.precio,
            p.costo
        FROM rotacion_insumos r
        JOIN platos p ON p.id_plato = r.id_plato
        """
    )
    rows = cur.fetchall()
    conn.close()

    if len(rows) == 0:
        return jsonify({"error": "No hay datos en rotacion_insumos"}), 404

    df = pd.DataFrame(
        rows,
        columns=[
            "mes",
            "id_plato",
            "cantidad_mes",
            "nombre",
            "categoria",
            "precio",
            "costo",
        ],
    )

    # mes como fecha (primer día del mes)
    df["mes"] = pd.to_datetime(df["mes"].astype(str) + "-01")

    # --- 2) Último mes y promedio últimos 3 meses ---
    ultimo_mes = df["mes"].max()

    # cantidades del último mes
    df_ultimo = df[df["mes"] == ultimo_mes].copy()
    g_ultimo = (
        df_ultimo.groupby(
            ["id_plato", "nombre", "categoria", "precio", "costo"]
        )["cantidad_mes"]
        .sum()
        .reset_index()
        .rename(columns={"cantidad_mes": "cantidad_mes_real"})
    )

    # promedio de los últimos 3 meses (si existen)
    meses_ordenados = sorted(df["mes"].unique())
    ultimos_3 = meses_ordenados[-3:] if len(meses_ordenados) >= 3 else meses_ordenados
    df_3m = df[df["mes"].isin(ultimos_3)].copy()

    g_3m = (
        df_3m.groupby(
            ["id_plato", "nombre", "categoria", "precio", "costo"]
        )["cantidad_mes"]
        .mean()
        .reset_index()
        .rename(columns={"cantidad_mes": "promedio_3m"})
    )

    # combinamos: por plato -> cantidad último mes + promedio 3 meses
    df_merge = pd.merge(
        g_ultimo,
        g_3m,
        on=["id_plato", "nombre", "categoria", "precio", "costo"],
        how="left",
    )
    # si no hay 3 meses, usamos el último mes como promedio
    df_merge["promedio_3m"] = df_merge["promedio_3m"].fillna(
        df_merge["cantidad_mes_real"]
    )

    # --- 3) Demanda histórica diaria y a N días ---
    # aprox: 1 mes = 30 días
    df_merge["demanda_hist_dia"] = df_merge["promedio_3m"] / 30.0
    df_merge["demanda_hist_n_dias"] = (
        df_merge["demanda_hist_dia"] * horizonte_dias
    )

    # clasificación histórica de rotación (baja / media / alta)
    df_merge["rotacion_hist_code"] = df_merge["cantidad_mes_real"].apply(
        _clas_rotacion
    )

    resultados = []

    # factores para ajustar la demanda según el modelo
    factores_rotacion = {
        0: 0.85,  # baja → un poco menos que el histórico
        1: 1.00,  # media → similar al histórico
        2: 1.15,  # alta → un poco más que el histórico
    }

    # --- 4) Aplicar el modelo para cada plato ---
    for _, row in df_merge.iterrows():
        # features para el modelo a partir del plato y su rotación promedio
        X_row = build_kpi3_feature_row_for_model(
            nombre_plato=row["nombre"],
            categoria_plato=row["categoria"],
            precio_plato=row["precio"],
            costo_plato=row["costo"],
            cant_prom_mes=row["promedio_3m"],
        )

        # el modelo devuelve [margen_linea, ticket_promedio_dia, rotacion_code]
        pred = model.predict(X_row)[0]
        rot_code_modelo = int(round(pred[2]))
        if rot_code_modelo < 0:
            rot_code_modelo = 0
        if rot_code_modelo > 2:
            rot_code_modelo = 2

        # demanda esperada ajustada por el modelo
        factor = factores_rotacion.get(rot_code_modelo, 1.0)
        demanda_hist_n = float(row["demanda_hist_n_dias"])
        demanda_modelo_n = demanda_hist_n * factor

        # construir mensaje / alerta simple
        rot_hist_label = _label_rotacion(int(row["rotacion_hist_code"]))
        rot_model_label = _label_rotacion(rot_code_modelo)

        if rot_model_label == "Alta" and demanda_modelo_n > demanda_hist_n * 1.05:
            alerta = (
                "Posible aumento de demanda. Revisa inventario y evita quiebres de stock."
            )
        elif rot_model_label == "Baja" and demanda_modelo_n < demanda_hist_n * 0.95:
            alerta = (
                "Demanda a la baja. Evita sobrecompras y considera promociones puntuales."
            )
        else:
            alerta = "Rotación estable según histórico y modelo."

        resultados.append(
            {
                "id_plato": int(row["id_plato"]),
                "nombre": row["nombre"],
                "categoria": row["categoria"],
                "cantidad_mes_real": float(row["cantidad_mes_real"]),
                "rotacion_historica": rot_hist_label,
                "rotacion_modelo": rot_model_label,
                "demanda_historica_n_dias": round(demanda_hist_n, 1),
                "demanda_modelo_n_dias": round(demanda_modelo_n, 1),
                "alerta": alerta,
            }
        )

    # ordenamos por demanda esperada (modelo) desc y limitamos
    resultados = sorted(
        resultados, key=lambda r: r["demanda_modelo_n_dias"], reverse=True
    )[:limit]

    return jsonify(
        {
            "horizonte_dias": horizonte_dias,
            "mes_base": ultimo_mes.strftime("%Y-%m"),
            "platos": resultados,
        }
    )


@app.post("/api/chat")
def api_chat():
    data = request.get_json(force=True)
    message = data.get("message", "")

    parsed = process_user_message(message)
    result = handle_action(parsed)

    return jsonify(result)


if __name__ == "__main__":
    #app.run(debug=True, port=8000)
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)