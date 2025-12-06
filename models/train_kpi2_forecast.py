# talita-backend/models/train_kpi2_forecast.py

import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# ============================================================
# RUTAS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # carpeta models
BACKEND_DIR = BASE_DIR  # si models está directamente dentro de talita-backend

DATA_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "data")
DB_PATH = os.path.join(DATA_DIR, "talita_realista.db")

MODEL_KPI2_PATH = os.path.join(BASE_DIR, "modelo_kpi2_cliente.pkl")
ENC_KPI2_PATH = os.path.join(BASE_DIR, "encoder_kpi2.pkl")
COLS_KPI2_PATH = os.path.join(BASE_DIR, "feature_columns_kpi2.npy")

print("Usando DB:", DB_PATH)

# ============================================================
# 1) CARGAR DATOS POR TICKET (solo modalidad local)
# ============================================================
conn = sqlite3.connect(DB_PATH)

# tickets + datos del cliente
q_tickets = """
SELECT
    t.id_ticket,
    t.id_cliente,
    t.fecha,
    t.total,
    t.ticket_promedio,
    c.edad,
    c.tipo_cliente
FROM tickets t
JOIN clientes c ON c.id_cliente = t.id_cliente
WHERE t.modalidad = 'local'
ORDER BY t.id_cliente, t.fecha;
"""
df_t = pd.read_sql_query(q_tickets, conn, parse_dates=["fecha"])

# número de platos por ticket
q_num_platos = """
SELECT id_ticket, SUM(cantidad) AS num_platos
FROM detalle_ticket
GROUP BY id_ticket;
"""
df_np = pd.read_sql_query(q_num_platos, conn)

# plato principal por ticket (el de mayor subtotal)
q_plato_principal = """
SELECT
    d.id_ticket,
    p.nombre        AS plato,
    SUM(d.cantidad) AS total_porciones,
    SUM(d.subtotal) AS monto
FROM detalle_ticket d
JOIN platos p ON p.id_plato = d.id_plato
GROUP BY d.id_ticket, p.id_plato
"""
df_pp = pd.read_sql_query(q_plato_principal, conn)
df_pp = df_pp.sort_values(["id_ticket", "monto"], ascending=[True, False])
df_top = (
    df_pp.groupby("id_ticket")
    .first()
    .reset_index()[["id_ticket", "plato"]]
    .rename(columns={"plato": "plato_principal"})
)

conn.close()

# merge de todo
df = (
    df_t
    .merge(df_np, on="id_ticket", how="left")
    .merge(df_top, on="id_ticket", how="left")
)

df["num_platos"] = df["num_platos"].fillna(1)

# ============================================================
# 2) CONSTRUIR DATASET DE PARES (estado hoy -> ticket futuro)
# ============================================================
df = df.sort_values(["id_cliente", "fecha"]).reset_index(drop=True)

rows = []
max_horizonte = 60  # para no irnos a horizontes enormes

for id_cli, g in df.groupby("id_cliente"):
    g = g.sort_values("fecha").reset_index(drop=True)
    n = len(g)
    if n < 2:
        continue

    for i in range(n - 1):
        for j in range(i + 1, n):
            dias = (g.loc[j, "fecha"] - g.loc[i, "fecha"]).days
            if dias <= 0 or dias > max_horizonte:
                continue

            rows.append(
                {
                    "id_cliente": id_cli,
                    "edad": g.loc[i, "edad"],
                    "tipo_cliente": g.loc[i, "tipo_cliente"],
                    "ticket_actual": g.loc[i, "ticket_promedio"],
                    "total_actual": g.loc[i, "total"],
                    "num_platos": g.loc[i, "num_platos"],
                    "plato_principal": g.loc[i, "plato_principal"] or "desconocido",
                    "dias_horizonte": dias,
                    "ticket_futuro": g.loc[j, "ticket_promedio"],
                }
            )

df_pairs = pd.DataFrame(rows)
df_pairs = df_pairs.dropna()

print("Filas de entrenamiento KPI2:", len(df_pairs))

if len(df_pairs) < 50:
    raise SystemExit("Muy pocas filas para entrenar el modelo KPI2. Revisa tus datos.")

# numéricas y categóricas
num_cols = ["edad", "ticket_actual", "total_actual", "num_platos", "dias_horizonte"]
cat_cols = ["tipo_cliente", "plato_principal"]

df_pairs[num_cols] = df_pairs[num_cols].apply(pd.to_numeric, errors="coerce")
df_pairs = df_pairs.dropna(subset=num_cols + cat_cols)

X_num = df_pairs[num_cols].reset_index(drop=True)

enc_kpi2 = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = enc_kpi2.fit_transform(df_pairs[cat_cols])
cat_feature_names = enc_kpi2.get_feature_names_out(cat_cols)
X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names)

X = pd.concat([X_num, X_cat_df], axis=1)
y = df_pairs["ticket_futuro"].astype(float).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_kpi2 = DecisionTreeRegressor(
    max_depth=6,
    min_samples_leaf=20,
    random_state=42,
)

model_kpi2.fit(X_train, y_train)

print("Train size:", len(X_train), "Test size:", len(X_test))
print("Rango ticket futuro:", y.min(), "->", y.max())

# ============================================================
# 3) GUARDAR MODELO + ENCODER + COLUMNAS
# ============================================================
joblib.dump(model_kpi2, MODEL_KPI2_PATH)
joblib.dump(enc_kpi2, ENC_KPI2_PATH)
np.save(COLS_KPI2_PATH, X.columns.to_numpy())

print("✅ Modelo KPI2 guardado en:", MODEL_KPI2_PATH)
print("✅ Encoder KPI2 guardado en:", ENC_KPI2_PATH)
print("✅ Columnas KPI2 guardadas en:", COLS_KPI2_PATH)