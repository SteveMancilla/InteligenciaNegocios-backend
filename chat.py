import os
import requests
import json
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

BACKEND_URL = "http://localhost:8000/api"


# --------------------------------------------------------
# Cargar datos SOLO cuando se usen (lazy loading)
# --------------------------------------------------------
PLATOS = None
CLIENTES = None


def get_platos():
    global PLATOS
    if PLATOS is None:
        try:
            resp = requests.get(f"{BACKEND_URL}/kpi1/platos").json()
            PLATOS = {p["nombre"].lower(): p["id_plato"] for p in resp}
        except:
            PLATOS = {}
    return PLATOS


def get_clientes():
    global CLIENTES
    if CLIENTES is None:
        try:
            resp = requests.get(f"{BACKEND_URL}/kpi2/clientes?limit=999").json()
            lista = resp.get("clientes", [])
            CLIENTES = {c["nombre"].lower(): c["id_cliente"] for c in lista}
        except:
            CLIENTES = {}
    return CLIENTES


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------
def extract_numbers(text):
    nums = re.findall(r"\d+\.?\d*", text)
    return [float(n) for n in nums]


def find_plato_in_text(msg):
    platos = get_platos()
    msg_clean = msg.lower().replace(" ", "_")

    for nombre_p in platos.keys():
        if nombre_p.replace(" ", "_") in msg_clean:
            return nombre_p
    return None


def find_cliente_in_text(msg):
    clientes = get_clientes()
    msg_clean = msg.lower()

    for nombre_c in clientes.keys():
        if nombre_c in msg_clean:
            return nombre_c
    return None


# --------------------------------------------------------
# INTENT PARSER
# --------------------------------------------------------
def process_user_message(message: str):
    msg = message.lower().strip()

    # SALUDOS
    if msg in ["hola", "buenas", "hey", "holaaa", "que tal", "hi"]:
        return {
            "accion": "texto",
            "parametros": {"texto": "Hola 😊 ¿En qué puedo ayudarte hoy?"}
        }

    # --------------------------------------------------------
    # KPI 1 — Simular margen por nombre de plato
    # --------------------------------------------------------
    if "simula" in msg and "margen" in msg:

        plato_name = find_plato_in_text(msg)
        if not plato_name:
            return {
                "accion": "texto",
                "parametros": {"texto": "No pude identificar el plato. ¿Puedes escribir su nombre?"}
            }

        id_plato = get_platos()[plato_name]
        nums = extract_numbers(msg)

        if len(nums) < 3:
            return {
                "accion": "texto",
                "parametros": {"texto": "Faltan datos. Indica costo, precio y margen objetivo."}
            }

        costo, precio, margen = nums[:3]

        return {
            "accion": "kpi1_simular",
            "parametros": {
                "id_plato": id_plato,
                "costo": costo,
                "precio_simulado": precio,
                "margen_objetivo": margen
            }
        }

    # --------------------------------------------------------
    # KPI 2 — Forecast por cliente usando nombre
    # --------------------------------------------------------
    if "cliente" in msg and ("predice" in msg or "forecast" in msg):

        cliente_name = find_cliente_in_text(msg)
        if not cliente_name:
            return {
                "accion": "texto",
                "parametros": {"texto": "No pude identificar al cliente. ¿Puedes darme su nombre?"}
            }

        id_cliente = get_clientes()[cliente_name]
        nums = extract_numbers(msg)
        dias = int(nums[0]) if nums else 7

        return {
            "accion": "kpi2_cliente_forecast",
            "parametros": {
                "id_cliente": id_cliente,
                "dias": dias
            }
        }

    # --------------------------------------------------------
    # KPI 3 — proyección de rotación
    # --------------------------------------------------------
    if "proyeccion" in msg or "proyección" in msg:
        nums = extract_numbers(msg)
        dias = int(nums[0]) if nums else 7

        return {
            "accion": "kpi3_proyeccion",
            "parametros": {"dias": dias}
        }

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------
    return {
        "accion": "texto",
        "parametros": {"texto": "¿Podrías darme más detalles?"}
    }


# --------------------------------------------------------
# HANDLE ACTION
# --------------------------------------------------------
def handle_action(data):
    accion = data.get("accion")
    params = data.get("parametros", {})

    if accion == "kpi1_simular":
        return call_backend("/kpi1/simular", "POST", params)

    if accion == "kpi2_cliente_forecast":
        return call_backend(f"/kpi2/cliente/{params['id_cliente']}/forecast?dias={params['dias']}")

    if accion == "kpi3_proyeccion":
        return call_backend(f"/kpi3/proyeccion?horizonte_dias={params['dias']}")

    if accion == "texto":
        return {"respuesta": params["texto"]}

    return {"error": "Acción no reconocida"}


def call_backend(endpoint, method="GET", data=None):
    url = f"{BACKEND_URL}{endpoint}"

    if method == "GET":
        return requests.get(url).json()
    return requests.post(url, json=data).json()