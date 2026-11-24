"""HyM · servidor de la demo (Fase 7).

FastAPI envuelve el estilista de `src/` y sirve el frontend (`app/static/`).

Variables de entorno (ver GUIA_EXPOSICION.txt):
  HM_DATA_DIR    carpeta del paquete exportado por el notebook 07 (por defecto ../hm-demo-data)
  LLM_BASE_URL   p. ej. http://localhost:13305/v1 (Lemonade) o http://localhost:11434/v1 (Ollama)
  LLM_MODEL      p. ej. Qwen3.6
  LLM_THINKING   0 para modelos con razonamiento (Qwen3.x)
  LLM_VISION     1 si el modelo ve imágenes
  HM_DEVICE      cpu | cuda (Fashion-CLIP); por defecto el que haya

Arranque:  uvicorn server:app --host 0.0.0.0 --port 8000   (desde la carpeta app/)
"""
import io
import json
import math
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT / "src"))

DATA = Path(os.environ.get("HM_DATA_DIR", ROOT.parent / "hm-demo-data")).resolve()
os.environ["HM_DATA_DIR"] = str(DATA)
os.environ.setdefault("HM_IMAGES_DIR", str(DATA / "images"))

from data import load_catalog          # noqa: E402  (después de fijar las variables de entorno)
from embeddings import FashionCLIP     # noqa: E402
from llm import LLM                    # noqa: E402
from ranking import Recommender        # noqa: E402
from stylist import Session, Stylist   # noqa: E402

LINEAS = {"mujer": "Ladieswear", "hombre": "Menswear", "divided": "Divided", "ninos": "Baby/Children",
          "sport": "Sport"}
CARD_FIELDS = ["article_id", "prod_name", "product_type_name", "product_group_name", "colour_group_name",
               "index_group_name", "section_name", "detail_desc"]

# ---------------------------------------------------------------- carga (una vez al arrancar)
t0 = time.time()
assert (DATA / "articles.parquet").exists(), f"No encontré el paquete de datos en {DATA} (ver GUIA_EXPOSICION.txt)"
CAT = load_catalog()
REC = Recommender(CAT)
ENC = FashionCLIP(device=os.environ.get("HM_DEVICE"))
LLM_CLIENT = LLM()
TH = json.load(open(DATA / "thresholds.json")) if (DATA / "thresholds.json").exists() else {}
STYLIST = Stylist(REC, ENC, LLM_CLIENT, tau_ooc=TH.get("tau_ooc", 0.76), tau_fashion=TH.get("tau_fashion", 0.6))
ARTS = CAT.articles.set_index("article_id")
BY_POP = CAT.avail[np.argsort(-CAT.pop[CAT.avail])]          # catálogo ordenado por popularidad
print(f"HyM listo: {len(CAT.avail):,} artículos · Fashion-CLIP en {ENC.device} · LLM {LLM_CLIENT.model} "
      f"@ {LLM_CLIENT.base_url} ({time.time() - t0:.0f}s)", flush=True)

LOCK = threading.Lock()          # el modelo y el LLM se usan de uno en uno (demo de un solo usuario)
SESSIONS: dict[str, tuple[Session, float]] = {}


LLM_STATE = {"revisado": False}


def _warm_up():
    """Carga el LLM en memoria antes de la primera pregunta (Lemonade/Ollama lo cargan en la 1.ª petición)."""
    ok = LLM_CLIENT.ping()
    LLM_STATE["revisado"] = True
    if ok:
        LLM_CLIENT.ask('Responde solo {"ok": true}', max_tokens=20)
    print(f"LLM {'listo' if ok else 'NO disponible: ' + str(LLM_CLIENT.last_error)}", flush=True)


threading.Thread(target=_warm_up, daemon=True).start()

app = FastAPI(title="HyM · estilista virtual")
app.mount("/static", StaticFiles(directory=APP / "static"), name="static")


# ---------------------------------------------------------------- utilidades
def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _card(aid, extra=None):
    a = ARTS.loc[aid]
    out = {"article_id": aid, **{c: _clean(a[c]) for c in CARD_FIELDS[1:] if c in a.index}}
    out["img"] = f"/img/{aid}.jpg"
    return {**out, **(extra or {})}


def _session(sid):
    now = time.time()
    for k in [k for k, (_, t) in SESSIONS.items() if now - t > 3 * 3600]:   # limpia sesiones viejas
        SESSIONS.pop(k, None)
    s = SESSIONS.get(sid, (Session(), now))[0]
    SESSIONS[sid] = (s, now)
    return s


# ---------------------------------------------------------------- rutas
@app.get("/")
def index():
    return FileResponse(APP / "static" / "index.html")


@app.get("/img/{article_id}.jpg")
def image(article_id: str):
    p = CAT.image_path(article_id) if article_id.isdigit() else None
    if p is None:
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/estado")
def estado():
    # Si el LLM estaba caído y ya pasó la pausa, se vuelve a probar en segundo plano (p. ej. Lemonade arrancó tarde)
    if LLM_STATE["revisado"] and LLM_CLIENT.last_error and time.time() > LLM_CLIENT._down_until:
        threading.Thread(target=LLM_CLIENT.ping, daemon=True).start()
    return {"articulos": int(len(CAT.avail)), "clip": ENC.device, "llm_url": LLM_CLIENT.base_url,
            "llm_modelo": LLM_CLIENT.model, "llm_ok": LLM_CLIENT.last_error is None,
            "llm_error": LLM_CLIENT.last_error, "vision": LLM_CLIENT.vision, "llm_revisado": LLM_STATE["revisado"]}


@app.get("/api/catalogo")
def catalogo(linea: str = "", grupo: str = "", pagina: int = 0, n: int = 24):
    rows = BY_POP
    arts = CAT.articles
    if linea in LINEAS:
        rows = rows[arts.index_group_name.to_numpy()[rows] == LINEAS[linea]]
    if grupo:
        rows = rows[arts.product_group_name.to_numpy()[rows] == grupo]
    # un artículo por producto (no repetir la misma prenda en varios colores)
    _, first = np.unique(CAT.pcode[rows], return_index=True)
    rows = rows[np.sort(first)]
    page = rows[pagina * n:(pagina + 1) * n]
    return {"total": int(len(rows)), "articulos": [_card(CAT.aid[i]) for i in page]}


@app.get("/api/grupos")
def grupos(linea: str = ""):
    arts = CAT.articles.iloc[CAT.avail]
    if linea in LINEAS:
        arts = arts[arts.index_group_name == LINEAS[linea]]
    vc = arts.product_group_name.value_counts()
    skip = {"Unknown", "Items", "Furniture", "Stationery", "Garment and Shoe care", "Interior textile", "Fun"}
    return [{"grupo": g, "n": int(c)} for g, c in vc.items() if g not in skip]


@app.post("/api/chat")
def chat(session_id: str = Form(""), mensaje: str = Form(""), article_id: str = Form(""),
         imagen: UploadFile | None = File(None)):
    sid = session_id or uuid.uuid4().hex
    img = None
    if imagen is not None and imagen.filename:
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(imagen.file.read()))).convert("RGB")
        except Exception:
            raise HTTPException(400, "No pude leer la imagen")
    elif article_id:
        img = CAT.load_image(article_id)                     # "¿con qué combino esto?" desde el catálogo
        if img is None:
            raise HTTPException(404, "Artículo sin imagen")
    if img is None and not mensaje.strip():
        raise HTTPException(400, "Escribe un mensaje o sube una foto")

    with LOCK:
        s = _session(sid)
        r = STYLIST.chat(s, mensaje, img)
    plan = r.get("plan") or {}
    items = [_card(a["article_id"], {"tipo": a["tipo"], "motivo": a["motivo"]}) for a in r["articulos"]]
    return JSONResponse({
        "session_id": sid,
        "mensaje": r["mensaje"],
        "articulos": items,
        "flags": r["flags"],
        "plan": {k: plan.get(k) for k in ("intencion", "consulta_en", "color", "excluir_color", "linea",
                                         "tipo_complemento")} if plan else None,
        "latencias": {k: round(v, 2) for k, v in r["latencias"].items()},
        "llm_ok": LLM_CLIENT.last_error is None,
    })


@app.post("/api/reiniciar")
def reiniciar(session_id: str = Form("")):
    SESSIONS.pop(session_id, None)
    return {"ok": True}
