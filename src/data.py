"""Carga de los artefactos que generan los notebooks 01, 03 y 04 (y 06 para `available.parquet`).

Busca los archivos en:
  - la variable de entorno HM_DATA_DIR, si existe;
  - /kaggle/input en Kaggle (Inputs del notebook);
  - data/ en local.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE = os.path.exists("/kaggle/input")
COMPETITION = "h-and-m-personalized-fashion-recommendations"


def data_roots():
    env = os.environ.get("HM_DATA_DIR")
    if env:
        return [Path(env)]
    return [Path("/kaggle/input")] if KAGGLE else [Path("data")]


def find(name, roots=None, required=True):
    """Primer archivo llamado `name` bajo las raíces. No entra a carpetas `images` (105k fotos: muy lento)."""
    roots = roots or data_roots()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != "images")
            if name in filenames:
                return Path(dirpath) / name
    if required:
        raise FileNotFoundError(f"No encontré {name} en {[str(r) for r in roots]}: "
                                "agrega como Input el notebook que lo genera")
    return None


def find_images_dir():
    """Carpeta de imágenes del catálogo (subcarpetas 010/, 011/, ...)."""
    candidates = [os.environ.get("HM_IMAGES_DIR"),
                  f"/kaggle/input/competitions/{COMPETITION}/images",
                  f"/kaggle/input/{COMPETITION}/images",
                  "data/images"]
    for c in candidates:
        if c and Path(c).is_dir():
            return Path(c)
    return None


@dataclass
class Catalog:
    """Catálogo alineado por fila: la fila i de `emb` y `pop` corresponde a `articles.iloc[i]`."""
    articles: pd.DataFrame
    emb: np.ndarray            # (n, 512) float32, L2-normalizado; ceros si el artículo no tiene imagen
    avail: np.ndarray          # filas disponibles (vendidas en las últimas semanas y con imagen), ordenadas
    pop: np.ndarray            # (n,) popularidad de la última semana en [0, 1]
    comp: pd.DataFrame         # complementos del 04: columnas a, b (filas), score
    images_dir: Path = None
    row: pd.Series = field(init=False, repr=False)

    def __post_init__(self):
        self.articles = self.articles.reset_index(drop=True)
        self.row = pd.Series(np.arange(len(self.articles)), index=self.articles.article_id)
        self.aid = self.articles.article_id.to_numpy()
        self.pcode = self.articles.product_code.to_numpy()
        self.pgroup = self.articles.product_group_name.to_numpy()

    def image_path(self, article_id):
        if self.images_dir is None:
            return None
        p = self.images_dir / article_id[:3] / f"{article_id}.jpg"
        return p if p.exists() else None

    def load_image(self, article_id):
        from PIL import Image
        p = self.image_path(article_id)
        return Image.open(p).convert("RGB") if p is not None else None

    def availability_table(self):
        return pd.DataFrame({"article_id": self.aid[self.avail], "pop": self.pop[self.avail]})


def load_catalog(window_weeks=4):
    """Carga todo lo necesario para recomendar. Si no hay `available.parquet`, lo calcula como en el 05:
    disponible = vendido en las últimas `window_weeks` semanas de train y con imagen;
    popularidad = ventas de la última semana / máximo."""
    articles = pd.read_parquet(find("articles.parquet"))
    row = pd.Series(np.arange(len(articles)), index=articles.article_id)

    emb = np.zeros((len(articles), 512), np.float32)
    ids = pd.read_parquet(find("emb_ids.parquet")).article_id
    emb[row.loc[ids].to_numpy()] = np.load(find("img_emb.npy"))

    comp = pd.read_parquet(find("complements.parquet"), columns=["anchor_id", "article_id", "score"])
    comp = pd.DataFrame({"a": row.loc[comp.anchor_id].to_numpy(), "b": row.loc[comp.article_id].to_numpy(),
                         "score": comp.score.to_numpy(np.float32)})

    pop = np.zeros(len(articles), np.float32)
    has_img = emb.any(1)
    av_file = find("available.parquet", required=False)
    if av_file is not None:
        av = pd.read_parquet(av_file)
        avail = row.loc[av.article_id].to_numpy()
        pop[avail] = av["pop"].to_numpy()
    else:
        tx = pd.read_parquet(find("train_small.parquet"), columns=["t_dat", "article_id"])
        end = tx.t_dat.max()
        i = row.loc[tx.article_id].to_numpy()
        recent = i[(tx.t_dat > end - pd.Timedelta(weeks=window_weeks)).to_numpy()]
        avail = np.intersect1d(recent, np.flatnonzero(has_img))
        counts = pd.Series(i[(tx.t_dat > end - pd.Timedelta(days=7)).to_numpy()]).value_counts()
        pop[counts.index.to_numpy()] = counts.to_numpy() / counts.max()
    avail = np.sort(avail[has_img[avail]])

    return Catalog(articles, emb, avail, pop, comp, find_images_dir())
