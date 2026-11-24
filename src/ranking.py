"""Recomendación en modo consulta (modo b del notebook 05): similares + complementos con re-ranking.

La consulta llega ya codificada como vector de Fashion-CLIP, así que el chat puede reutilizar el vector de la
imagen subida entre turnos sin volver a codificarla.
"""
import time
from collections import Counter

import numpy as np
import pandas as pd

# Columnas de `articles` sobre las que el chat puede filtrar
FILTER_COLUMNS = ["perceived_colour_master_name", "colour_group_name", "product_group_name",
                  "product_type_name", "index_group_name", "graphical_appearance_name"]
META_COLUMNS = ["article_id", "prod_name", "product_type_name", "product_group_name", "colour_group_name",
                "perceived_colour_master_name", "index_group_name", "section_name", "detail_desc"]


def knn(Q, E, k, chunk=1024):
    """Top-k de E por producto punto para cada fila de Q → (scores, índices), ordenados de mayor a menor."""
    Q = np.atleast_2d(Q)
    k = min(k, len(E))
    S_all, I_all = [], []
    for s in range(0, len(Q), chunk):
        S = Q[s:s + chunk] @ E.T
        I = np.argpartition(-S, k - 1, axis=1)[:, :k]
        Sk = np.take_along_axis(S, I, 1)
        o = np.argsort(-Sk, axis=1)
        S_all.append(np.take_along_axis(Sk, o, 1))
        I_all.append(np.take_along_axis(I, o, 1))
    return np.concatenate(S_all), np.concatenate(I_all)


def minmax(x):
    x = np.asarray(x, np.float32)
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.ones_like(x)


class Recommender:
    # Pesos del modo consulta: la señal principal (sim o cop) pesa 1 y la popularidad solo desempata.
    WQ = {"pop": 0.1}

    def __init__(self, catalog):
        self.cat = catalog
        self.avail = catalog.avail
        self.E = catalog.emb[self.avail]
        is_avail = np.zeros(len(catalog.articles), bool)
        is_avail[self.avail] = True
        self.is_avail = is_avail
        self.comp = catalog.comp[is_avail[catalog.comp.b.to_numpy()]].reset_index(drop=True)
        arts = catalog.articles
        self.meta = arts[[c for c in META_COLUMNS if c in arts.columns]]
        # Vocabulario de cada columna filtrable (solo artículos disponibles), para validar lo que pida el LLM
        self.vocab = {c: sorted(arts[c].iloc[self.avail].dropna().unique().tolist())
                      for c in FILTER_COLUMNS if c in arts.columns}
        self._lower = {c: {v.lower(): v for v in vals} for c, vals in self.vocab.items()}

    # ---------- utilidades ----------
    def clean_filters(self, filters):
        """Normaliza {columna: [valores]} contra el vocabulario del catálogo (sin distinguir mayúsculas).
        Devuelve (filtros válidos, lista de lo descartado)."""
        clean, dropped = {}, []
        for col, vals in (filters or {}).items():
            if col not in self._lower:
                dropped.append(col)
                continue
            vals = [vals] if isinstance(vals, str) else list(vals or [])
            ok = []
            for v in vals:
                m = self._lower[col].get(str(v).strip().lower())
                (ok.append(m) if m else dropped.append(f"{col}={v}"))
            if ok:
                clean[col] = ok
        return clean, dropped

    def _mask(self, filters=None, exclude=None):
        """Máscara booleana sobre los artículos disponibles."""
        arts = self.cat.articles
        ok = np.ones(len(self.avail), bool)
        for col, vals in (filters or {}).items():
            ok &= np.isin(arts[col].to_numpy()[self.avail], vals)
        for col, vals in (exclude or {}).items():
            ok &= ~np.isin(arts[col].to_numpy()[self.avail], vals)
        return ok

    def max_sim(self, q):
        return float((self.E @ q).max())

    def attributes(self, q, k=10, exclude_product_code=None):
        """Atributos por votación de los k artículos disponibles más parecidos (respaldo si no hay VLM)."""
        _, nn = knn(q, self.E, k + 20)
        rows = self.avail[nn[0]]
        if exclude_product_code is not None:
            rows = rows[self.cat.pcode[rows] != exclude_product_code]
        rows = rows[:k]
        arts = self.cat.articles
        out = {}
        for col in ["product_type_name", "product_group_name", "perceived_colour_master_name",
                    "colour_group_name", "index_group_name"]:
            v, n = Counter(arts[col].iloc[rows]).most_common(1)[0]
            out[col] = v
            out[col + "_conf"] = round(n / len(rows), 2)
        return out

    # ---------- recomendación ----------
    def recommend(self, q, q_anchor=None, filters=None, exclude=None, k=6, n_anchors=3,
                  kinds=("similar", "complemento"), filter_kinds=("similar",), comp_exclude_groups=(),
                  style_vec=None, style_w=0.5, anchor_group=None, comp_max_per_group=None):
        """
        q: vector de búsqueda (imagen, texto o suma normalizada de ambos).
        q_anchor: vector para elegir las anclas de los complementos (por defecto q). Con imagen subida conviene
                  pasar el vector de la imagen: las anclas son "lo que es" la prenda.
        filters / exclude: {columna de articles: [valores]}; se aplican a los tipos en `filter_kinds`.
        comp_exclude_groups: grupos de producto que no se proponen como complemento (p. ej. ropa interior).
        style_vec: vector de texto del estilo pedido ("formal"); re-ordena los complementos por afinidad visual a él.
        anchor_group: si se da, las anclas solo pueden ser de ese grupo (p. ej. los jeans de una foto de cuerpo entero).
        comp_max_per_group: máximo de complementos por grupo de producto (outfit variado: zapatos, accesorios...).
        Devuelve (DataFrame, latencias). Columnas: tipo ('similar' | 'complemento'), article_id, score, sim, cop,
        pop, ancla (para complementos: con qué artículo combina) y metadatos.
        """
        lat, t = {}, time.perf_counter()
        cat = self.cat
        sims = self.E @ q
        ok = self._mask(filters, exclude)
        ok_all = np.zeros(len(cat.articles), bool)
        ok_all[self.avail] = ok

        a_sims = sims if q_anchor is None else self.E @ q_anchor
        if anchor_group and (cat.pgroup[self.avail] == anchor_group).sum() >= n_anchors:
            a_sims = np.where(cat.pgroup[self.avail] == anchor_group, a_sims, -np.inf)
        a_order = np.argsort(-a_sims)[:n_anchors]
        anchors = self.avail[a_order]                                   # qué es la prenda (sin filtros)
        out = []

        if "similar" in kinds:
            order = np.argsort(-sims)
            if "similar" in filter_kinds:
                order = order[ok[order]]
            top = order[:200]
            out.append(("similar", pd.DataFrame({"i": self.avail[top], "sim": sims[top]}), "sim"))
        lat["búsqueda"] = time.perf_counter() - t
        t = time.perf_counter()

        if "complemento" in kinds:
            aw = pd.Series(minmax(a_sims[a_order]) * 0.5 + 0.5, index=anchors)   # peso de ancla en [0.5, 1]
            c = self.comp[self.comp.a.isin(anchors) & ~self.comp.b.isin(anchors)]
            c = c.assign(contrib=c.score * aw.loc[c.a].to_numpy()).sort_values("contrib", ascending=False)
            comp_c = (c.groupby("b", as_index=False, sort=False)
                       .agg(cop=("contrib", "sum"), ancla=("a", "first")).rename(columns={"b": "i"}))
            comp_c = comp_c[~np.isin(cat.pgroup[comp_c.i], cat.pgroup[anchors])]   # otro grupo que la prenda
            if len(comp_exclude_groups):
                comp_c = comp_c[~np.isin(cat.pgroup[comp_c.i], list(comp_exclude_groups))]
            if "complemento" in filter_kinds:
                comp_c = comp_c[ok_all[comp_c.i.to_numpy()]]
            out.append(("complemento", comp_c, "cop"))
        lat["complementos"] = time.perf_counter() - t
        t = time.perf_counter()

        res = []
        for tipo, d, main in out:
            if d.empty:
                continue
            d = d.assign(pop=cat.pop[d.i])
            d["score"] = minmax(d[main]) + self.WQ["pop"] * d["pop"]
            if tipo == "complemento" and style_vec is not None:
                d["score"] += style_w * minmax(cat.emb[d.i.to_numpy()] @ style_vec)
            d = d.sort_values("score", ascending=False)
            d = d[~pd.Series(cat.pcode[d.i]).duplicated().to_numpy()]            # un artículo por producto
            if tipo == "complemento" and comp_max_per_group:
                g = pd.Series(cat.pgroup[d.i.to_numpy()], index=d.index)
                varied = d[(g.groupby(g).cumcount() < comp_max_per_group).to_numpy()]
                d = varied if len(varied) >= min(k, len(d)) else pd.concat([varied, d.drop(varied.index)])
            d = d.head(k)
            res.append(d.assign(tipo=tipo))
        cols = ["tipo", "article_id", "score", "sim", "cop", "pop", "ancla"] + list(self.meta.columns[1:])
        if not res:
            lat["ranking"] = time.perf_counter() - t
            return pd.DataFrame(columns=cols), lat
        res = pd.concat(res, ignore_index=True)
        res["article_id"] = cat.aid[res.i]
        if "ancla" in res:
            res["ancla"] = res.ancla.map(lambda a: cat.aid[int(a)] if pd.notna(a) else None)
        res = res.merge(self.meta, on="article_id", how="left")
        lat["ranking"] = time.perf_counter() - t
        return res.reindex(columns=cols), lat
