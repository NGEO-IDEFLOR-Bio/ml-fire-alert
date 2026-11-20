"""Grade de 1 km vetorizada com as colunas esperadas pelo pipeline v1/v2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:5880"


def build_grid_v2(
    ucs_path: Path,
    cell_size_m: int = 1000,
    za_buffer_m: int = 10_000,
) -> pd.DataFrame:
    """Grade de ``cell_size_m`` sobre UC + ZA, vetorizada (288k celulas em ~5 min).

    ``dist_borda_uc_m`` segue a semantica da v1: 0 dentro da UC e distancia ao
    poligono da UC na ZA. O ``uc_id`` da ZA e o da UC mais proxima, de modo que o
    bloco de validacao nunca fica sem grupo.
    """
    to_metric = Transformer.from_crs(CRS_WGS84, CRS_METRIC, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(CRS_METRIC, CRS_WGS84, always_xy=True).transform

    dados = json.loads(ucs_path.read_text(encoding="utf-8"))
    ucs = []
    for feat in dados.get("features", []):
        geom_raw = feat.get("geometry")
        if not geom_raw:
            continue
        props = feat.get("properties", {})
        geom_m = shp_transform(to_metric, shapely.geometry.shape(geom_raw))
        if not geom_m.is_valid:
            geom_m = shapely.make_valid(geom_m)
        ucs.append(
            {
                "uc_id": int(props["id"]),
                "uc_nome": str(props.get("nome_") or props.get("name") or f"UC-{props['id']}"),
                "gerencia_id": str(props.get("gerencia") or ""),
                "geom": geom_m,
            }
        )
    ucs.sort(key=lambda u: u["uc_id"])
    if not ucs:
        raise ValueError(f"Nenhuma UC valida em {ucs_path}")

    area = unary_union([u["geom"] for u in ucs]).buffer(za_buffer_m)
    minx, miny, maxx, maxy = area.bounds
    passo = float(cell_size_m)
    xs = np.arange(np.floor(minx / passo) * passo + passo / 2, maxx, passo)
    ys = np.arange(np.floor(miny / passo) * passo + passo / 2, maxy, passo)
    grade_x, grade_y = np.meshgrid(xs, ys)
    px = grade_x.ravel()
    py = grade_y.ravel()

    dentro_area = shapely.contains_xy(area, px, py)
    px, py = px[dentro_area], py[dentro_area]
    pontos = shapely.points(px, py)

    uc_ids = np.full(len(px), -1, dtype=np.int16)
    dist_uc = np.full(len(px), np.inf, dtype=np.float64)
    dentro_uc = np.zeros(len(px), dtype=bool)
    for uc in ucs:
        x1, y1, x2, y2 = uc["geom"].buffer(za_buffer_m).bounds
        caixa = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
        if not caixa.any():
            continue
        idx = np.flatnonzero(caixa)
        d = shapely.distance(pontos[idx], uc["geom"])
        melhor = (d < dist_uc[idx]) | ((d == dist_uc[idx]) & (uc_ids[idx] == -1))
        alvo = idx[melhor]
        dist_uc[alvo] = d[melhor]
        uc_ids[alvo] = uc["uc_id"]

    if (uc_ids == -1).any():
        raise AssertionError("celula sem grupo de UC apos a poda por bbox")

    for uc in ucs:
        idx = np.flatnonzero(uc_ids == uc["uc_id"])
        if len(idx) == 0:
            continue
        interno = shapely.contains_xy(uc["geom"], px[idx], py[idx])
        dentro_uc[idx[interno]] = True
        dist_uc[idx[interno]] = 0.0

    lon, lat = to_wgs84(px, py)
    nomes = {u["uc_id"]: u["uc_nome"] for u in ucs}
    gerencia = {u["uc_id"]: u["gerencia_id"] for u in ucs}

    grade = pd.DataFrame(
        {
            "cell_id": [f"{int(x)}_{int(y)}" for x, y in zip(px, py, strict=True)],
            "uc_id": uc_ids.astype(np.int16),
            "dentro_uc": dentro_uc,
            "dentro_za": ~dentro_uc,
            "dist_borda_uc_m": dist_uc.astype(np.float32),
            "x_5880": px.astype(np.int64),
            "y_5880": py.astype(np.int64),
            "lon": np.round(lon, 6),
            "lat": np.round(lat, 6),
        }
    )
    grade["uc_nome"] = grade["uc_id"].map(nomes)
    grade["gerencia_id"] = grade["uc_id"].map(gerencia)
    return grade.sort_values(["y_5880", "x_5880"]).reset_index(drop=True)
