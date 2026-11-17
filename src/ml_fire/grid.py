"""Grade de análise de 1 km recortada nas UCs + Zona de Amortecimento (EPSG:5880)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:5880"


def _load_geojson_features(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("features", [])


def build_grid(
    ucs_path: Path,
    cell_size_m: int = 1000,
    za_buffer_m: int = 10_000,
) -> pd.DataFrame:
    """Gera grade regular de ``cell_size_m`` m sobre UCs + buffer ZA de ``za_buffer_m`` m.

    Cada célula recebe a UC mais próxima/contenedora, indicador ``dentro_uc``,
    ``dentro_za`` e ``dist_borda_uc_m`` (0 dentro da UC, >0 na ZA).
    """
    to_metric = Transformer.from_crs(CRS_WGS84, CRS_METRIC, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(CRS_METRIC, CRS_WGS84, always_xy=True).transform

    features = _load_geojson_features(ucs_path)
    ucs: list[dict[str, Any]] = []
    for feat in features:
        geom_raw = feat.get("geometry")
        if not geom_raw:
            continue
        props = feat.get("properties", {})
        geom_m = shp_transform(to_metric, shape(geom_raw))
        if not geom_m.is_valid:
            geom_m = geom_m.buffer(0)
        ucs.append(
            {
                "uc_id": int(props["id"]),
                "uc_nome": str(props.get("nome_") or props.get("name") or f"UC-{props['id']}"),
                "gerencia_id": str(props.get("gerencia") or props.get("GERENCIA") or ""),
                "geom": geom_m,
                "za_geom": geom_m.buffer(za_buffer_m),
            }
        )

    if not ucs:
        raise ValueError(f"Nenhuma UC valida encontrada em {ucs_path}")

    union_za = unary_union([u["za_geom"] for u in ucs])
    minx, miny, maxx, maxy = union_za.bounds

    xs = np.arange(np.floor(minx / cell_size_m) * cell_size_m + cell_size_m / 2, maxx, cell_size_m)
    ys = np.arange(np.floor(miny / cell_size_m) * cell_size_m + cell_size_m / 2, maxy, cell_size_m)

    rows: list[dict[str, Any]] = []
    for x in xs:
        for y in ys:
            pt = Point(float(x), float(y))
            if not union_za.contains(pt):
                continue

            # Associa à UC que contém o ponto ou à mais próxima dentro da ZA
            matched = None
            min_dist = float("inf")
            for u in ucs:
                if u["geom"].contains(pt):
                    matched = u
                    min_dist = 0.0
                    break
                if u["za_geom"].contains(pt):
                    d = u["geom"].distance(pt)
                    if d < min_dist:
                        min_dist = d
                        matched = u

            if matched is None:
                continue

            lon, lat = to_wgs84(float(x), float(y))
            rows.append(
                {
                    "cell_id": f"{int(x)}_{int(y)}",
                    "uc_id": matched["uc_id"],
                    "uc_nome": matched["uc_nome"],
                    "gerencia_id": matched["gerencia_id"],
                    "dentro_uc": min_dist == 0.0,
                    "dentro_za": min_dist > 0.0,
                    "dist_borda_uc_m": round(float(min_dist), 2),
                    "x_5880": float(x),
                    "y_5880": float(y),
                    "lon": round(float(lon), 6),
                    "lat": round(float(lat), 6),
                }
            )

    return pd.DataFrame(rows)
