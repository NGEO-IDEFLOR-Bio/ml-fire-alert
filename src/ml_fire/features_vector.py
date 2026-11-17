"""Features vetoriais de pressão antrópica e contexto territorial (EPSG:5880)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

from ml_fire.grid import CRS_METRIC, CRS_WGS84


def _load_metric_geoms(paths: list[Path]) -> list[BaseGeometry]:
    to_metric = Transformer.from_crs(CRS_WGS84, CRS_METRIC, always_xy=True).transform
    geoms: list[BaseGeometry] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for feat in data.get("features", []):
            raw = feat.get("geometry")
            if not raw:
                continue
            g = shp_transform(to_metric, shape(raw))
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty:
                geoms.append(g)
    return geoms


def _nearest_distances_m(points: list[Point], target_geoms: list[BaseGeometry]) -> list[float]:
    if not target_geoms:
        return [float("nan")] * len(points)
    tree = STRtree(target_geoms)
    dists: list[float] = []
    for pt in points:
        idx = tree.nearest(pt)
        dists.append(round(float(pt.distance(target_geoms[int(idx)])), 2))
    return dists


def compute_vector_distances(
    grid_df: pd.DataFrame,
    layers: dict[str, list[Path]],
) -> pd.DataFrame:
    """Calcula distâncias em metros (EPSG:5880) para cada camada em ``layers``."""
    out = grid_df.copy()
    points = [Point(float(x), float(y)) for x, y in zip(out["x_5880"], out["y_5880"], strict=True)]

    for col_name, paths in layers.items():
        geoms = _load_metric_geoms(paths)
        out[col_name] = np.array(_nearest_distances_m(points, geoms), dtype=float)

    return out
