"""Extração de rótulos de área queimada (MODIS/061/MCD64A1) para a Frente A."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

import ee
import pandas as pd


def _sample_label_chunk(
    image: ee.Image, chunk: list[dict[str, Any]], scale_m: int
) -> list[dict[str, Any]]:
    fc = ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point([r["lon"], r["lat"]]),
                {"cell_id": r["cell_id"]},
            )
            for r in chunk
        ]
    )
    sampled = image.sampleRegions(
        collection=fc,
        properties=["cell_id"],
        scale=scale_m,
        geometries=False,
    )
    info = sampled.getInfo()
    return [feat["properties"] for feat in info.get("features", [])]


def extract_mcd64a1_labels(
    grid_df: pd.DataFrame,
    ref_date: date,
    horizon_days: int = 30,
    batch_size: int = 2000,
    scale_m: int = 500,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Extrai rótulo binário ``y_queima`` (1 se ``BurnDate > 0`` na janela futura)."""
    start_str = ref_date.isoformat()
    end_str = (ref_date + timedelta(days=horizon_days)).isoformat()

    burned = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(start_str, end_str)
        .select("BurnDate")
        .max()
        .unmask(0)
    )
    y_img = burned.gt(0).rename("y_queima")
    img = ee.Image.cat([y_img, burned.rename("burn_date_doy")])

    rows = grid_df[["cell_id", "lon", "lat"]].to_dict("records")
    chunks = [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]

    records: list[dict[str, Any]] = []
    if max_workers <= 1 or len(chunks) <= 1:
        for chunk in chunks:
            records.extend(_sample_label_chunk(img, chunk, scale_m))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_sample_label_chunk, img, chunk, scale_m) for chunk in chunks]
            for fut in as_completed(futs):
                records.extend(fut.result())

    lbl_df = pd.DataFrame(records)
    if lbl_df.empty:
        out = grid_df.copy()
        out["y_queima"] = 0
        out["burn_date_doy"] = 0
        return out
    out = grid_df.merge(lbl_df, on="cell_id", how="left")
    out["y_queima"] = out["y_queima"].fillna(0).astype(int)
    out["burn_date_doy"] = out["burn_date_doy"].fillna(0).astype(int)
    return out
