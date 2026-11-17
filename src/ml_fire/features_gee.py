"""Extração de features ambientais e de detectabilidade via Google Earth Engine."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import ee
import pandas as pd


def init_ee(project_id: str, credentials_json: Path | None = None) -> None:
    """Inicializa sessão GEE com service account ou credencial padrão."""
    if not project_id:
        raise ValueError("GEE_PROJECT_ID nao configurado")
    if credentials_json is not None:
        if not credentials_json.exists():
            raise FileNotFoundError(f"Credencial GEE nao encontrada: {credentials_json}")
        creds = ee.ServiceAccountCredentials(None, str(credentials_json))
        ee.Initialize(credentials=creds, project=project_id)
    else:
        ee.Initialize(project=project_id)


def build_feature_image(ref_date: date) -> ee.Image:
    """Monta pilha multi-banda para a data de referência ``ref_date`` (sem vazamento futuro)."""
    end_str = ref_date.isoformat()
    start_7d = (ref_date - timedelta(days=7)).isoformat()
    start_16d = (ref_date - timedelta(days=16)).isoformat()
    start_30d = (ref_date - timedelta(days=30)).isoformat()
    start_12m = (ref_date - timedelta(days=365)).isoformat()

    # 1. LST MODIS (MOD11A1) -> °C
    mod11 = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate(start_16d, end_str)
        .select(["LST_Day_1km", "LST_Night_1km"])
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .rename(["lst_dia_c", "lst_noite_c"])
    )

    # 2. NDVI / EVI MODIS (MOD13A2) -> [-1, 1]
    mod13 = (
        ee.ImageCollection("MODIS/061/MOD13A2")
        .filterDate(start_30d, end_str)
        .select(["NDVI", "EVI"])
        .mean()
        .multiply(0.0001)
        .rename(["ndvi", "evi"])
    )

    # 3. Chuva CHIRPS (7d e 30d)
    chirps_7d = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start_7d, end_str)
        .select("precipitation")
        .sum()
        .rename("chuva_7d_mm")
    )
    chirps_30d = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start_30d, end_str)
        .select("precipitation")
        .sum()
        .rename("chuva_30d_mm")
    )

    # 4. Clima / seca atmosférica ERA5-Land (7d)
    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start_7d, end_str)
        .select(
            [
                "temperature_2m",
                "dewpoint_temperature_2m",
                "u_component_of_wind_10m",
                "v_component_of_wind_10m",
            ]
        )
        .mean()
    )
    temp_c = era5.select("temperature_2m").subtract(273.15).rename("temp_2m_c")
    dew_c = era5.select("dewpoint_temperature_2m").subtract(273.15).rename("dewpoint_2m_c")
    vento_ms = (
        era5.select("u_component_of_wind_10m")
        .pow(2)
        .add(era5.select("v_component_of_wind_10m").pow(2))
        .sqrt()
        .rename("vento_ms")
    )
    # VPD aproximado (kPa) pela equação de Tetens
    es = temp_c.multiply(17.27).divide(temp_c.add(237.3)).exp().multiply(0.6108)
    ea = dew_c.multiply(17.27).divide(dew_c.add(237.3)).exp().multiply(0.6108)
    vpd = es.subtract(ea).max(0).rename("vpd_kpa")

    # 5. Combustível estático (ESA WorldCover v200 10 m -> frações na escala de amostragem)
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    combustivel_classe = wc.rename("combustivel_classe")
    frac_floresta = wc.eq(10).rename("frac_floresta")
    frac_pastagem = wc.eq(30).Or(wc.eq(20)).rename("frac_pastagem_arbustiva")

    # 6. Histórico de queima 12 meses (MCD64A1)
    mcd64 = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(start_12m, end_str)
        .select("BurnDate")
        .map(lambda img: img.gt(0))
        .sum()
        .unmask(0)
        .rename("hist_queima_12m")
    )

    # 7. Detectabilidade / nuvem (nuisance obrigatória — PLANO §10)
    mod09_nuvem = (
        ee.ImageCollection("MODIS/061/MOD09GA")
        .filterDate(start_30d, end_str)
        .select("state_1km")
        .map(lambda img: img.bitwiseAnd(3).gt(0))
        .mean()
        .rename("nuvem_fracao_30d")
    )

    # 8. Relevo (Copernicus DEM GLO-30)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic()
    elevacao = dem.rename("elevacao_m")
    declividade = ee.Terrain.slope(dem).rename("declividade_deg")

    return ee.Image.cat(
        [
            mod11,
            mod13,
            chirps_7d,
            chirps_30d,
            temp_c,
            dew_c,
            vento_ms,
            vpd,
            combustivel_classe,
            frac_floresta,
            frac_pastagem,
            mcd64,
            mod09_nuvem,
            elevacao,
            declividade,
        ]
    )


def extract_gee_features(
    grid_df: pd.DataFrame,
    ref_date: date,
    batch_size: int = 400,
    scale_m: int = 1000,
) -> pd.DataFrame:
    """Amostra a pilha GEE nos centroides da grade em lotes para evitar limite de payload."""
    image = build_feature_image(ref_date)
    records: list[dict[str, Any]] = []

    rows = grid_df[["cell_id", "lon", "lat"]].to_dict("records")
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
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
        for feat in info.get("features", []):
            records.append(feat["properties"])

    gee_df = pd.DataFrame(records)
    if gee_df.empty:
        return grid_df.copy()
    return grid_df.merge(gee_df, on="cell_id", how="left")
