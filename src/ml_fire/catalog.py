"""Versionamento da feature table (Parquet + SHA-256) e catálogo de features."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

FEATURE_CATALOG_V1: list[dict[str, str]] = [
    {
        "coluna": "cell_id",
        "grupo": "identificacao",
        "fonte": "Grade 1 km (EPSG:5880)",
        "resolucao": "1000 m",
        "janela": "estatica",
        "tipo": "chave",
    },
    {
        "coluna": "dist_borda_uc_m",
        "grupo": "territorial",
        "fonte": "IDEFLOR-Bio (29 UCs DGMUC)",
        "resolucao": "vetorial (EPSG:5880)",
        "janela": "estatica",
        "tipo": "feature",
    },
    {
        "coluna": "dist_contrato_transicao_m",
        "grupo": "pressao_antropica",
        "fonte": "IDEFLOR-Bio (contratos_de_transicao.geojson)",
        "resolucao": "vetorial (EPSG:5880)",
        "janela": "estatica",
        "tipo": "feature",
    },
    {
        "coluna": "dist_concessao_m",
        "grupo": "pressao_antropica",
        "fonte": "IDEFLOR-Bio (concessoes_florestais_ideflor.geojson)",
        "resolucao": "vetorial (EPSG:5880)",
        "janela": "estatica",
        "tipo": "feature",
    },
    {
        "coluna": "dist_ti_m",
        "grupo": "contexto_territorial",
        "fonte": "FUNAI / visualizador (terras_indigenas.geojson)",
        "resolucao": "vetorial (EPSG:5880)",
        "janela": "estatica",
        "tipo": "feature",
    },
    {
        "coluna": "dist_quilombo_m",
        "grupo": "contexto_territorial",
        "fonte": "INCRA/ITERPA (quilombos_estaduais + federais)",
        "resolucao": "vetorial (EPSG:5880)",
        "janela": "estatica",
        "tipo": "feature",
    },
    {
        "coluna": "lst_dia_c",
        "grupo": "termica",
        "fonte": "MODIS/061/MOD11A1",
        "resolucao": "1000 m",
        "janela": "16 dias anteriores",
        "tipo": "feature",
    },
    {
        "coluna": "lst_noite_c",
        "grupo": "termica",
        "fonte": "MODIS/061/MOD11A1",
        "resolucao": "1000 m",
        "janela": "16 dias anteriores",
        "tipo": "feature",
    },
    {
        "coluna": "ndvi",
        "grupo": "vegetacao",
        "fonte": "MODIS/061/MOD13A2",
        "resolucao": "1000 m",
        "janela": "30 dias anteriores",
        "tipo": "feature",
    },
    {
        "coluna": "evi",
        "grupo": "vegetacao",
        "fonte": "MODIS/061/MOD13A2",
        "resolucao": "1000 m",
        "janela": "30 dias anteriores",
        "tipo": "feature",
    },
    {
        "coluna": "chuva_7d_mm",
        "grupo": "precipitacao",
        "fonte": "UCSB-CHG/CHIRPS/DAILY",
        "resolucao": "~5500 m",
        "janela": "7 dias anteriores",
        "tipo": "feature",
    },
    {
        "coluna": "chuva_30d_mm",
        "grupo": "precipitacao",
        "fonte": "UCSB-CHG/CHIRPS/DAILY",
        "resolucao": "~5500 m",
        "janela": "30 dias anteriores",
        "tipo": "feature",
    },
    {
        "coluna": "temp_2m_c",
        "grupo": "clima_seca",
        "fonte": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "resolucao": "~11132 m",
        "janela": "7 dias anteriores",
        "tipo": "feature",
    },
    {
        "coluna": "dewpoint_2m_c",
        "grupo": "clima_seca",
        "fonte": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "janela": "7 dias anteriores",
        "resolucao": "~11132 m",
        "tipo": "feature",
    },
    {
        "coluna": "vento_ms",
        "grupo": "clima_seca",
        "fonte": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "resolucao": "~11132 m",
        "janela": "7 dias anteriores",
        "tipo": "feature",
    },
    {
        "coluna": "vpd_kpa",
        "grupo": "clima_seca",
        "fonte": "ECMWF/ERA5_LAND/DAILY_AGGR (Tetens)",
        "resolucao": "~11132 m",
        "janela": "7 dias anteriores",
        "tipo": "feature",
    },
    {
        "coluna": "combustivel_classe",
        "grupo": "combustivel",
        "fonte": "ESA/WorldCover/v200",
        "resolucao": "10 m (amostrado 1000 m)",
        "janela": "estatica (2021)",
        "tipo": "feature",
    },
    {
        "coluna": "frac_floresta",
        "grupo": "combustivel",
        "fonte": "ESA/WorldCover/v200",
        "resolucao": "10 m (agregado 1000 m)",
        "janela": "estatica (2021)",
        "tipo": "feature",
    },
    {
        "coluna": "frac_pastagem_arbustiva",
        "grupo": "combustivel",
        "fonte": "ESA/WorldCover/v200",
        "resolucao": "10 m (agregado 1000 m)",
        "janela": "estatica (2021)",
        "tipo": "feature",
    },
    {
        "coluna": "hist_queima_12m",
        "grupo": "historico",
        "fonte": "MODIS/061/MCD64A1",
        "resolucao": "500 m",
        "janela": "12 meses anteriores",
        "tipo": "feature",
    },
    {
        "coluna": "nuvem_fracao_30d",
        "grupo": "detectabilidade",
        "fonte": "MODIS/061/MOD09GA (state_1km)",
        "resolucao": "1000 m",
        "janela": "30 dias anteriores",
        "tipo": "nuisance",
    },
    {
        "coluna": "elevacao_m",
        "grupo": "relevo",
        "fonte": "COPERNICUS/DEM/GLO30_2024_1",
        "resolucao": "30 m",
        "janela": "estatica",
        "tipo": "feature",
    },
    {
        "coluna": "declividade_deg",
        "grupo": "relevo",
        "fonte": "COPERNICUS/DEM/GLO30_2024_1",
        "resolucao": "30 m",
        "janela": "estatica",
        "tipo": "feature",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_parquet_with_digest(
    df: pd.DataFrame,
    out_path: Path,
    ref_date: date,
    metadata_path: Path,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path, index=False)
    digest = sha256_file(out_path)

    meta = {
        "versao": "v1",
        "ref_date": ref_date.isoformat(),
        "arquivo": out_path.name,
        "sha256": digest,
        "linhas": int(len(df)),
        "colunas": list(df.columns),
        "catalogo": FEATURE_CATALOG_V1,
    }
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
