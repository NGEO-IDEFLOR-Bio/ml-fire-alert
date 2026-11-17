"""Constrói painel espaço-temporal multi-data reaproveitando a grade estática 1 km.

Uso:
    python scripts/build_panel.py --dates 2025-08-31 2026-07-31 2026-08-31
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from ml_fire.catalog import write_parquet_with_digest
from ml_fire.config import get_settings
from ml_fire.features_gee import extract_gee_features, init_ee
from ml_fire.features_vector import compute_vector_distances
from ml_fire.grid import build_grid
from ml_fire.labels import extract_mcd64a1_labels

STATIC_COLS = [
    "cell_id",
    "uc_id",
    "uc_nome",
    "gerencia_id",
    "dentro_uc",
    "dentro_za",
    "dist_borda_uc_m",
    "x_5880",
    "y_5880",
    "lon",
    "lat",
    "dist_contrato_transicao_m",
    "dist_concessao_m",
    "dist_ti_m",
    "dist_quilombo_m",
]


def _load_or_build_base_grid(base_path: Path) -> pd.DataFrame:
    if base_path.exists():
        print(f"Reaproveitando grade estatica: {base_path}")
        return pd.read_parquet(base_path, columns=STATIC_COLS)

    cfg = get_settings()
    df = build_grid(cfg.ucs_geojson)
    layers = {
        "dist_contrato_transicao_m": [cfg.contratos_geojson],
        "dist_concessao_m": [cfg.concessoes_geojson],
        "dist_ti_m": [cfg.ti_geojson],
        "dist_quilombo_m": [cfg.quilombos_est_geojson, cfg.quilombos_fed_geojson],
    }
    df = compute_vector_distances(df, layers)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df[STATIC_COLS].to_parquet(base_path, index=False)
    return df[STATIC_COLS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", required=True, help="Lista de datas YYYY-MM-DD")
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--base-grid", default="", help="Parquet opcional com grade estática")
    args = parser.parse_args(argv)

    cfg = get_settings()
    base_path = (
        Path(args.base_grid) if args.base_grid else cfg.artifacts_dir / "base_grid_v1.parquet"
    )
    base_df = _load_or_build_base_grid(base_path)

    init_ee(cfg.gee_project_id, cfg.gee_credentials_json)

    frames: list[pd.DataFrame] = []
    for raw_date in args.dates:
        d = date.fromisoformat(raw_date)
        print(f"Extraindo features + rotulos para {d.isoformat()}...")
        df_d = extract_gee_features(base_df, ref_date=d)
        df_d = extract_mcd64a1_labels(df_d, ref_date=d, horizon_days=args.horizon_days)
        df_d["ref_date"] = d.isoformat()
        print(f"  {d.isoformat()}: {len(df_d)} linhas, {int(df_d['y_queima'].sum())} positivos")
        frames.append(df_d)

    panel = pd.concat(frames, ignore_index=True)
    last_d = date.fromisoformat(sorted(args.dates)[-1])
    out_pq = cfg.artifacts_dir / "panel_v1.parquet"
    meta_json = cfg.artifacts_dir / "panel_v1_catalog.json"
    meta = write_parquet_with_digest(panel, out_pq, ref_date=last_d, metadata_path=meta_json)
    print(f"Painel salvo: {out_pq} ({meta['linhas']} linhas, SHA-256={meta['sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
