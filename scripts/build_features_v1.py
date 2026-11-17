"""Gera a grade de 1 km e a tabela de features v1 (vetorial + GEE) com digest SHA-256.

Uso:
    python scripts/build_features_v1.py --ref-date 2026-10-15 --skip-gee
    python scripts/build_features_v1.py --ref-date 2026-10-15
"""

from __future__ import annotations

import argparse
from datetime import date

from ml_fire.catalog import write_parquet_with_digest
from ml_fire.config import get_settings
from ml_fire.features_gee import extract_gee_features, init_ee
from ml_fire.features_vector import compute_vector_distances
from ml_fire.grid import build_grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-date", default=date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--cell-size-m", type=int, default=1000)
    parser.add_argument("--za-buffer-m", type=int, default=10_000)
    parser.add_argument(
        "--skip-gee",
        action="store_true",
        help="Gera apenas grade + features vetoriais (sem chamada ao GEE).",
    )
    args = parser.parse_args(argv)

    ref_date = date.fromisoformat(args.ref_date)
    cfg = get_settings()

    print(f"Construindo grade {args.cell_size_m} m a partir de {cfg.ucs_geojson}...")
    df = build_grid(cfg.ucs_geojson, cell_size_m=args.cell_size_m, za_buffer_m=args.za_buffer_m)
    print(f"Grade gerada: {len(df)} celulas ({df['uc_id'].nunique()} UCs)")

    layers = {
        "dist_contrato_transicao_m": [cfg.contratos_geojson],
        "dist_concessao_m": [cfg.concessoes_geojson],
        "dist_ti_m": [cfg.ti_geojson],
        "dist_quilombo_m": [cfg.quilombos_est_geojson, cfg.quilombos_fed_geojson],
    }
    print("Calculando distancias vetoriais em EPSG:5880...")
    df = compute_vector_distances(df, layers)

    if not args.skip_gee:
        print(f"Inicializando GEE (project={cfg.gee_project_id})...")
        init_ee(cfg.gee_project_id, cfg.gee_credentials_json)
        print(f"Extraindo features GEE para {ref_date.isoformat()}...")
        df = extract_gee_features(df, ref_date=ref_date)

    out_parquet = cfg.artifacts_dir / f"features_v1_{ref_date.isoformat()}.parquet"
    meta_json = cfg.artifacts_dir / f"catalog_v1_{ref_date.isoformat()}.json"
    meta = write_parquet_with_digest(df, out_parquet, ref_date=ref_date, metadata_path=meta_json)
    print(f"Parquet salvo: {out_parquet} ({meta['linhas']} linhas)")
    print(f"SHA-256: {meta['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
