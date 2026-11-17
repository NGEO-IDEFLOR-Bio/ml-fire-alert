"""Treina e exporta o modelo v1 de risco de ignição + model_card.json (T7).

Uso:
    python scripts/train_v1.py --features artifacts/features_v1_2026-09-30.parquet \
        --ref-date 2026-09-30
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from ml_fire.catalog import sha256_file
from ml_fire.config import get_settings
from ml_fire.features_gee import init_ee
from ml_fire.labels import extract_mcd64a1_labels
from ml_fire.train import train_and_export


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Parquet de features v1")
    parser.add_argument("--ref-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--out-dir", default="", help="Diretório de saída do modelo/card")
    args = parser.parse_args(argv)

    cfg = get_settings()
    feat_path = Path(args.features)
    ref_date = date.fromisoformat(args.ref_date)
    out_dir = Path(args.out_dir) if args.out_dir else cfg.artifacts_dir

    df = pd.read_parquet(feat_path)
    if "y_queima" not in df.columns:
        print(f"Extraindo rotulos MCD64A1 ({ref_date.isoformat()} +{args.horizon_days}d)...")
        init_ee(cfg.gee_project_id, cfg.gee_credentials_json)
        df = extract_mcd64a1_labels(df, ref_date=ref_date, horizon_days=args.horizon_days)

    digest = sha256_file(feat_path)
    card = train_and_export(df, dataset_sha256=digest, ref_date=ref_date, out_dir=out_dir)
    print("Model card salvo em:", out_dir / "model_card.json")
    print("Block CV UC:", card["metricas"]["block_cv_espacial_por_uc"])
    print("Random CV:", card["metricas"]["random_cv_sem_bloqueio_espacial"])
    print("Baseline clima:", card["metricas"]["baseline_so_clima_block_cv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
