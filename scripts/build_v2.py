"""Constrói a grade e o painel de features da v2 (DETER, pastagem, estradas, fogo).

Uso:
    python scripts/build_v2.py --dates 2026-05-31 2026-06-30 2026-07-31 2026-08-31
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from ml_fire.catalog import write_parquet_with_digest
from ml_fire.config import get_settings
from ml_fire.features_v2 import build_panel_v2
from ml_fire.grid_v2 import build_grid_v2

PADRAO_DATAS = ["2026-05-31", "2026-06-30", "2026-07-31", "2026-08-31"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", default=PADRAO_DATAS)
    parser.add_argument("--base-grid", default="", help="Parquet com grade ja construida")
    args = parser.parse_args(argv)

    cfg = get_settings()
    datas = [date.fromisoformat(d) for d in args.dates]

    grid_path = (
        Path(args.base_grid) if args.base_grid else cfg.artifacts_dir / "base_grid_v2.parquet"
    )
    if grid_path.exists():
        print(f"Reaproveitando grade: {grid_path}")
        base = pd.read_parquet(grid_path)
    else:
        print("Construindo grade de 1 km (29 UCs + ZA)...")
        base = build_grid_v2(cfg.ucs_geojson)
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        base.to_parquet(grid_path, index=False)
        print(f"Grade: {len(base)} celulas, {base['uc_id'].nunique()} UCs -> {grid_path}")

    panel = build_panel_v2(base, datas, cfg)
    out_pq = cfg.artifacts_dir / "panel_v2.parquet"
    meta = write_parquet_with_digest(
        panel,
        out_pq,
        ref_date=max(datas),
        metadata_path=cfg.artifacts_dir / "panel_v2_catalog.json",
        versao="v2",
    )
    print(
        f"Painel v2: {meta['linhas']} linhas, "
        f"{int(panel['y_queima'].sum())} positivos, SHA-256={meta['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
