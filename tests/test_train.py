"""Testes unitários de treino, block CV por UC, baseline clima e SHAP (T7)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ml_fire.train import DOMAIN_FEATURES, NUISANCE_FEATURES, train_and_export


def _synthetic_dataset(n_ucs: int = 6, rows_per_uc: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for uc in range(1, n_ucs + 1):
        for i in range(rows_per_uc):
            vpd = float(rng.uniform(0.5, 3.0))
            dist_c = float(rng.uniform(100, 15000))
            logit = 1.2 * vpd - 0.0002 * dist_c - 1.5
            prob = 1.0 / (1.0 + np.exp(-logit))
            y = int(rng.random() < prob)
            rec = {
                "cell_id": f"{uc}_{i}",
                "uc_id": uc,
                "uc_nome": f"UC-{uc}",
                "y_queima": y,
            }
            for col in [*DOMAIN_FEATURES, *NUISANCE_FEATURES]:
                rec[col] = float(rng.uniform(0, 1))
            rec["vpd_kpa"] = vpd
            rec["dist_contrato_transicao_m"] = dist_c
            rows.append(rec)
    return pd.DataFrame(rows)


def test_train_and_export_block_cv_e_shap(tmp_path: Path) -> None:
    df = _synthetic_dataset()
    card = train_and_export(
        df,
        dataset_sha256="deadbeef",
        ref_date=date(2026, 8, 1),
        out_dir=tmp_path,
        n_splits=3,
    )
    assert (tmp_path / "model_v1.joblib").exists()
    assert (tmp_path / "model_card.json").exists()
    assert "block_cv_espacial_por_uc" in card["metricas"]
    assert "random_cv_sem_bloqueio_espacial" in card["metricas"]
    assert "baseline_so_clima_block_cv" in card["metricas"]
    assert card["nuisance_features_removidas_do_shap"] == ["nuvem_fracao_30d"]
    global_feats = {d["feature"] for d in card["shap"]["global_top"]}
    assert "nuvem_fracao_30d" not in global_feats
    assert len(card["shap"]["local_por_uc"]) == 6
