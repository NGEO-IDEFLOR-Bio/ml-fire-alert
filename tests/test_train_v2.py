"""Treino v2, calibracao cruzada e SHAP em dado sintetico (sem rede)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import box

from ml_fire.spatial_cv import folds_com_embargo
from ml_fire.train import NUISANCE_FEATURES
from ml_fire.train_v2 import (
    CLIMA_BASELINE,
    FEATURES_V2,
    avaliar,
    avaliar_holdout_temporal,
    matriz_shap,
    top_global,
    top_por_uc,
    treinar,
)


def _dados(n_por_uc: int = 120) -> tuple[pd.DataFrame, dict[int, object]]:
    rng = np.random.default_rng(11)
    ucs = {
        uc_id: box(uc_id * 100_000, 0, uc_id * 100_000 + 20_000, 20_000) for uc_id in range(1, 7)
    }
    linhas = []
    for uc_id, geom in ucs.items():
        for _ in range(n_por_uc):
            x = rng.uniform(geom.bounds[0], geom.bounds[2])
            y = rng.uniform(0, 20_000)
            risco = 1.4 * np.exp(-rng.exponential(3000) / 3000) + rng.normal(0, 0.3)
            linhas.append(
                {
                    "x_5880": x,
                    "y_5880": y,
                    "uc_id": uc_id,
                    "chuva_30d_mm": rng.gamma(2, 10),
                    "dist_deter_60d_m": rng.exponential(4000),
                    "dist_pastagem_m": rng.exponential(3000),
                    "dist_estradas_m": rng.exponential(2000),
                    "lst_dia_c": rng.normal(30, 3),
                    "vpd_kpa": rng.normal(1.6, 0.3),
                    "dist_borda_uc_m": rng.exponential(6000),
                    "frac_floresta": rng.beta(2, 3),
                    "declividade_deg": abs(rng.normal(2, 1)),
                    "tempo_desde_fogo_anos": rng.integers(0, 40),
                    "nuvem_fracao_30d": rng.beta(2, 4),
                    "y_queima": int(risco > 1.6),
                }
            )
    return pd.DataFrame(linhas), ucs


def test_treino_calibra_cruzado_e_avalia() -> None:
    df, ucs = _dados()
    features = [*FEATURES_V2, *NUISANCE_FEATURES]
    x = df[features].to_numpy()
    y = df["y_queima"].to_numpy()

    folds = folds_com_embargo(df, ucs, n_folds=3, embargo_m=0)
    treino = treinar(x, y, df, folds, params={"num_leaves": 7, "min_child_samples": 20})

    assert treino.prob_calibrada is not None
    assert not np.isnan(treino.prob_calibrada[treino.oof_mask]).any()
    assert treino.prob_calibrada[treino.oof_mask].min() >= 0.0
    assert treino.prob_calibrada[treino.oof_mask].max() <= 1.0
    assert len(treino.folds) == 3

    metricas = avaliar(df, y, treino.prob_calibrada[treino.oof_mask], treino.oof_mask)
    assert 0.0 <= metricas["auc_pr"] <= 1.0
    assert 0.0 <= metricas["recall_at_10pct"] <= 1.0


def test_shap_expurga_nuisance() -> None:
    df, ucs = _dados(60)
    features = [*FEATURES_V2, *NUISANCE_FEATURES]
    x = df[features].to_numpy()
    y = df["y_queima"].to_numpy()
    folds = folds_com_embargo(df, ucs, n_folds=2, embargo_m=0)
    treino = treinar(x, y, df, folds, params={"num_leaves": 7, "min_child_samples": 20})

    matriz, expurgadas, indices = matriz_shap(treino.modelo, x, features, n_amostra=200)
    assert NUISANCE_FEATURES[0] in expurgadas
    assert NUISANCE_FEATURES[0] not in matriz.columns
    assert matriz.shape[1] == len(features) - 1
    assert len(indices) == len(matriz)

    drivers = top_por_uc(matriz, df.iloc[: len(matriz)].reset_index(drop=True), k=3)
    assert {"1", "2"}.issubset(drivers)
    assert all(len(v) == 3 for v in drivers.values())
    assert all("mean_shap" in d for v in drivers.values() for d in v)

    globais = top_global(matriz, k=4)
    assert len(globais) == 4
    assert all(g["mean_abs_shap"] >= 0 for g in globais)


def test_holdout_temporal_v2() -> None:
    df, _ = _dados(100)
    # Atribui duas datas sinteticas
    metade = len(df) // 2
    df["ref_date"] = ["2026-07-31"] * metade + ["2026-08-31"] * (len(df) - metade)

    features = [*FEATURES_V2, *NUISANCE_FEATURES]
    res = avaliar_holdout_temporal(
        df,
        features,
        list(CLIMA_BASELINE),
        params={"num_leaves": 7, "min_child_samples": 10},
    )
    assert res is not None
    assert res["datas_treino"] == ["2026-07-31"]
    assert res["data_teste"] == "2026-08-31"
    assert "auc_pr" in res["modelo_completo"]
    assert "recall_at_10pct" in res["modelo_completo"]
    assert "auc_pr" in res["baseline_so_clima"]
