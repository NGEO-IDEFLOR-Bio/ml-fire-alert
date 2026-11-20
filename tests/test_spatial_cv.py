"""Spatial Block CV por UC com embargo (v2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import box

from ml_fire.spatial_cv import (
    atribuir_folds_por_uc,
    folds_com_embargo,
    recall_at_fracao,
    recall_por_uc,
)


def _ucs() -> dict[int, object]:
    return {
        1: box(0, 0, 2000, 2000),
        2: box(2000, 0, 4000, 2000),
        3: box(20_000, 0, 22_000, 2000),
        4: box(40_000, 0, 42_000, 2000),
    }


def _df() -> pd.DataFrame:
    linhas = []
    for uc_id, base in [(1, 0), (2, 2000), (3, 20_000), (4, 40_000)]:
        for dx in range(-2000, 4000, 500):
            for dy in range(0, 3000, 500):
                linhas.append(
                    {
                        "x_5880": base + dx,
                        "y_5880": dy,
                        "uc_id": uc_id,
                        "y_queima": int((dx == 0) and (dy == 0)),
                    }
                )
    return pd.DataFrame(linhas)


def test_folds_nao_compartilham_uc() -> None:
    df = _df()
    for fold in folds_com_embargo(df, _ucs(), n_folds=2, embargo_m=0):
        assert set(df.loc[fold.teste, "uc_id"]).isdisjoint(set(df.loc[fold.treino, "uc_id"]))


def test_embargo_remove_vizinhanca() -> None:
    df = _df()
    sem = {f.indice: f for f in folds_com_embargo(df, _ucs(), n_folds=2, embargo_m=0)}
    com = folds_com_embargo(df, _ucs(), n_folds=2, embargo_m=5000)
    for fold in com:
        assert len(fold.treino) < len(sem[fold.indice].treino)
        # UCs 1 e 2 sao vizinhas (borda comum): com embargo, uma nao treina a outra.
        for uc in set(df.loc[fold.teste, "uc_id"]):
            geom = _ucs()[uc]
            treino = df.loc[fold.treino]
            dentro = (
                (treino["x_5880"] >= geom.bounds[0] - 5000)
                & (treino["x_5880"] <= geom.bounds[2] + 5000)
                & (treino["y_5880"] >= geom.bounds[1] - 5000)
                & (treino["y_5880"] <= geom.bounds[3] + 5000)
            )
            assert not dentro.any()


def test_atribuicao_cobre_todas_as_ucs() -> None:
    atribuicao = atribuir_folds_por_uc(_df(), n_folds=2)
    assert set(atribuicao) == {1, 2, 3, 4}
    assert set(atribuicao.values()) == {0, 1}


def test_recall_por_uc_ignora_uc_sem_positivo() -> None:
    df = pd.DataFrame({"uc_id": [1, 1, 2, 2]})
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.9, 0.1])
    res = recall_por_uc(df, y, scores, fracao=0.5)
    assert res["ucs_avaliadas"] == 1
    assert res["media"] == 1.0
    assert recall_at_fracao(y, scores, 0.5) == 1.0
