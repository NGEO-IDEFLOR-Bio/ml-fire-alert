"""Spatial Block CV por UC com embargo de fronteira (v2)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform


@dataclass(frozen=True)
class Fold:
    indice: int
    ucs_teste: tuple[int, ...]
    treino: np.ndarray
    teste: np.ndarray


def carregar_geometrias_uc(caminho: Path) -> dict[int, shapely.geometry.base.BaseGeometry]:
    """Geometrias das UCs em EPSG:5880, indexadas por ``uc_id``."""
    to_metric = Transformer.from_crs("EPSG:4326", "EPSG:5880", always_xy=True).transform
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    geoms: dict[int, shapely.geometry.base.BaseGeometry] = {}
    for feat in dados.get("features", []):
        if not feat.get("geometry"):
            continue
        geom = shp_transform(to_metric, shape(feat["geometry"]))
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
        geoms[int(feat["properties"]["id"])] = geom
    return geoms


def atribuir_folds_por_uc(df: pd.DataFrame, *, n_folds: int = 5) -> dict[int, int]:
    """Distribui UCs inteiras em folds, equilibrando positivos (nunca células)."""
    por_uc = pd.DataFrame(
        {
            "positivos": df.groupby("uc_id")["y_queima"].sum().astype(int),
            "celulas": df.groupby("uc_id")["y_queima"].size().astype(int),
        }
    ).sort_values(["positivos", "celulas"], ascending=False)

    carga = {f: [0, 0] for f in range(n_folds)}
    atribuicao: dict[int, int] = {}
    for uc_id, linha in por_uc.iterrows():
        alvo = min(carga, key=lambda f: (carga[f][0], carga[f][1], f))
        atribuicao[int(uc_id)] = alvo
        carga[alvo][0] += int(linha["positivos"])
        carga[alvo][1] += int(linha["celulas"])
    return atribuicao


def folds_com_embargo(
    df: pd.DataFrame,
    geoms: dict[int, shapely.geometry.base.BaseGeometry],
    *,
    n_folds: int = 5,
    embargo_m: int = 5000,
    atribuicao: dict[int, int] | None = None,
) -> list[Fold]:
    """Folds de teste por UC e remoção da vizinhança do bloco no treino."""
    atribuicao = atribuicao or atribuir_folds_por_uc(df, n_folds=n_folds)
    fold_da_celula = df["uc_id"].map(atribuicao).to_numpy()
    xs = df["x_5880"].to_numpy()
    ys = df["y_5880"].to_numpy()

    folds: list[Fold] = []
    for indice in range(n_folds):
        teste = np.flatnonzero(fold_da_celula == indice)
        if len(teste) == 0:
            continue
        ucs_teste = tuple(sorted(df.loc[teste, "uc_id"].unique().tolist()))

        blocos = [geoms[u].buffer(embargo_m) for u in ucs_teste if u in geoms]
        treino = np.flatnonzero(fold_da_celula != indice)
        if blocos:
            area = shapely.unary_union(blocos)
            dentro = shapely.intersects_xy(area, xs, ys)
            treino = treino[~dentro[treino]]

        if np.isin(df.loc[treino, "uc_id"].to_numpy(), np.array(ucs_teste)).any():
            raise AssertionError("vazamento: UC do bloco de teste presente no treino")

        folds.append(Fold(indice, ucs_teste, treino, teste))

    if len(folds) < 2:
        raise ValueError("Spatial Block CV exige ao menos 2 folds com dados")
    return folds


def recall_at_fracao(y: np.ndarray, scores: np.ndarray, fracao: float = 0.10) -> float:
    """Fração dos positivos nas ``fracao`` células de maior score.

    Seleção por POSTO, não por limiar: probabilidade calibrada é uma função em
    degraus e ``scores >= limiar`` selecionaria todos os empatados no corte,
    inflando o recall (chegou a 0,97 com k efetivo de ~50%).
    """
    positivo = int(y.sum())
    if positivo == 0 or len(scores) == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(scores) * fracao)))
    topo = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(y[topo].sum() / positivo)


def recall_por_uc(
    df: pd.DataFrame, y: np.ndarray, scores: np.ndarray, *, fracao: float = 0.10
) -> dict[str, object]:
    """Recall@k calculado dentro de cada UC (k por UC, não global)."""
    valores: list[float] = []
    ucs: list[int] = []
    for uc_id, idx in df.groupby("uc_id").indices.items():
        if y[idx].sum() == 0:
            continue
        r = recall_at_fracao(y[idx], scores[idx], fracao)
        if not np.isnan(r):
            valores.append(r)
            ucs.append(int(uc_id))
    if not valores:
        return {"media": float("nan"), "ucs_avaliadas": 0, "por_uc": {}}
    return {
        "media": round(float(np.mean(valores)), 6),
        "ucs_avaliadas": len(valores),
        "por_uc": {str(u): round(float(v), 4) for u, v in zip(ucs, valores, strict=True)},
    }


def recall_global_por_uc(
    df: pd.DataFrame, y: np.ndarray, scores: np.ndarray, *, fracao: float = 0.10
) -> dict[str, object]:
    """Métrica no formato do card v1: top-k GLOBAL, recall calculado por UC.

    Não confundir com ``recall_por_uc`` (k dentro de cada UC). É esta a definição
    que produziu os 0,3039 do card v1, então é a única comparação honesta com ele.
    """
    n = len(scores)
    if n == 0:
        return {"media": float("nan"), "ucs_avaliadas": 0, "por_uc": {}}
    k = max(1, int(np.ceil(n * fracao)))
    selecionado = np.zeros(n, dtype=bool)
    selecionado[np.argsort(-np.asarray(scores), kind="stable")[:k]] = True

    ucs = df["uc_id"].to_numpy()
    recalls: dict[str, float] = {}
    for uc in np.unique(ucs):
        m = ucs == uc
        positivos = int(y[m].sum())
        if positivos == 0:
            continue
        recalls[str(int(uc))] = round(float(y[m & selecionado].sum() / positivos), 4)

    return {
        "media": round(float(np.mean(list(recalls.values()))), 6) if recalls else float("nan"),
        "ucs_avaliadas": len(recalls),
        "por_uc": recalls,
    }


def curva_calibracao(y: np.ndarray, prob: np.ndarray, n_bins: int = 5) -> dict[str, list[float]]:
    if len(y) == 0:
        return {"prob_media_predita": [], "fracao_positiva_observada": []}
    ordem = np.argsort(prob)
    bins = np.array_split(ordem, n_bins)
    return {
        "prob_media_predita": [round(float(prob[b].mean()), 6) for b in bins if len(b)],
        "fracao_positiva_observada": [round(float(y[b].mean()), 6) for b in bins if len(b)],
    }


def iterar_ucs(df: pd.DataFrame) -> Iterable[tuple[int, np.ndarray]]:
    for uc_id, idx in df.groupby("uc_id").indices.items():
        yield int(uc_id), idx
