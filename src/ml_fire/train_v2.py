"""Treino v2: grade de hiperparâmetros, calibração cruzada, métricas e TreeSHAP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from ml_fire.spatial_cv import (
    Fold,
    curva_calibracao,
    recall_at_fracao,
    recall_global_por_uc,
    recall_por_uc,
)
from ml_fire.train import NUISANCE_FEATURES

# Contrato da Seção 4 do PLANO_FRENTE_A_V2.
FEATURES_V2: tuple[str, ...] = (
    "chuva_30d_mm",
    "dist_deter_60d_m",
    "dist_pastagem_m",
    "dist_estradas_m",
    "lst_dia_c",
    "vpd_kpa",
    "dist_borda_uc_m",
    "frac_floresta",
    "declividade_deg",
    "tempo_desde_fogo_anos",
    "frac_sobreposicao_car",
)

CLIMA_BASELINE = ("chuva_30d_mm", "vpd_kpa", "lst_dia_c", "nuvem_fracao_30d")

GRADE = (
    {"num_leaves": 15, "min_child_samples": 200, "learning_rate": 0.05},
    {"num_leaves": 31, "min_child_samples": 400, "learning_rate": 0.03},
    {"num_leaves": 63, "min_child_samples": 1000, "learning_rate": 0.03, "feature_fraction": 0.7},
)

BASE = {
    "objective": "binary",
    "metric": "average_precision",
    "n_estimators": 1200,
    "max_bin": 127,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}


@dataclass
class ResultadoFold:
    indice: int
    ucs_teste: tuple[int, ...]
    melhor_iteracao: int
    auc_pr: float


@dataclass
class TreinoV2:
    modelo: LGBMClassifier
    calibrador: IsotonicRegression
    params: dict[str, Any]
    melhor_iteracao: int
    folds: list[ResultadoFold] = field(default_factory=list)
    oof: np.ndarray | None = None
    oof_mask: np.ndarray | None = None
    prob_calibrada: np.ndarray | None = None
    fold_da_celula: np.ndarray | None = None


def _ajustar(
    params: dict[str, Any],
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[LGBMClassifier, int]:
    modelo = LGBMClassifier(**{**BASE, **params})
    modelo.fit(
        x_tr,
        y_tr,
        eval_X=x_val,
        eval_y=y_val,
        eval_metric="average_precision",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    return modelo, max(int(modelo.best_iteration_ or BASE["n_estimators"]), 50)


def _subsplit(
    df: pd.DataFrame, treino: np.ndarray, n_grupos: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Separa UCs do treino para early stopping (nunca o fold de teste)."""
    ucs = np.sort(df.loc[treino, "uc_id"].unique())
    if len(ucs) < 2:
        return treino, treino
    validacao = ucs[-n_grupos:]
    mask = df.loc[treino, "uc_id"].isin(validacao).to_numpy()
    return treino[~mask], treino[mask]


def selecionar_hiperparametros(
    x: np.ndarray, y: np.ndarray, df: pd.DataFrame, folds: list[Fold]
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    historico: list[dict[str, float]] = []
    melhor, melhor_score = dict(GRADE[0]), -1.0
    for params in GRADE:
        scores = []
        for fold in folds:
            tr, va = _subsplit(df, fold.treino)
            if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            modelo, _ = _ajustar(params, x[tr], y[tr], x[va], y[va])
            prob = modelo.predict_proba(x[fold.teste])[:, 1]
            if y[fold.teste].sum() > 0:
                scores.append(average_precision_score(y[fold.teste], prob))
        if not scores:
            continue
        media = float(np.mean(scores))
        historico.append({**params, "auc_pr_media": round(media, 6)})
        if media > melhor_score:
            melhor, melhor_score = dict(params), media
    return melhor, historico


def treinar(
    x: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    folds: list[Fold],
    *,
    params: dict[str, Any] | None = None,
) -> TreinoV2:
    params = params or dict(GRADE[0])
    oof = np.full(len(y), np.nan, dtype=np.float64)
    resultados: list[ResultadoFold] = []
    iteracoes: list[int] = []

    for fold in folds:
        tr, va = _subsplit(df, fold.treino)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        modelo, iteracao = _ajustar(params, x[tr], y[tr], x[va], y[va])
        oof[fold.teste] = modelo.predict_proba(x[fold.teste])[:, 1]
        iteracoes.append(iteracao)
        auc = (
            average_precision_score(y[fold.teste], oof[fold.teste])
            if y[fold.teste].sum() > 0
            else float("nan")
        )
        resultados.append(ResultadoFold(fold.indice, fold.ucs_teste, iteracao, float(auc)))

    mask = ~np.isnan(oof)
    if mask.sum() == 0:
        raise RuntimeError("nenhuma predicao out-of-fold produzida")

    calibrador = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrador.fit(oof[mask], y[mask])

    # Calibração com ajuste cruzado: o isotônico de cada fold vem dos outros folds.
    prob_cal = np.full(len(y), np.nan, dtype=np.float64)
    fold_da_celula = np.full(len(y), -1, dtype=np.int16)
    for fold in folds:
        fold_da_celula[fold.teste] = fold.indice
    for fold in folds:
        fora = mask & (fold_da_celula != fold.indice)
        dentro = mask & (fold_da_celula == fold.indice)
        if dentro.sum() == 0:
            continue
        if fora.sum() == 0 or len(np.unique(y[fora])) < 2:
            prob_cal[dentro] = calibrador.predict(oof[dentro])
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(oof[fora], y[fora])
        prob_cal[dentro] = iso.predict(oof[dentro])

    melhor_iteracao = int(np.median(iteracoes)) if iteracoes else 500
    final = LGBMClassifier(**{**BASE, **params, "n_estimators": melhor_iteracao})
    final.fit(x, y)

    return TreinoV2(
        modelo=final,
        calibrador=calibrador,
        params=params,
        melhor_iteracao=melhor_iteracao,
        folds=resultados,
        oof=oof,
        oof_mask=mask,
        prob_calibrada=prob_cal,
        fold_da_celula=fold_da_celula,
    )


def avaliar(df: pd.DataFrame, y: np.ndarray, prob: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Métricas em um subconjunto (``mask``); ``prob`` e ``y`` têm o tamanho do painel."""
    sub = df.loc[mask].reset_index(drop=True)
    y_m, p_m = y[mask], prob[mask]
    por_uc = recall_por_uc(sub, y_m, p_m, fracao=0.10)
    return {
        "n_celulas": int(mask.sum()),
        "prevalencia": round(float(y_m.mean()), 6),
        "auc_pr": round(float(average_precision_score(y_m, p_m)), 6),
        "brier": round(float(brier_score_loss(y_m, p_m)), 6),
        "recall_at_10pct": round(float(recall_at_fracao(y_m, p_m, 0.10)), 6),
        "recall_at_10pct_por_uc": por_uc,
        "curva_calibracao": curva_calibracao(y_m, p_m),
    }


def avaliar_cv(
    df: pd.DataFrame,
    y: np.ndarray,
    prob: np.ndarray,
    folds: list[Fold],
    *,
    ranking: np.ndarray | None = None,
) -> dict[str, Any]:
    """Agrega métricas **por fold** (média), não agrupando OOF de folds desiguais.

    ``ranking`` é o score usado para AUC-PR e recall@k (padrão: ``prob``). Com
    probabilidade calibrada isotônica há muitos empates, então o ranking deve usar
    o score bruto do modelo; a calibração fica para Brier e curva.
    """
    notas = prob if ranking is None else ranking
    por_fold: list[dict[str, Any]] = []
    recall_uc: dict[str, float] = {}
    curvas: list[dict[str, list[float]]] = []

    for fold in folds:
        mask = np.zeros(len(y), dtype=bool)
        mask[fold.teste] = True
        y_m, p_m = y[mask], prob[mask]
        r_m = notas[mask]
        if y_m.sum() == 0:
            continue
        sub = df.loc[mask].reset_index(drop=True)
        recall_uc.update(recall_por_uc(sub, y_m, r_m, fracao=0.10)["por_uc"])
        curvas.append(curva_calibracao(y_m, p_m))
        por_fold.append(
            {
                "fold": fold.indice,
                "n_celulas": int(mask.sum()),
                "positivos": int(y_m.sum()),
                "prevalencia": round(float(y_m.mean()), 6),
                "auc_pr": round(float(average_precision_score(y_m, r_m)), 6),
                "brier": round(float(brier_score_loss(y_m, p_m)), 6),
                "recall_at_10pct": round(float(recall_at_fracao(y_m, r_m, 0.10)), 6),
            }
        )

    if not por_fold:
        raise ValueError("nenhum fold com positivos para avaliar")

    aucs = [f["auc_pr"] for f in por_fold]
    return {
        "n_celulas": int(sum(f["n_celulas"] for f in por_fold)),
        "n_folds": len(por_fold),
        "prevalencia": round(
            float(sum(f["positivos"] for f in por_fold) / sum(f["n_celulas"] for f in por_fold)), 6
        ),
        "auc_pr": round(float(np.mean(aucs)), 6),
        "auc_pr_desvio": round(float(np.std(aucs)), 6),
        "auc_pr_por_fold": [round(a, 6) for a in aucs],
        "brier": round(float(np.mean([f["brier"] for f in por_fold])), 6),
        "recall_at_10pct": round(float(np.mean([f["recall_at_10pct"] for f in por_fold])), 6),
        "recall_at_10pct_por_fold": [f["recall_at_10pct"] for f in por_fold],
        "recall_at_10pct_por_uc": {
            "media": round(float(np.mean(list(recall_uc.values()))), 6)
            if recall_uc
            else float("nan"),
            "ucs_avaliadas": len(recall_uc),
            "por_uc": recall_uc,
        },
        "recall_at_10pct_v1_style": recall_global_por_uc(
            df.loc[np.concatenate([f.teste for f in folds])].reset_index(drop=True),
            y[np.concatenate([f.teste for f in folds])],
            notas[np.concatenate([f.teste for f in folds])],
        ),
        "curva_calibracao": _media_curvas(curvas),
        "por_fold": por_fold,
    }


def _media_curvas(curvas: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    """Curva média por bin entre folds (não empilha escalas de folds distintos)."""
    if not curvas:
        return {"prob_media_predita": [], "fracao_positiva_observada": []}
    n = min(len(c["prob_media_predita"]) for c in curvas)
    return {
        chave: [round(float(np.mean([c[chave][i] for c in curvas])), 6) for i in range(n)]
        for chave in ("prob_media_predita", "fracao_positiva_observada")
    }


def matriz_shap(
    modelo: Any,
    x: np.ndarray,
    features: list[str],
    *,
    indices: np.ndarray | None = None,
    n_amostra: int = 20_000,
    semente: int = 42,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Matriz SHAP (linhas x features) já sem as nuisance features.

    Devolve também os índices amostrados, para que o chamador alinhe células e
    atribuições (sem isso, o SHAP de uma UC pode ser creditado a outra).
    """
    if indices is None:
        rng = np.random.default_rng(semente)
        indices = (
            np.arange(len(x))
            if len(x) <= n_amostra
            else rng.choice(len(x), n_amostra, replace=False)
        )

    explainer = shap.TreeExplainer(modelo)
    valores = explainer.shap_values(x[indices])
    if isinstance(valores, list):
        valores = valores[1] if len(valores) > 1 else valores[0]
    valores = np.asarray(valores)
    if valores.ndim == 3:
        valores = valores[:, :, 1]

    matriz = pd.DataFrame(valores, columns=features)
    expurgadas = [f for f in NUISANCE_FEATURES if f in matriz.columns]
    return matriz.drop(columns=expurgadas), expurgadas, indices


def amostra_estratificada_por_uc(
    df: pd.DataFrame, *, n_total: int = 20_000, minimo_por_uc: int = 400, semente: int = 42
) -> np.ndarray:
    """Amostra que garante presença de toda UC no SHAP local."""
    rng = np.random.default_rng(semente)
    grupos = df.groupby("uc_id").indices
    por_uc = min(minimo_por_uc, n_total // max(1, len(grupos)))
    escolhidos: list[int] = []
    for idx in grupos.values():
        n = min(len(idx), por_uc)
        escolhidos.extend(rng.choice(idx, n, replace=False).tolist())
    resto = n_total - len(escolhidos)
    if resto > 0:
        pool = np.setdiff1d(np.arange(len(df)), np.array(escolhidos), assume_unique=False)
        if len(pool):
            escolhidos.extend(rng.choice(pool, min(resto, len(pool)), replace=False).tolist())
    return np.array(sorted(escolhidos))


def top_global(matriz: pd.DataFrame, k: int = 5) -> list[dict[str, float]]:
    medias = matriz.mean(axis=0)
    ordem = medias.abs().sort_values(ascending=False).index[:k]
    return [{"feature": str(f), "mean_abs_shap": round(abs(float(medias[f])), 5)} for f in ordem]


def top_por_uc(matriz: pd.DataFrame, df: pd.DataFrame, *, k: int = 5) -> dict[str, list[dict]]:
    saida: dict[str, list[dict]] = {}
    for uc_id, posicoes in df.groupby("uc_id").indices.items():
        sub = matriz.iloc[posicoes]
        medias = sub.mean(axis=0)
        ordem = medias.abs().sort_values(ascending=False).index[:k]
        saida[str(int(uc_id))] = [
            {"feature": str(f), "mean_shap": round(float(medias[f]), 5)} for f in ordem
        ]
    return saida


def avaliar_holdout_temporal(
    df: pd.DataFrame,
    features: list[str],
    clima_features: list[str],
    *,
    target_col: str = "y_queima",
    date_col: str = "ref_date",
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Avalia o modelo completo e baseline climatico em particao temporal estrita."""
    if date_col not in df.columns:
        return None
    datas = sorted(df[date_col].astype(str).unique())
    if len(datas) < 2:
        return None

    last_date = datas[-1]
    tr_mask = df[date_col].astype(str) < last_date
    te_mask = df[date_col].astype(str) == last_date
    if not tr_mask.any() or not te_mask.any():
        return None

    y_tr = df.loc[tr_mask, target_col].to_numpy(dtype=int)
    y_te = df.loc[te_mask, target_col].to_numpy(dtype=int)
    sub_te = df.loc[te_mask].reset_index(drop=True)

    hiper = {**BASE, **(params or {}), "n_estimators": 300}

    def _fit_eval(cols: list[str]) -> dict[str, Any]:
        x_tr = df.loc[tr_mask, cols].astype(float)
        x_te = df.loc[te_mask, cols].astype(float)
        m = LGBMClassifier(**hiper)
        m.fit(x_tr, y_tr)
        p_tr = m.predict_proba(x_tr)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_tr, y_tr)
        raw_te = m.predict_proba(x_te)[:, 1]
        cal_te = iso.transform(raw_te)

        k = 0.10
        top_n = max(1, int(len(y_te) * k))
        ord_idx = np.argsort(raw_te)[::-1][:top_n]
        recall_10pct = float(y_te[ord_idx].sum() / max(1, y_te.sum()))
        por_uc = recall_por_uc(sub_te, y_te, raw_te, fracao=0.10)

        return {
            "n_celulas": int(len(y_te)),
            "positivos": int(y_te.sum()),
            "prevalencia": round(float(y_te.mean()), 6),
            "auc_pr": round(float(average_precision_score(y_te, raw_te)), 4),
            "brier": round(float(brier_score_loss(y_te, cal_te)), 4),
            "recall_at_10pct": round(recall_10pct, 4),
            "recall_at_10pct_por_uc": por_uc,
            "curva_calibracao": curva_calibracao(y_te, cal_te),
        }

    return {
        "datas_treino": datas[:-1],
        "data_teste": last_date,
        "n_celulas_treino": int(tr_mask.sum()),
        "positivos_treino": int(y_tr.sum()),
        "modelo_completo": _fit_eval(features),
        "baseline_so_clima": _fit_eval(clima_features),
    }
