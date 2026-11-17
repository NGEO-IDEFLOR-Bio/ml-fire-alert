"""Treino, validação espacial (block CV por UC), calibração e SHAP (T7)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold

CLIMATE_FEATURES: list[str] = [
    "temp_2m_c",
    "dewpoint_2m_c",
    "vento_ms",
    "vpd_kpa",
    "chuva_7d_mm",
    "chuva_30d_mm",
]

NUISANCE_FEATURES: list[str] = ["nuvem_fracao_30d"]

DOMAIN_FEATURES: list[str] = [
    "dist_borda_uc_m",
    "dist_contrato_transicao_m",
    "dist_concessao_m",
    "dist_ti_m",
    "dist_quilombo_m",
    "lst_dia_c",
    "lst_noite_c",
    "ndvi",
    "evi",
    "combustivel_classe",
    "frac_floresta",
    "frac_pastagem_arbustiva",
    "hist_queima_12m",
    "elevacao_m",
    "declividade_deg",
    *CLIMATE_FEATURES,
]


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    top_n = max(1, int(np.ceil(n * k)))
    order = np.argsort(y_prob)[::-1][:top_n]
    return float(np.mean(y_true[order]))


def recall_at_k_by_uc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uc_ids: np.ndarray,
    k: float = 0.10,
) -> dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {"mean_recall_at_k": 0.0, "ucs_avaliadas": 0}
    top_n = max(1, int(np.ceil(n * k)))
    selected = np.zeros(n, dtype=bool)
    selected[np.argsort(y_prob)[::-1][:top_n]] = True

    recalls: list[float] = []
    for uc in np.unique(uc_ids):
        mask = uc_ids == uc
        pos = int(np.sum(y_true[mask]))
        if pos == 0:
            continue
        hits = int(np.sum(y_true[mask] & selected[mask]))
        recalls.append(hits / pos)

    return {
        "mean_recall_at_k": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "ucs_avaliadas": len(recalls),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uc_ids: np.ndarray | None = None,
    k: float = 0.10,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos_rate = float(np.mean(y_true)) if len(y_true) else 0.0
    auc_pr = float(average_precision_score(y_true, y_prob)) if pos_rate > 0 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))
    p_at_k = precision_at_k(y_true, y_prob, k=k)

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=min(5, max(2, len(np.unique(y_prob)))), strategy="quantile"
    )
    out: dict[str, Any] = {
        "prevalencia": round(pos_rate, 6),
        "auc_pr": round(auc_pr, 4),
        "brier": round(brier, 4),
        f"precision_at_{int(k * 100)}pct": round(p_at_k, 4),
        "curva_calibracao": {
            "prob_media_predita": [round(float(x), 4) for x in prob_pred],
            "fracao_positiva_observada": [round(float(x), 4) for x in prob_true],
        },
    }
    if uc_ids is not None:
        out["recall_at_k_por_uc"] = recall_at_k_by_uc(y_true, y_prob, uc_ids, k=k)
    return out


def _make_lgbm(random_state: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        subsample_for_bin=50_000,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def _run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    uc_ids: np.ndarray,
    n_splits: int = 5,
) -> dict[str, Any]:
    oof_raw = np.zeros(len(y), dtype=float)
    if groups is not None:
        unique_groups = len(np.unique(groups))
        splits = min(n_splits, unique_groups)
        if splits < 2:
            raise ValueError("Precisa de pelo menos 2 UCs para block CV espacial")
        splitter = GroupKFold(n_splits=splits)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)

    for tr_idx, va_idx in split_iter:
        model = _make_lgbm()
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof_raw[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    oof_cal = iso.fit_transform(oof_raw, y)
    return compute_metrics(y, oof_cal, uc_ids=uc_ids)


def evaluate_experiments(
    df: pd.DataFrame,
    target_col: str = "y_queima",
    n_splits: int = 5,
) -> dict[str, Any]:
    """Avalia modelo completo com/sem block CV por UC e compara contra baseline só-clima."""
    full_cols = [c for c in [*DOMAIN_FEATURES, *NUISANCE_FEATURES] if c in df.columns]
    climate_cols = [c for c in CLIMATE_FEATURES if c in df.columns]

    y = df[target_col].to_numpy(dtype=int)
    uc_ids = df["uc_id"].to_numpy()

    X_full = df[full_cols].astype(float)
    X_climate = df[climate_cols].astype(float)

    block_cv_full = _run_cv(X_full, y, groups=uc_ids, uc_ids=uc_ids, n_splits=n_splits)
    random_cv_full = _run_cv(X_full, y, groups=None, uc_ids=uc_ids, n_splits=n_splits)
    block_cv_climate = _run_cv(X_climate, y, groups=uc_ids, uc_ids=uc_ids, n_splits=n_splits)

    return {
        "features_usadas": full_cols,
        "features_climate_baseline": climate_cols,
        "block_cv_espacial_por_uc": block_cv_full,
        "random_cv_sem_bloqueio_espacial": random_cv_full,
        "baseline_so_clima_block_cv": block_cv_climate,
    }


def compute_shap_summary(
    model: LGBMClassifier,
    X: pd.DataFrame,
    uc_ids: np.ndarray,
    uc_names: dict[int, str] | None = None,
    nuisance_cols: tuple[str, ...] = tuple(NUISANCE_FEATURES),
    top_k: int = 5,
    max_sample: int = 5000,
) -> dict[str, Any]:
    """Calcula SHAP global e local por UC, removendo variáveis nuisance antes do ranqueamento."""
    if len(X) > max_sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_sample, replace=False)
        X_sub = X.iloc[idx]
        uc_sub = uc_ids[idx]
    else:
        X_sub = X
        uc_sub = uc_ids

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sub)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv, dtype=float)

    keep_cols = [i for i, c in enumerate(X_sub.columns) if c not in nuisance_cols]
    names = [X_sub.columns[i] for i in keep_cols]
    sv_clean = sv[:, keep_cols]

    global_imp = np.mean(np.abs(sv_clean), axis=0)
    order = np.argsort(global_imp)[::-1]
    global_top = [
        {"feature": names[i], "mean_abs_shap": round(float(global_imp[i]), 5)}
        for i in order[:top_k]
    ]

    local_por_uc: dict[str, Any] = {}
    for uc in np.unique(uc_sub):
        mask = uc_sub == uc
        mean_sv = np.mean(sv_clean[mask], axis=0)
        uc_order = np.argsort(mean_sv)[::-1]
        uc_key = str(int(uc))
        local_por_uc[uc_key] = {
            "uc_nome": (uc_names or {}).get(int(uc), f"UC-{uc_key}"),
            "top_drivers": [
                {"feature": names[i], "mean_shap": round(float(mean_sv[i]), 5)}
                for i in uc_order[:3]
            ],
        }

    return {"global_top": global_top, "local_por_uc": local_por_uc}


def train_and_export(
    df: pd.DataFrame,
    dataset_sha256: str,
    ref_date: date,
    out_dir: Path,
    valid_days: int = 30,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Executa avaliação completa, treina modelo final calibrado e exporta artefato + model card."""
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_res = evaluate_experiments(df, n_splits=n_splits)
    full_cols = eval_res["features_usadas"]

    X = df[full_cols].astype(float)
    y = df["y_queima"].to_numpy(dtype=int)
    uc_ids = df["uc_id"].to_numpy()
    uc_names = (
        df[["uc_id", "uc_nome"]].drop_duplicates().set_index("uc_id")["uc_nome"].to_dict()
        if "uc_nome" in df.columns
        else {}
    )

    model = _make_lgbm()
    model.fit(X, y)
    raw_probs = model.predict_proba(X)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_probs, y)

    shap_summary = compute_shap_summary(model, X, uc_ids, uc_names=uc_names)

    model_path = out_dir / "model_v1.joblib"
    joblib.dump({"model": model, "calibrator": calibrator, "features": full_cols}, model_path)

    model_card = {
        "versao": "v1",
        "alvo": "P(ignicao na UC nos proximos 30 dias)",
        "ref_date": ref_date.isoformat(),
        "validade_ate": (ref_date + timedelta(days=valid_days)).isoformat(),
        "dataset": {
            "sha256": dataset_sha256,
            "linhas": int(len(df)),
            "positivos": int(np.sum(y)),
        },
        "features": full_cols,
        "nuisance_features_removidas_do_shap": [c for c in NUISANCE_FEATURES if c in full_cols],
        "metricas": {
            "block_cv_espacial_por_uc": eval_res["block_cv_espacial_por_uc"],
            "random_cv_sem_bloqueio_espacial": eval_res["random_cv_sem_bloqueio_espacial"],
            "baseline_so_clima_block_cv": eval_res["baseline_so_clima_block_cv"],
        },
        "shap": shap_summary,
        "limitacoes_declaradas": [
            "SHAP explica a previsao do modelo, nao causalidade.",
            (
                "Rotulo MCD64A1 herda vies de deteccao por nuvem "
                "(mitigado por nuisance nuvem_fracao_30d)."
            ),
            "Confundidor de fogo de manejo permanece sem curadoria (D-009).",
            (
                "SHAP local de modelo global representa transferencia de drivers, "
                "nao modelo exclusivo da UC."
            ),
        ],
    }
    card_path = out_dir / "model_card.json"
    card_path.write_text(json.dumps(model_card, indent=2, ensure_ascii=False), encoding="utf-8")
    return model_card
