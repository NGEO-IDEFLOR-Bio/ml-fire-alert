"""Contrato de exportacao do ``frente_a_model_card.json`` (Secao 4 do plano).

O schema canonico da v2 usa ``versao_modelo``/``metricas_validacao``/``top_drivers_ucs``.
O consumidor operacional (``services/frente_a.py``) le ``versao``/``metricas``/
``shap.local_por_uc``; por isso o card carrega, alem do schema canonico, uma camada
de compatibilidade com os mesmos valores. Nada e duplicado com valor diferente.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ml_fire.train import NUISANCE_FEATURES
from ml_fire.train_v2 import FEATURES_V2

VERSAO = "v2"
ALGORITMO = "LightGBM + TreeSHAP + Calibracao Isotonica"

MESES_PT = (
    "Janeiro",
    "Fevereiro",
    "Marco",
    "Abril",
    "Maio",
    "Junho",
    "Julho",
    "Agosto",
    "Setembro",
    "Outubro",
    "Novembro",
    "Dezembro",
)


def validar_card(card: dict[str, Any]) -> list[str]:
    """Levanta os defeitos do card; lista vazia = contrato cumprido."""
    problemas: list[str] = []
    obrigatorios = [
        "versao_modelo",
        "data_treino",
        "base_referencia",
        "algoritmo",
        "metricas_validacao",
        "features_utilizadas",
        "top_drivers_ucs",
    ]
    for chave in obrigatorios:
        if chave not in card:
            problemas.append(f"chave obrigatoria ausente: {chave}")

    metricas = card.get("metricas_validacao") or {}
    for chave in ["protocolo", "auc_pr", "prevalencia_natural", "recall_at_10pct"]:
        if chave not in metricas:
            problemas.append(f"metrica ausente: {chave}")

    expostas = set(card.get("features_utilizadas") or [])
    for nuisance in NUISANCE_FEATURES:
        if nuisance in expostas:
            problemas.append(f"nuisance feature exposta em features_utilizadas: {nuisance}")

    for uc, drivers in (card.get("top_drivers_ucs") or {}).items():
        for driver in drivers:
            if driver.get("feature") in NUISANCE_FEATURES:
                problemas.append(f"nuisance feature exposta nos drivers da UC {uc}")
            if "mean_shap" not in driver:
                problemas.append(f"driver sem mean_shap na UC {uc}")

    if metricas.get("protocolo") and "Spatial Block" not in str(metricas["protocolo"]):
        problemas.append("protocolo de validacao nao e Spatial Block CV")

    return problemas


def montar_card(
    *,
    data_treino: date,
    ref_date: date,
    validade_dias: int,
    metricas: dict[str, Any],
    features: list[str],
    top_drivers_ucs: dict[str, list[dict[str, float]]],
    shap_global: list[dict[str, float]],
    dataset: dict[str, Any],
    hiperparametros: dict[str, Any],
    limitacoes: list[str],
    pasta_ucs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Monta o card v2 (schema da Secao 4 + camada de compatibilidade)."""
    expostas = [f for f in features if f not in NUISANCE_FEATURES]
    base_referencia = f"{MESES_PT[ref_date.month - 1]}/{ref_date.year}"

    validacao: dict[str, Any] = {
        "protocolo": "Spatial Block CV por UC",
        "auc_pr": metricas.get("auc_pr"),
        "prevalencia_natural": metricas.get("prevalencia"),
        "recall_at_10pct": metricas.get("recall_at_10pct"),
        "baseline_climatico_recall": metricas.get("baseline_climatico_recall"),
        "auc_pr_desvio": metricas.get("auc_pr_desvio"),
        "auc_pr_por_fold": metricas.get("auc_pr_por_fold"),
        "recall_at_10pct_por_fold": metricas.get("recall_at_10pct_por_fold"),
        "recall_at_10pct_por_uc": metricas.get("recall_at_10pct_por_uc"),
        "recall_at_10pct_v1_style": metricas.get("recall_at_10pct_v1_style"),
        "curva_calibracao": metricas.get("curva_calibracao"),
        "brier": metricas.get("brier"),
        "n_celulas": metricas.get("n_celulas"),
        "n_folds": metricas.get("n_folds"),
        "por_fold": metricas.get("por_fold"),
        "detalhe_baseline": metricas.get("baseline_climatico"),
        "holdout_temporal": metricas.get("holdout_temporal"),
    }

    card: dict[str, Any] = {
        # --- schema canonico da Secao 4 ---------------------------------
        "versao_modelo": VERSAO,
        "data_treino": data_treino.isoformat(),
        "base_referencia": base_referencia,
        "algoritmo": ALGORITMO,
        "metricas_validacao": validacao,
        "features_utilizadas": expostas,
        "top_drivers_ucs": top_drivers_ucs,
        # --- camada de compatibilidade com o consumidor operacional -----
        "versao": VERSAO,
        "alvo": "P(ignicao na UC nos proximos 30 dias)",
        "ref_date": ref_date.isoformat(),
        "validade_ate": (ref_date + timedelta(days=validade_dias)).isoformat(),
        "dataset": dataset,
        "features": features,
        "nuisance_features_removidas_do_shap": list(NUISANCE_FEATURES),
        "hiperparametros": hiperparametros,
        "metricas": metricas,
        "shap": {
            "global_top": shap_global,
            "local_por_uc": _compat_local(top_drivers_ucs, pasta_ucs or {}),
        },
        "limitacoes_declaradas": limitacoes,
    }
    return card


def _compat_local(
    top_drivers_ucs: dict[str, list[dict[str, float]]],
    pasta_ucs: dict[str, str],
) -> dict[str, Any]:
    """Formato que ``obter_drivers_uc`` consome (``local_por_uc``)."""
    return {
        uc: {
            "uc_nome": pasta_ucs.get(uc),
            "top_drivers": drivers,
        }
        for uc, drivers in top_drivers_ucs.items()
    }


def gravar_card(card: dict[str, Any], destino: Path) -> Path:
    problemas = validar_card(card)
    if problemas:
        raise ValueError("card invalido: " + "; ".join(problemas))
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destino


def features_do_contrato() -> list[str]:
    return list(FEATURES_V2)
