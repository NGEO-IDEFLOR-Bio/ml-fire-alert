"""Contrato do model card: schema da Secao 4 e expurgo da nuisance feature."""

from __future__ import annotations

from datetime import date

import pytest

from ml_fire.model_card_v2 import montar_card, validar_card
from ml_fire.train import NUISANCE_FEATURES
from ml_fire.train_v2 import FEATURES_V2


def _card() -> dict:
    return montar_card(
        data_treino=date(2026, 9, 1),
        ref_date=date(2026, 8, 31),
        validade_dias=30,
        metricas={
            "prevalencia": 0.0037,
            "auc_pr": 0.012,
            "recall_at_10pct": 0.362,
            "brier": 0.0037,
            "baseline_climatico_recall": 0.141,
            "recall_at_10pct_por_uc": {"media": 0.35, "ucs_avaliadas": 29, "por_uc": {}},
            "curva_calibracao": {
                "prob_media_predita": [0.001],
                "fracao_positiva_observada": [0.001],
            },
            "n_celulas": 100,
        },
        features=[*FEATURES_V2, *NUISANCE_FEATURES],
        top_drivers_ucs={"1": [{"feature": "dist_deter_60d_m", "mean_shap": 0.485}]},
        shap_global=[{"feature": "dist_deter_60d_m", "mean_abs_shap": 0.485}],
        dataset={"sha256": "x", "linhas": 10, "positivos": 1, "ref_dates": ["2026-08-31"]},
        hiperparametros={"num_leaves": 31},
        limitacoes=["SHAP explica a previsao do modelo, nao causalidade."],
        pasta_ucs={"1": "UC Teste"},
    )


def test_card_tem_schema_da_secao_4() -> None:
    card = _card()
    for chave in [
        "versao_modelo",
        "data_treino",
        "base_referencia",
        "algoritmo",
        "metricas_validacao",
        "features_utilizadas",
        "top_drivers_ucs",
    ]:
        assert chave in card
    assert card["versao_modelo"] == "v2"
    assert card["metricas_validacao"]["protocolo"] == "Spatial Block CV por UC"
    assert card["metricas_validacao"]["prevalencia_natural"] == 0.0037
    assert card["metricas_validacao"]["recall_at_10pct"] == 0.362
    assert card["base_referencia"] == "Agosto/2026"


def test_nuisance_expurgada_da_camada_explicativa() -> None:
    card = _card()
    for nuisance in NUISANCE_FEATURES:
        assert nuisance not in card["features_utilizadas"]
        assert nuisance in card["features"]
        for drivers in card["top_drivers_ucs"].values():
            assert all(d["feature"] != nuisance for d in drivers)
        assert all(g["feature"] != nuisance for g in card["shap"]["global_top"])
    assert card["nuisance_features_removidas_do_shap"] == list(NUISANCE_FEATURES)


def test_camada_de_compatibilidade_preserva_consumidor() -> None:
    card = _card()
    assert card["versao"] == card["versao_modelo"]
    assert card["ref_date"] == "2026-08-31"
    assert card["validade_ate"] == "2026-09-30"
    assert card["shap"]["local_por_uc"]["1"]["uc_nome"] == "UC Teste"
    assert card["metricas"]["auc_pr"] == 0.012
    assert card["shap"]["local_por_uc"]["1"]["top_drivers"][0]["feature"] == "dist_deter_60d_m"


def test_validar_card_pega_nuisance_exposta() -> None:
    card = _card()
    card["top_drivers_ucs"]["1"].append({"feature": NUISANCE_FEATURES[0], "mean_shap": 0.1})
    problemas = validar_card(card)
    assert any("nuisance" in p for p in problemas)


def test_validar_card_pega_protocolo_errado() -> None:
    card = _card()
    card["metricas_validacao"]["protocolo"] = "Random CV"
    assert any("protocolo" in p for p in validar_card(card))


def test_gravar_card_recusa_card_invalido(tmp_path) -> None:
    from ml_fire.model_card_v2 import gravar_card

    card = _card()
    del card["features_utilizadas"]
    with pytest.raises(ValueError):
        gravar_card(card, tmp_path / "card.json")
