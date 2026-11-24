"""Exporta a média SHAP de **todas** as features por UC para o model card.

O card v2 guardava só o top-5 por UC (``shap.local_por_uc``), suficiente para o
painel mas insuficiente para a nota pública mostrar a decomposição inteira.
Este script recalcula a matriz SHAP na mesma amostra estratificada do treino
(20.000 células, semente 42) e grava ``shap.por_uc_completo`` no card, sem
retreinar.

Uso::

    python scripts/export_shap_completo.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from ml_fire.model_card_v2 import gravar_card, validar_card
from ml_fire.train_v2 import amostra_estratificada_por_uc, matriz_shap, medias_por_uc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=Path("artifacts/frente_a_model_card.json"))
    parser.add_argument("--panel", type=Path, default=Path("artifacts/panel_v2.parquet"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model_v2.joblib"))
    args = parser.parse_args(argv)

    card = json.loads(args.card.read_text(encoding="utf-8"))
    features = list(card["features"])
    artefato = joblib.load(args.model)
    modelo = artefato["model"] if isinstance(artefato, dict) else artefato

    painel = pd.read_parquet(args.panel)
    faltando = [f for f in features if f not in painel.columns]
    if faltando:
        raise SystemExit(f"painel sem features do card: {faltando}")

    x = painel[features].to_numpy()
    indices = amostra_estratificada_por_uc(painel, n_total=20_000, minimo_por_uc=400)
    matriz, expurgadas, indices = matriz_shap(modelo, x, features, indices=indices)
    celulas = painel.iloc[indices].reset_index(drop=True)
    completos = medias_por_uc(matriz, celulas)

    card.setdefault("shap", {})["por_uc_completo"] = completos
    problemas = validar_card(card)
    if problemas:
        raise SystemExit("card invalido: " + "; ".join(problemas))

    destino = gravar_card(card, args.card)
    print(f"card atualizado: {destino}")
    print(f"amostra: {len(indices)} celulas; features expurgadas: {expurgadas}")
    uc8 = completos.get("8") or []
    for linha in uc8:
        print(f"  {linha['feature']}: {linha['mean_shap']:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
