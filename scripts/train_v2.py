"""Treina a Frente A v2 e emite ``frente_a_model_card.json`` (Secao 4 do plano).

Uso:
    python scripts/train_v2.py --panel artifacts/panel_v2.parquet
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_fire.catalog import sha256_file
from ml_fire.config import get_settings
from ml_fire.model_card_v2 import gravar_card, montar_card
from ml_fire.spatial_cv import carregar_geometrias_uc, folds_com_embargo
from ml_fire.train import NUISANCE_FEATURES
from ml_fire.train_v2 import (
    CLIMA_BASELINE,
    FEATURES_V2,
    amostra_estratificada_por_uc,
    avaliar_cv,
    avaliar_holdout_temporal,
    matriz_shap,
    medias_por_uc,
    selecionar_hiperparametros,
    top_global,
    top_por_uc,
    treinar,
)

# Escolhidos na busca de hiperparâmetros do primeiro treino completo (T11).
PARAMS_PADRAO = {"num_leaves": 31, "min_child_samples": 400, "learning_rate": 0.03}


def _matriz(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    faltando = [f for f in features if f not in df.columns]
    if faltando:
        raise KeyError(f"features ausentes no painel: {faltando}")
    return df[features].astype(float).to_numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="", help="Parquet do painel v2")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo-m", type=int, default=5000)
    parser.add_argument("--sem-grade", action="store_true", help="pula a busca de hiperparametros")
    parser.add_argument("--out-dir", default="", help="Diretorio do artefato")
    parser.add_argument(
        "--copiar-sistema",
        action="store_true",
        help="copia o card para backend/artifacts do sistema operacional",
    )
    args = parser.parse_args(argv)

    cfg = get_settings()
    panel_path = Path(args.panel) if args.panel else cfg.artifacts_dir / "panel_v2.parquet"
    out_dir = Path(args.out_dir) if args.out_dir else cfg.artifacts_dir
    df = pd.read_parquet(panel_path)
    features = [*FEATURES_V2, *NUISANCE_FEATURES]
    x = _matriz(df, features)
    y = df["y_queima"].to_numpy(dtype=int)

    geoms = carregar_geometrias_uc(cfg.ucs_geojson)
    folds = folds_com_embargo(df, geoms, n_folds=args.n_folds, embargo_m=args.embargo_m)
    print(f"folds: {[f.indice for f in folds]} | embargo {args.embargo_m} m")

    params = PARAMS_PADRAO if args.sem_grade else None
    historico: list[dict[str, float]] = []
    if not args.sem_grade:
        params, historico = selecionar_hiperparametros(x, y, df, folds)
    print("hiperparametros:", params)

    treino = treinar(x, y, df, folds, params=params)
    metricas = avaliar_cv(df, y, treino.prob_calibrada, folds, ranking=treino.oof)

    x_clima = _matriz(df, list(CLIMA_BASELINE))
    baseline = treinar(x_clima, y, df, folds, params=params or PARAMS_PADRAO)
    base_metricas = avaliar_cv(df, y, baseline.prob_calibrada, folds, ranking=baseline.oof)
    metricas["baseline_climatico_recall"] = base_metricas["recall_at_10pct"]
    metricas["baseline_climatico"] = {
        "auc_pr": base_metricas["auc_pr"],
        "auc_pr_por_fold": base_metricas["auc_pr_por_fold"],
        "recall_at_10pct": base_metricas["recall_at_10pct"],
        "recall_at_10pct_por_fold": base_metricas["recall_at_10pct_por_fold"],
        "brier": base_metricas["brier"],
    }
    print("por fold:", metricas["por_fold"])
    print("baseline por fold:", base_metricas["por_fold"])

    holdout = avaliar_holdout_temporal(
        df,
        features,
        list(CLIMA_BASELINE),
        params=params or PARAMS_PADRAO,
    )
    if holdout:
        metricas["holdout_temporal"] = holdout
        print(
            "holdout temporal:",
            {
                "datas_treino": holdout["datas_treino"],
                "data_teste": holdout["data_teste"],
                "modelo_auc_pr": holdout["modelo_completo"]["auc_pr"],
                "modelo_recall@10%": holdout["modelo_completo"]["recall_at_10pct"],
                "baseline_auc_pr": holdout["baseline_so_clima"]["auc_pr"],
                "baseline_recall@10%": holdout["baseline_so_clima"]["recall_at_10pct"],
            },
        )

    indices = amostra_estratificada_por_uc(df, n_total=20_000, minimo_por_uc=400)
    shap_matriz, expurgadas, indices = matriz_shap(treino.modelo, x, features, indices=indices)
    celulas = df.iloc[indices].reset_index(drop=True)
    drivers = top_por_uc(shap_matriz, celulas)
    drivers_completos = medias_por_uc(shap_matriz, celulas)
    shap_global = top_global(shap_matriz)

    nomes = (
        df[["uc_id", "uc_nome"]].drop_duplicates().set_index("uc_id")["uc_nome"].to_dict()
        if "uc_nome" in df.columns
        else {}
    )
    nomes_str = {str(int(k)): v for k, v in nomes.items()}

    ref_final = date.fromisoformat(str(pd.to_datetime(df["ref_date"]).max().date()))
    card = montar_card(
        data_treino=date.today(),
        ref_date=ref_final,
        validade_dias=30,
        metricas=metricas,
        features=features,
        top_drivers_ucs=drivers,
        shap_global=shap_global,
        dataset={
            "sha256": sha256_file(panel_path),
            "linhas": int(len(df)),
            "positivos": int(y.sum()),
            "ref_dates": sorted(df["ref_date"].astype(str).unique().tolist()),
        },
        hiperparametros={
            **treino.params,
            "melhor_iteracao": treino.melhor_iteracao,
            "grade": historico,
        },
        limitacoes=[
            "SHAP explica a previsao do modelo, nao causalidade.",
            "nuvem_fracao_30d entra no ajuste como controle de detectabilidade e e "
            "expurgada da camada explicativa.",
            "Rotulo MCD64A1 herda vies de deteccao por nuvem e resolucao de 500 m.",
            "Ramais clandestinos sao sub-mapeados no OSM; a ausencia nao e completada "
            "por inferencia.",
            "frac_sobreposicao_car usa extrato estatico do SICAR-PA (via ver-o-car-api, "
            "~390 mil imoveis); imoveis nao declarados nao entram.",
            "Distancias vetoriais rasterizadas a 250 m; erro posicional de ate 250 m.",
            "Distancias saturadas em 50 km; NaN indica fonte sem feicao no recorte.",
            f"SHAP local de modelo global representa transferencia de drivers "
            f"(nuisance expurgada: {', '.join(expurgadas)}).",
        ],
        pasta_ucs=nomes_str,
        shap_por_uc_completo=drivers_completos,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "oof_v2.npz",
        oof=treino.oof,
        prob_calibrada=treino.prob_calibrada,
        fold=treino.fold_da_celula,
        y=y,
    )
    joblib.dump(
        {"model": treino.modelo, "calibrator": treino.calibrador, "features": features},
        out_dir / "model_v2.joblib",
    )
    card_path = out_dir / "frente_a_model_card.json"
    gravar_card(card, card_path)
    print("card:", card_path)
    print(json.dumps(card["metricas_validacao"], ensure_ascii=False, indent=2)[:1200])

    if args.copiar_sistema:
        destino = (
            cfg.repo_root.parent
            / "sistema_alerta_incendio"
            / "backend"
            / "artifacts"
            / "frente_a_model_card.json"
        )
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(card_path, destino)
        print("copiado para:", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
