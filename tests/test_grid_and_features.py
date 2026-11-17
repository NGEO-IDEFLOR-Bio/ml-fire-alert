"""Testes unitários da grade 1 km, distâncias vetoriais, catálogo e GEE (sem rede)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from ml_fire.catalog import sha256_file, write_parquet_with_digest
from ml_fire.features_gee import extract_gee_features
from ml_fire.features_vector import compute_vector_distances
from ml_fire.grid import build_grid


def _write_geojson(path: Path, features: list[dict]) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def test_build_grid_e_distancias_vetoriais(tmp_path: Path) -> None:
    ucs_path = tmp_path / "ucs.geojson"
    contratos_path = tmp_path / "contratos.geojson"

    # Quadrado ~2km x 2km no Pará
    _write_geojson(
        ucs_path,
        [
            {
                "type": "Feature",
                "properties": {"id": 1, "nome_": "UC Teste", "gerencia": "GRCN-I"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-52.00, -1.00],
                            [-51.98, -1.00],
                            [-51.98, -0.98],
                            [-52.00, -0.98],
                            [-52.00, -1.00],
                        ]
                    ],
                },
            }
        ],
    )
    _write_geojson(
        contratos_path,
        [
            {
                "type": "Feature",
                "properties": {"id": 10},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-51.95, -1.00],
                            [-51.94, -1.00],
                            [-51.94, -0.99],
                            [-51.95, -0.99],
                            [-51.95, -1.00],
                        ]
                    ],
                },
            }
        ],
    )

    df = build_grid(ucs_path, cell_size_m=1000, za_buffer_m=2000)
    assert not df.empty
    assert (df["uc_id"] == 1).all()
    assert df["dentro_uc"].any()
    assert df["dentro_za"].any()

    df_vec = compute_vector_distances(df, {"dist_contrato_transicao_m": [contratos_path]})
    assert "dist_contrato_transicao_m" in df_vec.columns
    assert (df_vec["dist_contrato_transicao_m"] >= 0).all()


def test_write_parquet_with_digest(tmp_path: Path) -> None:
    df = pd.DataFrame({"cell_id": ["1_2", "3_4"], "valor": [1.5, 2.5]})
    out_pq = tmp_path / "features.parquet"
    meta_json = tmp_path / "catalog.json"

    meta = write_parquet_with_digest(df, out_pq, date(2026, 10, 15), meta_json)
    assert out_pq.exists()
    assert meta_json.exists()
    assert meta["sha256"] == sha256_file(out_pq)
    assert meta["linhas"] == 2


def test_extract_gee_features_mockado() -> None:
    grid_df = pd.DataFrame(
        {
            "cell_id": ["100_200", "300_400"],
            "lon": [-52.0, -52.1],
            "lat": [-1.0, -1.1],
        }
    )
    fake_sampled = MagicMock()
    fake_sampled.getInfo.return_value = {
        "features": [
            {"properties": {"cell_id": "100_200", "lst_dia_c": 31.2, "ndvi": 0.72}},
            {"properties": {"cell_id": "300_400", "lst_dia_c": 29.8, "ndvi": 0.81}},
        ]
    }
    fake_img = MagicMock()
    fake_img.sampleRegions.return_value = fake_sampled

    with (
        patch("ml_fire.features_gee.build_feature_image", return_value=fake_img),
        patch("ml_fire.features_gee.ee.FeatureCollection", side_effect=lambda x: x),
        patch("ml_fire.features_gee.ee.Feature", side_effect=lambda g, p: p),
        patch("ml_fire.features_gee.ee.Geometry.Point", side_effect=lambda c: c),
    ):
        res = extract_gee_features(grid_df, date(2026, 10, 15), batch_size=10)

    assert len(res) == 2
    assert res.loc[res["cell_id"] == "100_200", "lst_dia_c"].iloc[0] == 31.2
    assert res.loc[res["cell_id"] == "300_400", "ndvi"].iloc[0] == 0.81
