"""Extrator DETER contra fixture real do WFS (sem rede)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from shapely.geometry import Polygon

from ml_fire import deter

FIXTURE = Path(__file__).parent / "fixtures" / "deter_amz_sample.json"


def test_cql_inclui_janela_e_uf() -> None:
    cql = deter.montar_cql(date(2026, 7, 2), date(2026, 8, 31), "PA")
    assert "view_date >= '2026-07-02'" in cql
    assert "uf = 'PA'" in cql


def test_area_geodesica_de_quadrado_de_1km() -> None:
    quadrado = Polygon([(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)])
    assert 115 < deter._area_ha(quadrado) < 130


def test_filtra_classes_de_corte_raso(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    chamadas: list[dict] = []

    def fake_get_json(url, params=None, **_kwargs):  # noqa: ANN001
        chamadas.append(params or {})
        return payload if len(chamadas) == 1 else {"features": []}

    monkeypatch.setattr(deter, "get_json", fake_get_json)
    alertas = deter.buscar_deter(
        url_wfs="https://exemplo/wfs",
        typename="deter-amz:deter_amz",
        inicio=date(2025, 8, 1),
        fim=date(2025, 8, 10),
        uf="PA",
    )
    assert {a.classe for a in alertas} == {"DESMATAMENTO_CR", "DESMATAMENTO_VEG"}
    assert len(alertas) == 9
    assert chamadas[0]["typeNames"] == "deter-amz:deter_amz"
    assert all(a.area_ha > 0 for a in alertas)
