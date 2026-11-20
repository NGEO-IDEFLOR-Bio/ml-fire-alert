"""Extrator DETER (INPE / TerraBrasilis) para a feature ``dist_deter_60d_m``.

Correcao em relacao ao plano: o typename ``deter-amz-nb:deter_amz`` nao existe no
GeoServer publicado (erro ``Unknown namespace [deter-amz-nb]``). O namespace real e
``deter-amz``. O campo de data e ``view_date`` (nao ``date``) e a classe de corte raso
vem em ``classname``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pyproj import Geod
from shapely.geometry import shape

from ml_fire.http import get_json

# Corte raso no DETER-Amazonia. ``CS_*`` e corte seletivo, ``DEGRADACAO`` e
# degradacao: nenhum dos dois e o combustivel derrubado e dessecado que a v2 quer.
CLASSES_CORTE_RASO: tuple[str, ...] = ("DESMATAMENTO_CR", "DESMATAMENTO_VEG")

GEOD = Geod(ellps="WGS84")
PAGINA = 5000


@dataclass(frozen=True)
class AlertaDeter:
    gid: str
    classe: str
    view_date: date
    uf: str
    municipio: str | None
    area_ha: float
    geometry: object


def _area_ha(geom) -> float:
    """Area geodesica em hectares (nao confia no atributo ``areauckm`` do WFS)."""
    try:
        area_m2, _ = GEOD.geometry_area_perimeter(geom)
    except (TypeError, ValueError):
        return 0.0
    return abs(area_m2) / 10_000.0


def montar_cql(inicio: date, fim: date, uf: str | None = None) -> str:
    cql = f"view_date >= '{inicio.isoformat()}' AND view_date <= '{fim.isoformat()}'"
    if uf:
        cql += f" AND uf = '{uf}'"
    return cql


def buscar_deter(
    *,
    url_wfs: str,
    typename: str,
    inicio: date,
    fim: date,
    uf: str | None = "PA",
    classes: tuple[str, ...] = CLASSES_CORTE_RASO,
    max_paginas: int = 20,
) -> list[AlertaDeter]:
    """Baixa os alertas de corte raso do periodo (paginado por ``startIndex``)."""
    cql = montar_cql(inicio, fim, uf)
    alertas: list[AlertaDeter] = []
    vistos: set[str] = set()

    for pagina in range(max_paginas):
        params = {
            "service": "wfs",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": typename,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "count": str(PAGINA),
            "startIndex": str(pagina * PAGINA),
            "CQL_FILTER": cql,
            "sortBy": "gid",
        }
        payload = get_json(url_wfs, params=params)
        features = payload.get("features", [])
        if not features:
            break

        for feat in features:
            props = feat.get("properties") or {}
            gid = str(props.get("gid"))
            if gid in vistos:
                continue
            classe = props.get("classname")
            if classe not in classes:
                continue
            geom = feat.get("geometry")
            if not geom:
                continue
            shape_geom = shape(geom)
            if shape_geom.is_empty:
                continue
            vistos.add(gid)
            alertas.append(
                AlertaDeter(
                    gid=gid,
                    classe=str(classe),
                    view_date=date.fromisoformat(str(props["view_date"])),
                    uf=str(props.get("uf") or ""),
                    municipio=props.get("municipality"),
                    area_ha=_area_ha(shape_geom),
                    geometry=shape_geom,
                )
            )

        if len(features) < PAGINA:
            break

    return alertas
