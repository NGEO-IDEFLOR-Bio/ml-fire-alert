"""Malha viaria a partir do extrato estatico do Geofabrik (OSM, ODbL).

O Overpass publico nao sustenta 29 bboxes grandes (429/504 e quedas no meio da
coleta); o extrato do Geofabrik e deterministico e cobre todo o Norte do Brasil.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests
from shapely.geometry import LineString

from ml_fire.osm_overpass import TrechoVia

URL_NORTE = "https://download.geofabrik.de/south-america/brazil/norte-latest-free.shp.zip"
MEMBROS = (
    "gis_osm_roads_free_1.shp",
    "gis_osm_roads_free_1.shx",
    "gis_osm_roads_free_1.dbf",
    "gis_osm_roads_free_1.prj",
    "gis_osm_roads_free_1.cpg",
)
# Mesmo recorte de classes do extrator Overpass (sem residential/service).
CLASSES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "track",
}


def garantir_shapefile(diretorio: Path, timeout: int = 1800) -> Path:
    """Baixa (uma vez) e extrai apenas as camadas de rodovias do extrato Norte."""
    diretorio.mkdir(parents=True, exist_ok=True)
    shp = diretorio / "gis_osm_roads_free_1.shp"
    if shp.exists():
        return shp

    compactado = diretorio / "norte-latest-free.shp.zip"
    if not compactado.exists():
        temporario = compactado.with_suffix(".part")
        with requests.get(URL_NORTE, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length") or 0)
            baixado = 0
            with temporario.open("wb") as fh:
                for bloco in resp.iter_content(chunk_size=1 << 20):
                    fh.write(bloco)
                    baixado += len(bloco)
            print(f"  geofabrik: {baixado / 1e6:.1f} MB de {total / 1e6:.1f} MB", flush=True)
        temporario.rename(compactado)

    with zipfile.ZipFile(compactado) as zf:
        disponiveis = set(zf.namelist())
        for membro in MEMBROS:
            if membro in disponiveis:
                zf.extract(membro, diretorio)
    if not shp.exists():
        raise FileNotFoundError(f"{MEMBROS[0]} ausente no extrato {compactado}")
    return shp


def ler_vias(
    shp: Path,
    *,
    bbox: tuple[float, float, float, float],
) -> list[TrechoVia]:
    """Le as vias de uma bbox (EPSG:4326) filtrando no proprio GDAL."""
    import pyogrio

    tabela = pyogrio.read_dataframe(
        shp, bbox=bbox, columns=["osm_id", "fclass", "name"], read_geometry=True
    )
    if tabela.empty:
        return []

    tabela = tabela[tabela["fclass"].isin(CLASSES)]
    trechos: list[TrechoVia] = []
    for osm_id, fclass, nome, geom in zip(
        tabela["osm_id"].to_numpy(),
        tabela["fclass"].to_numpy(),
        tabela["name"].to_numpy(),
        tabela.geometry.to_numpy(),
        strict=True,
    ):
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            geom = max(geom.geoms, key=lambda g: g.length)
        if not isinstance(geom, LineString) or geom.length == 0:
            continue
        trechos.append(
            TrechoVia(
                osm_id=int(osm_id),
                classe=str(fclass),
                nome=str(nome) if nome is not None else None,
                geometry=geom,
            )
        )
    return trechos
