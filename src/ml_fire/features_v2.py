"""Features da v2: DETER 60d, pastagem MapBiomas, estradas/ramais e histórico de fogo.

As distâncias vetoriais são calculadas sobre máscaras rasterizadas a 250 m (EDT):
288 mil células contra dezenas de milhares de segmentos inviabiliza a consulta
ponto-a-ponto via STRtree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import shape as shp_shape
from shapely.ops import transform as shp_transform

from ml_fire.config import Settings
from ml_fire.deter import AlertaDeter, buscar_deter
from ml_fire.features_gee import init_ee
from ml_fire.gee_export import baixar_geotiff
from ml_fire.mapbiomas import (
    anos_disponiveis,
    descobrir_asset_fogo,
    descobrir_asset_lulc,
    fracao_por_classe,
    mascara_fina,
    mascara_floresta,
    mascara_pastagem,
    ultimo_ano_fogo,
)
from ml_fire.osm_extract import garantir_shapefile, ler_vias
from ml_fire.raster_ops import (
    amostrar,
    distancia_metros,
    ler_geotiff,
    rasterizar_linhas,
    rasterizar_poligonos,
    transform_da_grade,
)

RAIO_AREA_DETER_M = 3000.0
SEM_FOGO_ANOS = 40.0
DIST_SATURACAO_M = 50_000.0
RESOLUCAO_MASCARA_M = 250
RESOLUCAO_GRID_M = 1000

_TO_METRICO = Transformer.from_crs("EPSG:4326", "EPSG:5880", always_xy=True).transform


@dataclass(frozen=True)
class Recorte:
    """Bounds em EPSG:5880."""

    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)

    def regiao(self, ee: Any) -> Any:
        return ee.Geometry.Rectangle(list(self.bounds), proj="EPSG:5880", geodesic=False)


def recorte_do_grid(base: pd.DataFrame, margem_m: float = 20_000.0) -> Recorte:
    """Recorte derivado da grade, com margem para distâncias de borda."""
    return Recorte(
        minx=float(base["x_5880"].min()) - margem_m,
        miny=float(base["y_5880"].min()) - margem_m,
        maxx=float(base["x_5880"].max()) + margem_m,
        maxy=float(base["y_5880"].max()) + margem_m,
    )


def _dir_dados(settings: Settings) -> Path:
    pasta = settings.repo_root / "data" / "raw"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def _elevacao(regiao: Any) -> Any:
    import ee

    return (
        ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1")
        .filterBounds(regiao)
        .select("DEM")
        .mosaic()
        .rename("elevacao_m")
    )


def baixar_estaticos(settings: Settings, recorte: Recorte) -> dict[str, Path]:
    """Pastagem/floresta (MapBiomas), último ano de fogo e elevação (uma vez)."""
    import ee

    init_ee(settings.gee_project_id, settings.gee_credentials_json)
    saida = _dir_dados(settings)
    asset_lulc = descobrir_asset_lulc()
    asset_fogo = descobrir_asset_fogo()
    ano_lulc = max(anos_disponiveis(asset_lulc))
    anos_fogo = anos_disponiveis(asset_fogo)

    pastagem = mascara_pastagem(asset_lulc, ano_lulc)
    floresta = mascara_floresta(asset_lulc, ano_lulc)
    fogo = ultimo_ano_fogo(asset_fogo, min(anos_fogo), max(anos_fogo))

    alvos = {
        "pastagem_250m.tif": (mascara_fina(pastagem, RESOLUCAO_MASCARA_M), RESOLUCAO_MASCARA_M),
        "frac_pastagem_1000m.tif": (
            fracao_por_classe(pastagem, RESOLUCAO_GRID_M),
            RESOLUCAO_GRID_M,
        ),
        "frac_floresta_1000m.tif": (
            fracao_por_classe(floresta, RESOLUCAO_GRID_M),
            RESOLUCAO_GRID_M,
        ),
        "ultimo_ano_fogo_1000m.tif": (fogo, RESOLUCAO_GRID_M),
        "elevacao_1000m.tif": (_elevacao(recorte.regiao(ee)), RESOLUCAO_GRID_M),
    }

    caminhos: dict[str, Path] = {}
    for nome, (imagem, escala) in alvos.items():
        destino = saida / nome
        if not destino.exists():
            baixar_geotiff(
                imagem, destino=destino, bounds=recorte.bounds, scale=escala, crs="EPSG:5880"
            )
        caminhos[nome] = destino
    return caminhos


def baixar_janela(settings: Settings, recorte: Recorte, ref_date: date) -> Path:
    """Chuva, VPD, LST, nuvem e rótulo MCD64A1 da janela de 30 d após ``ref_date``."""
    import ee

    init_ee(settings.gee_project_id, settings.gee_credentials_json)
    saida = _dir_dados(settings) / "janelas"
    saida.mkdir(parents=True, exist_ok=True)
    destino = saida / f"janela_{ref_date.isoformat()}.tif"
    if destino.exists():
        return destino

    start = (ref_date - timedelta(days=30)).isoformat()
    end = ref_date.isoformat()

    def mean_collection(colecao: str, banda: str) -> Any:
        return ee.ImageCollection(colecao).filterDate(start, end).select(banda).mean()

    chuva = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .select("precipitation")
        .sum()
        .rename("chuva_30d_mm")
    )

    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, end).mean()
    t = era5.select("temperature_2m").subtract(273.15)
    td = era5.select("dewpoint_temperature_2m").subtract(273.15)
    vpd = (
        t.expression("0.6108 * exp(17.27 * t / (t + 237.3))", {"t": t})
        .subtract(td.expression("0.6108 * exp(17.27 * td / (td + 237.3))", {"td": td}))
        .rename("vpd_kpa")
    )

    lst = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate(start, end)
        .select("LST_Day_1km")
        .map(lambda img: img.multiply(0.02).subtract(273.15))
        .mean()
        .rename("lst_dia_c")
    )

    nuvem = (
        ee.ImageCollection("MODIS/061/MOD09GA")
        .filterDate(start, end)
        .select("state_1km")
        .map(
            lambda img: (
                img.select("state_1km")
                .bitwiseAnd(3)
                .eq(1)
                .add(img.select("state_1km").bitwiseAnd(3).eq(2))
                .gt(0)
            )
        )
        .mean()
        .rename("nuvem_fracao_30d")
    )

    burnt = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(ref_date.isoformat(), (ref_date + timedelta(days=31)).isoformat())
        .select("BurnDate")
    )
    projecao = burnt.first().projection()
    queimou = (
        burnt.max()
        .setDefaultProjection(projecao)
        .gt(0)
        .toByte()
        .reduceResolution(reducer=ee.Reducer.max(), maxPixels=8192)
        .reproject(crs="EPSG:5880", scale=RESOLUCAO_GRID_M)
        .rename("y_queima")
    )

    composicao = ee.Image.cat([chuva, vpd, lst, nuvem, queimou])
    baixar_geotiff(
        composicao,
        destino=destino,
        bounds=recorte.bounds,
        scale=RESOLUCAO_GRID_M,
        crs="EPSG:5880",
        max_extensao_m=600_000.0,
    )
    return destino


def baixar_estradas(settings: Settings, bbox_4326: tuple[float, float, float, float]) -> list[dict]:
    """Malha viária do OSM (Geofabrik, ODbL) no recorte das UCs + ZA."""
    cache = _dir_dados(settings) / "estradas_geofabrik.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    shp = garantir_shapefile(_dir_dados(settings) / "osm")
    trechos = ler_vias(shp, bbox=bbox_4326)
    dados = [
        {
            "osm_id": t.osm_id,
            "classe": t.classe,
            "nome": t.nome,
            "geometry": shapely.geometry.mapping(t.geometry),
        }
        for t in trechos
    ]
    cache.write_text(json.dumps(dados), encoding="utf-8")
    return dados


def _area_deter_3km(
    alertas: list[AlertaDeter],
    xs: np.ndarray,
    ys: np.ndarray,
    raio_m: float = RAIO_AREA_DETER_M,
) -> np.ndarray:
    acumulado = np.zeros(len(xs), dtype=np.float64)
    for alerta in alertas:
        geom = shp_transform(_TO_METRICO, alerta.geometry)
        area_ha = float(shapely.area(geom) / 10_000.0)
        busca = geom.buffer(raio_m)
        minx, miny, maxx, maxy = busca.bounds
        caixa = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
        if not caixa.any():
            continue
        idx = np.flatnonzero(caixa)
        dentro = shapely.intersects_xy(busca, xs[idx], ys[idx])
        acumulado[idx[dentro]] += area_ha
    return acumulado


def build_panel_v2(base: pd.DataFrame, ref_dates: list[date], settings: Settings) -> pd.DataFrame:
    """Painel v2: grade + features novas + ``y_queima`` por data de referência."""
    recorte = recorte_do_grid(base)
    estaticos = baixar_estaticos(settings, recorte)

    xs = base["x_5880"].to_numpy().astype(np.float64)
    ys = base["y_5880"].to_numpy().astype(np.float64)

    past_250, tr_past = ler_geotiff(estaticos["pastagem_250m.tif"])
    dist_pastagem = distancia_metros(past_250[0], RESOLUCAO_MASCARA_M)
    base["dist_pastagem_m"] = np.clip(
        amostrar(dist_pastagem, tr_past, xs, ys), None, DIST_SATURACAO_M
    )
    del past_250, dist_pastagem

    for nome, coluna in [
        ("frac_pastagem_1000m.tif", "frac_pastagem"),
        ("frac_floresta_1000m.tif", "frac_floresta"),
    ]:
        banda, transform = ler_geotiff(estaticos[nome])
        base[coluna] = amostrar(banda[0], transform, xs, ys)

    fogo, tr_fogo = ler_geotiff(estaticos["ultimo_ano_fogo_1000m.tif"])
    ultimo_ano = amostrar(fogo[0], tr_fogo, xs, ys)

    elev, tr_elev = ler_geotiff(estaticos["elevacao_1000m.tif"])
    dz_dy, dz_dx = np.gradient(np.asarray(elev[0], dtype=np.float64), abs(tr_elev.a))
    declividade = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)
    base["declividade_deg"] = amostrar(declividade, tr_elev, xs, ys)

    bbox = (
        float(base["lon"].min()),
        float(base["lat"].min()),
        float(base["lon"].max()),
        float(base["lat"].max()),
    )
    estradas = baixar_estradas(settings, bbox)
    tr_mascara, altura, largura = transform_da_grade(recorte.bounds, RESOLUCAO_MASCARA_M)
    linhas = [shp_transform(_TO_METRICO, shp_shape(e["geometry"])) for e in estradas]
    mascara_vias = rasterizar_linhas(linhas, tr_mascara, altura, largura)
    dist_vias = distancia_metros(mascara_vias, RESOLUCAO_MASCARA_M)
    base["dist_estradas_m"] = np.clip(
        amostrar(dist_vias, tr_mascara, xs, ys), None, DIST_SATURACAO_M
    )
    del linhas, mascara_vias, dist_vias

    blocos: list[pd.DataFrame] = []
    for ref_date in ref_dates:
        janela_path = baixar_janela(settings, recorte, ref_date)
        banda, transform = ler_geotiff(janela_path)
        nomes = ["chuva_30d_mm", "vpd_kpa", "lst_dia_c", "nuvem_fracao_30d", "y_queima"]
        bloco = base.copy()
        for i, nome in enumerate(nomes[: len(banda)]):
            bloco[nome] = amostrar(banda[i], transform, xs, ys)

        for nome in ["vpd_kpa", "lst_dia_c"]:
            bloco[nome] = np.where(bloco[nome] == 0, np.nan, bloco[nome])

        alertas = buscar_deter(
            url_wfs=settings.deter_wfs,
            typename=settings.deter_typename,
            inicio=ref_date - timedelta(days=60),
            fim=ref_date,
            uf="PA",
        )
        poly = [shp_transform(_TO_METRICO, a.geometry) for a in alertas]
        mascara_deter = rasterizar_poligonos(poly, tr_mascara, altura, largura)
        dist_deter = distancia_metros(mascara_deter, RESOLUCAO_MASCARA_M)
        bloco["dist_deter_60d_m"] = np.clip(
            amostrar(dist_deter, tr_mascara, xs, ys), None, DIST_SATURACAO_M
        )
        bloco["area_deter_3km_ha"] = _area_deter_3km(alertas, xs, ys)

        anos = np.where(np.isnan(ultimo_ano), np.nan, ref_date.year - ultimo_ano)
        anos = np.where(ultimo_ano <= 0, SEM_FOGO_ANOS, anos)
        bloco["tempo_desde_fogo_anos"] = np.clip(anos, 0, SEM_FOGO_ANOS)

        bloco["y_queima"] = (bloco.pop("y_queima").fillna(0) > 0).astype(int)
        bloco["ref_date"] = ref_date.isoformat()
        blocos.append(bloco)

    return pd.concat(blocos, ignore_index=True)
