"""Leitura, amostragem e transformada de distancia sobre grades metricas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely
from affine import Affine
from scipy.ndimage import distance_transform_edt


def ler_geotiff(caminho: Path) -> tuple[np.ndarray, Affine]:
    """Le todas as bandas de um GeoTIFF (n_bandas, altura, largura) + transform."""
    import rasterio

    with rasterio.open(caminho) as ds:
        dados = ds.read(masked=False)
        return dados, ds.transform


def indices_pixel(
    transform: Affine, xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Converte coordenadas (x, y) em (linha, coluna) de pixel (vizinho mais proximo)."""
    inv = ~transform
    cols, linhas = inv @ (xs, ys)
    return np.floor(linhas).astype(np.int64), np.floor(cols).astype(np.int64)


def amostrar(
    banda: np.ndarray,
    transform: Affine,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    nodata: float = np.nan,
) -> np.ndarray:
    """Amostra uma banda nos pontos dados; fora da imagem -> ``nodata``."""
    linhas, cols = indices_pixel(transform, xs, ys)
    altura, largura = banda.shape
    valido = (linhas >= 0) & (linhas < altura) & (cols >= 0) & (cols < largura)
    saida = np.full(len(xs), nodata, dtype=np.float64)
    saida[valido] = banda[linhas[valido], cols[valido]]
    return saida


def transform_da_grade(
    bounds: tuple[float, float, float, float], resolucao_m: float
) -> tuple[Affine, int, int]:
    """Transform alinhada a multiplos da resolucao (canto superior esquerdo)."""
    minx, miny, maxx, maxy = bounds
    x0 = np.floor(minx / resolucao_m) * resolucao_m
    y1 = np.ceil(maxy / resolucao_m) * resolucao_m
    x1 = np.ceil(maxx / resolucao_m) * resolucao_m
    y0 = np.floor(miny / resolucao_m) * resolucao_m
    largura = int(round((x1 - x0) / resolucao_m))
    altura = int(round((y1 - y0) / resolucao_m))
    return Affine(resolucao_m, 0, x0, 0, -resolucao_m, y1), altura, largura


def rasterizar_poligonos(
    geometrias: list,
    transform: Affine,
    altura: int,
    largura: int,
    valores: list[float] | None = None,
) -> np.ndarray:
    """Rasteriza poligonos (EPSG:5880) em float32; sem valor -> 1.0."""
    from rasterio.features import rasterize

    pares = [
        (g, float(valores[i]) if valores is not None else 1.0)
        for i, g in enumerate(geometrias)
        if g is not None and not g.is_empty
    ]
    if not pares:
        return np.zeros((altura, largura), dtype=np.float32)
    return rasterize(
        pares,
        out_shape=(altura, largura),
        transform=transform,
        fill=0.0,
        all_touched=False,
        dtype="float32",
    )


def rasterizar_linhas(
    geometrias: list,
    transform: Affine,
    altura: int,
    largura: int,
) -> np.ndarray:
    """Rasteriza linhas como corredor de 1 pixel (buffer minimo + ``all_touched``)."""
    from rasterio.features import rasterize

    pares = [(g.buffer(1.0), 1.0) for g in geometrias if g is not None and not g.is_empty]
    if not pares:
        return np.zeros((altura, largura), dtype=np.uint8)
    return rasterize(
        pares,
        out_shape=(altura, largura),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )


def distancia_metros(mascara: np.ndarray, resolucao_m: float) -> np.ndarray:
    """Distancia euclidiana (m) de cada pixel ao pixel mais proximo com mascara > 0."""
    origem = mascara <= 0
    if not origem.any() or origem.all():
        return np.full(mascara.shape, np.nan, dtype=np.float32)
    return distance_transform_edt(origem, sampling=resolucao_m).astype(np.float32)


def distancia_a_geometrias(
    geometrias: list,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    raio_busca_m: float | None = None,
) -> np.ndarray:
    """Distancia exata (shapely) de cada ponto a geometria mais proxima.

    Usado para ``area_deter_3km_ha`` e para conferir a aproximacao raster.
    """
    pontos = shapely.points(xs, ys)
    if not geometrias:
        return np.full(len(xs), np.nan)
    uniao = shapely.unary_union([g for g in geometrias if g is not None and not g.is_empty])
    if raio_busca_m is None:
        return shapely.distance(pontos, uniao)
    return shapely.distance(pontos, uniao)
