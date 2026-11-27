"""Extrai e rasteriza a base de imóveis do CAR (SICAR-PA) para a grade da Frente A.

Lê sicar_imoveis_pa.geojson (390.811 imóveis rurais) de ver-o-car-api,
recorta na área de influência das 29 UCs do IDEFLOR-Bio, projeta para EPSG:5880,
gera os GeoTIFFs de fração de sobreposição (1000 m) e máscara fina (250 m),
e amostra frac_sobreposicao_car para a grade e o painel v2.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

# Fix para conflito de PROJ com PostGIS no Windows
import pyproj

pyproj_dir = pyproj.datadir.get_data_dir()
if pyproj_dir and Path(pyproj_dir).exists():
    os.environ["PROJ_DATA"] = str(pyproj_dir)
    os.environ["PROJ_LIB"] = str(pyproj_dir)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from shapely.geometry import box, shape  # noqa: E402
from shapely.ops import transform as shp_transform  # noqa: E402

from ml_fire.config import get_settings  # noqa: E402
from ml_fire.raster_ops import amostrar, transform_da_grade  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_car")

CAR_GEOJSON_SRC = Path(os.environ.get("CAR_GEOJSON_SRC", ""))
TO_METRICO = Transformer.from_crs("EPSG:4674", "EPSG:5880", always_xy=True).transform


def main() -> int:
    cfg = get_settings()
    if not CAR_GEOJSON_SRC.exists():
        logger.error("Arquivo fonte não encontrado: %s", CAR_GEOJSON_SRC)
        return 1

    grid_path = cfg.artifacts_dir / "base_grid_v2.parquet"
    if not grid_path.exists():
        logger.error("Grade base_grid_v2.parquet não encontrada em %s", grid_path)
        return 1

    logger.info("Carregando grade métrica...")
    grid = pd.read_parquet(grid_path)
    minx = float(grid["x_5880"].min()) - 10_000
    miny = float(grid["y_5880"].min()) - 10_000
    maxx = float(grid["x_5880"].max()) + 10_000
    maxy = float(grid["y_5880"].max()) + 10_000
    bounds_m = (minx, miny, maxx, maxy)

    lon_min = float(grid["lon"].min()) - 0.2
    lat_min = float(grid["lat"].min()) - 0.2
    lon_max = float(grid["lon"].max()) + 0.2
    lat_max = float(grid["lat"].max()) + 0.2
    caixa_filtro = box(lon_min, lat_min, lon_max, lat_max)
    logger.info(
        "Filtro WGS84: lon [%.2f, %.2f], lat [%.2f, %.2f]",
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    )

    t0 = time.time()
    logger.info("Carregando GeoJSON do SICAR (%s)...", CAR_GEOJSON_SRC)
    with CAR_GEOJSON_SRC.open("r", encoding="utf-8") as fh:
        raw_data = json.load(fh)
    logger.info(
        "GeoJSON carregado em %.1f s. Total: %d feições.",
        time.time() - t0,
        len(raw_data.get("features", [])),
    )

    logger.info("Filtrando e projetando imóveis sobre a área das UCs...")
    t1 = time.time()
    geometrias_m = []
    total_lidos = 0
    for feat in raw_data.get("features", []):
        total_lidos += 1
        g = feat.get("geometry")
        if not g:
            continue
        try:
            geom = shape(g)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty or not geom.intersects(caixa_filtro):
                continue
            geom_m = shp_transform(TO_METRICO, geom)
            if geom_m.is_valid and not geom_m.is_empty:
                geometrias_m.append(geom_m)
        except Exception:
            continue

    logger.info(
        "Filtrados e convertidos %d imóveis em %.1f s.",
        len(geometrias_m),
        time.time() - t1,
    )

    # 1. Rasteriza máscara fina de 250 m
    res_fina = 250.0
    tr_250, alt_250, larg_250 = transform_da_grade(bounds_m, res_fina)
    logger.info("Rasterizando máscara 250 m (%d x %d pixels)...", alt_250, larg_250)
    pares = [(g, 1.0) for g in geometrias_m]
    mascara_250 = rasterize(
        pares,
        out_shape=(alt_250, larg_250),
        transform=tr_250,
        fill=0.0,
        all_touched=True,
        dtype="float32",
    )

    out_dir = cfg.repo_root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    car_250_path = out_dir / "car_250m.tif"
    with rasterio.open(
        car_250_path,
        "w",
        driver="GTiff",
        height=alt_250,
        width=larg_250,
        count=1,
        dtype="float32",
        crs="EPSG:5880",
        transform=tr_250,
        compress="lzw",
    ) as ds:
        ds.write(mascara_250, 1)
    logger.info("Salvo: %s", car_250_path)

    # 2. Rasteriza fração em 1000 m via média em blocos 4x4
    res_1000 = 1000.0
    tr_1000, alt_1000, larg_1000 = transform_da_grade(bounds_m, res_1000)
    logger.info("Calculando fração contínua 1000 m...")
    h_trim = min((alt_250 // 4) * 4, alt_1000 * 4)
    w_trim = min((larg_250 // 4) * 4, larg_1000 * 4)
    mascara_trim = mascara_250[:h_trim, :w_trim]
    frac_1000 = (
        mascara_trim.reshape(h_trim // 4, 4, w_trim // 4, 4).mean(axis=(1, 3)).astype(np.float32)
    )

    frac_car_path = out_dir / "frac_car_1000m.tif"
    with rasterio.open(
        frac_car_path,
        "w",
        driver="GTiff",
        height=frac_1000.shape[0],
        width=frac_1000.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:5880",
        transform=tr_1000,
        compress="lzw",
    ) as ds:
        ds.write(frac_1000, 1)
    logger.info("Salvo: %s (fração média de sobreposição CAR)", frac_car_path)

    # 3. Amostra para a grade e o painel v2
    xs = grid["x_5880"].to_numpy().astype(np.float64)
    ys = grid["y_5880"].to_numpy().astype(np.float64)
    frac_amostrada = amostrar(frac_1000, tr_1000, xs, ys, nodata=0.0)
    frac_amostrada = np.clip(np.nan_to_num(frac_amostrada, nan=0.0), 0.0, 1.0)
    logger.info(
        "Amostragem concluída para 288.607 células: min=%.4f, max=%.4f, média=%.4f, com CAR: %d",
        float(frac_amostrada.min()),
        float(frac_amostrada.max()),
        float(frac_amostrada.mean()),
        int((frac_amostrada > 0).sum()),
    )

    grid["frac_sobreposicao_car"] = frac_amostrada
    grid.to_parquet(grid_path, index=False)
    logger.info("Grade %s atualizada com frac_sobreposicao_car.", grid_path)

    panel_path = cfg.artifacts_dir / "panel_v2.parquet"
    if panel_path.exists():
        logger.info("Atualizando panel_v2.parquet...")
        panel = pd.read_parquet(panel_path)
        mapa_car = dict(zip(grid["cell_id"], frac_amostrada, strict=False))
        panel["frac_sobreposicao_car"] = (
            panel["cell_id"].map(mapa_car).fillna(0.0).astype(np.float32)
        )
        panel.to_parquet(panel_path, index=False)
        logger.info(
            "panel_v2.parquet (%d linhas) atualizado com sucesso! (Células com CAR no painel: %d)",
            len(panel),
            int((panel["frac_sobreposicao_car"] > 0).sum()),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
