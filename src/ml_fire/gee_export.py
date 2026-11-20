"""Download de recortes GEE (GeoTIFF) em tiles para a grade de analise."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


def listar_assets(prefixo: str) -> list[str]:
    """Lista nomes completos de assets sob um prefixo (descoberta, nao chute)."""
    import ee

    resposta = ee.data.listAssets({"parent": prefixo})
    return [a["name"] for a in resposta.get("assets", [])]


def baixar_geotiff(
    imagem: Any,
    *,
    destino: Path,
    bounds: tuple[float, float, float, float],
    scale: float,
    crs: str,
    max_bytes_requisicao: int = 30_000_000,
    max_extensao_m: float = 200_000.0,
    timeout: int = 900,
) -> Path:
    """Baixa o recorte em GeoTIFF, em tiles, e funde no arquivo ``destino``.

    O ``getDownloadURL`` recusa requisicoes acima de 48 MiB (medidos antes de
    comprimir), entao o recorte e fatiado por numero de bandas e pixels.
    """
    import tempfile

    import rasterio
    from rasterio.merge import merge

    destino.parent.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = bounds
    n_bandas = int(imagem.bandNames().size().getInfo()) or 1

    largura_total = int(round((maxx - minx) / scale))
    altura_total = int(round((maxy - miny) / scale))
    pixels_por_bytes = max(1, max_bytes_requisicao // 4 // n_bandas)
    lado = max(256, int(pixels_por_bytes**0.5))
    # O gargalo real nao e o tamanho do arquivo, e a reprojecao a partir da
    # resolucao nativa (30 m no MapBiomas): tiles acima de ~200 km estouram
    # "Reprojection output too large" mesmo com poucos pixels de saida.
    lado = min(lado, int(max_extensao_m / scale))

    colunas = list(range(0, largura_total, lado))
    linhas = list(range(0, altura_total, lado))
    if len(colunas) == 1 and len(linhas) == 1:
        _baixar_tile(imagem, destino, (minx, miny, maxx, maxy), scale, crs, timeout, n_bandas)
        return destino

    temporarios: list[Path] = []
    pasta = Path(tempfile.mkdtemp(prefix="gee_", dir=str(destino.parent)))
    try:
        for i, linha in enumerate(linhas):
            for j, coluna in enumerate(colunas):
                x0 = minx + coluna * scale
                y1 = maxy - linha * scale
                x1 = min(maxx, x0 + lado * scale)
                y0 = max(miny, y1 - lado * scale)
                tile = pasta / f"tile_{i}_{j}.tif"
                _baixar_tile(imagem, tile, (x0, y0, x1, y1), scale, crs, timeout, n_bandas)
                temporarios.append(tile)

        fontes = [rasterio.open(t) for t in temporarios]
        try:
            mosaico, transform = merge(fontes, bounds=(minx, miny, maxx, maxy), res=scale)
            perfil = fontes[0].profile.copy()
            perfil.update(
                height=mosaico.shape[1],
                width=mosaico.shape[2],
                transform=transform,
                compress="deflate",
            )
            with rasterio.open(destino, "w", **perfil) as saida:
                saida.write(mosaico)
        finally:
            for fonte in fontes:
                fonte.close()
    finally:
        for t in temporarios:
            t.unlink(missing_ok=True)
        pasta.rmdir()

    return destino


def _baixar_tile(
    imagem: Any,
    destino: Path,
    bounds: tuple[float, float, float, float],
    scale: float,
    crs: str,
    timeout: int,
    n_bandas: int,
) -> None:
    minx, miny, maxx, maxy = bounds
    largura = max(1, int(round((maxx - minx) / scale)))
    altura = max(1, int(round((maxy - miny) / scale)))
    params = {
        "crs": crs,
        "crs_transform": [scale, 0, minx, 0, -scale, maxy],
        "dimensions": [largura, altura],
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    url = imagem.getDownloadURL(params)
    if isinstance(url, dict):
        url = url["url"]

    with requests.get(url, stream=True, timeout=timeout) as resp:
        if resp.status_code != 200:
            raise RuntimeError(
                f"GEE recusou o tile {bounds}: HTTP {resp.status_code} {resp.text[:500]}"
            )
        with destino.open("wb") as fh:
            for bloco in resp.iter_content(chunk_size=1 << 20):
                fh.write(bloco)
