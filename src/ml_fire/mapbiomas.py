"""MapBiomas (GEE): pastagem classe 15, cobertura florestal e historico de fogo.

Os nomes de asset sao descobertos por listagem (``ee.data.listAssets``), nunca
fixados no codigo: a colecao avanca e o caminho muda.
"""

from __future__ import annotations

import re
from typing import Any

LULC_RAIZ = "projects/mapbiomas-public/assets/brazil/lulc"
FOGO_RAIZ = "projects/mapbiomas-public/assets/brazil/fire"

CLASSE_PASTAGEM = 15
# Cobertura florestal natural: Formacao Florestal (3), Formacao Savanica (4),
# Mangue (5) e Floresta Alagavel (6). Mangue e floresta alagavel pesam nas UCs
# costeiras (Combu, Marajo) e omiti-las produziria frac_floresta = 0 falso.
CLASSES_FLORESTA = (3, 4, 5, 6)


def _numero(nome: str) -> int:
    achado = re.findall(r"(\d+)", nome.rsplit("/", 1)[-1])
    return int(achado[0]) if achado else 0


def _versao(nome: str) -> int:
    achado = re.search(r"_v(\d+)$", nome)
    return int(achado.group(1)) if achado else 0


def descobrir_asset_lulc() -> str:
    """Asset de cobertura (classificacao anual) da colecao MapBiomas mais recente."""
    from ml_fire.gee_export import listar_assets

    colecoes = [a for a in listar_assets(LULC_RAIZ) if "collection" in a.lower()]
    for pasta in sorted(colecoes, key=_numero, reverse=True):
        candidatos = [a for a in listar_assets(pasta) if "coverage" in a.lower()]
        if candidatos:
            return max(candidatos, key=_versao)
        candidatos = [a for a in listar_assets(pasta) if "integration" in a.lower()]
        if candidatos:
            return max(candidatos, key=_versao)
    raise RuntimeError(f"nenhum asset LULC encontrado sob {LULC_RAIZ}")


def descobrir_asset_fogo() -> str:
    """Asset anual de cicatrizes de queima do MapBiomas Fogo mais recente."""
    from ml_fire.gee_export import listar_assets

    colecoes = [a for a in listar_assets(FOGO_RAIZ) if "collection" in a.lower()]
    for pasta in sorted(colecoes, key=_numero, reverse=True):
        candidatos = [a for a in listar_assets(pasta) if "annual_burned_v" in a.lower()]
        if not candidatos:
            candidatos = [a for a in listar_assets(pasta) if "annual_burned" in a.lower()]
        if candidatos:
            return max(candidatos, key=_versao)
    raise RuntimeError(f"nenhum asset de fogo encontrado sob {FOGO_RAIZ}")


def anos_disponiveis(asset: str) -> list[int]:
    """Anos presentes nas bandas ``classification_AAAA``."""
    import ee

    nomes = ee.Image(asset).bandNames().getInfo()
    return sorted({int(m.group(0)) for n in nomes if (m := re.search(r"(?:19|20)\d{2}", n))})


def bandas(img: Any) -> list[str]:
    return list(img.bandNames().getInfo())


def banda_do_ano(img: Any, ano: int) -> str:
    """Seleciona a banda do ano, tolerando sufixo/sem sufixo."""
    nomes = bandas(img)
    alvo = re.compile(rf"(^|_){ano}(_|$)")
    for nome in nomes:
        if alvo.search(nome):
            return nome
    for nome in nomes:
        if str(ano) in nome:
            return nome
    raise KeyError(f"banda de {ano} ausente em {nomes[:5]}...")


def mascara_pastagem(asset: str, ano: int) -> Any:
    import ee

    img = ee.Image(asset)
    return img.select(banda_do_ano(img, ano)).eq(CLASSE_PASTAGEM).rename("pastagem")


def mascara_floresta(asset: str, ano: int) -> Any:
    import ee

    img = ee.Image(asset)
    banda = img.select(banda_do_ano(img, ano))
    florestal = ee.Image(0)
    for classe in CLASSES_FLORESTA:
        florestal = florestal.Or(banda.eq(classe))
    return florestal.rename("floresta")


def ultimo_ano_fogo(asset: str, ano_inicial: int, ano_final: int) -> Any:
    """Ano da cicatriz mais recente (0 = nunca queimou na serie)."""
    import ee

    img = ee.Image(asset)
    ultimo = ee.Image(0).rename("ultimo_ano_fogo")
    for ano in range(ano_inicial, ano_final + 1):
        try:
            cicatriz = img.select(banda_do_ano(img, ano))
        except KeyError:
            continue
        ultimo = ultimo.where(cicatriz.gt(0), ano)
    return ultimo.toInt16()


def fracao_por_classe(mascara: Any, escala_m: int) -> Any:
    """Converte mascara booleana de 30 m em fracao (0..1) na escala pedida."""
    import ee

    return (
        mascara.toFloat()
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=8192)
        .reproject(crs="EPSG:5880", scale=escala_m)
        .rename("fracao")
    )


def mascara_fina(mascara: Any, escala_m: int) -> Any:
    """Presenca (max) na escala fina, para a transformada de distancia."""
    import ee

    return (
        mascara.toByte()
        .reduceResolution(reducer=ee.Reducer.max(), maxPixels=8192)
        .reproject(crs="EPSG:5880", scale=escala_m)
        .rename("presenca")
    )
