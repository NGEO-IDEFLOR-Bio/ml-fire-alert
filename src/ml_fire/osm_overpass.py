"""Extrator da malha viaria (OSM / Overpass) para ``dist_estradas_m``.

Rodovias oficiais vem do OSM (``highway``). Ramais clandestinos de arraste madeireiro
sao sub-mapeados no OSM e essa ausencia e declarada no card, nunca completada por
inferencia.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from shapely.geometry import LineString

from ml_fire.http import ErroFonteExterna, post_text

# Vias que carregam acesso humano ao combustivel. ``residential``/``service`` ficam
# de fora: dominam o trafego urbano (milhares de vias por UC) sem informacao de
# ignicao; os ramais rurais entram por ``track``/``path``/``unclassified``.
VIAS = "motorway|trunk|primary|secondary|tertiary|unclassified|track"

CONSULTA = """
[out:json][timeout:{timeout}];
(
  way["highway"~"^({vias})$"]({sul},{oeste},{norte},{leste});
);
out geom;
"""


@dataclass(frozen=True)
class TrechoVia:
    osm_id: int
    classe: str
    nome: str | None
    geometry: LineString


def _consulta_bbox(bbox: tuple[float, float, float, float], timeout_s: int) -> str:
    oeste, sul, leste, norte = bbox
    return CONSULTA.format(
        timeout=timeout_s,
        vias=VIAS,
        sul=f"{sul:.6f}",
        oeste=f"{oeste:.6f}",
        norte=f"{norte:.6f}",
        leste=f"{leste:.6f}",
    )


def parse_overpass(payload: dict[str, Any]) -> list[TrechoVia]:
    """Converte a resposta Overpass (``out geom``) em linhas shapely."""
    trechos: list[TrechoVia] = []
    for el in payload.get("elements", []):
        if el.get("type") != "way":
            continue
        coords = [(p["lon"], p["lat"]) for p in (el.get("geometry") or [])]
        if len(coords) < 2:
            continue
        tags = el.get("tags") or {}
        trechos.append(
            TrechoVia(
                osm_id=int(el["id"]),
                classe=str(tags.get("highway", "")),
                nome=tags.get("name"),
                geometry=LineString(coords),
            )
        )
    return trechos


def buscar_vias_bbox(
    *,
    url: str,
    bbox: tuple[float, float, float, float],
    timeout_s: int = 120,
    pausa_s: float = 1.0,
) -> list[TrechoVia]:
    """Baixa as vias de uma bbox ``(oeste, sul, leste, norte)`` em EPSG:4326."""
    import json

    texto = post_text(url, _consulta_bbox(bbox, timeout_s), timeout=timeout_s + 60)
    time.sleep(pausa_s)
    try:
        payload = json.loads(texto)
    except ValueError as exc:
        raise ErroFonteExterna(f"resposta nao-JSON do Overpass: {texto[:200]}") from exc
    return parse_overpass(payload)


def buscar_vias_ucs(
    *,
    url: str,
    bboxes: dict[int, tuple[float, float, float, float]],
    pausa_s: float = 1.5,
) -> dict[int, list[TrechoVia]]:
    """Baixa e deduplica as vias por UC, tolerando falha isolada.

    Uma UC que o Overpass recusar nao derruba as outras; a cobertura faltante vira
    limitacao declarada, nunca interpolacao.
    """
    from mlfire.http import ErroFonteExterna

    por_uc: dict[int, list[TrechoVia]] = {}
    for uc_id, bbox in bboxes.items():
        try:
            por_uc[uc_id] = buscar_vias_bbox(url=url, bbox=bbox, pausa_s=pausa_s)
            print(f"  vias UC {uc_id}: {len(por_uc[uc_id])}", flush=True)
        except ErroFonteExterna as exc:
            print(f"  vias UC {uc_id}: FALHOU ({str(exc)[:120]})", flush=True)
            por_uc[uc_id] = []
    return por_uc
