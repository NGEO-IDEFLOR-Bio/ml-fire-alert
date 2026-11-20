"""Cliente HTTP com retry/backoff exponencial para fontes externas."""

from __future__ import annotations

import time
from typing import Any

import requests

SESSAO = requests.Session()
SESSAO.headers.update({"User-Agent": "ml-fire-alert/0.2 (IDEFLOR-Bio/NGEO)"})


class ErroFonteExterna(RuntimeError):
    """Falha definitiva de uma fonte externa (apos os retries)."""


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: int = 180,
    tentativas: int = 4,
    espera_base_s: float = 2.0,
) -> dict[str, Any]:
    """GET que retorna JSON; repete em erro de rede ou 5xx."""
    ultimo: Exception | None = None
    for i in range(tentativas):
        try:
            resp = SESSAO.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code < 500 and resp.status_code != 429:
                raise ErroFonteExterna(f"{url} -> HTTP {resp.status_code}: {resp.text[:300]}")
            ultimo = ErroFonteExterna(f"HTTP {resp.status_code}")
        except (requests.RequestException, ValueError) as exc:
            ultimo = exc
        if i + 1 < tentativas:
            time.sleep(espera_base_s * (2**i))
    raise ErroFonteExterna(f"falha apos {tentativas} tentativas em {url}: {ultimo}")


def post_text(
    url: str,
    data: str,
    *,
    timeout: int = 180,
    tentativas: int = 3,
    espera_base_s: float = 30.0,
) -> str:
    """POST (Overpass) que retorna texto; respeita 429 e 5xx com espera longa."""
    ultimo: Exception | None = None
    for i in range(tentativas):
        try:
            resp = SESSAO.post(url, data=data.encode("utf-8"), timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            ultimo = ErroFonteExterna(f"HTTP {resp.status_code}: {resp.text[:300]}")
            if resp.status_code not in (429, 500, 502, 503, 504):
                break
            espera = resp.headers.get("Retry-After")
            if espera:
                time.sleep(min(float(espera), 120.0))
                continue
        except requests.RequestException as exc:
            ultimo = exc
        if i + 1 < tentativas:
            time.sleep(espera_base_s * (2**i))
    raise ErroFonteExterna(f"falha apos {tentativas} tentativas em {url}: {ultimo}")
