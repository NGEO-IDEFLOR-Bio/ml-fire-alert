"""Testes de amostragem, transform e distancia (sem rede)."""

from __future__ import annotations

import numpy as np
from affine import Affine

from ml_fire.raster_ops import (
    amostrar,
    distancia_metros,
    indices_pixel,
    transform_da_grade,
)


def test_transform_alinhado() -> None:
    transform, altura, largura = transform_da_grade((10, 20, 3010, 2020), 1000)
    assert transform.a == 1000
    assert altura == 3
    assert largura == 4


def test_amostrar_vizinho_mais_proximo() -> None:
    banda = np.array([[1.0, 2.0], [3.0, 4.0]])
    transform = Affine(1000, 0, 0, 0, -1000, 2000)
    valores = amostrar(banda, transform, np.array([500, 1500, 5000]), np.array([1500, 500, 500]))
    assert valores[0] == 1.0
    assert valores[1] == 4.0
    assert np.isnan(valores[2])


def test_indices_pixel() -> None:
    transform = Affine(250, 0, 0, 0, -250, 1000)
    linhas, cols = indices_pixel(transform, np.array([0, 250, 499]), np.array([1000, 750, 500]))
    assert cols.tolist() == [0, 1, 1]
    assert linhas.tolist() == [0, 1, 2]


def test_distancia_metros() -> None:
    mascara = np.zeros((5, 5), dtype=np.uint8)
    mascara[0, 2] = 1
    dist = distancia_metros(mascara, 250)
    assert dist[0, 2] == 0
    assert dist[2, 2] == 500
    assert dist[4, 2] == 1000
