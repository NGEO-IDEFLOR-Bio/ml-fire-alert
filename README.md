# ml-fire-alert

Frente A (pesquisa e modelagem de risco de ignição com ML + SHAP) do Sistema de Alerta de Incêndio Florestal (NGEO / IDEFLOR-Bio).

## Estrutura

- `src/ml_fire/grid.py`: grade de análise de 1 km (EPSG:5880) sobre UCs + ZA.
- `src/ml_fire/features_vector.py`: distâncias vetoriais (contratos de transição, concessões, TIs, quilombos).
- `src/ml_fire/features_gee.py`: extração GEE (MOD11A1, MOD13A2, CHIRPS, ERA5-Land, WorldCover, MCD64A1, MOD09GA, Copernicus DEM).
- `src/ml_fire/catalog.py`: geração de Parquet versionado com digest SHA-256.
- `scripts/build_features_v1.py`: CLI para geração da tabela de features v1.
- `docs/CATALOGO_FEATURES_V1.md`: catálogo e proveniência das features.

## v2 (DETER, pastagem MapBiomas, estradas e histórico de fogo)

- `src/ml_fire/deter.py`: DETER-Amz via WFS TerraBrasilis. O namespace do plano
  (`deter-amz-nb`) não existe; o publicado é `deter-amz`, com data em `view_date`.
  Corte raso = `DESMATAMENTO_CR` + `DESMATAMENTO_VEG`.
- `src/ml_fire/mapbiomas.py`: assets descobertos em runtime (cobertura e fogo);
  pastagem = classe 15; floresta = classes 3, 4, 5 e 6.
- `src/ml_fire/osm_extract.py`: malha viária do Geofabrik (OSM/ODbL). O Overpass
  público não sustenta 29 recortes grandes (429/504).
- `src/ml_fire/raster_ops.py` + `gee_export.py`: máscaras a 250 m e EDT, com download
  GEE em tiles (limite de 32 MB e 10.000 px por requisição).
- `src/ml_fire/features_v2.py`: painel v2 (features + `y_queima` por data).
- `src/ml_fire/spatial_cv.py`: block CV por UC com embargo de fronteira.
- `src/ml_fire/train_v2.py`: LightGBM, calibração isotônica com ajuste cruzado,
  TreeSHAP com expurgo de `nuvem_fracao_30d`.
- `src/ml_fire/model_card_v2.py`: contrato do `frente_a_model_card.json` (Seção 4).
- `scripts/build_v2.py` e `scripts/train_v2.py`.

```bash
python scripts/build_v2.py --dates 2026-05-31 2026-06-30 2026-07-31 2026-08-31
python scripts/train_v2.py --panel artifacts/panel_v2.parquet --copiar-sistema
```
