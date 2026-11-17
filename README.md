# ml-fire-alert

Frente A (pesquisa e modelagem de risco de ignição com ML + SHAP) do Sistema de Alerta de Incêndio Florestal (NGEO / IDEFLOR-Bio).

## Estrutura

- `src/ml_fire/grid.py`: grade de análise de 1 km (EPSG:5880) sobre UCs + ZA.
- `src/ml_fire/features_vector.py`: distâncias vetoriais (contratos de transição, concessões, TIs, quilombos).
- `src/ml_fire/features_gee.py`: extração GEE (MOD11A1, MOD13A2, CHIRPS, ERA5-Land, WorldCover, MCD64A1, MOD09GA, Copernicus DEM).
- `src/ml_fire/catalog.py`: geração de Parquet versionado com digest SHA-256.
- `scripts/build_features_v1.py`: CLI para geração da tabela de features v1.
- `docs/CATALOGO_FEATURES_V1.md`: catálogo e proveniência das features.
