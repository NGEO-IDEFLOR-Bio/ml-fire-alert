# Catálogo de Features v1 (`ml-fire-alert`)

Unidade de análise: **célula de 1 km × 1 km (EPSG:5880)** recortada nas **29 UCs estaduais (DGMUC)** + **Zona de Amortecimento (buffer 10 km)**.

## Regras de domínio aplicadas
- **Zonas de plano de manejo não entram** como feature (D-009).
- **Distância ao fogo ativo mais próximo não entra** (ortogonalidade com Frente B).
- **Variáveis de detectabilidade (`nuvem_fracao_30d`)** entram como *nuisances* no treino e são removidas antes do SHAP no alerta.

## Tabela de features

| Coluna | Grupo | Fonte | Resolução nativa | Janela | Papel |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dist_borda_uc_m` | territorial | IDEFLOR-Bio (29 UCs) | vetorial (EPSG:5880) | estática | feature |
| `dist_contrato_transicao_m` | pressão antrópica | `contratos_de_transicao.geojson` | vetorial (EPSG:5880) | estática | feature |
| `dist_concessao_m` | pressão antrópica | `concessoes_florestais_ideflor.geojson` | vetorial (EPSG:5880) | estática | feature |
| `dist_ti_m` | contexto | `terras_indigenas.geojson` | vetorial (EPSG:5880) | estática | feature |
| `dist_quilombo_m` | contexto | `quilombos_{estaduais,federais}.geojson` | vetorial (EPSG:5880) | estática | feature |
| `lst_dia_c`, `lst_noite_c` | térmica | `MODIS/061/MOD11A1` | 1000 m | 16d | feature |
| `ndvi`, `evi` | vegetação | `MODIS/061/MOD13A2` | 1000 m | 30d | feature |
| `chuva_7d_mm`, `chuva_30d_mm` | precipitação | `UCSB-CHG/CHIRPS/DAILY` | ~5500 m | 7d / 30d | feature |
| `temp_2m_c`, `dewpoint_2m_c`, `vento_ms`, `vpd_kpa` | clima / seca | `ECMWF/ERA5_LAND/DAILY_AGGR` | ~11132 m | 7d | feature |
| `combustivel_classe`, `frac_floresta`, `frac_pastagem_arbustiva` | combustível | `ESA/WorldCover/v200` | 10 m | estática (2021) | feature |
| `hist_queima_12m` | histórico | `MODIS/061/MCD64A1` | 500 m | 12 meses | feature |
| `nuvem_fracao_30d` | detectabilidade | `MODIS/061/MOD09GA` | 1000 m | 30d | **nuisance** |
| `elevacao_m`, `declividade_deg` | relevo | `COPERNICUS/DEM/GLO30` | 30 m | estática | feature |

## Pendências registradas para v2 (issue #04)
- Estradas (OSM/DNIT) e densidade de raios não incluídas na v1.
