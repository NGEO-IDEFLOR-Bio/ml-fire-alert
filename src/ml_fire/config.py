"""Configuração de caminhos e credenciais para extração da Frente A."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SISTEMA_ROOT = REPO_ROOT.parent / "sistema_alerta_incendio"
VISUALIZADOR_DATA = Path(os.environ.get("VISUALIZADOR_DATA", ""))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    ucs_geojson: Path
    contratos_geojson: Path
    concessoes_geojson: Path
    ti_geojson: Path
    quilombos_est_geojson: Path
    quilombos_fed_geojson: Path
    gee_project_id: str
    gee_credentials_json: Path | None
    artifacts_dir: Path
    repo_root: Path = REPO_ROOT
    deter_wfs: str = "https://terrabrasilis.dpi.inpe.br/geoserver/wfs"
    # O plano cita "deter-amz-nb:deter_amz"; o namespace publicado e "deter-amz".
    deter_typename: str = "deter-amz:deter_amz"


def get_settings() -> Settings:
    _load_dotenv(REPO_ROOT / ".env")
    _load_dotenv(SISTEMA_ROOT / ".env")

    cred_raw = os.environ.get("GEE_CREDENTIALS_JSON", "").strip()
    cred_path: Path | None = None
    if cred_raw:
        p = Path(cred_raw)
        if not p.is_absolute():
            p = (SISTEMA_ROOT / p).resolve()
        cred_path = p

    return Settings(
        ucs_geojson=Path(
            os.environ.get(
                "UCS_GEOJSON",
                str(SISTEMA_ROOT / "data" / "shapes_de_todas_as_unidades_com_gerencia.geojson"),
            )
        ),
        contratos_geojson=Path(
            os.environ.get(
                "CONTRATOS_GEOJSON",
                str(VISUALIZADOR_DATA / "contratos_de_transicao.geojson"),
            )
        ),
        concessoes_geojson=Path(
            os.environ.get(
                "CONCESSOES_GEOJSON",
                str(VISUALIZADOR_DATA / "concessoes_florestais_ideflor.geojson"),
            )
        ),
        ti_geojson=Path(
            os.environ.get("TI_GEOJSON", str(VISUALIZADOR_DATA / "terras_indigenas.geojson"))
        ),
        quilombos_est_geojson=Path(
            os.environ.get(
                "QUILOMBOS_EST_GEOJSON", str(VISUALIZADOR_DATA / "quilombos_estaduais.geojson")
            )
        ),
        quilombos_fed_geojson=Path(
            os.environ.get(
                "QUILOMBOS_FED_GEOJSON", str(VISUALIZADOR_DATA / "quilombos_federais.geojson")
            )
        ),
        gee_project_id=os.environ.get("GEE_PROJECT_ID", "").strip(),
        gee_credentials_json=cred_path,
        artifacts_dir=REPO_ROOT / "artifacts",
    )
