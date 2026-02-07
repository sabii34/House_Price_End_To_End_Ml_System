from dataclasses import dataclass
from typing import Any, Dict
import yaml
from pathlib import Path


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Settings:
    project_name: str
    seed: int

    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path

    test_size: float
    val_size: float
    target: str
    save_model_as: str

    log_level: str


def load_settings(config_path: str = "configs/config.yaml") -> Settings:
    cfg = load_yaml(config_path)

    return Settings(
        project_name=cfg["project"]["name"],
        seed=int(cfg["project"]["seed"]),

        data_dir=Path(cfg["paths"]["data_dir"]),
        raw_dir=Path(cfg["paths"]["raw_dir"]),
        processed_dir=Path(cfg["paths"]["processed_dir"]),
        models_dir=Path(cfg["paths"]["models_dir"]),
        reports_dir=Path(cfg["paths"]["reports_dir"]),

        test_size=float(cfg["training"]["test_size"]),
        val_size=float(cfg["training"]["val_size"]),
        target=str(cfg["training"]["target"]),
        save_model_as=str(cfg["training"]["save_model_as"]),

        log_level=str(cfg["logging"]["level"]).upper(),
    )
