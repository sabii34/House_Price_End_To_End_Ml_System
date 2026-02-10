import pandas as pd
from pathlib import Path

from .config import load_settings, load_yaml
from .logger import get_logger
from .datasets import load_split, get_xy


def main():
    settings = load_settings()
    cfg = load_yaml("configs/config.yaml")
    log = get_logger("baseline", settings.log_level)

    monitoring_dir = Path(cfg["monitoring"]["monitoring_dir"])
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    df_train = load_split("train")
    X_train, _ = get_xy(df_train)

    baseline_path = Path(cfg["monitoring"]["baseline_file"])
    X_train.to_csv(baseline_path, index=False)

    log.info("[bold green]✅ Baseline created[/bold green]")
    log.info(f"Saved baseline: {baseline_path.resolve()}")
    log.info(f"Rows: {X_train.shape[0]} | Cols: {X_train.shape[1]}")


if __name__ == "__main__":
    main()
