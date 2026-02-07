import pandas as pd
from sklearn.datasets import fetch_california_housing

from .config import load_settings, load_yaml
from .logger import get_logger
from .paths import ensure_dirs


def load_california_housing_df() -> pd.DataFrame:
    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame.copy()
    # sklearn names target as "MedHouseVal" already in frame for as_frame=True
    return df


def main():
    settings = load_settings()
    cfg = load_yaml("configs/config.yaml")
    log = get_logger("data_loading", settings.log_level)

    ensure_dirs(settings)

    dataset_name = cfg["dataset"]["name"]
    raw_filename = cfg["dataset"]["raw_filename"]

    if dataset_name != "california_housing":
        raise ValueError(f"Unsupported dataset in Phase 1: {dataset_name}")

    df = load_california_housing_df()

    raw_path = settings.raw_dir / raw_filename
    df.to_csv(raw_path, index=False)

    log.info("[bold green]✅ Raw dataset saved[/bold green]")
    log.info(f"Rows: {df.shape[0]} | Cols: {df.shape[1]}")
    log.info(f"Saved to: {raw_path.resolve()}")


if __name__ == "__main__":
    main()
