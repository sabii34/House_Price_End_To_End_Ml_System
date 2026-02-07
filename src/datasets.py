import pandas as pd
from .config import load_settings, load_yaml


def load_split(split_name: str) -> pd.DataFrame:
    """
    split_name: one of ["train", "val", "test"]
    """
    settings = load_settings()
    path = settings.processed_dir / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run Phase 1 split first: python -m src.split"
        )
    return pd.read_csv(path)


def get_xy(df: pd.DataFrame):
    cfg = load_yaml("configs/config.yaml")
    target = cfg["training"]["target"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    X = df.drop(columns=[target])
    y = df[target].astype(float)
    return X, y
