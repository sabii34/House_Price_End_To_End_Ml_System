import pandas as pd

from .config import load_settings, load_yaml
from .logger import get_logger


class DataValidationError(Exception):
    pass


def validate_dataframe(df: pd.DataFrame, cfg: dict) -> None:
    req_cols = cfg["validation"]["required_columns"]
    max_missing = float(cfg["validation"]["max_missing_ratio_per_column"])
    min_rows = int(cfg["validation"]["min_rows"])

    # 1) row count
    if df.shape[0] < min_rows:
        raise DataValidationError(f"Too few rows: {df.shape[0]} < {min_rows}")

    # 2) required columns present
    missing_cols = [c for c in req_cols if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    # 3) missing ratio check
    miss_ratio = df[req_cols].isna().mean()
    bad = miss_ratio[miss_ratio > max_missing]
    if len(bad) > 0:
        raise DataValidationError(
            f"Columns exceed missing threshold {max_missing}: {bad.to_dict()}"
        )

    # 4) target sanity
    target = cfg["training"]["target"]
    if target not in df.columns:
        raise DataValidationError(f"Target column not found: {target}")
    if df[target].isna().mean() > 0:
        raise DataValidationError("Target contains missing values.")

    # 5) basic numeric sanity
    for col in req_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Non-numeric column detected: {col}")

    # 6) duplicates (rare here, but industry check)
    if df.duplicated().any():
        # don’t fail hard; just warn later in pipeline
        pass


def main():
    settings = load_settings()
    cfg = load_yaml("configs/config.yaml")
    log = get_logger("data_validation", settings.log_level)

    raw_path = settings.raw_dir / cfg["dataset"]["raw_filename"]
    df = pd.read_csv(raw_path)

    validate_dataframe(df, cfg)

    log.info("[bold green]✅ Data validation passed[/bold green]")
    log.info(f"Validated file: {raw_path.resolve()}")


if __name__ == "__main__":
    main()
