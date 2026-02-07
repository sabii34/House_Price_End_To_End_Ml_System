import pandas as pd
from sklearn.model_selection import train_test_split

from .config import load_settings, load_yaml
from .logger import get_logger
from .paths import ensure_dirs
from .data_validation import validate_dataframe


def main():
    settings = load_settings()
    cfg = load_yaml("configs/config.yaml")
    log = get_logger("split", settings.log_level)

    ensure_dirs(settings)

    raw_path = settings.raw_dir / cfg["dataset"]["raw_filename"]
    df = pd.read_csv(raw_path)

    # validate before splitting
    validate_dataframe(df, cfg)

    target = cfg["training"]["target"]
    test_size = float(cfg["training"]["test_size"])
    val_size = float(cfg["training"]["val_size"])

    # 1) split test first
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=settings.seed,
        shuffle=True,
    )

    # 2) split train/val from remaining
    # val_size is fraction of FULL data; convert to fraction of train_val
    val_ratio_of_train_val = val_size / (1.0 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=val_ratio_of_train_val,
        random_state=settings.seed,
        shuffle=True,
    )

    # save
    train_path = settings.processed_dir / "train.csv"
    val_path = settings.processed_dir / "val.csv"
    test_path = settings.processed_dir / "test.csv"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    log.info("[bold green]✅ Split complete[/bold green]")
    log.info(f"Train: {train.shape} -> {train_path.resolve()}")
    log.info(f"Val:   {val.shape} -> {val_path.resolve()}")
    log.info(f"Test:  {test.shape} -> {test_path.resolve()}")

    # quick target stats
    log.info(f"Target mean (train): {train[target].mean():.4f}")
    log.info(f"Target mean (val):   {val[target].mean():.4f}")
    log.info(f"Target mean (test):  {test[target].mean():.4f}")


if __name__ == "__main__":
    main()
