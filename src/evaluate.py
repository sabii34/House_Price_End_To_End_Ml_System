import json
from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import load_settings, load_yaml
from .logger import get_logger
from .datasets import load_split, get_xy


def regression_metrics(y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def main():
    settings = load_settings()
    cfg = load_yaml("configs/config.yaml")
    log = get_logger("evaluate", settings.log_level)

    model_path = Path(settings.models_dir) / cfg["training"]["save_model_as"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}. Run: python -m src.train")

    pipe = joblib.load(model_path)

    df_test = load_split("test")
    X_test, y_test = get_xy(df_test)

    pred = pipe.predict(X_test)
    metrics = regression_metrics(y_test, pred)

    reports_dir = Path(settings.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_path = reports_dir / "metrics_test.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log.info("[bold green]✅ Test evaluation complete[/bold green]")
    log.info(f"TEST metrics: RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f}")
    log.info(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
