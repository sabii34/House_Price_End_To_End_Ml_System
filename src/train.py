import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import load_settings, load_yaml
from .logger import get_logger
from .paths import ensure_dirs
from .datasets import load_split, get_xy
from .features import get_feature_spec_from_config_or_infer, build_preprocessor


def regression_metrics(y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def train_and_eval(model_name: str, model, preprocessor, X_train, y_train, X_val, y_val) -> tuple[Pipeline, dict]:
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)
    pred_val = pipeline.predict(X_val)
    metrics = regression_metrics(y_val, pred_val)
    metrics["model"] = model_name
    return pipeline, metrics


def main():
    settings = load_settings()
    cfg = load_yaml("configs/config.yaml")
    log = get_logger("train", settings.log_level)

    ensure_dirs(settings)

    # Load splits
    df_train = load_split("train")
    df_val = load_split("val")

    X_train, y_train = get_xy(df_train)
    X_val, y_val = get_xy(df_val)

    # Build preprocessor
    spec, _ = get_feature_spec_from_config_or_infer(df_train)
    preprocessor = build_preprocessor(spec)

    # Models
    ridge_cfg = cfg["models"]["baseline_ridge"]
    hgb_cfg = cfg["models"]["strong_hist_gb"]

    ridge = Ridge(alpha=float(ridge_cfg["alpha"]), random_state=settings.seed)
    hgb = HistGradientBoostingRegressor(
        max_depth=int(hgb_cfg["max_depth"]),
        learning_rate=float(hgb_cfg["learning_rate"]),
        max_iter=int(hgb_cfg["max_iter"]),
        l2_regularization=float(hgb_cfg["l2_regularization"]),
        random_state=settings.seed,
    )

    # Train + evaluate on VAL
    results = []
    best_pipeline = None
    best_rmse = float("inf")

    for name, model in [("ridge_baseline", ridge), ("hist_gb_strong", hgb)]:
        pipe, metrics = train_and_eval(name, model, preprocessor, X_train, y_train, X_val, y_val)
        results.append(metrics)

        log.info(f"[bold]{name}[/bold] VAL metrics: RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f}")

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_pipeline = pipe

    # Save comparison report
    reports_dir = Path(settings.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = reports_dir / "model_comparison.csv"
    pd.DataFrame(results).to_csv(comparison_path, index=False)

    # Save best pipeline
    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / cfg["training"]["save_model_as"]

    import joblib
    joblib.dump(best_pipeline, model_path)

    # Save metrics summary
    summary = {
        "best_model": [r for r in results if r["rmse"] == best_rmse][0]["model"],
        "val_results": results,
        "feature_spec": asdict(spec),
        "artifact": str(model_path),
        "seed": settings.seed,
    }
    metrics_path = reports_dir / "metrics_val.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("[bold green]✅ Training complete[/bold green]")
    log.info(f"Saved best pipeline: {model_path.resolve()}")
    log.info(f"Saved comparison: {comparison_path.resolve()}")
    log.info(f"Saved metrics: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
