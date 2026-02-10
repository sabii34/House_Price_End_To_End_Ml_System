import joblib
import mlflow
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import load_settings, load_yaml
from .logger import get_logger
from .datasets import load_split, get_xy
from .features import get_feature_spec_from_config_or_infer, build_preprocessor


def regression_metrics(y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def main():
    settings = load_settings()
    cfg = load_yaml("configs/config.yaml")
    log = get_logger("train_mlflow", settings.log_level)

    mlflow_cfg = cfg["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    df_train = load_split("train")
    df_val = load_split("val")
    X_train, y_train = get_xy(df_train)
    X_val, y_val = get_xy(df_val)

    spec, _ = get_feature_spec_from_config_or_infer(df_train)
    preprocessor = build_preprocessor(spec)

    ridge_cfg = cfg["models"]["baseline_ridge"]
    hgb_cfg = cfg["models"]["strong_hist_gb"]

    candidates = [
        ("ridge_baseline", Ridge(alpha=float(ridge_cfg["alpha"]), random_state=settings.seed)),
        ("hist_gb_strong", HistGradientBoostingRegressor(
            max_depth=int(hgb_cfg["max_depth"]),
            learning_rate=float(hgb_cfg["learning_rate"]),
            max_iter=int(hgb_cfg["max_iter"]),
            l2_regularization=float(hgb_cfg["l2_regularization"]),
            random_state=settings.seed,
        )),
    ]

    best = None
    best_rmse = float("inf")

    with mlflow.start_run(run_name="train_house_price") as run:
        mlflow.log_param("seed", settings.seed)

        for name, model in candidates:
            pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val)
            m = regression_metrics(y_val, pred)

            mlflow.log_metrics({f"{name}_val_rmse": m["rmse"], f"{name}_val_mae": m["mae"], f"{name}_val_r2": m["r2"]})
            log.info(f"{name} VAL: RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:.4f}")

            if m["rmse"] < best_rmse:
                best_rmse = m["rmse"]
                best = (name, pipe)

        best_name, best_pipe = best
        mlflow.log_param("best_model", best_name)

        # Save local artifact too (keep your old behavior)
        model_path = Path(settings.models_dir) / cfg["training"]["save_model_as"]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipe, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_local")

        # Register to MLflow Model Registry
        registered_name = mlflow_cfg["registered_model_name"]
        mlflow.sklearn.log_model(
            sk_model=best_pipe,
            artifact_path="model",
            registered_model_name=registered_name,
        )

        log.info("[bold green]✅ Registered to MLflow[/bold green]")
        log.info(f"Registered name: {registered_name}")
        log.info(f"Local model: {model_path.resolve()}")


if __name__ == "__main__":
    main()
