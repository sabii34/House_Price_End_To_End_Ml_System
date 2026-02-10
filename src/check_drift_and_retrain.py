from pathlib import Path
import pandas as pd

from evidently.report import Report
from evidently.metrics import DataDriftTable

from .config import load_yaml, load_settings
from .logger import get_logger


def main():
    settings = load_settings()
    cfg = load_yaml("configs/config.yaml")
    log = get_logger("drift", settings.log_level)

    baseline_path = Path(cfg["monitoring"]["baseline_file"])
    live_path = Path(cfg["monitoring"]["live_file"])
    report_path = Path(cfg["monitoring"]["drift_report_html"])
    threshold = float(cfg["monitoring"]["drift_threshold_share"])

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}. Run: python -m src.monitoring_baseline")
    if not live_path.exists():
        log.info("No live data yet. Call /predict a few times first.")
        return

    baseline = pd.read_csv(baseline_path)
    live = pd.read_csv(live_path)

    # keep same columns intersection
    cols = [c for c in baseline.columns if c in live.columns]
    baseline = baseline[cols]
    live = live[cols]

    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=baseline, current_data=live)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_path))
    log.info(f"[bold green]✅ Drift report saved[/bold green] {report_path.resolve()}")

    # extract drift share
    res = report.as_dict()
    drift_share = res["metrics"][0]["result"]["share_of_drifted_columns"]

    log.info(f"Drifted columns share: {drift_share:.3f} | threshold: {threshold:.3f}")

    if drift_share >= threshold:
        log.info("[bold yellow]⚠ Drift threshold reached → trigger retrain[/bold yellow]")
        # trigger retrain by importing and running your MLflow training
        from .train_mlflow import main as retrain_main
        retrain_main()
        log.info("[bold green]✅ Retrain completed & registered[/bold green]")
    else:
        log.info("[bold green]No retrain needed[/bold green]")


if __name__ == "__main__":
    main()
