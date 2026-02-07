import json
from pathlib import Path

import joblib
import pandas as pd

from .config import load_settings, load_yaml
from .logger import get_logger


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON file containing one record or a list of records.")
    args = parser.parse_args()

    settings = load_settings()
    cfg = load_yaml("configs/config.yaml")
    log = get_logger("predict", settings.log_level)

    model_path = Path(settings.models_dir) / cfg["training"]["save_model_as"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}. Run: python -m src.train")

    pipe = joblib.load(model_path)

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]

    X = pd.DataFrame(payload)
    preds = pipe.predict(X)

    for i, p in enumerate(preds):
        print(f"prediction[{i}] = {float(p):.4f}")

    log.info("[bold green]✅ Prediction finished[/bold green]")


if __name__ == "__main__":
    main()
