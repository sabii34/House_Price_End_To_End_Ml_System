from pathlib import Path
import joblib
import pandas as pd

from src.config import load_settings, load_yaml


class ModelService:
    def __init__(self):
        self._pipe = None
        self._artifact_path = None

    def load(self):
        settings = load_settings()
        cfg = load_yaml("configs/config.yaml")

        artifact = cfg["training"]["save_model_as"]
        model_path = Path(settings.models_dir) / artifact

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run: python -m src.train"
            )

        self._pipe = joblib.load(model_path)
        self._artifact_path = str(model_path)

    @property
    def artifact_path(self) -> str:
        return self._artifact_path or ""

    def predict_one(self, features: dict) -> float:
        if self._pipe is None:
            raise RuntimeError("Model not loaded")

        X = pd.DataFrame([features])
        pred = self._pipe.predict(X)[0]
        return float(pred)


model_service = ModelService()
