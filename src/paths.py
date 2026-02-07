from pathlib import Path
from .config import Settings


def ensure_dirs(settings: Settings) -> None:
    for d in [settings.data_dir, settings.raw_dir, settings.processed_dir,
              settings.models_dir, settings.reports_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
