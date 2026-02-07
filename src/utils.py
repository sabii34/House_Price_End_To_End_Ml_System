from .config import load_settings
from .logger import get_logger
from .paths import ensure_dirs


def main():
    settings = load_settings()
    log = get_logger("setup", settings.log_level)
    ensure_dirs(settings)
    log.info(f"[bold green]✅ Setup OK[/bold green] | Project: {settings.project_name}")
    log.info(f"Data dir: {settings.data_dir.resolve()}")


if __name__ == "__main__":
    main()
