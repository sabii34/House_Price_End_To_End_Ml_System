import mlflow
from mlflow.tracking import MlflowClient

from .config import load_yaml


def main():
    cfg = load_yaml("configs/config.yaml")
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    name = cfg["mlflow"]["registered_model_name"]
    client = MlflowClient()

    # pick latest version
    versions = client.search_model_versions(f"name='{name}'")
    latest = max(int(v.version) for v in versions)

    # Promote to Production
    (client.set_registered_model_alias(name, "prod", str(latest)))

    print(f"✅ Promoted {name} version {latest} to alias: prod")


if __name__ == "__main__":
    main()
