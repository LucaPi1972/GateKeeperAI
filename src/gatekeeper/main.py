from gatekeeper.core.config import DEFAULT_CONFIG_PATH, load_config, validate_storage_paths
from gatekeeper.core.database import initialize_database
from gatekeeper.core.logging_setup import configure_logging


def startup(config_path: str = str(DEFAULT_CONFIG_PATH)) -> dict[str, object]:
    config = load_config(config_path)
    validate_storage_paths(config)
    configure_logging(config["storage"]["log_folder"], config["application"]["version"])
    initialize_database(config["database"]["path"])
    return config


if __name__ == "__main__":
    startup()
