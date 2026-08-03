from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    config: dict[str, Any] = {}
    current_section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1]
            config[current_section] = {}
            continue
        if current_section and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            config[current_section][key] = value.strip()

    return config


def validate_storage_paths(config: dict[str, Any]) -> None:
    database_path = Path(config["database"]["path"])
    log_folder = Path(config["storage"]["log_folder"])
    image_folder = Path(config["storage"]["image_folder"])

    database_path.parent.mkdir(parents=True, exist_ok=True)
    log_folder.mkdir(parents=True, exist_ok=True)
    image_folder.mkdir(parents=True, exist_ok=True)
