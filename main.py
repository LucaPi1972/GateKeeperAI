#!/usr/bin/env python3
"""GateKeeper AI application entry point.

The module is intentionally conservative at import time so it can start on a
Raspberry Pi before optional camera/GPIO dependencies are configured.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # dependency is installed by install.sh/requirements.txt
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
LOG_DIR = PROJECT_ROOT / "logs"
IMAGE_DIR = PROJECT_ROOT / "images"
LATEST_IMAGE = IMAGE_DIR / "latest.jpg"



def _load_minimal_yaml(contents: str) -> dict[str, Any]:
    """Load the simple bundled YAML config when PyYAML is unavailable."""
    data: dict[str, Any] = {}
    current_section: str | None = None

    for raw_line in contents.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            data[current_section] = {}
            continue
        if current_section and ":" in line:
            key, value = line.split(":", 1)
            section = data.setdefault(current_section, {})
            if isinstance(section, dict):
                section[key.strip()] = value.strip()
            continue
        raise ValueError("Unsupported YAML syntax; install PyYAML for full YAML support.")

    return data

def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the YAML application configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        if yaml is not None:
            loaded = yaml.safe_load(config_file) or {}
        else:
            loaded = _load_minimal_yaml(config_file.read())

    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")

    return loaded


def configure_logging(verbose: bool = False) -> None:
    """Configure console and file logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "gatekeeper.log", encoding="utf-8"),
        ],
    )


class CameraManager:
    """Manage a persistent OpenCV camera connection."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self._capture: Any | None = None

    def start(self) -> None:
        """Initialize the camera if it is not already open."""
        if self._capture is not None and self._capture.isOpened():
            return

        import cv2

        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"Unable to open camera index {self.camera_index}.")

    def capture_latest(self, output_path: Path = LATEST_IMAGE) -> Path:
        """Capture a frame to latest.jpg using OpenCV without stopping the camera."""
        self.start()
        if self._capture is None:
            raise RuntimeError("Camera is not initialized.")

        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("Unable to capture frame from camera.")

        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Unable to write captured image: {output_path}")
        return output_path

    def stop(self) -> None:
        """Release camera resources during application shutdown."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def check_runtime() -> list[str]:
    """Return non-fatal runtime warnings for this host."""
    warnings: list[str] = []

    if sys.version_info < (3, 13):
        warnings.append(
            f"Python 3.13 is recommended; detected {platform.python_version()}."
        )

    machine = platform.machine().lower()
    if machine not in {"armv7l", "aarch64", "arm64"}:
        warnings.append(
            f"Raspberry Pi hardware not detected (machine={platform.machine()}); "
            "hardware features will only be available on the Pi."
        )

    return warnings


def import_optional_hardware() -> dict[str, bool]:
    """Probe optional Raspberry Pi hardware libraries without failing startup."""
    availability = {"gpiozero": False, "picamera2": False, "cv2": False, "flask": False}

    try:
        import gpiozero  # noqa: F401
    except ImportError:
        pass
    else:
        availability["gpiozero"] = True

    try:
        import picamera2  # noqa: F401
    except ImportError:
        pass
    else:
        availability["picamera2"] = True

    try:
        import cv2  # noqa: F401
    except ImportError:
        pass
    else:
        availability["cv2"] = True

    try:
        import flask  # noqa: F401
    except ImportError:
        pass
    else:
        availability["flask"] = True

    return availability


def main(argv: list[str] | None = None) -> int:
    """Start the GateKeeper AI application."""
    parser = argparse.ArgumentParser(description="GateKeeper AI startup")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and imports, then exit without running services.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    logger = logging.getLogger("gatekeeper")

    config = load_config(args.config)
    app_config = config.get("application", {})
    app_name = app_config.get("name", "GateKeeper AI")
    app_version = app_config.get("version", "unknown")

    logger.info("Starting %s version %s", app_name, app_version)
    for warning in check_runtime():
        logger.warning(warning)

    availability = import_optional_hardware()
    logger.info("Optional dependency availability: %s", availability)

    if args.check:
        logger.info("Startup check completed successfully.")
        return 0

    logger.info("No long-running services are configured yet; exiting cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
