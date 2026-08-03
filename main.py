#!/usr/bin/env python3
"""GateKeeper AI application entry point."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # dependency is installed by requirements.txt
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
VERSION_FILE = PROJECT_ROOT / "VERSION"
LOG_DIR = PROJECT_ROOT / "logs"
IMAGE_DIR = PROJECT_ROOT / "images"
LATEST_IMAGE = IMAGE_DIR / "latest.jpg"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
PID_FILE = RUNTIME_DIR / "gatekeeper.pid"
DATABASE_PATH = RUNTIME_DIR / "gatekeeper.db"
CAMERA_BACKEND = "Picamera2"


def get_version() -> str:
    """Return the application version from the VERSION file."""
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def get_git_commit() -> str:
    """Return the current short Git commit hash, or development outside Git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "development"
    return result.stdout.strip() or "development"


def get_opencv_version() -> str:
    """Return the installed OpenCV version or unavailable."""
    try:
        import cv2
    except ImportError:
        return "unavailable"
    return str(cv2.__version__)


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
                value = value.strip()
                if value.lower() in {"true", "false"}:
                    section[key.strip()] = value.lower() == "true"
                else:
                    try:
                        section[key.strip()] = int(value)
                    except ValueError:
                        section[key.strip()] = value
            continue
        raise ValueError(
            "Unsupported YAML syntax; install PyYAML for full YAML support."
        )

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


def configure_logging(config: dict[str, Any], verbose: bool = False) -> None:
    """Configure console and file logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging_config = config.get("logging", {})
    configured_level = str(logging_config.get("level", "INFO")).upper()
    level = (
        logging.DEBUG if verbose else getattr(logging, configured_level, logging.INFO)
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "gatekeeper.log", encoding="utf-8"),
        ],
        force=True,
    )


class CameraManager:
    """Manage the Raspberry Pi camera through Picamera2."""

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._camera: Any | None = None
        self._pixel_format = "RGB888"

    def initialize(self) -> None:
        """Initialize and start Picamera2 if needed."""
        if self._camera is not None:
            return

        from picamera2 import Picamera2

        self._camera = Picamera2()
        configuration = self._camera.create_still_configuration(
            main={"size": (self.width, self.height), "format": self._pixel_format}
        )
        self._camera.configure(configuration)
        self._camera.start()
        time.sleep(max(0.1, 1 / max(self.fps, 1)))

    def health_check(self) -> bool:
        """Return True when the camera can be initialized."""
        self.initialize()
        return self._camera is not None

    def capture_frame(self) -> Any:
        """Capture a frame while keeping the camera open."""
        self.initialize()
        if self._camera is None:
            raise RuntimeError("Picamera2 is not initialized.")
        return self._camera.capture_array()

    def capture(self, output_path: Path = LATEST_IMAGE) -> Path:
        """Capture a JPEG image to the requested path."""
        self.initialize()
        if self._camera is None:
            raise RuntimeError("Picamera2 is not initialized.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._camera.capture_file(str(output_path))
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"Captured image is empty or missing: {output_path}")
        return output_path

    def save_latest(self) -> Path:
        """Capture and save images/latest.jpg."""
        return self.capture(LATEST_IMAGE)

    def get_info(self) -> dict[str, Any]:
        """Return camera backend, resolution, pixel format, and model details."""
        model = "unknown"
        if self._camera is not None:
            properties = getattr(self._camera, "camera_properties", {}) or {}
            model = properties.get("Model", model)
        return {
            "backend": CAMERA_BACKEND,
            "resolution": f"{self.width}x{self.height}",
            "pixel_format": self._pixel_format,
            "camera_model": model,
        }

    def stop(self) -> None:
        """Stop and close Picamera2 resources during shutdown."""
        if self._camera is not None:
            self._camera.stop()
            self._camera.close()
            self._camera = None


class EventDatabase:
    """Persist GateKeeper AI events to SQLite."""

    def __init__(self, path: Path = DATABASE_PATH) -> None:
        self.path = path
        self.connection: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Open the database and create the events table."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL,
                event_id TEXT,
                start_time TEXT,
                end_time TEXT,
                duration REAL,
                max_contour_area REAL,
                image_start TEXT,
                image_end TEXT
            )
            """)
        self._ensure_event_columns()
        self.connection.commit()

    def _ensure_event_columns(self) -> None:
        """Add motion event columns to existing databases when necessary."""
        if self.connection is None:
            raise RuntimeError("Database is not initialized.")
        existing = {
            row[1] for row in self.connection.execute("PRAGMA table_info(events)")
        }
        columns = {
            "event_id": "TEXT",
            "start_time": "TEXT",
            "end_time": "TEXT",
            "duration": "REAL",
            "max_contour_area": "REAL",
            "image_start": "TEXT",
            "image_end": "TEXT",
        }
        for name, definition in columns.items():
            if name not in existing:
                self.connection.execute(
                    f"ALTER TABLE events ADD COLUMN {name} {definition}"
                )

    def insert_event(
        self,
        timestamp: str,
        event_type: str,
        details: str,
        *,
        event_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        duration: float | None = None,
        max_contour_area: float | None = None,
        image_start: str | None = None,
        image_end: str | None = None,
    ) -> None:
        """Insert a single event row."""
        if self.connection is None:
            raise RuntimeError("Database is not initialized.")
        self.connection.execute(
            """
            INSERT INTO events (
                timestamp, event_type, details, event_id, start_time, end_time,
                duration, max_contour_area, image_start, image_end
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                event_type,
                details,
                event_id,
                start_time,
                end_time,
                duration,
                max_contour_area,
                image_start,
                image_end,
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None


class MotionDetector:
    """Detect motion between consecutive camera frames using OpenCV."""

    def __init__(self, min_area: int = 1000, threshold: int = 25) -> None:
        self.min_area = min_area
        self.threshold = threshold
        self.previous_frame: Any | None = None

    def _prepare_frame(self, frame: Any) -> Any:
        import cv2

        gray = (
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
        )
        return cv2.GaussianBlur(gray, (21, 21), 0)

    def detect(self, frame: Any) -> tuple[bool, float]:
        """Return whether motion exists and the largest contour area."""
        import cv2

        prepared = self._prepare_frame(frame)
        if self.previous_frame is None:
            self.previous_frame = prepared
            return False, 0.0

        delta = cv2.absdiff(self.previous_frame, prepared)
        self.previous_frame = prepared
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        areas = [float(cv2.contourArea(contour)) for contour in contours]
        max_area = max(areas, default=0.0)
        return max_area >= self.min_area, max_area

    @staticmethod
    def save_motion_image(frame: Any, output_path: Path) -> Path:
        """Save the motion frame without overwriting latest.jpg."""
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = (
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if len(frame.shape) == 3 else frame
        )
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Unable to save motion image: {output_path}")
        return output_path


@contextmanager
def pid_file(path: Path = PID_FILE):
    """Create a runtime PID file and remove it on shutdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def print_startup_banner(version: str, git_commit: str) -> None:
    """Print release and runtime details to stdout."""
    print("========================================")
    print(f" GateKeeper AI v{version}")
    print("========================================")
    print(f"Build: {git_commit}")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.system()}")
    print(f"Camera backend: {CAMERA_BACKEND}")
    print(f"OpenCV: {get_opencv_version()}")
    print()


def print_startup_status() -> None:
    """Print the startup status checklist."""
    for item in ("Configuration", "Logger", "Database", "Camera", "Capture"):
        print(f"[OK] {item}")
    print("Waiting...")


def log_startup_metadata(
    logger: logging.Logger,
    version: str,
    git_commit: str,
    camera_info: dict[str, Any],
    capture_time: str,
    image_path: Path,
) -> None:
    """Write startup metadata to logs/gatekeeper.log."""
    logger.info("Version: %s", version)
    logger.info("Git commit: %s", git_commit)
    logger.info("Python version: %s", platform.python_version())
    logger.info("Platform: %s", platform.system())
    logger.info("Camera backend: %s", camera_info["backend"])
    logger.info("Camera resolution: %s", camera_info["resolution"])
    logger.info("Capture time: %s", capture_time)
    logger.info("Image path: %s", image_path)
    logger.info("Image size: %d bytes", image_path.stat().st_size)


def capture_and_log_latest(camera: CameraManager, logger: logging.Logger) -> Path:
    """Capture latest.jpg and log the UTC capture time and size."""
    captured_at = datetime.now(timezone.utc).isoformat()
    output_path = camera.save_latest()
    image_size = output_path.stat().st_size
    logger.info("Capture time: %s", captured_at)
    logger.info("Image path: %s", output_path)
    logger.info("Image size: %d bytes", image_size)
    return output_path


def motion_image_path(timestamp: str, marker: str) -> Path:
    """Build a filesystem-safe motion event image path from an ISO timestamp."""
    safe_timestamp = timestamp.replace(":", "").replace("+", "Z")
    return IMAGE_DIR / f"motion_{marker}_{safe_timestamp}.jpg"


class MotionEventManager:
    """Convert frame-level detections into start/end motion events."""

    IDLE = "IDLE"
    MOTION_STARTED = "MOTION_STARTED"
    MOTION_ACTIVE = "MOTION_ACTIVE"
    MOTION_FINISHED = "MOTION_FINISHED"

    def __init__(
        self,
        detector: MotionDetector,
        database: EventDatabase | None,
        logger: logging.Logger,
        end_delay_seconds: float = 2,
    ) -> None:
        self.detector = detector
        self.database = database
        self.logger = logger
        self.end_delay_seconds = end_delay_seconds
        self.state = self.IDLE
        self.event_id: str | None = None
        self.start_time: datetime | None = None
        self.last_motion_time: datetime | None = None
        self.max_contour_area = 0.0
        self.image_start: Path | None = None

    def process_frame(self, frame: Any) -> None:
        """Process one frame and create events only on motion boundaries."""
        motion, contour_area = self.detector.detect(frame)
        now = datetime.now(timezone.utc)
        if motion:
            if self.state in {self.IDLE, self.MOTION_FINISHED}:
                self._start(frame, now, contour_area)
            else:
                self.state = self.MOTION_ACTIVE
                self.max_contour_area = max(self.max_contour_area, contour_area)
                self.logger.info("Motion active")
            self.last_motion_time = now
        elif (
            self.state in {self.MOTION_STARTED, self.MOTION_ACTIVE}
            and self.last_motion_time is not None
        ):
            if (now - self.last_motion_time).total_seconds() >= self.end_delay_seconds:
                self._finish(frame, now)

    def _start(self, frame: Any, now: datetime, contour_area: float) -> None:
        self.state = self.MOTION_STARTED
        self.event_id = str(uuid.uuid4())
        self.start_time = now
        self.last_motion_time = now
        self.max_contour_area = contour_area
        self.image_start = self.detector.save_motion_image(
            frame, motion_image_path(now.isoformat(), "START")
        )
        self.logger.info("Motion started")
        if self.database is not None:
            self.database.insert_event(
                now.isoformat(),
                "MOTION_START",
                "Motion started",
                event_id=self.event_id,
                start_time=now.isoformat(),
                max_contour_area=self.max_contour_area,
                image_start=str(self.image_start),
            )

    def _finish(self, frame: Any, now: datetime) -> None:
        self.state = self.MOTION_FINISHED
        image_end = self.detector.save_motion_image(
            frame, motion_image_path(now.isoformat(), "END")
        )
        duration = (
            (now - self.start_time).total_seconds()
            if self.start_time is not None
            else 0.0
        )
        self.logger.info("Motion finished")
        self.logger.info("Duration")
        self.logger.info("Max contour area")
        if self.database is not None:
            self.database.insert_event(
                now.isoformat(),
                "MOTION_END",
                "Motion finished",
                event_id=self.event_id,
                start_time=self.start_time.isoformat() if self.start_time else None,
                end_time=now.isoformat(),
                duration=duration,
                max_contour_area=self.max_contour_area,
                image_start=str(self.image_start) if self.image_start else None,
                image_end=str(image_end),
            )
        self.state = self.IDLE
        self.event_id = None
        self.start_time = None
        self.last_motion_time = None
        self.max_contour_area = 0.0
        self.image_start = None


def run_until_interrupted(
    camera: CameraManager,
    logger: logging.Logger,
    stop_event: Any | None = None,
    install_signal_handlers: bool = True,
    fps: int = 1,
    motion_enabled: bool = True,
    motion_detector: MotionDetector | None = None,
    database: EventDatabase | None = None,
    motion_event_manager: MotionEventManager | None = None,
    end_delay_seconds: float = 2,
) -> None:
    """Capture frames continuously until shutdown, detecting motion when enabled."""
    running = True

    def request_shutdown(signum: int, _frame: Any) -> None:
        nonlocal running
        logger.info("Received signal %s; shutting down camera.", signum)
        running = False
        if stop_event is not None:
            stop_event.set()

    previous_sigint: Any | None = None
    previous_sigterm: Any | None = None
    if install_signal_handlers:
        previous_sigint = signal.signal(signal.SIGINT, request_shutdown)
        previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
    interval = 1 / max(fps, 1)
    motion_detector = motion_detector or MotionDetector()
    motion_event_manager = motion_event_manager or MotionEventManager(
        motion_detector, database, logger, end_delay_seconds
    )
    try:
        logger.info("Application remains running; press Ctrl+C to stop.")
        while running and (stop_event is None or not stop_event.is_set()):
            loop_started = time.monotonic()
            frame = camera.capture_frame()
            if motion_enabled:
                motion_event_manager.process_frame(frame)
            elapsed = time.monotonic() - loop_started
            time.sleep(max(0, interval - elapsed))
    finally:
        camera.stop()
        if database is not None:
            database.close()
            logger.info("Database closed cleanly.")
        logger.info("Camera shut down cleanly.")
        if install_signal_handlers:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv: list[str] | None = None) -> int:
    """Start the GateKeeper AI application."""
    parser = argparse.ArgumentParser(description="GateKeeper AI startup")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check", action="store_true", help="Validate startup metadata and exit."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    version = get_version()
    git_commit = get_git_commit()
    config = load_config(args.config)
    configure_logging(config, args.verbose)
    logger = logging.getLogger("gatekeeper")

    camera_config = config.get("camera", {})
    fps = int(camera_config.get("fps", 1))
    camera = CameraManager(
        width=int(camera_config.get("width", 1640)),
        height=int(camera_config.get("height", 1232)),
        fps=fps,
    )
    database = EventDatabase()
    motion_config = config.get("motion", {})
    motion_detector = MotionDetector(
        min_area=int(motion_config.get("min_area", 1000)),
        threshold=int(motion_config.get("threshold", 25)),
    )
    end_delay_seconds = float(motion_config.get("end_delay_seconds", 2))

    try:
        print_startup_banner(version, git_commit)
        if args.check:
            logger.info("Startup check completed successfully.")
            return 0

        with pid_file():
            database.initialize()
            camera.health_check()
            image_path = camera.save_latest()
            capture_time = datetime.now(timezone.utc).isoformat()
            log_startup_metadata(
                logger, version, git_commit, camera.get_info(), capture_time, image_path
            )
            print_startup_status()
            run_until_interrupted(
                camera,
                logger,
                fps=fps,
                motion_enabled=bool(motion_config.get("enabled", True)),
                motion_detector=motion_detector,
                database=database,
                end_delay_seconds=end_delay_seconds,
            )
    except Exception:
        camera.stop()
        database.close()
        logger.exception("Startup failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
