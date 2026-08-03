from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import main


class FakeCamera:
    def __init__(self, payload: bytes = b"jpeg-bytes", frames: list[object] | None = None) -> None:
        self.payload = payload
        self.stopped = False
        self.initialized = False
        self.frames = frames or []
        self.capture_count = 0

    def initialize(self) -> None:
        self.initialized = True

    def health_check(self) -> bool:
        self.initialize()
        return True

    def capture_frame(self):
        self.initialize()
        frame = self.frames[min(self.capture_count, len(self.frames) - 1)] if self.frames else None
        self.capture_count += 1
        return frame

    def capture(self, output_path: Path = main.LATEST_IMAGE) -> Path:
        self.initialize()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self.payload)
        return output_path

    def save_latest(self) -> Path:
        return self.capture(main.LATEST_IMAGE)

    def get_info(self) -> dict[str, str]:
        return {
            "backend": main.CAMERA_BACKEND,
            "resolution": "1640x1232",
            "pixel_format": "RGB888",
            "camera_model": "fake-camera",
        }

    def stop(self) -> None:
        self.stopped = True


class FakeMotionDetector:
    def __init__(self) -> None:
        self.calls = 0
        self.saved_paths: list[Path] = []

    def detect(self, frame: object) -> tuple[bool, float]:
        self.calls += 1
        return self.calls >= 2, 1234.0

    def save_motion_image(self, frame: object, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"motion-jpeg")
        self.saved_paths.append(output_path)
        return output_path


def test_version_comes_from_version_file():
    assert main.get_version() == "0.4.0"
    assert main.get_version() == main.VERSION_FILE.read_text(encoding="utf-8").strip()


def test_startup_banner_contains_release_version(capsys):
    version = main.get_version()
    main.print_startup_banner(version, "development")

    output = capsys.readouterr().out

    assert "GateKeeper AI v0.4.0" in output
    assert "Build: development" in output
    assert f"Camera backend: {main.CAMERA_BACKEND}" in output


def test_default_config_contains_motion_detection_settings():
    config = main.load_config()

    assert config["camera"]["fps"] == 1
    assert config["motion"] == {"enabled": True, "min_area": 1000, "threshold": 25}


def test_capture_and_log_latest_creates_non_empty_jpeg_and_logs_time(tmp_path, caplog):
    image_path = tmp_path / "images" / "latest.jpg"
    camera = FakeCamera()
    logger = logging.getLogger("test-gatekeeper")

    original_latest = main.LATEST_IMAGE
    main.LATEST_IMAGE = image_path
    try:
        with caplog.at_level(logging.INFO, logger="test-gatekeeper"):
            captured = main.capture_and_log_latest(camera, logger)
    finally:
        main.LATEST_IMAGE = original_latest

    assert captured == image_path
    assert image_path.is_file()
    assert image_path.stat().st_size > 0
    assert "Capture time:" in caplog.text
    assert "Image path:" in caplog.text
    assert "Image size:" in caplog.text


def test_pid_file_exists_and_is_removed(tmp_path):
    pid_path = tmp_path / "runtime" / "gatekeeper.pid"

    with main.pid_file(pid_path):
        assert pid_path.is_file()
        assert pid_path.read_text(encoding="utf-8").strip().isdigit()

    assert not pid_path.exists()


def test_motion_detector_detects_large_contour():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    detector = main.MotionDetector(min_area=1000, threshold=25)
    first = np.zeros((120, 120, 3), dtype=np.uint8)
    second = first.copy()
    cv2.rectangle(second, (20, 20), (90, 90), (255, 255, 255), -1)

    assert detector.detect(first) == (False, 0.0)
    motion, area = detector.detect(second)

    assert motion is True
    assert area >= 1000


def test_event_database_inserts_motion_event(tmp_path):
    database = main.EventDatabase(tmp_path / "gatekeeper.db")
    database.initialize()
    database.insert_event("2026-08-03T00:00:00+00:00", "MOTION", "Motion detected")
    database.close()

    with sqlite3.connect(tmp_path / "gatekeeper.db") as connection:
        row = connection.execute("SELECT timestamp, event_type, details FROM events").fetchone()

    assert row == ("2026-08-03T00:00:00+00:00", "MOTION", "Motion detected")


def test_run_until_interrupted_captures_frames_detects_motion_and_stops_cleanly(tmp_path, caplog):
    camera = FakeCamera(frames=[object(), object(), object()])
    detector = FakeMotionDetector()
    database = main.EventDatabase(tmp_path / "gatekeeper.db")
    database.initialize()
    logger = logging.getLogger("test-gatekeeper")
    stop_event = threading.Event()

    original_image_dir = main.IMAGE_DIR
    main.IMAGE_DIR = tmp_path / "images"
    thread = threading.Thread(
        target=main.run_until_interrupted,
        args=(camera, logger, stop_event, False),
        kwargs={"fps": 20, "motion_enabled": True, "motion_detector": detector, "database": database},
    )

    try:
        with caplog.at_level(logging.INFO, logger="test-gatekeeper"):
            thread.start()
            time.sleep(0.2)
            assert thread.is_alive()
            stop_event.set()
            thread.join(timeout=2)
    finally:
        main.IMAGE_DIR = original_image_dir

    assert not thread.is_alive()
    assert camera.stopped
    assert camera.capture_count >= 2
    assert detector.saved_paths
    assert detector.saved_paths[0].name.startswith("motion_")
    assert detector.saved_paths[0].name != "latest.jpg"
    assert "Application remains running" in caplog.text
    assert "Motion detected" in caplog.text
    assert "Camera shut down cleanly" in caplog.text

    with sqlite3.connect(tmp_path / "gatekeeper.db") as connection:
        row = connection.execute("SELECT event_type, details FROM events").fetchone()

    assert row[0] == "MOTION"
    assert "contour_area=1234.0" in row[1]
