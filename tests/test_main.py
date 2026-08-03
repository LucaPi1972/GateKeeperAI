from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import main


class FakeCamera:
    def __init__(self, payload: bytes = b"jpeg-bytes") -> None:
        self.payload = payload
        self.stopped = False
        self.initialized = False

    def initialize(self) -> None:
        self.initialized = True

    def health_check(self) -> bool:
        self.initialize()
        return True

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


def test_version_comes_from_version_file():
    assert main.get_version() == main.VERSION_FILE.read_text(encoding="utf-8").strip()


def test_startup_banner_contains_release_version(capsys):
    version = main.get_version()
    main.print_startup_banner(version, "development")

    output = capsys.readouterr().out

    assert f"GateKeeper AI v{main.get_version()}" in output
    assert "Build: development" in output
    assert f"Camera backend: {main.CAMERA_BACKEND}" in output


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


def test_run_until_interrupted_keeps_running_and_stops_camera_cleanly(caplog):
    camera = FakeCamera()
    logger = logging.getLogger("test-gatekeeper")
    stop_event = threading.Event()
    thread = threading.Thread(
        target=main.run_until_interrupted,
        args=(camera, logger, stop_event, False),
    )

    with caplog.at_level(logging.INFO, logger="test-gatekeeper"):
        thread.start()
        time.sleep(0.7)
        assert thread.is_alive()
        stop_event.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert camera.stopped
    assert "Application remains running" in caplog.text
    assert "Camera shut down cleanly" in caplog.text
