from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import main


class FakeCamera:
    def __init__(
        self, payload: bytes = b"jpeg-bytes", frames: list[object] | None = None
    ) -> None:
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
        frame = (
            self.frames[min(self.capture_count, len(self.frames) - 1)]
            if self.frames
            else None
        )
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


class FakePlateDetector:
    def __init__(self, detection=None) -> None:
        self.detection = detection
        self.calls = 0
        self.crops = 0

    def initialize(self) -> None:
        return None

    def detect(self, frame):
        self.calls += 1
        return self.detection

    def crop(self, frame, detection):
        self.crops += 1
        return frame


class FakeMotionDetector:
    def __init__(self, detections: list[tuple[bool, float]] | None = None) -> None:
        self.calls = 0
        self.saved_paths: list[Path] = []
        self.detections = detections or [(False, 0.0), (True, 1234.0)]

    def detect(self, frame: object) -> tuple[bool, float]:
        result = self.detections[min(self.calls, len(self.detections) - 1)]
        self.calls += 1
        return result

    def save_motion_image(self, frame: object, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"motion-jpeg")
        self.saved_paths.append(output_path)
        return output_path


def test_version_comes_from_version_file():
    assert main.get_version() == "0.5.2"
    assert main.get_version() == main.VERSION_FILE.read_text(encoding="utf-8").strip()


def test_startup_banner_contains_release_version(capsys):
    version = main.get_version()
    main.print_startup_banner(version, "development")

    output = capsys.readouterr().out

    assert "GateKeeper AI v0.5.2" in output
    assert "Build: development" in output
    assert f"Camera backend: {main.CAMERA_BACKEND}" in output


def test_default_config_contains_motion_detection_settings():
    config = main.load_config()

    assert config["camera"]["fps"] == 1
    assert config["motion"] == {
        "enabled": True,
        "threshold": 25,
        "min_area": 1000,
        "end_delay_seconds": 2,
    }


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


def test_event_database_inserts_motion_event_metadata(tmp_path):
    database = main.EventDatabase(tmp_path / "gatekeeper.db")
    database.initialize()
    database.insert_event(
        "2026-08-03T00:00:01+00:00",
        "MOTION_END",
        "Motion finished",
        event_id="event-1",
        start_time="2026-08-03T00:00:00+00:00",
        end_time="2026-08-03T00:00:01+00:00",
        duration=1.0,
        max_contour_area=1234.0,
        image_start="images/motion_START_2026.jpg",
        image_end="images/motion_END_2026.jpg",
    )
    database.close()

    with sqlite3.connect(tmp_path / "gatekeeper.db") as connection:
        row = connection.execute("""
            SELECT event_type, event_id, start_time, end_time, duration,
                   max_contour_area, image_start, image_end
            FROM events
            """).fetchone()

    assert row == (
        "MOTION_END",
        "event-1",
        "2026-08-03T00:00:00+00:00",
        "2026-08-03T00:00:01+00:00",
        1.0,
        1234.0,
        "images/motion_START_2026.jpg",
        "images/motion_END_2026.jpg",
    )


def test_event_database_inserts_plate_record(tmp_path):
    database = main.EventDatabase(tmp_path / "gatekeeper.db")
    database.initialize()
    database.insert_plate(
        "event-1",
        "images/plate_2026.jpg",
        0.75,
        "2026-08-03T00:00:01+00:00",
    )
    database.close()

    with sqlite3.connect(tmp_path / "gatekeeper.db") as connection:
        row = connection.execute(
            "SELECT event_id, image_path, confidence, created_at FROM plates"
        ).fetchone()

    assert row == (
        "event-1",
        "images/plate_2026.jpg",
        0.75,
        "2026-08-03T00:00:01+00:00",
    )


def test_plate_detector_detects_and_crops_quadrilateral_candidate():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    detector = main.PlateDetector(min_area=100)
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40, 45), (190, 75), (255, 255, 255), 2)

    detection = detector.detect(frame)

    assert detection is not None
    assert detection.confidence > 0
    x, y, width, height = detection.bounding_box
    assert width / height >= 2
    crop = detector.crop(frame, detection)
    assert crop.shape[0] == height
    assert crop.shape[1] == width


def test_motion_start_runs_plate_detector_saves_crop_and_records_database(
    tmp_path, caplog
):
    np = pytest.importorskip("numpy")
    frame = np.zeros((20, 60, 3), dtype=np.uint8)
    detector = FakeMotionDetector([(True, 1234.0)])
    detection = main.PlateDetection((5, 5, 30, 10), 0.8, object())
    plate_detector = FakePlateDetector(detection)
    database = main.EventDatabase(tmp_path / "gatekeeper.db")
    database.initialize()
    logger = logging.getLogger("test-gatekeeper")

    original_image_dir = main.IMAGE_DIR
    main.IMAGE_DIR = tmp_path / "images"
    manager = main.MotionEventManager(
        detector,
        database,
        logger,
        end_delay_seconds=0.05,
        plate_detector=plate_detector,
    )
    try:
        with caplog.at_level(logging.INFO, logger="test-gatekeeper"):
            manager.process_frame(frame)
    finally:
        main.IMAGE_DIR = original_image_dir
        database.close()

    assert plate_detector.calls == 1
    assert plate_detector.crops == 1
    assert "Plate detected" in caplog.text
    assert "Confidence: 0.800" in caplog.text
    assert "Bounding box: (5, 5, 30, 10)" in caplog.text
    assert "Crop path:" in caplog.text

    with sqlite3.connect(tmp_path / "gatekeeper.db") as connection:
        row = connection.execute(
            "SELECT event_id, image_path, confidence FROM plates"
        ).fetchone()

    assert row is not None
    assert row[0]
    assert Path(row[1]).is_file()
    assert Path(row[1]).name.startswith("plate_")
    assert row[2] == 0.8


def test_run_until_interrupted_captures_frames_detects_motion_and_stops_cleanly(
    tmp_path, caplog
):
    camera = FakeCamera(frames=[object(), object(), object(), object(), object()])
    detector = FakeMotionDetector(
        [(False, 0.0), (True, 1234.0), (True, 1500.0), (False, 0.0), (False, 0.0)]
    )
    database = main.EventDatabase(tmp_path / "gatekeeper.db")
    database.initialize()
    logger = logging.getLogger("test-gatekeeper")
    stop_event = threading.Event()

    original_image_dir = main.IMAGE_DIR
    main.IMAGE_DIR = tmp_path / "images"
    thread = threading.Thread(
        target=main.run_until_interrupted,
        args=(camera, logger, stop_event, False),
        kwargs={
            "fps": 20,
            "motion_enabled": True,
            "motion_detector": detector,
            "database": database,
            "end_delay_seconds": 0.05,
        },
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
    assert len(detector.saved_paths) == 2
    assert detector.saved_paths[0].name.startswith("motion_START_")
    assert detector.saved_paths[1].name.startswith("motion_END_")
    assert all(path.name != "latest.jpg" for path in detector.saved_paths)
    assert "Application remains running" in caplog.text
    assert "Motion started" in caplog.text
    assert "Motion active" in caplog.text
    assert "Motion finished" in caplog.text
    assert "Duration" in caplog.text
    assert "Max contour area" in caplog.text
    assert "Camera shut down cleanly" in caplog.text

    with sqlite3.connect(tmp_path / "gatekeeper.db") as connection:
        rows = connection.execute(
            "SELECT event_type, duration, max_contour_area, image_start, image_end FROM events ORDER BY id"
        ).fetchall()

    assert len(rows) == 2
    assert rows[0][0] == "MOTION_START"
    assert rows[1][0] == "MOTION_END"
    assert rows[1][1] is not None and rows[1][1] > 0
    assert rows[1][2] == 1500.0
    assert rows[0][3] is not None
    assert rows[1][4] is not None


def test_default_config_contains_debug_vision_settings():
    config = main.load_config()

    assert config["debug"] == {
        "enabled": True,
        "live_preview": True,
        "save_annotated_frames": True,
    }


def test_debug_vision_disables_preview_when_headless(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(main.platform, "system", lambda: "Linux")

    debug_vision = main.DebugVision(enabled=True, live_preview=True)

    assert debug_vision.enabled is True
    assert debug_vision.live_preview is False


def test_debug_vision_preview_can_be_enabled(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(main.platform, "system", lambda: "Linux")

    debug_vision = main.DebugVision(enabled=True, live_preview=True)

    assert debug_vision.live_preview is True


def test_debug_vision_saves_annotated_frame_and_draws_bounding_box(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    detection = main.PlateDetection((20, 30, 70, 20), 0.9, object())
    debug_vision = main.DebugVision(enabled=True, save_annotated_frames=True)

    original_debug_dir = main.DEBUG_DIR
    main.DEBUG_DIR = tmp_path / "debug"
    try:
        annotated = debug_vision.annotate(
            frame,
            motion_state=main.MotionEventManager.MOTION_STARTED,
            fps=12.5,
            camera_resolution="160x80",
            timestamp=main.datetime(2026, 8, 3, tzinfo=main.timezone.utc),
            plate_detection=detection,
        )
        output_path = debug_vision.save_frame(
            annotated, main.datetime(2026, 8, 3, tzinfo=main.timezone.utc)
        )
    finally:
        main.DEBUG_DIR = original_debug_dir

    assert output_path.is_file()
    assert output_path.name.startswith("frame_")
    assert output_path.suffix == ".jpg"
    assert annotated[30, 20].any()
    saved = cv2.imread(str(output_path))
    assert saved is not None


def test_debug_vision_q_key_stops_run_cleanly(monkeypatch, tmp_path, caplog):
    np = pytest.importorskip("numpy")
    camera = FakeCamera(frames=[np.zeros((20, 20, 3), dtype=np.uint8)])
    logger = logging.getLogger("test-gatekeeper")
    stop_event = threading.Event()
    debug_vision = main.DebugVision(enabled=True, live_preview=True, logger=logger)
    debug_vision.live_preview = True
    monkeypatch.setattr(
        main.cv2 if hasattr(main, "cv2") else __import__("cv2"),
        "imshow",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(__import__("cv2"), "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(__import__("cv2"), "destroyWindow", lambda name: None)

    with caplog.at_level(logging.INFO, logger="test-gatekeeper"):
        main.run_until_interrupted(
            camera,
            logger,
            stop_event,
            False,
            fps=20,
            motion_enabled=False,
            database=main.EventDatabase(tmp_path / "gatekeeper.db"),
            debug_vision=debug_vision,
        )

    assert camera.stopped
    assert "Debug Vision quit requested" in caplog.text
    assert "Camera shut down cleanly" in caplog.text


def test_default_config_contains_display_settings():
    config = main.load_config()

    assert config["display"] == {
        "enabled": True,
        "fullscreen": False,
        "window_name": "GateKeeper AI",
        "show_fps": True,
        "show_motion": True,
        "show_plate_box": True,
        "show_confidence": True,
        "show_timestamp": True,
        "save_snapshot_key": "s",
    }


def test_display_manager_disables_when_headless(monkeypatch, caplog):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(main.platform, "system", lambda: "Linux")
    monkeypatch.setattr("src.gatekeeper.display_manager.platform.system", lambda: "Linux")
    logger = logging.getLogger("test-gatekeeper")

    with caplog.at_level(logging.INFO, logger="test-gatekeeper"):
        display = main.DisplayManager(enabled=True, logger=logger)

    assert display.requested_enabled is True
    assert display.enabled is False
    assert "No graphical display detected. Running headless." in caplog.text


def test_display_manager_draws_green_plate_overlay_and_snapshot(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.zeros((90, 180, 3), dtype=np.uint8)
    detection = main.PlateDetection((25, 35, 80, 20), 0.85, object())
    display = main.DisplayManager(enabled=False, version="0.5.2", git_commit="abc123")
    monkeypatch.setattr("src.gatekeeper.display_manager.SNAPSHOT_DIR", tmp_path / "snapshots")

    annotated = display.draw_overlays(
        frame,
        motion_state=main.MotionEventManager.MOTION_STARTED,
        fps=10.0,
        timestamp=main.datetime(2026, 8, 3, tzinfo=main.timezone.utc),
        plate_detection=detection,
    )
    output_path = display.save_snapshot(
        annotated, main.datetime(2026, 8, 3, tzinfo=main.timezone.utc)
    )

    assert annotated[35, 25, 1] > 0
    assert output_path.is_file()
    assert output_path.name.startswith("snapshot_")
    assert output_path.suffix == ".jpg"
    assert cv2.imread(str(output_path)) is not None


def test_display_manager_keyboard_shortcuts(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    display = main.DisplayManager(enabled=False)
    monkeypatch.setattr("src.gatekeeper.display_manager.SNAPSHOT_DIR", tmp_path / "snapshots")

    display.handle_key(ord("d"), frame)
    assert display.overlays_enabled is False
    display.handle_key(ord("f"), frame)
    assert display.fullscreen is True
    display.handle_key(ord("s"), frame)
    assert list((tmp_path / "snapshots").glob("snapshot_*.jpg"))
    display.handle_key(ord("q"), frame)
    assert display.quit_requested is True


def test_display_manager_window_creation_and_close(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr("src.gatekeeper.display_manager.platform.system", lambda: "Linux")
    cv2 = pytest.importorskip("cv2")
    calls = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: calls.append(("named", args)))
    monkeypatch.setattr(cv2, "setWindowProperty", lambda *args: calls.append(("prop", args)))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: calls.append(("destroy", args)))

    display = main.DisplayManager(enabled=True, fullscreen=True, window_name="Test")
    display.create_window()
    display.close()

    assert calls[0][0] == "named"
    assert any(call[0] == "prop" for call in calls)
    assert calls[-1][0] == "destroy"
