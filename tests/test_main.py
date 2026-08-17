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
    assert main.get_version() == "0.7.1"
    assert main.get_version() == main.VERSION_FILE.read_text(encoding="utf-8").strip()


def test_startup_banner_contains_release_version(capsys):
    version = main.get_version()
    main.print_startup_banner(version, "development")

    output = capsys.readouterr().out

    assert "GateKeeper AI v0.7.1" in output
    assert "Build: development" in output
    assert f"Camera backend: {main.CAMERA_BACKEND}" in output


def test_default_config_contains_motion_detection_settings():
    config = main.load_config()

    assert config["camera"] == {
        "width": 1640,
        "height": 1232,
        "fps": 1,
        "rotation": 180,
        "flip_horizontal": False,
        "flip_vertical": False,
        "diagnostics": True,
        "pipeline": "rgb",
        "controls": {
            "AwbEnable": True,
            "AeEnable": True,
            "Brightness": 0.0,
            "Contrast": 1.0,
            "Saturation": 1.0,
            "Sharpness": 1.0,
            "ExposureValue": 0.0,
            "AnalogueGain": 1.0,
        },
    }
    assert config["motion"] == {
        "enabled": True,
        "threshold": 25,
        "min_area": 1000,
        "end_delay_seconds": 2,
    }
    assert config["plate_calibration"] == {
        "enabled": True,
        "save_frames": True,
        "show_candidates": True,
        "show_rejected": True,
        "show_metrics": True,
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


def test_default_config_contains_preview_settings():
    config = main.load_config()

    assert config["preview"] == {"pipeline": "bgr"}


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
        "show_crosshair": True,
        "show_grid": True,
        "show_safe_area": True,
        "show_bbox": True,
        "show_status": True,
        "detection_area": {"x": 164, "y": 123, "width": 1312, "height": 986},
        "plate_ideal_area": {"x": 574, "y": 493, "width": 492, "height": 246},
    }


def test_default_config_contains_plate_detector_thresholds():
    config = main.load_config()

    assert config["plate_detector"] == {
        "debug": True,
        "confidence_threshold": 0.70,
        "aspect_ratio_min": 3.5,
        "aspect_ratio_max": 6.5,
        "min_area": 2500,
        "max_area": 150000,
        "min_rectangularity": 0.80,
        "max_rotation": 15,
        "border_margin": 20,
    }


def test_display_manager_disables_when_headless(monkeypatch, caplog):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(main.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        "src.gatekeeper.display_manager.platform.system", lambda: "Linux"
    )
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
    display = main.DisplayManager(enabled=False, version="0.6.9", git_commit="abc123")
    monkeypatch.setattr(
        "src.gatekeeper.display_manager.SNAPSHOT_DIR", tmp_path / "snapshots"
    )

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
    monkeypatch.setattr(
        "src.gatekeeper.display_manager.SNAPSHOT_DIR", tmp_path / "snapshots"
    )

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
    monkeypatch.setattr(
        "src.gatekeeper.display_manager.platform.system", lambda: "Linux"
    )
    cv2 = pytest.importorskip("cv2")
    calls = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: calls.append(("named", args)))
    monkeypatch.setattr(
        cv2, "setWindowProperty", lambda *args: calls.append(("prop", args))
    )
    monkeypatch.setattr(
        cv2, "destroyWindow", lambda *args: calls.append(("destroy", args))
    )

    display = main.DisplayManager(enabled=True, fullscreen=True, window_name="Test")
    display.create_window()
    display.close()

    assert calls[0][0] == "named"
    assert any(call[0] == "prop" for call in calls)
    assert calls[-1][0] == "destroy"



def test_index_html_displays_release_and_no_pipeline_selection_buttons():
    html = Path("src/gatekeeper/web/templates/index.html").read_text(encoding="utf-8")

    assert "GateKeeper AI 0.7.1" in html
    assert "Plate Calibration" in html
    assert "Save Calibration Frame" in html
    assert "Live Preview Pipeline: BGR" in html
    for forbidden in ("Use RGB", "Use BGR", "Use RAW", "Use SWAP RB", "preview pipeline selection", "pipeline selection buttons"):
        assert forbidden not in html

def test_default_config_contains_web_live_preview_settings():
    config = main.load_config()

    assert config["web"] == {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8080,
        "stream_fps": 5,
    }


def test_live_preview_state_serves_shared_frame_and_metadata():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = np.zeros((5, 10, 3), dtype=np.uint8)
    detection = main.PlateDetection((1, 2, 10, 5), 0.77, object())
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")

    state.update_frame(
        frame,
        motion_state=main.MotionEventManager.MOTION_STARTED,
        fps=5.5,
        resolution="30x20",
        plate_detection=detection,
        plate_crop=crop,
    )
    state.record_motion_event({"type": "MOTION_START", "timestamp": "now"})

    status = state.snapshot()
    assert status == {
        "version": "0.6.9",
        "camera": main.CAMERA_BACKEND,
        "motion_state": main.MotionEventManager.MOTION_STARTED,
        "fps": 5.5,
        "resolution": "30x20",
        "git_commit": "abc123",
    }
    assert state.events_snapshot()[0]["type"] == "MOTION_START"
    assert (
        cv2.imdecode(
            np.frombuffer(state.latest_frame_jpeg(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        is not None
    )
    assert (
        cv2.imdecode(
            np.frombuffer(state.latest_plate_jpeg(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        is not None
    )

class FakeCrop:
    shape = (5, 10, 3)
    dtype = "uint8"
    strides = (30, 3, 1)


class FakeFrame:
    shape = (20, 30, 3)


class CountingJpegCameraManager:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def encode_jpeg(self, frame):
        self.calls.append(frame)
        return f"jpeg-{len(self.calls)}".encode()

    def frame_checksum(self, frame):
        return "checksum"


def test_live_preview_plate_crop_jpeg_cache_reuses_same_crop():
    frame = FakeFrame()
    crop = FakeCrop()
    camera_manager = CountingJpegCameraManager()
    state = main.LivePreviewState(version="0.7.1", git_commit="abc123")
    state.camera_manager = camera_manager
    state.preview_swap_rb = "false"

    state.update_frame(frame, motion_state=main.MotionEventManager.IDLE, fps=1.0, resolution="30x20", plate_crop=crop)
    first_plate_jpeg = state.latest_plate_jpeg()
    state.update_frame(frame, motion_state=main.MotionEventManager.IDLE, fps=1.0, resolution="30x20", plate_crop=crop)

    assert state.latest_plate_jpeg() == first_plate_jpeg
    assert sum(call is crop for call in camera_manager.calls) == 1


def test_live_preview_plate_crop_jpeg_cache_encodes_new_crop():
    frame = FakeFrame()
    first_crop = FakeCrop()
    second_crop = FakeCrop()
    camera_manager = CountingJpegCameraManager()
    state = main.LivePreviewState(version="0.7.1", git_commit="abc123")
    state.camera_manager = camera_manager
    state.preview_swap_rb = "false"

    state.update_frame(frame, motion_state=main.MotionEventManager.IDLE, fps=1.0, resolution="30x20", plate_crop=first_crop)
    first_plate_jpeg = state.latest_plate_jpeg()
    state.update_frame(frame, motion_state=main.MotionEventManager.IDLE, fps=1.0, resolution="30x20", plate_crop=second_crop)

    assert state.latest_plate_jpeg() != first_plate_jpeg
    assert sum(call is first_crop for call in camera_manager.calls) == 1
    assert sum(call is second_crop for call in camera_manager.calls) == 1


def test_live_preview_plate_crop_none_invalidates_jpeg_cache():
    frame = FakeFrame()
    crop = FakeCrop()
    camera_manager = CountingJpegCameraManager()
    state = main.LivePreviewState(version="0.7.1", git_commit="abc123")
    state.camera_manager = camera_manager
    state.preview_swap_rb = "false"

    state.update_frame(frame, motion_state=main.MotionEventManager.IDLE, fps=1.0, resolution="30x20", plate_crop=crop)
    assert state.latest_plate_jpeg() is not None
    state.update_frame(frame, motion_state=main.MotionEventManager.IDLE, fps=1.0, resolution="30x20", plate_crop=None)
    state.update_frame(frame, motion_state=main.MotionEventManager.IDLE, fps=1.0, resolution="30x20", plate_crop=crop)

    assert state.latest_plate_jpeg() is not None
    assert sum(call is crop for call in camera_manager.calls) == 2


def test_live_preview_plate_crop_cached_jpeg_matches_existing_encoder_output():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = np.full((5, 10, 3), 128, dtype=np.uint8)
    state = main.LivePreviewState(version="0.7.1", git_commit="abc123")

    state.update_frame(frame, motion_state=main.MotionEventManager.IDLE, fps=1.0, resolution="30x20", plate_crop=crop)
    cached_jpeg = state.latest_plate_jpeg()

    assert cached_jpeg == main.CameraManager.encode_jpeg(crop)
    assert cv2.imdecode(np.frombuffer(cached_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR) is not None


def test_live_preview_server_routes_use_shared_state(monkeypatch):
    pytest.importorskip("flask")
    np = pytest.importorskip("numpy")
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")
    state.update_frame(
        np.zeros((20, 30, 3), dtype=np.uint8),
        motion_state=main.MotionEventManager.IDLE,
        fps=1.0,
        resolution="30x20",
    )
    server = main.LivePreviewServer(state=state)
    monkeypatch.setattr(main.threading.Thread, "start", lambda self: None)

    assert server.start() is True
    client = server._app.test_client()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_data(as_text=True) == "OK\n"
    index = client.get("/")
    assert index.status_code == 200
    assert "GateKeeper AI 0.7.1" in index.get_data(as_text=True)
    status = client.get("/api/status").json
    assert status["motion_state"] == main.MotionEventManager.IDLE
    assert status["resolution"] == "30x20"
    stream = client.get("/stream")
    assert stream.mimetype == "multipart/x-mixed-replace"


def test_encode_jpeg_preserves_rgb_red_channel():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)

    jpeg = main.encode_jpeg(frame, color_order="RGB")
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded[..., 2].mean() > 200
    assert decoded[..., 0].mean() < 50


def test_live_preview_overlay_crosshair_grid_bbox_and_frame_info_snapshot(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.zeros((90, 120, 3), dtype=np.uint8)
    detection = main.PlateDetection((10, 20, 30, 15), 0.88, object())
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")
    state.display_config = {
        "show_crosshair": True,
        "show_grid": True,
        "show_safe_area": True,
        "show_bbox": True,
        "show_status": True,
        "detection_area": {"x": 5, "y": 6, "width": 40, "height": 30},
        "plate_ideal_area": {"x": 50, "y": 40, "width": 30, "height": 20},
    }
    monkeypatch.setattr(main, "SNAPSHOT_DIR", tmp_path / "snapshots")

    state.update_frame(
        frame,
        motion_state=main.MotionEventManager.MOTION_STARTED,
        fps=12.345,
        resolution="120x90",
        plate_detection=detection,
    )
    info = state.frame_info()
    output_path = state.save_snapshot()
    decoded = cv2.imdecode(np.frombuffer(state.latest_frame_jpeg(), dtype=np.uint8), cv2.IMREAD_COLOR)

    assert info["width"] == 120
    assert info["height"] == 90
    assert info["fps"] == 12.35
    assert info["motion"] == main.MotionEventManager.MOTION_STARTED
    assert info["confidence"] == 0.88
    assert info["plate_found"] is True
    assert info["plate_box"] == (10, 20, 30, 15)
    assert output_path.is_file()
    assert output_path.name.startswith("snapshot_")
    assert decoded[45, 60, 1] > 0
    assert decoded[30, 40].any()
    assert decoded[20, 10, 1] > 0


def test_live_preview_server_frame_info_and_snapshot_route(monkeypatch, tmp_path):
    pytest.importorskip("flask")
    np = pytest.importorskip("numpy")
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")
    state.update_frame(
        np.zeros((20, 30, 3), dtype=np.uint8),
        motion_state=main.MotionEventManager.IDLE,
        fps=1.0,
        resolution="30x20",
    )
    monkeypatch.setattr(main, "SNAPSHOT_DIR", tmp_path / "snapshots")
    server = main.LivePreviewServer(state=state)
    monkeypatch.setattr(main.threading.Thread, "start", lambda self: None)
    server.start()
    client = server._app.test_client()

    frame_info = client.get("/api/frame_info")
    snapshot = client.post("/api/snapshot")

    assert frame_info.status_code == 200
    assert frame_info.json["width"] == 30
    assert frame_info.json["plate_found"] is False
    assert snapshot.status_code == 200
    assert (tmp_path / "snapshots" / snapshot.json["filename"]).is_file()


def test_apply_orientation_rotation_and_flips():
    np = pytest.importorskip("numpy")
    frame = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))

    rotated = main.apply_orientation(frame, rotation=180)
    flipped = main.apply_orientation(frame, flip_horizontal=True)
    vertical = main.apply_orientation(frame, flip_vertical=True)

    assert rotated[0, 0].tolist() == frame[1, 1].tolist()
    assert flipped[0, 0].tolist() == frame[0, 1].tolist()
    assert vertical[0, 0].tolist() == frame[1, 0].tolist()


def test_camera_capture_applies_orientation_before_pipeline(monkeypatch):
    np = pytest.importorskip("numpy")
    raw = np.arange(18, dtype=np.uint8).reshape((2, 3, 3))

    class FakePicamera2:
        def create_still_configuration(self, **_kwargs):
            return {}

        def configure(self, _configuration):
            return None

        def start(self):
            return None

        def capture_array(self):
            return raw.copy()

    monkeypatch.setitem(__import__("sys").modules, "picamera2", type("M", (), {"Picamera2": FakePicamera2}))
    manager = main.CameraManager(3, 2, 1, rotation=90, flip_horizontal=True, pipeline="rgb")

    frame = manager.capture_frame()
    expected = main.apply_orientation(raw, rotation=90, flip_horizontal=True)

    assert frame.shape == (3, 2, 3)
    assert frame.tolist() == expected.tolist()


def test_apply_orientation_supports_all_permanent_rotations():
    np = pytest.importorskip("numpy")
    frame = np.arange(24, dtype=np.uint8).reshape((2, 4, 3))

    assert main.apply_orientation(frame, rotation=0).tolist() == frame.tolist()
    assert main.apply_orientation(frame, rotation=90).shape == (4, 2, 3)
    assert main.apply_orientation(frame, rotation=180)[0, 0].tolist() == frame[1, 3].tolist()
    assert main.apply_orientation(frame, rotation=270).shape == (4, 2, 3)
    with pytest.raises(ValueError):
        main.apply_orientation(frame, rotation=45)


def test_encode_jpeg_bgr_does_not_double_convert_blue():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)  # BGR blue

    jpeg = main.encode_jpeg(frame, color_order="BGR")
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded[..., 0].mean() > 200
    assert decoded[..., 2].mean() < 50


def test_plate_detector_filters_rejected_and_selects_candidate():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    detector = main.PlateDetector(
        min_area=100,
        max_area=20000,
        min_aspect_ratio=3.0,
        max_aspect_ratio=7.0,
        confidence_threshold=0.05,
        min_rectangularity=0.5,
        border_margin=5,
        debug=True,
    )
    frame = np.zeros((160, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40, 60), (240, 100), (255, 255, 255), 2)
    cv2.rectangle(frame, (1, 1), (80, 40), (255, 255, 255), 2)

    detection = detector.detect(frame)

    assert detection is not None
    assert detection.confidence >= detector.confidence_threshold
    assert detection.bounding_box[0] >= 35
    assert detector.last_candidates
    assert any(not candidate.valid for candidate in detector.last_candidates)
    assert any(candidate.valid for candidate in detector.last_candidates)


def test_live_preview_server_debug_endpoints(monkeypatch):
    pytest.importorskip("flask")
    np = pytest.importorskip("numpy")
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")
    state.update_frame(np.zeros((20, 30, 3), dtype=np.uint8), motion_state=main.MotionEventManager.IDLE, fps=1.0, resolution="30x20")
    server = main.LivePreviewServer(state=state)
    monkeypatch.setattr(main.threading.Thread, "start", lambda self: None)
    server.start()
    client = server._app.test_client()

    for endpoint in ("/debug/gray", "/debug/edges", "/debug/contours", "/debug/candidates", "/debug/final"):
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.mimetype == "multipart/x-mixed-replace"


def test_generate_camera_diagnostics_writes_all_pipeline_images(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)

    result = main.generate_camera_diagnostics(frame, tmp_path)

    assert set(result) == {"frame_raw.jpg", "frame_rgb.jpg", "frame_bgr.jpg", "frame_swap_rb.jpg"}
    for name, metadata in result.items():
        assert metadata["pipeline"] in {"raw", "rgb", "bgr", "swap_rb"}
        assert (tmp_path / name).is_file()
        assert cv2.imread(str(tmp_path / name)) is not None


def test_apply_color_pipeline_selection_changes_channels():
    np = pytest.importorskip("numpy")
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)

    assert main.apply_color_pipeline(frame, "rgb")[0, 0].tolist() == [1, 2, 3]
    assert main.apply_color_pipeline(frame, "raw")[0, 0].tolist() == [1, 2, 3]
    assert main.apply_color_pipeline(frame, "swap_rb")[0, 0].tolist() == [3, 2, 1]
    assert main.apply_color_pipeline(frame, "bgr")[0, 0].tolist() == [3, 2, 1]


def test_live_preview_state_persists_pipeline_orientation_and_controls(tmp_path):
    pytest.importorskip("yaml")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("camera:\n  pipeline: rgb\n  rotation: 0\n  flip_horizontal: false\n  flip_vertical: false\n", encoding="utf-8")
    camera = main.CameraManager(10, 8, 1)
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")
    state.config_path = config_path
    state.camera_manager = camera

    state.update_camera_config({"pipeline": "swap_rb", "rotation": 90, "flip_horizontal": True, "flip_vertical": True, "controls": {"Brightness": 0.2}})
    persisted = main.load_config(config_path)

    assert camera.pipeline == "swap_rb"
    assert camera.rotation == 90
    assert camera.flip_horizontal is True
    assert camera.flip_vertical is True
    assert camera.controls == {"Brightness": 0.2}
    assert persisted["camera"]["pipeline"] == "swap_rb"
    assert persisted["camera"]["rotation"] == 90
    assert persisted["camera"]["flip_horizontal"] is True
    assert persisted["camera"]["flip_vertical"] is True
    assert persisted["camera"]["controls"]["Brightness"] == 0.2


def test_live_preview_server_diagnostics_page_and_camera_update(monkeypatch, tmp_path):
    pytest.importorskip("flask")
    pytest.importorskip("yaml")
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")
    state.config_path = tmp_path / "config.yaml"
    state.config_path.write_text("camera:\n  pipeline: rgb\n", encoding="utf-8")
    state.diagnostics_dir = tmp_path
    server = main.LivePreviewServer(state=state)
    monkeypatch.setattr(main.threading.Thread, "start", lambda self: None)
    server.start()
    client = server._app.test_client()

    page = client.get("/diagnostics")
    update = client.post("/api/camera", json={"pipeline": "bgr", "flip_vertical": True})

    assert page.status_code == 200
    assert "Camera Calibration" in page.get_data(as_text=True)
    assert "Sensor Calibration" in page.get_data(as_text=True)
    assert "Runtime Preview" in page.get_data(as_text=True)
    assert "Current Runtime Pipeline" in page.get_data(as_text=True)
    assert update.status_code == 200
    pipeline = client.get("/api/pipeline")
    assert pipeline.status_code == 200
    assert pipeline.get_json()["camera_pipeline"] == "rgb"
    assert main.load_config(state.config_path)["camera"]["pipeline"] == "rgb"
    assert main.load_config(state.config_path)["camera"]["flip_vertical"] is True


def test_snapshot_motion_and_plate_use_selected_pipeline(tmp_path):
    np = pytest.importorskip("numpy")
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:, :] = (1, 2, 3)
    processed = main.apply_color_pipeline(frame, "swap_rb")
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")
    monkeypatch_dir = tmp_path / "snapshots"
    original_snapshot = main.SNAPSHOT_DIR
    main.SNAPSHOT_DIR = monkeypatch_dir
    try:
        state.update_frame(processed, motion_state=main.MotionEventManager.IDLE, fps=1, resolution="30x20")
        assert state.save_snapshot().is_file()
    finally:
        main.SNAPSHOT_DIR = original_snapshot
    detector = FakeMotionDetector([(True, 1234.0)])
    plate_detector = FakePlateDetector(main.PlateDetection((1, 1, 5, 5), 0.8, object()))
    manager = main.MotionEventManager(detector, None, logging.getLogger("test-gatekeeper"), plate_detector=plate_detector)
    original_image_dir = main.IMAGE_DIR
    main.IMAGE_DIR = tmp_path / "images"
    try:
        manager.process_frame(processed)
    finally:
        main.IMAGE_DIR = original_image_dir
    assert detector.saved_paths
    assert plate_detector.calls == 1


def test_motion_detector_resets_when_frame_shape_changes(caplog):
    np = pytest.importorskip("numpy")
    detector = main.MotionDetector(min_area=1, threshold=1)
    detector.detect(np.zeros((4, 6, 3), dtype=np.uint8))

    with caplog.at_level(logging.WARNING, logger="gatekeeper"):
        motion, area = detector.detect(np.zeros((6, 4, 3), dtype=np.uint8))

    assert motion is False
    assert area == 0.0
    assert "MotionDetector reset because frame size changed" in caplog.text
    assert detector.previous_frame.shape == (6, 4)


def test_pipeline_and_camera_reconfigure_logging(caplog):
    with caplog.at_level(logging.INFO, logger="gatekeeper"):
        main.log_pipeline_changed(logging.getLogger("gatekeeper"), "rgb", "bgr")
        main.log_camera_reconfigure(logging.getLogger("gatekeeper"), 0, 180, False, True, False, False)

    assert "PIPELINE CHANGED" in caplog.text
    assert "Old pipeline: rgb" in caplog.text
    assert "New pipeline: bgr" in caplog.text
    assert "CAMERA RECONFIGURE" in caplog.text
    assert "Reset previous_frame" in caplog.text


def test_startup_metadata_logs_frame_pipeline(caplog, tmp_path):
    image_path = tmp_path / "latest.jpg"
    image_path.write_bytes(b"jpeg")
    camera_info = {
        "backend": main.CAMERA_BACKEND,
        "resolution": "10x8",
        "width": 10,
        "height": 8,
        "pixel_format": "RGB888",
        "frame_format": "RGB",
        "shape": (8, 10, 3),
        "dtype": "uint8",
        "color_pipeline": "rgb",
        "jpeg_encoder_format": main.JPEG_ENCODER_INPUT_FORMAT,
        "camera_model": "fake-camera",
        "sensor": "fake-sensor",
        "orientation": {"rotation": 180, "flip_horizontal": False, "flip_vertical": False},
    }

    with caplog.at_level(logging.INFO, logger="gatekeeper"):
        main.log_startup_metadata(logging.getLogger("gatekeeper"), "0.6.9", "abc123", {**camera_info, "preview_swap_rb": True}, "now", image_path)

    assert "GateKeeper AI v0.6.9" in caplog.text
    assert "FRAME PIPELINE" in caplog.text
    assert "Picamera2" in caplog.text
    assert "CameraManager" in caplog.text
    assert "JPEG Encoder input format: BGR" in caplog.text
    assert "Live Preview pipeline: BGR" in caplog.text
    assert "Live Preview configuration: inherited from 0.6.9" in caplog.text
    assert "Pipeline selection UI: disabled" in caplog.text
    assert "Live Preview source: FRAME_MASTER" in caplog.text


def test_runtime_consumers_share_frame_master_identity_and_checksum():
    np = pytest.importorskip("numpy")
    frame_master = np.arange(27, dtype=np.uint8).reshape((3, 3, 3))
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")

    state.update_frame(
        frame_master,
        motion_state=main.MotionEventManager.IDLE,
        fps=1.0,
        resolution="3x3",
    )
    pipeline = state.pipeline_snapshot()

    assert pipeline["master_frame_id"] == id(frame_master)
    assert pipeline["preview_frame_id"] == id(frame_master)
    assert pipeline["snapshot_frame_id"] == id(frame_master)
    assert pipeline["motion_frame_id"] == id(frame_master)
    assert pipeline["plate_frame_id"] == id(frame_master)
    assert pipeline["checksum"] == main.CameraManager.frame_checksum(frame_master)
    assert pipeline["preview_checksum"] == pipeline["checksum"]
    assert pipeline["snapshot_checksum"] == pipeline["checksum"]
    assert pipeline["verification"] == "PASS"


def test_diagnostics_are_generated_from_raw_frame_after_pipeline_change(tmp_path):
    np = pytest.importorskip("numpy")

    class FakePicamera:
        camera_properties = {"Model": "fake"}
        def __init__(self):
            self.frame = np.array([[[10, 20, 30]]], dtype=np.uint8)
        def capture_array(self):
            return self.frame.copy()

    manager = main.CameraManager(1, 1, 1, pipeline="swap_rb", rotation=180, flip_horizontal=True)
    manager._camera = FakePicamera()

    diagnostics = manager.generate_diagnostics(tmp_path)

    assert set(diagnostics) == {"frame_raw.jpg", "frame_rgb.jpg", "frame_bgr.jpg", "frame_swap_rb.jpg"}
    assert manager.diagnostics_source_id is not None
    assert manager.raw_frame_id == manager.diagnostics_source_id
    assert manager._last_frame_master is None


def test_motion_and_plate_receive_frame_master_object():
    np = pytest.importorskip("numpy")
    frame_master = np.zeros((20, 60, 3), dtype=np.uint8)
    motion = FakeMotionDetector([(True, 1234.0)])
    plate = FakePlateDetector(main.PlateDetection((1, 1, 5, 5), 0.8, object()))
    manager = main.MotionEventManager(motion, None, logging.getLogger("test-gatekeeper"), plate_detector=plate)

    manager.process_frame(frame_master)

    assert motion.calls == 1
    assert plate.calls == 1
    assert manager.last_plate_crop is frame_master


def test_preview_swap_rb_is_isolated_from_snapshot(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame_master = np.zeros((16, 16, 3), dtype=np.uint8)
    frame_master[:, :] = (255, 0, 0)
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")
    state.camera_pipeline = "bgr"
    state.preview_swap_rb = "true"

    original_snapshot = main.SNAPSHOT_DIR
    main.SNAPSHOT_DIR = tmp_path / "snapshots"
    try:
        state.update_frame(frame_master, motion_state=main.MotionEventManager.IDLE, fps=1, resolution="16x16")
        preview = cv2.imdecode(np.frombuffer(state.latest_frame_jpeg(), dtype=np.uint8), cv2.IMREAD_COLOR)
        snapshot_path = state.save_snapshot()
        snapshot = cv2.imread(str(snapshot_path))
    finally:
        main.SNAPSHOT_DIR = original_snapshot

    assert preview[..., 0].mean() > 200
    assert preview[..., 2].mean() < 50
    assert snapshot[..., 2].mean() > 200
    assert snapshot[..., 0].mean() < 50



def test_live_preview_uses_0_6_9_swap_rb_configuration(monkeypatch):
    np = pytest.importorskip("numpy")
    frame_master = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    encoded_inputs = []

    def fake_encode(frame, *, color_order="RGB"):
        encoded_inputs.append(frame.copy())
        return b"jpeg"

    monkeypatch.setattr(main.CameraManager, "encode_jpeg", staticmethod(fake_encode))
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")
    state.preview_pipeline = "bgr"

    state.update_frame(frame_master, motion_state=main.MotionEventManager.IDLE, fps=1, resolution="2x1")

    assert state.latest_frame_jpeg() == b"jpeg"
    assert encoded_inputs[0].tolist() == [[[30, 20, 10], [60, 50, 40]]]
    assert np.array_equal(frame_master, np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8))


def test_live_preview_reproduces_0_6_9_preview_encoding(monkeypatch):
    np = pytest.importorskip("numpy")
    frame_master = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    original = frame_master.copy()
    encoded_inputs = []

    def fake_encode(frame, *, color_order="RGB"):
        encoded_inputs.append(frame.copy())
        return b"jpeg"

    monkeypatch.setattr(main.CameraManager, "encode_jpeg", staticmethod(fake_encode))
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")
    state.preview_pipeline = "bgr"

    state.update_frame(frame_master, motion_state=main.MotionEventManager.IDLE, fps=1, resolution="2x1")
    pipeline = state.pipeline_snapshot()

    assert state.latest_frame_jpeg() == b"jpeg"
    assert encoded_inputs[0].tolist() == [[[30, 20, 10], [60, 50, 40]]]
    assert np.array_equal(frame_master, original)
    assert pipeline["master_frame_id"] == id(frame_master)
    assert pipeline["preview_frame_id"] == id(frame_master)
    assert pipeline["preview_pipeline"] == "bgr"
    assert pipeline["preview_swap_rb"] is True
    assert pipeline["preview_configuration"] == "inherited from 0.6.9"



def test_api_camera_ignores_pipeline_changes_but_keeps_orientation(tmp_path, monkeypatch):
    pytest.importorskip("flask")
    pytest.importorskip("yaml")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("camera:\n  pipeline: rgb\n  rotation: 0\n  flip_horizontal: false\n  flip_vertical: false\npreview:\n  pipeline: bgr\n", encoding="utf-8")
    state = main.LivePreviewState(version="0.7.1", git_commit="abc123")
    state.config_path = config_path
    state.camera_pipeline = "rgb"
    server = main.LivePreviewServer(state=state)
    monkeypatch.setattr(main.threading.Thread, "start", lambda self: None)
    server.start()
    client = server._app.test_client()

    response = client.post("/api/camera", json={"pipeline": "swap_rb", "rotation": 90})
    pipeline = client.get("/api/pipeline").get_json()

    assert response.status_code == 200
    assert pipeline["camera_pipeline"] == "rgb"
    assert pipeline["preview_pipeline"] == "bgr"
    persisted = main.load_config(config_path)
    assert persisted["camera"]["pipeline"] == "rgb"
    assert persisted["camera"]["rotation"] == 90


def test_live_preview_conversion_does_not_change_master_motion_plate_snapshot_or_diagnostics(monkeypatch):
    np = pytest.importorskip("numpy")
    frame_master = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    original = frame_master.copy()
    encoded_inputs = []

    def fake_encode(frame, *, color_order="RGB"):
        encoded_inputs.append((frame.copy(), color_order))
        return b"jpeg"

    monkeypatch.setattr(main.CameraManager, "encode_jpeg", staticmethod(fake_encode))
    state = main.LivePreviewState(version="0.7.1", git_commit="abc123")
    state.update_frame(frame_master, motion_state=main.MotionEventManager.IDLE, fps=1, resolution="2x1")
    pipeline = state.pipeline_snapshot()

    assert len(encoded_inputs) == 1
    assert np.array_equal(encoded_inputs[0][0], np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8))
    assert encoded_inputs[0][1] == "BGR"
    assert np.array_equal(frame_master, original)
    assert pipeline["master_frame_id"] == id(frame_master)
    assert pipeline["motion_frame_id"] == id(frame_master)
    assert pipeline["plate_frame_id"] == id(frame_master)
    assert pipeline["snapshot_frame_id"] == id(frame_master)
    assert pipeline["diagnostics_pipeline"] == "RAW_FRAME"

def test_diagnostics_unchanged_by_preview_swap_rb(tmp_path):
    np = pytest.importorskip("numpy")
    raw = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    state = main.LivePreviewState(version="0.6.9", git_commit="abc123")
    state.preview_swap_rb = "true"

    diagnostics = main.generate_camera_diagnostics(raw, tmp_path)

    assert set(diagnostics) == {"frame_raw.jpg", "frame_rgb.jpg", "frame_bgr.jpg", "frame_swap_rb.jpg"}
    assert diagnostics["frame_raw.jpg"]["pipeline"] == "raw"
    assert state.resolved_preview_swap_rb() is True

def test_diagnostics_runtime_orientation_uses_raw_frame_only(tmp_path):
    np = pytest.importorskip("numpy")
    raw = np.arange(18, dtype=np.uint8).reshape((2, 3, 3))

    off = main.generate_camera_diagnostics(raw, tmp_path / "off")
    on = main.generate_camera_diagnostics(
        raw,
        tmp_path / "on",
        runtime_orientation={"rotation": 90, "flip_horizontal": False, "flip_vertical": False},
    )

    assert set(off) == {"frame_raw.jpg", "frame_rgb.jpg", "frame_bgr.jpg", "frame_swap_rb.jpg"}
    assert set(on) == set(off)
    assert off["frame_raw.jpg"]["pipeline"] == "raw"
    assert on["frame_raw.jpg"]["pipeline"] == "raw"


def test_plate_calibration_candidate_metadata_and_reasons():
    candidate = main.PlateCandidate((1, 2, 30, 10), 0.4, object(), 3.0, 300.0, 0.7, 0.0, False, "confidence")
    metadata = candidate.metadata()

    assert metadata["bounding_box"] == (1, 2, 30, 10)
    assert metadata["area"] == 300.0
    assert metadata["aspect_ratio"] == 3.0
    assert metadata["rectangularity"] == 0.7
    assert metadata["confidence"] == 0.4
    assert metadata["selected"] is False
    assert metadata["rejected"] is True
    assert metadata["rejection_reason"] == "confidence"


def test_plate_calibration_json_and_save_route(monkeypatch, tmp_path):
    pytest.importorskip("flask")
    np = pytest.importorskip("numpy")
    candidate = main.PlateCandidate((2, 3, 20, 8), 0.8, object(), 2.5, 160.0, 0.9, 0.0, True, "", True)
    detection = main.PlateDetection((2, 3, 20, 8), 0.8, object(), candidates=(candidate,))
    state = main.LivePreviewState(version="0.7.1", git_commit="abc123")
    state.plate_calibration_config = {"enabled": True, "save_frames": True}
    state.detector_thresholds = {"min_area": 100, "max_area": 1000, "aspect_ratio_min": 2.0, "aspect_ratio_max": 6.0, "min_rectangularity": 0.8, "confidence_threshold": 0.7}
    monkeypatch.setattr(main, "IMAGE_DIR", tmp_path / "images")
    state.update_frame(np.zeros((30, 50, 3), dtype=np.uint8), motion_state=main.MotionEventManager.MOTION_STARTED, fps=1.0, resolution="50x30", plate_detection=detection)
    server = main.LivePreviewServer(state=state)
    monkeypatch.setattr(main.threading.Thread, "start", lambda self: None)
    server.start()
    client = server._app.test_client()

    payload = client.get("/api/plate_calibration").json
    saved = client.post("/api/plate_calibration/save").json

    assert payload["enabled"] is True
    assert payload["candidate_count"] == 1
    assert payload["selected_candidate"]["confidence"] == 0.8
    assert payload["frame_width"] == 50
    assert payload["frame_height"] == 30
    assert saved["filename"].startswith("calibration_")
    assert (tmp_path / "images" / saved["filename"]).is_file()


def test_annotated_calibration_frame_draws_selected_and_rejected():
    np = pytest.importorskip("numpy")
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    selected = main.PlateCandidate((5, 5, 20, 8), 0.9, object(), 2.5, 160.0, 0.9, 0.0, True, "", True)
    rejected = main.PlateCandidate((35, 5, 20, 8), 0.2, object(), 2.5, 160.0, 0.9, 0.0, False, "confidence")
    detection = main.PlateDetection((5, 5, 20, 8), 0.9, object(), candidates=(selected, rejected))

    annotated = main.annotate_calibration_frame(frame, detection, [selected, rejected])

    assert annotated[5, 5, 1] > 0
    assert annotated[5, 35, 0] > 0


def test_live_preview_color_pipeline_unchanged_by_calibration():
    np = pytest.importorskip("numpy")
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)
    plain = main.LivePreviewState(version="0.7.1", git_commit="abc123")
    calibrated = main.LivePreviewState(version="0.7.1", git_commit="abc123")
    calibrated.plate_calibration_config = {"enabled": True}

    plain.update_frame(frame, motion_state=main.MotionEventManager.IDLE, fps=1, resolution="24x24")
    calibrated.update_frame(frame, motion_state=main.MotionEventManager.IDLE, fps=1, resolution="24x24")

    assert plain.latest_frame_jpeg() == calibrated.latest_frame_jpeg()



def test_live_preview_downscales_large_frame_without_changing_master(monkeypatch):
    np = pytest.importorskip("numpy")
    frame_master = np.zeros((1232, 1640, 3), dtype=np.uint8)
    encoded_inputs = []

    def fake_encode(frame, *, color_order="RGB"):
        encoded_inputs.append(frame.copy())
        return b"jpeg"

    monkeypatch.setattr(main.CameraManager, "encode_jpeg", staticmethod(fake_encode))
    state = main.LivePreviewState(version="0.8.19", git_commit="abc123")

    state.update_frame(frame_master, motion_state=main.MotionEventManager.IDLE, fps=1, resolution="1640x1232")

    assert state.latest_frame_jpeg() == b"jpeg"
    assert encoded_inputs[0].shape == (616, 820, 3)
    assert state.width == 1640
    assert state.height == 1232
    assert state.pipeline_snapshot()["master_frame_id"] == id(frame_master)


def test_live_preview_does_not_upscale_small_frame(monkeypatch):
    np = pytest.importorskip("numpy")
    frame_master = np.zeros((120, 160, 3), dtype=np.uint8)
    encoded_inputs = []

    def fake_encode(frame, *, color_order="RGB"):
        encoded_inputs.append(frame.copy())
        return b"jpeg"

    monkeypatch.setattr(main.CameraManager, "encode_jpeg", staticmethod(fake_encode))
    state = main.LivePreviewState(version="0.8.19", git_commit="abc123")

    state.update_frame(frame_master, motion_state=main.MotionEventManager.IDLE, fps=1, resolution="160x120")

    assert encoded_inputs[0].shape == (120, 160, 3)

def test_motion_detection_unchanged_thresholds_and_state():
    detector = main.MotionDetector(min_area=123, threshold=45)

    assert detector.min_area == 123
    assert detector.threshold == 45
    assert detector.previous_frame is None


def test_plate_detector_default_max_area_is_150000_only_area_change():
    detector = main.PlateDetector()

    assert detector.min_area == 2500.0
    assert detector.max_area == 150000.0
    assert detector.min_aspect_ratio == 3.5
    assert detector.max_aspect_ratio == 6.5
    assert detector.min_rectangularity == 0.80
    assert detector.max_rotation == 15.0
    assert detector.confidence_threshold == 0.70


def test_plate_calibration_overlay_draws_selected_prominently_and_threshold_summary():
    np = pytest.importorskip("numpy")
    frame = np.zeros((120, 220, 3), dtype=np.uint8)
    selected = main.PlateCandidate((20, 70, 80, 20), 0.912, object(), 4.0, 1600.0, 0.91, 2.0, True, "", True)
    valid = main.PlateCandidate((120, 70, 70, 20), 0.8, object(), 3.5, 1400.0, 0.9, 1.0, True, "", False)
    rejected = main.PlateCandidate((20, 20, 50, 15), 0.2, object(), 3.3, 750.0, 0.5, 3.0, False, "rectangularity")
    detection = main.PlateDetection(selected.bounding_box, selected.confidence, object(), candidates=(selected, valid, rejected))

    annotated = main.annotate_calibration_frame(frame, detection, [selected, valid, rejected], thresholds={"min_area": 2500, "max_area": 150000, "min_aspect_ratio": 3.5, "max_aspect_ratio": 6.5, "min_rectangularity": 0.8, "max_rotation": 15, "confidence_threshold": 0.7})

    assert annotated[70, 20, 1] > 0  # selected green border
    assert annotated[70, 120, 0] > 0 and annotated[70, 120, 1] > 0  # valid yellow border
    assert annotated[20, 20, 0] > 0  # rejected red border
    assert annotated[68, 20, 1] > 0  # prominent selected border is thicker than normal
    assert annotated.sum() > 0


def test_plate_calibration_endpoint_exposes_live_rejected_count_and_thresholds():
    np = pytest.importorskip("numpy")
    selected = main.PlateCandidate((1, 1, 20, 6), 0.8, object(), 3.33, 120.0, 0.9, 0.0, True, "", True)
    rejected = main.PlateCandidate((30, 1, 20, 6), 0.2, object(), 3.33, 120.0, 0.5, 0.0, False, "confidence")
    detection = main.PlateDetection(selected.bounding_box, selected.confidence, object(), candidates=(selected, rejected))
    state = main.LivePreviewState(version="0.7.1", git_commit="abc123")
    state.plate_calibration_config = {"enabled": True}
    state.detector_thresholds = {"min_area": 2500, "max_area": 150000, "aspect_ratio_min": 3.5, "aspect_ratio_max": 6.5, "min_rectangularity": 0.8, "max_rotation": 15, "confidence_threshold": 0.7}

    state.update_frame(np.zeros((40, 80, 3), dtype=np.uint8), motion_state=main.MotionEventManager.IDLE, fps=1.0, resolution="80x40", plate_detection=detection)
    payload = state.plate_calibration_snapshot()

    assert payload["candidate_count"] == 2
    assert payload["selected_candidate"]["bounding_box"] == selected.bounding_box
    assert len([c for c in payload["candidates"] if c["rejected"]]) == 1
    assert payload["candidates"][1]["rejection_reason"] == "confidence"
    assert payload["thresholds"]["max_area"] == 150000
    assert payload["thresholds"]["max_rotation"] == 15
