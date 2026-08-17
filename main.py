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
import zlib
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import deque
from dataclasses import dataclass

from src.gatekeeper.display_manager import DisplayManager
from src.gatekeeper.plate_detector import PlateCandidate, PlateDetection, PlateDetector

try:
    import yaml
except ImportError:  # dependency is installed by requirements.txt
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
VERSION_FILE = PROJECT_ROOT / "VERSION"
LOG_DIR = PROJECT_ROOT / "logs"
IMAGE_DIR = PROJECT_ROOT / "images"
DEBUG_DIR = PROJECT_ROOT / "debug"
SNAPSHOT_DIR = PROJECT_ROOT / "snapshots"
DIAGNOSTICS_DIR = PROJECT_ROOT / "diagnostics"
LATEST_IMAGE = IMAGE_DIR / "latest.jpg"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
PID_FILE = RUNTIME_DIR / "gatekeeper.pid"
DATABASE_PATH = RUNTIME_DIR / "gatekeeper.db"
CAMERA_BACKEND = "Picamera2"
PIPELINES = {"raw": "As returned by Picamera2", "rgb": "Interpret frame as RGB", "bgr": "Convert RGB->BGR before JPEG encoding", "swap_rb": "Swap red and blue channels explicitly"}
PREVIEW_SWAP_RB_VALUES = {"true", "false"}
LIVE_PREVIEW_PIPELINE = "bgr"
LIVE_PREVIEW_CONFIGURATION = "inherited from 0.6.9"
JPEG_ENCODER_INPUT_FORMAT = "BGR"


@dataclass(frozen=True)
class FrameMaster:
    """Metadata describing the one processed frame shared by all consumers."""

    frame_id: int
    shape: tuple[int, ...] | None
    dtype: str | None
    pipeline: str
    rotation: int
    flip_horizontal: bool
    flip_vertical: bool

    @classmethod
    def from_frame(cls, frame: Any, *, pipeline: str, rotation: int, flip_horizontal: bool, flip_vertical: bool) -> "FrameMaster":
        return cls(
            frame_id=id(frame),
            shape=tuple(frame.shape) if hasattr(frame, "shape") else None,
            dtype=str(frame.dtype) if hasattr(frame, "dtype") else None,
            pipeline=pipeline,
            rotation=rotation,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "shape": self.shape,
            "dtype": self.dtype,
            "pipeline": self.pipeline,
            "rotation": self.rotation,
            "flip_horizontal": self.flip_horizontal,
            "flip_vertical": self.flip_vertical,
        }


def frame_channels(frame: Any) -> int:
    """Return the number of channels represented by a frame shape."""
    if frame is None or not hasattr(frame, "shape"):
        return 0
    return int(frame.shape[2]) if len(frame.shape) >= 3 else 1


def describe_frame(frame: Any, *, pipeline: str, rotation: int, flip_horizontal: bool, flip_vertical: bool) -> dict[str, Any]:
    """Return runtime frame metadata for pipeline inspection."""
    return {
        "frame_id": id(frame),
        "shape": tuple(frame.shape) if hasattr(frame, "shape") else None,
        "dtype": str(frame.dtype) if hasattr(frame, "dtype") else None,
        "channels": frame_channels(frame),
        "pipeline": pipeline,
        "rotation": rotation,
        "flip_horizontal": flip_horizontal,
        "flip_vertical": flip_vertical,
    }


def log_frame_step(logger: logging.Logger, name: str, frame: Any, *, pipeline: str, rotation: int, flip_horizontal: bool, flip_vertical: bool) -> None:
    """Log one frame path inspection step."""
    info = describe_frame(frame, pipeline=pipeline, rotation=rotation, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)
    logger.info("%s", name)
    logger.info("id(frame): %s", info["frame_id"])
    logger.info("shape: %s", info["shape"])
    logger.info("dtype: %s", info["dtype"])
    logger.info("channels: %s", info["channels"])
    logger.info("pipeline: %s", info["pipeline"])
    logger.info("rotation: %s", info["rotation"])
    logger.info("flip H: %s", info["flip_horizontal"])
    logger.info("flip V: %s", info["flip_vertical"])


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


def _parse_minimal_value(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _load_minimal_yaml(contents: str) -> dict[str, Any]:
    """Load the simple bundled YAML config when PyYAML is unavailable."""
    data: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, data)]

    for raw_line in contents.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError("Unsupported YAML syntax; install PyYAML for full YAML support.")
        key, raw_value = line.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = raw_value.strip()
        if value == "":
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _parse_minimal_value(value)

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


def save_config(config: dict[str, Any], path: Path = DEFAULT_CONFIG) -> None:
    """Persist application configuration without changing the schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        raise RuntimeError("PyYAML is required to persist configuration.")
    with path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)


def apply_color_pipeline(frame: Any, pipeline: str = "rgb") -> Any:
    """Backward-compatible wrapper for CameraManager.apply_pipeline()."""
    return CameraManager.apply_pipeline(frame, pipeline)


def diagnostic_frames(raw_frame: Any) -> dict[str, tuple[Any, str]]:
    """Return diagnostic variants generated only from RAW_FRAME."""
    return CameraManager.diagnostic_frames(raw_frame)


def generate_camera_diagnostics(
    raw_frame: Any,
    output_dir: Path = DIAGNOSTICS_DIR,
    *,
    runtime_orientation: dict[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Save RAW_FRAME diagnostic variants, optionally with runtime orientation only."""
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, str]] = {}
    for filename, (frame, pipeline) in diagnostic_frames(raw_frame).items():
        if runtime_orientation:
            frame = apply_orientation(
                frame,
                rotation=int(runtime_orientation.get("rotation", 0) or 0),
                flip_horizontal=bool(runtime_orientation.get("flip_horizontal", False)),
                flip_vertical=bool(runtime_orientation.get("flip_vertical", False)),
            )
        path = output_dir / filename
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Unable to save diagnostic image: {path}")
        results[filename] = {"path": str(path), "pipeline": pipeline}
    return results


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


class LivePreviewState:
    """Thread-safe state shared by the camera loop and HTTP live preview."""

    def __init__(self, *, version: str, git_commit: str, max_events: int = 20) -> None:
        self.version = version
        self.git_commit = git_commit
        self._lock = threading.RLock()
        self._frame: Any | None = None
        self._frame_jpeg: bytes | None = None
        self._frame_master: FrameMaster | None = None
        self._consumer_frames: dict[str, FrameMaster] = {}
        self._last_master_log = 0.0
        self._plate_jpeg: bytes | None = None
        self._plate_jpeg_cache_key: tuple[int, tuple[int, ...] | None, str | None, tuple[int, ...] | None] | None = None
        self.motion_state = (
            MotionEventManager.IDLE if "MotionEventManager" in globals() else "IDLE"
        )
        self.plate_bounding_box: tuple[int, int, int, int] | None = None
        self.confidence: float | None = None
        self.resolution = "unknown"
        self.fps = 0.0
        self.last_motion_event: dict[str, Any] | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.updated_at: str | None = None
        self.width: int | None = None
        self.height: int | None = None
        self.display_config: dict[str, Any] = {}
        self.camera_orientation: dict[str, Any] = {}
        self.camera_pipeline = "rgb"
        self.camera_controls: dict[str, Any] = {}
        self.diagnostics: dict[str, Any] = {}
        self.diagnostics_dir = DIAGNOSTICS_DIR
        self.config_path: Path = DEFAULT_CONFIG
        self.camera_manager: CameraManager | None = None
        self.detector_thresholds: dict[str, Any] = {}
        self.candidates: list[Any] = []
        self.motion_event_manager: Any | None = None
        self.preview_pipeline = LIVE_PREVIEW_PIPELINE
        self.preview_swap_rb = "true"
        self.diagnostics_mode = "RAW_FRAME"
        self.runtime_orientation = False
        self.plate_calibration_config = {"enabled": False, "save_frames": False, "show_candidates": False, "show_rejected": False, "show_metrics": False}
        self._calibration_jpeg: bytes | None = None
        self._calibration_frame: Any | None = None

    def update_frame(
        self,
        frame: Any,
        *,
        motion_state: str,
        fps: float,
        resolution: str,
        plate_detection: PlateDetection | None = None,
        plate_crop: Any | None = None,
        plate_candidates: list[Any] | None = None,
    ) -> None:
        """Store the latest captured frame and metadata for web consumers."""
        now = datetime.now(timezone.utc).isoformat()
        encoder = self.camera_manager.encode_jpeg if self.camera_manager is not None else CameraManager.encode_jpeg
        encoded_frame = None
        plate_cache_key = self._plate_crop_cache_key(plate_crop)
        encoded_plate = None
        reuse_plate_jpeg = False
        if plate_crop is not None:
            with self._lock:
                reuse_plate_jpeg = plate_cache_key == self._plate_jpeg_cache_key
            if not reuse_plate_jpeg:
                encoded_plate = encoder(plate_crop)
        master = FrameMaster.from_frame(frame, pipeline=self.camera_pipeline, rotation=int(self.camera_orientation.get("rotation", 0) or 0), flip_horizontal=bool(self.camera_orientation.get("flip_horizontal", False)), flip_vertical=bool(self.camera_orientation.get("flip_vertical", False)))
        consumers = {name: master for name in ("preview", "snapshot", "motion", "plate")}
        height = int(frame.shape[0]) if hasattr(frame, "shape") else None
        width = int(frame.shape[1]) if hasattr(frame, "shape") and len(frame.shape) > 1 else None
        candidates = list(plate_candidates or getattr(plate_detection, "candidates", []) or [])
        with self._lock:
            self._frame = frame
            self._frame_master = master
            self._consumer_frames = consumers
            self.width = width
            self.height = height
            if plate_crop is None:
                self._plate_jpeg = None
                self._plate_jpeg_cache_key = None
            elif not reuse_plate_jpeg:
                self._plate_jpeg = encoded_plate
                self._plate_jpeg_cache_key = plate_cache_key
            self.motion_state = motion_state
            self.fps = fps
            self.resolution = resolution
            self.updated_at = now
            self.plate_bounding_box = (
                plate_detection.bounding_box if plate_detection is not None else None
            )
            self.confidence = plate_detection.confidence if plate_detection is not None else None
            self.candidates = candidates
            if self.plate_calibration_config.get("enabled"):
                all_candidates = candidates or list(getattr(plate_detection, "candidates", []) or [])
                self._calibration_frame = annotate_calibration_frame(frame, plate_detection, all_candidates, thresholds=self.calibration_thresholds())
                self._calibration_jpeg = encoder(self._calibration_frame)
                encoded_frame = self.encode_preview_jpeg(self._calibration_frame)
            else:
                self._calibration_frame = None
                self._calibration_jpeg = None
                encoded_frame = self.encode_preview_jpeg(frame)
            self._frame_jpeg = encoded_frame
            now_monotonic = time.monotonic()
            if now_monotonic - self._last_master_log >= 10:
                snapshot = self.pipeline_snapshot()
                logging.getLogger("gatekeeper").info(
                    "RAW_FRAME id=%s FRAME_MASTER id=%s Motion id=%s Preview id=%s Snapshot id=%s Plate id=%s Diagnostics source id=%s Pipeline=%s Rotation=%s Flip=%s",
                    snapshot.get("raw_frame_id"),
                    snapshot.get("master_frame_id"),
                    snapshot.get("motion_frame_id"),
                    snapshot.get("preview_frame_id"),
                    snapshot.get("snapshot_frame_id"),
                    snapshot.get("plate_frame_id"),
                    snapshot.get("diagnostics_source_id"),
                    snapshot.get("pipeline"),
                    snapshot.get("rotation"),
                    snapshot.get("flip"),
                )
                self._last_master_log = now_monotonic

    @staticmethod
    def _plate_crop_cache_key(plate_crop: Any | None) -> tuple[int, tuple[int, ...] | None, str | None, tuple[int, ...] | None] | None:
        """Return a lightweight identity/metadata key for the current plate crop."""
        if plate_crop is None:
            return None
        shape = tuple(plate_crop.shape) if hasattr(plate_crop, "shape") else None
        dtype = str(plate_crop.dtype) if hasattr(plate_crop, "dtype") else None
        strides = tuple(plate_crop.strides) if hasattr(plate_crop, "strides") else None
        return id(plate_crop), shape, dtype, strides

    def record_motion_event(self, event: dict[str, Any]) -> None:
        """Store a motion event summary for the dashboard and API."""
        with self._lock:
            self.last_motion_event = event
            self.events.appendleft(event)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable HTTP status snapshot."""
        with self._lock:
            return {
                "version": self.version,
                "camera": CAMERA_BACKEND,
                "motion_state": self.motion_state,
                "fps": round(self.fps, 2),
                "resolution": self.resolution,
                "git_commit": self.git_commit,
            }

    def frame_info(self) -> dict[str, Any]:
        """Return calibration metadata for the latest streamed frame."""
        with self._lock:
            return {
                "width": self.width,
                "height": self.height,
                "fps": round(self.fps, 2),
                "motion": self.motion_state,
                "confidence": self.confidence,
                "plate_found": self.plate_bounding_box is not None,
                "plate_box": self.plate_bounding_box,
                "timestamp": self.updated_at,
                "candidate_count": len(self.candidates),
                "selected_candidate": self.plate_bounding_box,
                "rejected_candidates": len([c for c in self.candidates if not getattr(c, "valid", False)]),
                "thresholds": self.calibration_thresholds(),
                "camera_orientation": self.camera_orientation,
                "camera_pipeline": self.camera_pipeline,
                "active_pipeline": self.camera_pipeline,
                "rotation": self.camera_orientation.get("rotation"),
                "flip_horizontal": self.camera_orientation.get("flip_horizontal"),
                "flip_vertical": self.camera_orientation.get("flip_vertical"),
                "stream_pipeline": LIVE_PREVIEW_PIPELINE,
                "preview_pipeline": LIVE_PREVIEW_PIPELINE,
                "snapshot_pipeline": self.camera_pipeline,
                "motion_pipeline": self.camera_pipeline,
                "plate_pipeline": self.camera_pipeline,
                "camera_controls": self.camera_controls,
                "diagnostics": self.diagnostics,
                "diagnostics_mode": self.diagnostics_mode,
                "runtime_orientation": self.runtime_orientation,
                "preview_swap_rb": self.resolved_preview_swap_rb(),
                "preview_configuration": LIVE_PREVIEW_CONFIGURATION,
                "runtime_pipeline": self.pipeline_snapshot(),
            }

    def pipeline_snapshot(self) -> dict[str, Any]:
        """Return current runtime camera pipeline mappings for diagnostics."""
        with self._lock:
            pipeline = self.camera_pipeline
            if self.camera_manager is not None:
                pipeline = str(getattr(self.camera_manager, "pipeline", pipeline))
                rotation = int(getattr(self.camera_manager, "rotation", self.camera_orientation.get("rotation", 0)))
                flip_horizontal = bool(getattr(self.camera_manager, "flip_horizontal", self.camera_orientation.get("flip_horizontal", False)))
                flip_vertical = bool(getattr(self.camera_manager, "flip_vertical", self.camera_orientation.get("flip_vertical", False)))
            else:
                rotation = int(self.camera_orientation.get("rotation", 0) or 0)
                flip_horizontal = bool(self.camera_orientation.get("flip_horizontal", False))
                flip_vertical = bool(self.camera_orientation.get("flip_vertical", False))
            frame_id = self._frame_master.frame_id if self._frame_master is not None else None
            checksum = self.camera_manager.frame_checksum(self._frame) if self.camera_manager is not None else CameraManager.frame_checksum(self._frame)
            snapshot = {
                "raw_frame_id": getattr(self.camera_manager, "raw_frame_id", None) if self.camera_manager is not None else None,
                "master_frame_id": frame_id,
                "preview_frame_id": self._consumer_frames.get("preview").frame_id if self._consumer_frames.get("preview") else None,
                "snapshot_frame_id": self._consumer_frames.get("snapshot").frame_id if self._consumer_frames.get("snapshot") else None,
                "motion_frame_id": self._consumer_frames.get("motion").frame_id if self._consumer_frames.get("motion") else None,
                "plate_frame_id": self._consumer_frames.get("plate").frame_id if self._consumer_frames.get("plate") else None,
                "diagnostics_source_id": getattr(self.camera_manager, "diagnostics_source_id", None) if self.camera_manager is not None else None,
                "pipeline": pipeline,
                "flip": {"horizontal": flip_horizontal, "vertical": flip_vertical},
                "checksum": checksum,
                "preview_checksum": checksum,
                "snapshot_checksum": checksum,
                "verification": "PASS" if self._verify_consumers(frame_id, checksum) else "FAIL",
                "camera_pipeline": pipeline,
                "preview_pipeline": LIVE_PREVIEW_PIPELINE,
                "motion_pipeline": pipeline,
                "plate_pipeline": pipeline,
                "snapshot_pipeline": pipeline,
                "diagnostics_pipeline": "RAW_FRAME",
                "diagnostics_mode": self.diagnostics_mode,
                "runtime_orientation": self.runtime_orientation,
                "preview_swap_rb": self.resolved_preview_swap_rb(),
                "preview_configuration": LIVE_PREVIEW_CONFIGURATION,
                "preview_source": "FRAME_MASTER",
                "diagnostics_source": "RAW_FRAME",
                "frame_master_status": "available" if self._frame_master is not None else "pending",
                "jpeg_pipeline": LIVE_PREVIEW_PIPELINE,
                "jpeg_encoder": JPEG_ENCODER_INPUT_FORMAT,
                "frame_id": frame_id,
                "shape": self._frame_master.shape if self._frame_master is not None else None,
                "dtype": self._frame_master.dtype if self._frame_master is not None else None,
                "rotation": rotation,
                "flip_horizontal": flip_horizontal,
                "flip_vertical": flip_vertical,
            }
            for name, meta in self._consumer_frames.items():
                if meta.pipeline != pipeline or meta.rotation != rotation or meta.flip_horizontal != flip_horizontal or meta.flip_vertical != flip_vertical or meta.frame_id != frame_id:
                    logging.getLogger("gatekeeper").error("FRAME PIPELINE MISMATCH: %s %s", name, meta.as_dict())
            return snapshot

    def _verify_consumers(self, frame_id: int | None, checksum: str | None) -> bool:
        expected = {"preview", "snapshot", "motion", "plate"}
        same_ids = all(self._consumer_frames.get(name) is not None and self._consumer_frames[name].frame_id == frame_id for name in expected)
        ok = same_ids and checksum is not None
        logging.getLogger("gatekeeper").info("FRAME VERIFIED" if ok else "FRAME MISMATCH")
        if not ok:
            logging.getLogger("gatekeeper").error("FRAME PIPELINE ERROR")
        return ok

    def save_snapshot(self) -> Path:
        """Save FRAME_MASTER JPEG bytes into snapshots/."""
        with self._lock:
            frame = self._frame
        encoder = self.camera_manager.encode_jpeg if self.camera_manager is not None else CameraManager.encode_jpeg
        jpeg = encoder(frame)
        if jpeg is None:
            raise RuntimeError("No live preview frame is available to snapshot.")
        timestamp = datetime.now(timezone.utc).isoformat().replace(":", "").replace("+", "Z")
        output_path = SNAPSHOT_DIR / f"snapshot_{timestamp}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(jpeg)
        return output_path

    def update_camera_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Persist camera updates and apply them to the live CameraManager."""
        logger = logging.getLogger("gatekeeper")
        config = load_config(self.config_path)
        camera_config = config.setdefault("camera", {})
        old_pipeline = str(camera_config.get("pipeline", self.camera_pipeline))
        old_rotation = int(camera_config.get("rotation", self.camera_orientation.get("rotation", 0) or 0))
        old_flip_horizontal = bool(camera_config.get("flip_horizontal", self.camera_orientation.get("flip_horizontal", False)))
        old_flip_vertical = bool(camera_config.get("flip_vertical", self.camera_orientation.get("flip_vertical", False)))
        camera_config.update(updates)
        new_pipeline = str(camera_config.get("pipeline", old_pipeline))
        new_rotation = int(camera_config.get("rotation", old_rotation))
        new_flip_horizontal = bool(camera_config.get("flip_horizontal", old_flip_horizontal))
        new_flip_vertical = bool(camera_config.get("flip_vertical", old_flip_vertical))
        pipeline_changed = "pipeline" in updates and new_pipeline != old_pipeline
        orientation_changed = any(key in updates for key in ("rotation", "flip_horizontal", "flip_vertical")) and (new_rotation, new_flip_horizontal, new_flip_vertical) != (old_rotation, old_flip_horizontal, old_flip_vertical)
        if pipeline_changed:
            log_pipeline_changed(logger, old_pipeline, new_pipeline)
        if orientation_changed:
            log_camera_reconfigure(logger, old_rotation, new_rotation, old_flip_horizontal, new_flip_horizontal, old_flip_vertical, new_flip_vertical)
            if self.motion_event_manager is not None and hasattr(self.motion_event_manager, "reset"):
                self.motion_event_manager.reset()
        save_config(config, self.config_path)
        if self.camera_manager is not None:
            self.camera_manager.reconfigure(camera_config)
        self.camera_pipeline = str(camera_config.get("pipeline", self.camera_pipeline))
        self.camera_orientation = {
            "rotation": int(camera_config.get("rotation", 0)),
            "flip_horizontal": bool(camera_config.get("flip_horizontal", False)),
            "flip_vertical": bool(camera_config.get("flip_vertical", False)),
        }
        self.camera_controls = dict(camera_config.get("controls", {}) or {})
        return {"camera": camera_config}



    def calibration_thresholds(self) -> dict[str, Any]:
        """Expose existing PlateDetector thresholds using calibration labels."""
        thresholds = dict(self.detector_thresholds or {})
        return {
            "min_area": thresholds.get("min_area"),
            "max_area": thresholds.get("max_area"),
            "min_aspect_ratio": thresholds.get("min_aspect_ratio", thresholds.get("aspect_ratio_min")),
            "max_aspect_ratio": thresholds.get("max_aspect_ratio", thresholds.get("aspect_ratio_max")),
            "min_rectangularity": thresholds.get("min_rectangularity"),
            "max_rotation": thresholds.get("max_rotation"),
            "confidence_threshold": thresholds.get("confidence_threshold"),
        }

    def plate_calibration_snapshot(self) -> dict[str, Any]:
        """Return the current runtime plate calibration state."""
        with self._lock:
            selected_box = self.plate_bounding_box
            candidates = [plate_candidate_metadata(candidate, selected_box) for candidate in self.candidates]
            selected = next((candidate for candidate in candidates if candidate["selected"]), None)
            return {
                "enabled": bool(self.plate_calibration_config.get("enabled", False)),
                "candidate_count": len(candidates),
                "selected_candidate": selected,
                "candidates": candidates,
                "timestamp": self.updated_at,
                "frame_width": self.width or 0,
                "frame_height": self.height or 0,
                "thresholds": self.calibration_thresholds(),
            }

    def save_current_calibration_frame(self) -> Path:
        """Persist the latest annotated plate calibration frame."""
        with self._lock:
            frame = self._calibration_frame
            source = self._frame
            candidates = list(self.candidates)
            detection = None
            if self.plate_bounding_box is not None:
                detection = PlateDetection(self.plate_bounding_box, float(self.confidence or 0.0), None, candidates=tuple(candidates))
        if frame is None:
            if source is None:
                raise RuntimeError("No calibration frame is available.")
            return save_calibration_frame(source, detection, candidates)
        return CameraManager.write_image(frame, calibration_image_path(datetime.now(timezone.utc).isoformat()))

    def events_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.events)

    def resolved_preview_swap_rb(self) -> bool:
        return str(self.preview_swap_rb or "true").lower() == "true"

    def encode_preview_jpeg(self, frame: Any) -> bytes | None:
        """Encode HTTP Live Preview with the Release 0.6.9 preview-only swap behavior."""
        preview_frame = CameraManager.apply_pipeline(frame, "swap_rb") if self.resolved_preview_swap_rb() else frame
        encoder = self.camera_manager.encode_jpeg if self.camera_manager is not None else CameraManager.encode_jpeg
        return encoder(preview_frame)

    def latest_frame_jpeg(self) -> bytes | None:
        with self._lock:
            return self._frame_jpeg

    def latest_plate_jpeg(self) -> bytes | None:
        with self._lock:
            return self._plate_jpeg

    def debug_jpeg(self, name: str) -> bytes | None:
        with self._lock:
            if name == "final":
                return self._frame_jpeg
            if name in {"candidates", "contours"}:
                return self._frame_jpeg
            frame = self._frame
        if frame is None:
            return None
        return self._frame_jpeg



def apply_orientation(frame: Any, *, rotation: int = 0, flip_horizontal: bool = False, flip_vertical: bool = False) -> Any:
    """Backward-compatible wrapper for CameraManager rotation/flip stages."""
    return CameraManager.apply_flip(CameraManager.apply_rotation(frame, rotation), flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)

def encode_jpeg(frame: Any, *, color_order: str = "RGB") -> bytes | None:
    """Backward-compatible wrapper around CameraManager.encode_jpeg()."""
    if color_order.upper() != "RGB":
        return CameraManager.encode_jpeg(frame, color_order=color_order)
    return CameraManager.encode_jpeg(frame)


def _rect_from_config(value: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(value, dict):
        return None
    x = int(value.get("x", 0)); y = int(value.get("y", 0))
    w = int(value.get("width", width)); h = int(value.get("height", height))
    return max(0, x), max(0, y), max(1, w), max(1, h)


def render_live_preview_frame(
    frame: Any, *, version: str, git_commit: str, motion_state: str, fps: float,
    resolution: str, timestamp: str, confidence: float | None,
    plate_detection: PlateDetection | None, display_config: dict[str, Any] | None = None,
) -> Any:
    """Draw HTTP calibration overlays directly onto a copy of an RGB frame."""
    import cv2

    cfg = display_config or {}
    output = frame.copy() if hasattr(frame, "copy") else frame
    if not hasattr(output, "shape") or len(output.shape) < 2:
        return output
    height, width = output.shape[:2]
    overlay = output.copy()
    green = (0, 255, 0)
    if cfg.get("show_grid", True):
        for x in (width // 3, 2 * width // 3):
            cv2.line(output, (x, 0), (x, height), (80, 120, 80), 1)
        for y in (height // 3, 2 * height // 3):
            cv2.line(output, (0, y), (width, y), (80, 120, 80), 1)
    if cfg.get("show_safe_area", True):
        for key, color in (("detection_area", (0, 255, 255)), ("plate_ideal_area", (255, 200, 0))):
            rect = _rect_from_config(cfg.get(key), width, height)
            if rect:
                x, y, w, h = rect; cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
    if cfg.get("show_crosshair", True):
        cx, cy = width // 2, height // 2
        cv2.line(overlay, (cx, 0), (cx, height), green, 1)
        cv2.line(overlay, (0, cy), (width, cy), green, 1)
        cv2.addWeighted(overlay, 0.45, output, 0.55, 0, output)
    candidates = list(getattr(plate_detection, "candidates", []) or [])
    if cfg.get("debug_candidates", False) and candidates:
        selected_box = plate_detection.bounding_box if plate_detection is not None else None
        for candidate in candidates:
            x, y, w, h = candidate.bounding_box
            color = green if candidate.bounding_box == selected_box else ((255, 255, 0) if candidate.valid else (255, 0, 0))
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            label = f"S:{candidate.confidence:.2f} AR:{candidate.aspect_ratio:.2f} A:{candidate.area:.0f} R:{candidate.rectangularity:.2f}"
            cv2.putText(output, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    elif cfg.get("show_bbox", True) and plate_detection is not None:
        x, y, w, h = plate_detection.bounding_box
        cv2.rectangle(output, (x, y), (x + w, y + h), green, 2)
        cv2.putText(output, f"{plate_detection.confidence:.3f} ({x},{y},{w},{h})", (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1, cv2.LINE_AA)
    if cfg.get("show_status", True):
        top = [f"GateKeeper AI v{version}", f"Git: {git_commit}", f"FPS: {fps:.2f}", f"Resolution: {resolution}", f"Timestamp: {timestamp}", f"Motion: {motion_state}"]
        for i, line in enumerate(top):
            cv2.putText(output, line, (10, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, green, 1, cv2.LINE_AA)
        bottom = [f"Confidence: {confidence:.3f}" if confidence is not None else "Confidence: n/a", "Plate detected" if plate_detection is not None else "Not detected"]
        for i, line in enumerate(reversed(bottom)):
            size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.putText(output, line, (max(10, width - size[0] - 10), height - 12 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, green, 1, cv2.LINE_AA)
    return output


class CameraManager:
    """Manage the Raspberry Pi camera through Picamera2."""

    def __init__(self, width: int, height: int, fps: int, rotation: int = 0, flip_horizontal: bool = False, flip_vertical: bool = False, pipeline: str = "rgb", controls: dict[str, Any] | None = None) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.rotation = rotation
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.pipeline = pipeline
        self.controls = controls or {}
        self._camera: Any | None = None
        self._lock = threading.RLock()
        self._paused = False
        self._pixel_format = "RGB888"
        self._frame_format = "RGB"
        self._last_shape: tuple[int, ...] | None = None
        self._last_dtype: str | None = None
        self._inspect_next_frame = True
        self._last_frame_master: FrameMaster | None = None
        self.raw_frame_id: int | None = None
        self.diagnostics_source_id: int | None = None
        self._last_master_log_at = 0.0
        self.logger = logging.getLogger("gatekeeper")

    def initialize(self) -> None:
        """Initialize and start Picamera2 if needed."""
        with self._lock:
            if self._camera is not None:
                return

            from picamera2 import Picamera2

            self._camera = Picamera2()
            configuration = self._camera.create_still_configuration(
                main={"size": (self.width, self.height), "format": self._pixel_format}
            )
            self._camera.configure(configuration)
            self._camera.start()
            self._paused = False
            self.apply_controls(self.controls)
            time.sleep(max(0.1, 1 / max(self.fps, 1)))

    def health_check(self) -> bool:
        """Return True when the camera can be initialized."""
        self.initialize()
        return self._camera is not None

    @staticmethod
    def apply_pipeline(frame: Any, pipeline: str = "rgb") -> Any:
        """Generate the configured color pipeline from RAW_FRAME."""
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 3:
            return frame
        import cv2
        normalized = str(pipeline or "rgb").lower()
        if normalized in {"raw", "rgb"}:
            return frame
        if normalized == "bgr":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if normalized == "swap_rb":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        raise ValueError(f"Unsupported camera pipeline: {pipeline}")

    @staticmethod
    def apply_rotation(frame: Any, rotation: int = 0) -> Any:
        """Apply deterministic FRAME_MASTER rotation."""
        if frame is None or not hasattr(frame, "shape"):
            return frame
        import cv2
        normalized = rotation % 360
        if normalized == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if normalized == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if normalized == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if normalized != 0:
            raise ValueError(f"Unsupported camera rotation: {rotation}")
        return frame

    @staticmethod
    def apply_flip(frame: Any, *, flip_horizontal: bool = False, flip_vertical: bool = False) -> Any:
        """Apply deterministic FRAME_MASTER flips."""
        if frame is None or not hasattr(frame, "shape"):
            return frame
        import cv2
        if flip_horizontal and flip_vertical:
            return cv2.flip(frame, -1)
        if flip_horizontal:
            return cv2.flip(frame, 1)
        if flip_vertical:
            return cv2.flip(frame, 0)
        return frame

    @staticmethod
    def to_gray(frame: Any) -> Any:
        """Convert a FRAME_MASTER consumer frame to grayscale inside CameraManager."""
        if len(frame.shape) != 3:
            return frame
        import cv2
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def frame_checksum(frame: Any) -> str | None:
        """Return a lightweight CRC32 checksum for a frame object."""
        if frame is None:
            return None
        payload = frame.tobytes() if hasattr(frame, "tobytes") else bytes(str(frame), "utf-8")
        return f"{zlib.crc32(payload) & 0xffffffff:08x}"

    @staticmethod
    def diagnostic_frames(raw_frame: Any) -> dict[str, tuple[Any, str]]:
        """Generate raw/rgb/bgr/swap_rb diagnostics from RAW_FRAME only."""
        return {
            "frame_raw.jpg": (raw_frame, "raw"),
            "frame_rgb.jpg": (raw_frame, "rgb"),
            "frame_bgr.jpg": (CameraManager.apply_pipeline(raw_frame, "bgr"), "bgr"),
            "frame_swap_rb.jpg": (CameraManager.apply_pipeline(raw_frame, "swap_rb"), "swap_rb"),
        }

    @staticmethod
    def write_image(frame: Any, output_path: Path) -> Path:
        """Write an image file using CameraManager as the pixel boundary."""
        import cv2
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = CameraManager.apply_pipeline(frame, "bgr") if hasattr(frame, "shape") and len(frame.shape) == 3 else frame
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Unable to save image: {output_path}")
        return output_path

    def get_processed_frame(self) -> Any:
        """Return the single processed RGB frame used by every consumer."""
        with self._lock:
            self.initialize()
            if self._camera is None:
                raise RuntimeError("Picamera2 is not initialized.")
            raw = self._camera.capture_array()
            self.raw_frame_id = id(raw)
            self._last_shape = tuple(raw.shape) if hasattr(raw, "shape") else None
            self._last_dtype = str(raw.dtype) if hasattr(raw, "dtype") else None
            frame = self.apply_pipeline(raw, self.pipeline)
            frame = self.apply_rotation(frame, self.rotation)
            frame = self.apply_flip(frame, flip_horizontal=self.flip_horizontal, flip_vertical=self.flip_vertical)
            if self._inspect_next_frame:
                self.logger.info("======================================================")
                self.logger.info("FRAME INSPECT")
                self.logger.info("======================================================")
                log_frame_step(self.logger, "Capture", raw, pipeline="Picamera2", rotation=0, flip_horizontal=False, flip_vertical=False)
                log_frame_step(self.logger, "CameraManager", frame, pipeline=self.pipeline, rotation=self.rotation, flip_horizontal=self.flip_horizontal, flip_vertical=self.flip_vertical)
                for component in ("Motion Detector", "Plate Detector", "HTTP Preview", "Snapshot"):
                    self.logger.info("↓")
                    log_frame_step(self.logger, component, frame, pipeline=self.pipeline, rotation=self.rotation, flip_horizontal=self.flip_horizontal, flip_vertical=self.flip_vertical)
                self.logger.info("↓")
                self.logger.info("JPEG encoder")
                self.logger.info("input format: %s", JPEG_ENCODER_INPUT_FORMAT)
                self.logger.info("======================================================")
                self._inspect_next_frame = False
            self._last_frame_master = FrameMaster.from_frame(frame, pipeline=self.pipeline, rotation=self.rotation, flip_horizontal=self.flip_horizontal, flip_vertical=self.flip_vertical)
            now = time.monotonic()
            if now - self._last_master_log_at >= 10:
                self.logger.info("FRAME MASTER frame id=%s shape=%s pipeline=%s rotation=%s flip=(%s,%s) consumers=%s", self._last_frame_master.frame_id, self._last_frame_master.shape, self.pipeline, self.rotation, self.flip_horizontal, self.flip_vertical, "Motion Detector, Plate Detector, HTTP Preview, Snapshot, Diagnostics, Future OCR")
                self._last_master_log_at = now
            self.apply_controls(self.controls)
            return frame

    def capture_frame(self) -> Any:
        """Capture the shared processed RGB frame while keeping the camera open."""
        return self.get_processed_frame()

    def capture_raw_frame(self) -> Any:
        """Capture exactly what Picamera2 returns for diagnostics only."""
        with self._lock:
            self.initialize()
            if self._camera is None:
                raise RuntimeError("Picamera2 is not initialized.")
            raw = self._camera.capture_array()
            self.raw_frame_id = id(raw)
            self.diagnostics_source_id = id(raw)
            self._last_shape = tuple(raw.shape) if hasattr(raw, "shape") else None
            self._last_dtype = str(raw.dtype) if hasattr(raw, "dtype") else None
            return raw

    def generate_diagnostics(self, output_dir: Path = DIAGNOSTICS_DIR, *, runtime_orientation: bool = False) -> dict[str, dict[str, str]]:
        """Generate diagnostics from RAW_FRAME only; never from FRAME_MASTER."""
        raw_frame = self.capture_raw_frame()
        orientation = {"rotation": self.rotation, "flip_horizontal": self.flip_horizontal, "flip_vertical": self.flip_vertical} if runtime_orientation else None
        return generate_camera_diagnostics(raw_frame, output_dir, runtime_orientation=orientation)

    def apply_controls(self, controls: dict[str, Any] | None = None) -> None:
        """Apply supported Picamera2 controls immediately."""
        self.controls = dict(controls or {})
        if self._camera is not None and self.controls:
            self._camera.set_controls(self.controls)

    def reconfigure(self, camera_config: dict[str, Any]) -> None:
        """Safely apply live camera pipeline, orientation, flips, and controls."""
        with self._lock:
            self._paused = True
            was_initialized = self._camera is not None
            camera = self._camera
            if camera is not None and hasattr(camera, "stop"):
                camera.stop()
            self.rotation = int(camera_config.get("rotation", self.rotation))
            self.flip_horizontal = bool(camera_config.get("flip_horizontal", self.flip_horizontal))
            self.flip_vertical = bool(camera_config.get("flip_vertical", self.flip_vertical))
            self.pipeline = str(camera_config.get("pipeline", self.pipeline))
            self.controls = dict(camera_config.get("controls", self.controls) or {})
            if was_initialized and camera is not None:
                self._inspect_next_frame = True
                configuration = camera.create_still_configuration(
                    main={"size": (self.width, self.height), "format": self._pixel_format}
                )
                if hasattr(camera, "configure"):
                    camera.configure(configuration)
                if hasattr(camera, "start"):
                    camera.start()
                self.apply_controls(self.controls)
                time.sleep(max(0.1, 1 / max(self.fps, 1)))
            self._paused = False

    @staticmethod
    def encode_jpeg(frame: Any, *, color_order: str = "RGB") -> bytes | None:
        """Prepare JPEG bytes for every HTTP, snapshot, and diagnostic consumer."""
        if frame is None:
            return None
        import cv2

        if len(frame.shape) == 3:
            normalized = color_order.upper()
            if normalized == "RGB":
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif normalized in {"BGR", "GRAY"}:
                image = frame
            else:
                raise ValueError(f"Unsupported frame color order: {color_order}")
        else:
            image = frame
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            return None
        return buffer.tobytes()

    def capture(self, output_path: Path = LATEST_IMAGE) -> Path:
        """Capture a JPEG image to the requested path."""
        self.initialize()
        if self._camera is None:
            raise RuntimeError("Picamera2 is not initialized.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.get_processed_frame()
        jpeg = self.encode_jpeg(frame)
        if jpeg is None:
            raise RuntimeError(f"Unable to encode capture: {output_path}")
        output_path.write_bytes(jpeg)
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
            "width": self.width,
            "height": self.height,
            "pixel_format": self._pixel_format,
            "sensor": properties.get("Sensor", "unknown") if self._camera is not None else "unknown",
            "camera_model": model,
            "frame_format": self._frame_format,
            "shape": self._last_shape,
            "dtype": self._last_dtype,
            "color_pipeline": self.pipeline,
            "jpeg_encoder_format": JPEG_ENCODER_INPUT_FORMAT,
            "orientation": {"rotation": self.rotation, "flip_horizontal": self.flip_horizontal, "flip_vertical": self.flip_vertical},
            "preview_pipeline": LIVE_PREVIEW_PIPELINE,
        }

    def stop(self) -> None:
        """Stop and close Picamera2 resources during shutdown."""
        with self._lock:
            if self._camera is not None:
                self._camera.stop()
                self._camera.close()
                self._camera = None
                self._inspect_next_frame = True


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
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
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

    def insert_plate(
        self,
        event_id: str,
        image_path: str,
        confidence: float,
        created_at: str,
    ) -> None:
        """Insert one detected plate crop record."""
        if self.connection is None:
            raise RuntimeError("Database is not initialized.")
        self.connection.execute(
            """
            INSERT INTO plates (event_id, image_path, confidence, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (event_id, image_path, confidence, created_at),
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
        self.logger = logging.getLogger("gatekeeper")

    def reset(self) -> None:
        """Clear previous frame state after camera shape/orientation changes."""
        self.previous_frame = None

    def _prepare_frame(self, frame: Any) -> Any:
        import cv2

        gray = CameraManager.to_gray(frame)
        return cv2.GaussianBlur(gray, (21, 21), 0)

    def detect(self, frame: Any) -> tuple[bool, float]:
        """Return whether motion exists and the largest contour area."""
        import cv2

        prepared = self._prepare_frame(frame)
        if self.previous_frame is None:
            self.previous_frame = prepared
            return False, 0.0

        if getattr(self.previous_frame, "shape", None) != getattr(prepared, "shape", None):
            self.logger.warning("MotionDetector reset because frame size changed")
            self.logger.warning("old shape: %s", getattr(self.previous_frame, "shape", None))
            self.logger.warning("new shape: %s", getattr(prepared, "shape", None))
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
        return CameraManager.write_image(frame, output_path)


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
    orientation = camera_info.get("orientation", {})
    logger.info("======================================================")
    logger.info("GateKeeper AI v%s", version)
    logger.info("FRAME PIPELINE")
    logger.info("======================================================")
    logger.info("Picamera2")
    logger.info("---------")
    logger.info("Camera model: %s", camera_info.get("camera_model"))
    logger.info("Sensor: %s", camera_info.get("sensor"))
    logger.info("Pixel format: %s", camera_info.get("pixel_format"))
    logger.info("Frame format: %s", camera_info.get("frame_format"))
    logger.info("Width: %s", camera_info.get("width"))
    logger.info("Height: %s", camera_info.get("height"))
    logger.info("dtype: %s", camera_info.get("dtype"))
    logger.info("CameraManager")
    logger.info("-------------")
    logger.info("Selected pipeline: %s", camera_info.get("color_pipeline"))
    logger.info("Rotation: %s", orientation.get("rotation"))
    logger.info("Flip Horizontal: %s", orientation.get("flip_horizontal"))
    logger.info("Flip Vertical: %s", orientation.get("flip_vertical"))
    logger.info("Motion Detector: %s", camera_info.get("color_pipeline"))
    logger.info("Plate Detector: %s", camera_info.get("color_pipeline"))
    logger.info("HTTP Preview: %s", LIVE_PREVIEW_PIPELINE)
    logger.info("Snapshot: %s", camera_info.get("color_pipeline"))
    logger.info("Live Preview pipeline: BGR")
    logger.info("Live Preview configuration: inherited from 0.6.9")
    logger.info("Pipeline selection UI: disabled")
    logger.info("Preview swap RB: %s", str(camera_info.get("preview_swap_rb", "true")).lower())
    logger.info("Live Preview source: FRAME_MASTER")
    logger.info("Diagnostics source: RAW_FRAME")
    logger.info("Preview source: FRAME_MASTER")
    logger.info("JPEG Encoder input format: %s", camera_info.get("jpeg_encoder_format"))
    logger.info("======================================================")
    logger.info("Version: %s", version)
    logger.info("Git commit: %s", git_commit)
    logger.info("Python version: %s", platform.python_version())
    logger.info("Platform: %s", platform.system())
    logger.info("Camera backend: %s", camera_info["backend"])
    logger.info("Camera model: %s", camera_info.get("camera_model"))
    logger.info("Sensor: %s", camera_info.get("sensor"))
    logger.info("Camera resolution: %s", camera_info["resolution"])
    logger.info("Camera pixel format: %s", camera_info.get("pixel_format"))
    logger.info("Frame format: %s", camera_info.get("frame_format"))
    logger.info("Shape: %s", camera_info.get("shape"))
    logger.info("dtype: %s", camera_info.get("dtype"))
    logger.info("Camera initialized")
    logger.info("Selected pipeline: %s", camera_info.get("color_pipeline"))
    logger.info("Rotation: %s", orientation.get("rotation"))
    logger.info("Flip H: %s", orientation.get("flip_horizontal"))
    logger.info("Flip V: %s", orientation.get("flip_vertical"))
    logger.info("Motion pipeline: %s", camera_info.get("color_pipeline"))
    logger.info("Plate pipeline: %s", camera_info.get("color_pipeline"))
    logger.info("Stream pipeline: %s", LIVE_PREVIEW_PIPELINE)
    logger.info("Snapshot pipeline: %s", camera_info.get("color_pipeline"))
    logger.info("JPEG encoder format: %s", camera_info.get("jpeg_encoder_format"))
    logger.info("Capture time: %s", capture_time)
    logger.info("Image path: %s", image_path)
    logger.info("Image size: %d bytes", image_path.stat().st_size)



def log_pipeline_changed(logger: logging.Logger, old_pipeline: str, new_pipeline: str) -> None:
    """Log runtime pipeline replacement sequencing."""
    logger.info("======================================================")
    logger.info("PIPELINE CHANGED")
    logger.info("======================================================")
    logger.info("Old pipeline: %s", old_pipeline)
    logger.info("New pipeline: %s", new_pipeline)
    logger.info("Stopping Motion Detector")
    logger.info("Stopping Preview")
    logger.info("Stopping Camera")
    logger.info("Restart Camera")
    logger.info("Restart Preview")
    logger.info("Restart Motion")
    logger.info("Completed")
    logger.info("======================================================")


def log_camera_reconfigure(logger: logging.Logger, old_rotation: int, new_rotation: int, old_flip_h: bool, new_flip_h: bool, old_flip_v: bool, new_flip_v: bool) -> None:
    """Log orientation reconfiguration sequencing."""
    logger.info("======================================================")
    logger.info("CAMERA RECONFIGURE")
    logger.info("======================================================")
    logger.info("Old rotation: %s", old_rotation)
    logger.info("New rotation: %s", new_rotation)
    logger.info("Old flip H: %s", old_flip_h)
    logger.info("New flip H: %s", new_flip_h)
    logger.info("Old flip V: %s", old_flip_v)
    logger.info("New flip V: %s", new_flip_v)
    logger.info("Reset Motion Detector")
    logger.info("Reset previous_frame")
    logger.info("Restart camera")
    logger.info("Completed")
    logger.info("======================================================")

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


def plate_image_path(timestamp: str) -> Path:
    """Build a unique filesystem-safe plate crop path from an ISO timestamp."""
    safe_timestamp = timestamp.replace(":", "").replace("+", "Z")
    return IMAGE_DIR / f"plate_{safe_timestamp}.jpg"


def calibration_image_path(timestamp: str) -> Path:
    """Build a unique filesystem-safe plate calibration frame path."""
    safe_timestamp = timestamp.replace(":", "").replace("+", "Z")
    return IMAGE_DIR / f"calibration_{safe_timestamp}.jpg"


def plate_candidate_metadata(candidate: Any, selected_box: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    """Return JSON-ready plate candidate calibration metadata."""
    if hasattr(candidate, "metadata"):
        return candidate.metadata()
    bbox = tuple(getattr(candidate, "bounding_box", ()))
    selected = bool(getattr(candidate, "selected", False)) or (selected_box is not None and bbox == selected_box)
    valid = bool(getattr(candidate, "valid", selected))
    reason = str(getattr(candidate, "rejected_reason", "") or ("not_selected" if valid and not selected else ""))
    return {
        "bounding_box": bbox,
        "area": float(getattr(candidate, "area", 0.0)),
        "aspect_ratio": float(getattr(candidate, "aspect_ratio", 0.0)),
        "rectangularity": float(getattr(candidate, "rectangularity", 0.0)),
        "confidence": float(getattr(candidate, "confidence", 0.0)),
        "selected": selected,
        "rejected": not selected,
        "rejection_reason": "" if selected else reason,
    }


def annotate_calibration_frame(
    frame: Any,
    plate_detection: PlateDetection | None = None,
    candidates: list[Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> Any:
    """Draw plate calibration candidate overlays on a copy of FRAME_MASTER."""
    import cv2

    annotated = frame.copy() if hasattr(frame, "copy") else frame
    selected_box = plate_detection.bounding_box if plate_detection is not None else None
    items = list(candidates or getattr(plate_detection, "candidates", []) or [])
    selected_count = 0
    rejected_count = 0
    for candidate in items:
        data = plate_candidate_metadata(candidate, selected_box)
        selected_count += 1 if data["selected"] else 0
        rejected_count += 0 if data["selected"] else 1
        x, y, w, h = [int(v) for v in data["bounding_box"]]
        color = (0, 255, 0) if data["selected"] else ((255, 255, 0) if data["rejection_reason"] == "not_selected" else (255, 0, 0))
        thickness = 4 if data["selected"] else 2
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        lines = [
            f'A:{data["area"]:.0f}',
            f'AR:{data["aspect_ratio"]:.2f}',
            f'R:{data["rectangularity"]:.2f}',
            f'C:{data["confidence"]:.3f}',
        ]
        if data["selected"]:
            lines.extend(["SELECTED PLATE", f'Confidence: {data["confidence"]:.3f}'])
        elif data["rejection_reason"] and data["rejection_reason"] != "not_selected":
            lines.append(f'REJECT:{data["rejection_reason"]}')
        elif data["rejection_reason"] == "not_selected":
            lines.append("VALID")
        for idx, label in enumerate(lines):
            yy = max(16, y - 60 + idx * 14)
            if yy < 16 + idx * 14:
                yy = min(y + h + 16 + idx * 14, annotated.shape[0] - 4) if hasattr(annotated, "shape") else y + h + 16 + idx * 14
            cv2.putText(annotated, label, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    t = thresholds or {}
    summary = [
        "PLATE CALIBRATION",
        f"Candidates: {len(items)}",
        f"Selected: {selected_count}",
        f"Rejected: {rejected_count}",
        f"Min area: {t.get('min_area', 'n/a')}",
        f"Max area: {t.get('max_area', 'n/a')}",
        f"Aspect: {float(t.get('min_aspect_ratio') or 0):.2f} - {float(t.get('max_aspect_ratio') or 0):.2f}",
        f"Rectangularity: {float(t.get('min_rectangularity') or 0):.2f}",
        f"Rotation: {int(float(t.get('max_rotation') or 0))}°",
        f"Confidence: {float(t.get('confidence_threshold') or 0):.2f}",
    ]
    for i, line in enumerate(summary):
        cv2.putText(annotated, line, (10, 22 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return annotated


def save_calibration_frame(frame: Any, plate_detection: PlateDetection | None, candidates: list[Any], timestamp: datetime | None = None) -> Path:
    """Save an annotated calibration frame under images/calibration_*.jpg."""
    timestamp = timestamp or datetime.now(timezone.utc)
    annotated = annotate_calibration_frame(frame, plate_detection, candidates)
    return CameraManager.write_image(annotated, calibration_image_path(timestamp.isoformat()))


def is_display_available() -> bool:
    """Return whether an OpenCV preview window can be shown."""
    if platform.system() in {"Windows", "Darwin"}:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def debug_frame_path(timestamp: str) -> Path:
    """Build a filesystem-safe annotated debug frame path."""
    safe_timestamp = timestamp.replace(":", "").replace("+", "Z")
    return DEBUG_DIR / f"frame_{safe_timestamp}.jpg"


class DebugVision:
    """Render and optionally save annotated frames for detector tuning."""

    WINDOW_NAME = "GateKeeper AI Debug Vision"

    def __init__(
        self,
        *,
        enabled: bool = False,
        live_preview: bool = False,
        save_annotated_frames: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.enabled = enabled
        self.live_preview = live_preview and is_display_available()
        self.save_annotated_frames = save_annotated_frames
        self.logger = logger or logging.getLogger(__name__)
        self.overlays_enabled = True
        self.quit_requested = False
        if enabled and live_preview and not self.live_preview:
            self.logger.info("DISPLAY unavailable; Debug Vision preview disabled.")

    def annotate(
        self,
        frame: Any,
        *,
        motion_state: str,
        fps: float,
        camera_resolution: str,
        timestamp: datetime,
        plate_detection: PlateDetection | None = None,
    ) -> Any:
        """Return a copy of ``frame`` with Debug Vision overlays."""
        import cv2

        annotated = frame.copy() if hasattr(frame, "copy") else frame
        if not self.enabled or not self.overlays_enabled:
            return annotated

        text_color = (0, 255, 0)
        line_height = 24
        lines = (
            f"Motion: {motion_state}",
            f"FPS: {fps:.2f}",
            f"Resolution: {camera_resolution}",
            f"Timestamp: {timestamp.isoformat()}",
        )
        for index, line in enumerate(lines):
            cv2.putText(
                annotated,
                line,
                (10, 25 + (index * line_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
                cv2.LINE_AA,
            )

        if plate_detection is not None:
            x, y, width, height = plate_detection.bounding_box
            cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 0, 255), 2)
            label = (
                f"Plate {plate_detection.confidence:.3f} " f"({x},{y},{width},{height})"
            )
            cv2.putText(
                annotated,
                label,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def save_frame(
        self, annotated_frame: Any, timestamp: datetime | None = None
    ) -> Path:
        """Save one annotated frame into the debug directory."""
        import cv2

        timestamp = timestamp or datetime.now(timezone.utc)
        output_path = debug_frame_path(timestamp.isoformat())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        CameraManager.write_image(annotated_frame, output_path)
        self.logger.info("Debug frame saved: %s", output_path)
        return output_path

    def show(self, annotated_frame: Any) -> None:
        """Display a frame and process Debug Vision keyboard shortcuts."""
        if not self.enabled or not self.live_preview:
            return
        import cv2

        cv2.imshow(self.WINDOW_NAME, CameraManager.apply_pipeline(annotated_frame, "bgr") if len(annotated_frame.shape) == 3 else annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.quit_requested = True
        elif key == ord("s"):
            self.save_frame(annotated_frame)
        elif key == ord("d"):
            self.overlays_enabled = not self.overlays_enabled

    def close(self) -> None:
        """Close Debug Vision preview resources."""
        if self.live_preview:
            import cv2

            cv2.destroyWindow(self.WINDOW_NAME)


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
        plate_detector: PlateDetector | None = None,
        debug_vision: DebugVision | None = None,
        live_preview_state: LivePreviewState | None = None,
        plate_calibration: dict[str, Any] | None = None,
    ) -> None:
        self.detector = detector
        self.database = database
        self.logger = logger
        self.end_delay_seconds = end_delay_seconds
        self.plate_detector = plate_detector
        self.debug_vision = debug_vision
        self.live_preview_state = live_preview_state
        self.plate_calibration = plate_calibration or {"enabled": False}
        self.last_plate_crop: Any | None = None
        self.last_plate_detection: PlateDetection | None = None
        self.last_plate_candidates: list[Any] = []
        self.state = self.IDLE
        self.event_id: str | None = None
        self.start_time: datetime | None = None
        self.last_motion_time: datetime | None = None
        self.max_contour_area = 0.0
        self.image_start: Path | None = None

    def reset(self) -> None:
        """Reset motion event state and detector frame cache after camera changes."""
        if hasattr(self.detector, "reset"):
            self.detector.reset()
        elif hasattr(self.detector, "previous_frame"):
            self.detector.previous_frame = None
        self.state = self.IDLE
        self.event_id = None
        self.start_time = None
        self.last_motion_time = None
        self.max_contour_area = 0.0
        self.image_start = None

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
                if self.plate_calibration.get("enabled"):
                    self._detect_plate(frame, now, persist_crop=False)
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
        if self.live_preview_state is not None:
            self.live_preview_state.record_motion_event(
                {
                    "type": "MOTION_START",
                    "timestamp": now.isoformat(),
                    "event_id": self.event_id,
                    "image_start": str(self.image_start),
                    "max_contour_area": self.max_contour_area,
                }
            )
        self._detect_plate(frame, now)
        if self.debug_vision is not None and self.debug_vision.save_annotated_frames:
            annotated = self.debug_vision.annotate(
                frame,
                motion_state=self.state,
                fps=0.0,
                camera_resolution=(
                    f"{frame.shape[1]}x{frame.shape[0]}"
                    if hasattr(frame, "shape")
                    else "unknown"
                ),
                timestamp=now,
                plate_detection=self.last_plate_detection,
            )
            self.debug_vision.save_frame(annotated, now)
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

    def _detect_plate(self, frame: Any, now: datetime, *, persist_crop: bool = True) -> None:
        if self.plate_detector is None or self.event_id is None:
            return
        detection = self.plate_detector.detect(frame)
        self.last_plate_detection = detection
        candidates = list(getattr(detection, "candidates", []) or getattr(self.plate_detector, "last_candidates", []) or [])
        self.last_plate_candidates = candidates
        selected_box = detection.bounding_box if detection is not None else None
        for candidate in candidates:
            data = plate_candidate_metadata(candidate, selected_box)
            self.logger.info("Plate candidate bbox=%s area=%.0f aspect=%.3f rectangularity=%.3f confidence=%.3f selected=%s reason=%s", data["bounding_box"], data["area"], data["aspect_ratio"], data["rectangularity"], data["confidence"], data["selected"], data["rejection_reason"] or "selected")
        if self.plate_calibration.get("enabled") and self.plate_calibration.get("save_frames", False):
            annotated = annotate_calibration_frame(frame, detection, candidates)
            path = calibration_image_path(now.isoformat())
            threading.Thread(target=CameraManager.write_image, args=(annotated, path), daemon=True).start()
            self.logger.info("Calibration frame saved: %s", path)
        if detection is None:
            return
        crop_path = self._save_plate_crop(frame, detection, plate_image_path(now.isoformat())) if persist_crop else None
        self.logger.info("Plate detected")
        self.logger.info("Confidence: %.3f", detection.confidence)
        self.logger.info("Bounding box: %s", detection.bounding_box)
        if crop_path is not None:
            self.logger.info("Crop path: %s", crop_path)
        self.logger.info("Plate selected bbox=%s confidence=%.3f crop=%s", detection.bounding_box, detection.confidence, crop_path)
        if self.debug_vision is not None and self.debug_vision.save_annotated_frames:
            annotated = self.debug_vision.annotate(
                frame,
                motion_state=self.state,
                fps=0.0,
                camera_resolution=(
                    f"{frame.shape[1]}x{frame.shape[0]}"
                    if hasattr(frame, "shape")
                    else "unknown"
                ),
                timestamp=now,
                plate_detection=detection,
            )
            self.debug_vision.save_frame(annotated, now)
        if persist_crop and crop_path is not None and self.database is not None:
            self.database.insert_plate(
                self.event_id,
                str(crop_path),
                detection.confidence,
                now.isoformat(),
            )

    def _save_plate_crop(
        self, frame: Any, detection: PlateDetection, output_path: Path
    ) -> Path:
        import cv2

        crop = (
            self.plate_detector.crop(frame, detection) if self.plate_detector else frame
        )
        self.last_plate_crop = crop
        return CameraManager.write_image(crop, output_path)

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
        if self.live_preview_state is not None:
            self.live_preview_state.record_motion_event(
                {
                    "type": "MOTION_END",
                    "timestamp": now.isoformat(),
                    "event_id": self.event_id,
                    "start_time": (
                        self.start_time.isoformat() if self.start_time else None
                    ),
                    "end_time": now.isoformat(),
                    "duration": duration,
                    "max_contour_area": self.max_contour_area,
                    "image_start": str(self.image_start) if self.image_start else None,
                    "image_end": str(image_end),
                }
            )
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


from src.gatekeeper.web.server import LivePreviewServer


def validate_frame_pipeline(camera: CameraManager, logger: logging.Logger) -> bool:
    """Validate startup consumers agree on the latest FRAME_MASTER metadata."""
    meta = getattr(camera, "_last_frame_master", None)
    if meta is None:
        logger.error("FRAME PIPELINE ERROR")
        return False
    consumers = {name: meta for name in ("Preview", "Snapshot", "Motion", "Plate")}
    ok = all(item == meta for item in consumers.values())
    logger.info("FRAME PIPELINE VERIFIED" if ok else "FRAME PIPELINE ERROR")
    return ok


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
    plate_detector: PlateDetector | None = None,
    debug_vision: DebugVision | None = None,
    plate_calibration: dict[str, Any] | None = None,
    display_manager: DisplayManager | None = None,
    live_preview_state: LivePreviewState | None = None,
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
        motion_detector,
        database,
        logger,
        end_delay_seconds,
        plate_detector,
        debug_vision,
        live_preview_state,
        plate_calibration,
    )
    camera_resolution = camera.get_info().get("resolution", "unknown")
    last_frame_started = time.monotonic()
    try:
        logger.info("Application remains running; press Ctrl+C to stop.")
        while running and (stop_event is None or not stop_event.is_set()):
            loop_started = time.monotonic()
            frame = camera.capture_frame()
            if motion_enabled:
                motion_event_manager.process_frame(frame)
            now = datetime.now(timezone.utc)
            frame_interval = max(loop_started - last_frame_started, 0.000001)
            measured_fps = 1 / frame_interval
            if live_preview_state is not None:
                live_preview_state.update_frame(
                    frame,
                    motion_state=motion_event_manager.state,
                    fps=measured_fps,
                    resolution=camera_resolution,
                    plate_detection=motion_event_manager.last_plate_detection,
                    plate_crop=motion_event_manager.last_plate_crop,
                    plate_candidates=motion_event_manager.last_plate_candidates,
                )
            if display_manager is not None and display_manager.requested_enabled:
                display_manager.update_frame(
                    frame,
                    motion_state=motion_event_manager.state,
                    fps=measured_fps,
                    timestamp=now,
                    plate_detection=motion_event_manager.last_plate_detection,
                )
                if display_manager.quit_requested:
                    logger.info("Display quit requested; shutting down cleanly.")
                    running = False
                    if stop_event is not None:
                        stop_event.set()
            if debug_vision is not None and debug_vision.enabled:
                annotated = debug_vision.annotate(
                    frame,
                    motion_state=motion_event_manager.state,
                    fps=measured_fps,
                    camera_resolution=camera_resolution,
                    timestamp=now,
                    plate_detection=motion_event_manager.last_plate_detection,
                )
                debug_vision.show(annotated)
                if debug_vision.quit_requested:
                    logger.info("Debug Vision quit requested; shutting down cleanly.")
                    running = False
                    if stop_event is not None:
                        stop_event.set()
            last_frame_started = loop_started
            elapsed = time.monotonic() - loop_started
            time.sleep(max(0, interval - elapsed))
    finally:
        camera.stop()
        if display_manager is not None:
            display_manager.close()
        if debug_vision is not None:
            debug_vision.close()
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
        rotation=int(camera_config.get("rotation", 0)),
        flip_horizontal=bool(camera_config.get("flip_horizontal", False)),
        flip_vertical=bool(camera_config.get("flip_vertical", False)),
        pipeline=str(camera_config.get("pipeline", "rgb")),
        controls=dict(camera_config.get("controls", {}) or {}),
    )
    database = EventDatabase()
    motion_config = config.get("motion", {})
    motion_detector = MotionDetector(
        min_area=int(motion_config.get("min_area", 1000)),
        threshold=int(motion_config.get("threshold", 25)),
    )
    end_delay_seconds = float(motion_config.get("end_delay_seconds", 2))
    plate_config = config.get("plate_detector", {})
    plate_detector = PlateDetector(
        min_aspect_ratio=float(plate_config.get("aspect_ratio_min", 3.5)),
        max_aspect_ratio=float(plate_config.get("aspect_ratio_max", 6.5)),
        min_area=float(plate_config.get("min_area", 2500)),
        max_area=float(plate_config.get("max_area", 150000)),
        confidence_threshold=float(plate_config.get("confidence_threshold", 0.70)),
        min_rectangularity=float(plate_config.get("min_rectangularity", 0.80)),
        max_rotation=float(plate_config.get("max_rotation", 15)),
        border_margin=int(plate_config.get("border_margin", 20)),
        debug=bool(plate_config.get("debug", False)),
    )
    debug_config = config.get("debug", {})
    calibration_config = config.get("plate_calibration", {})
    web_config = config.get("web", {})
    display_config = config.get("display", {})
    display_manager = DisplayManager(
        enabled=bool(display_config.get("enabled", True)),
        fullscreen=bool(display_config.get("fullscreen", False)),
        window_name=str(display_config.get("window_name", "GateKeeper AI")),
        show_fps=bool(display_config.get("show_fps", True)),
        show_motion=bool(display_config.get("show_motion", True)),
        show_plate_box=bool(display_config.get("show_plate_box", True)),
        show_confidence=bool(display_config.get("show_confidence", True)),
        show_timestamp=bool(display_config.get("show_timestamp", True)),
        save_snapshot_key=str(display_config.get("save_snapshot_key", "s")),
        version=version,
        git_commit=git_commit,
        logger=logger,
    )
    debug_vision = DebugVision(
        enabled=bool(debug_config.get("enabled", False)),
        live_preview=bool(debug_config.get("live_preview", False)),
        save_annotated_frames=bool(debug_config.get("save_annotated_frames", False)),
        logger=logger,
    )
    live_preview_state = LivePreviewState(version=version, git_commit=git_commit)
    live_preview_state.display_config = {**display_config, "debug_candidates": bool(config.get("plate_detector", {}).get("debug", False))}
    live_preview_state.config_path = args.config
    live_preview_state.camera_manager = camera
    live_preview_state.camera_pipeline = str(camera_config.get("pipeline", "rgb"))
    live_preview_state.camera_orientation = {"rotation": int(camera_config.get("rotation", 0)), "flip_horizontal": bool(camera_config.get("flip_horizontal", False)), "flip_vertical": bool(camera_config.get("flip_vertical", False))}
    preview_config = config.get("preview", {})
    live_preview_state.preview_pipeline = LIVE_PREVIEW_PIPELINE
    live_preview_state.preview_swap_rb = "true"
    live_preview_state.camera_controls = dict(camera_config.get("controls", {}) or {})
    live_preview_state.detector_thresholds = config.get("plate_detector", {})
    live_preview_state.plate_calibration_config = calibration_config
    motion_event_manager = MotionEventManager(
        motion_detector,
        database,
        logger,
        end_delay_seconds,
        plate_detector,
        debug_vision,
        live_preview_state,
        calibration_config,
    )
    live_preview_state.motion_event_manager = motion_event_manager
    live_preview_server = LivePreviewServer(
        state=live_preview_state,
        host=str(web_config.get("host", "0.0.0.0")),
        port=int(web_config.get("port", 8080)),
        stream_fps=int(web_config.get("stream_fps", 5)),
        logger=logger,
    )

    try:
        print_startup_banner(version, git_commit)
        if args.check:
            logger.info("Startup check completed successfully.")
            return 0

        with pid_file():
            database.initialize()
            plate_detector.initialize()
            camera.health_check()
            if bool(camera_config.get("diagnostics", False)):
                live_preview_state.diagnostics = camera.generate_diagnostics(runtime_orientation=live_preview_state.runtime_orientation)
            image_path = camera.save_latest()
            validate_frame_pipeline(camera, logger)
            if bool(web_config.get("enabled", True)):
                live_preview_server.start()
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
                motion_event_manager=motion_event_manager,
                end_delay_seconds=end_delay_seconds,
                plate_detector=plate_detector,
                debug_vision=debug_vision,
                plate_calibration=calibration_config,
                display_manager=display_manager,
                live_preview_state=(
                    live_preview_state
                    if bool(web_config.get("enabled", True))
                    else None
                ),
            )
    except Exception:
        camera.stop()
        database.close()
        logger.exception("Startup failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
