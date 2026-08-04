"""OpenCV display, overlay, and calibration preview support."""

from __future__ import annotations

import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "snapshots"


def is_display_available() -> bool:
    """Return True when this host should be able to create GUI windows."""
    if platform.system() in {"Windows", "Darwin"}:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def snapshot_path(timestamp: datetime | None = None) -> Path:
    """Build a snapshot path using a filesystem-safe UTC timestamp."""
    timestamp = timestamp or datetime.now(timezone.utc)
    safe_timestamp = timestamp.isoformat().replace(":", "").replace("+", "Z")
    return SNAPSHOT_DIR / f"snapshot_{safe_timestamp}.jpg"


class DisplayManager:
    """Manage the local OpenCV preview window and interactive display controls."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        fullscreen: bool = False,
        window_name: str = "GateKeeper AI",
        show_fps: bool = True,
        show_motion: bool = True,
        show_plate_box: bool = True,
        show_confidence: bool = True,
        show_timestamp: bool = True,
        save_snapshot_key: str = "s",
        version: str = "development",
        git_commit: str = "development",
        logger: logging.Logger | None = None,
    ) -> None:
        self.requested_enabled = enabled
        self.enabled = enabled and is_display_available()
        self.fullscreen = fullscreen
        self.window_name = window_name
        self.show_fps = show_fps
        self.show_motion = show_motion
        self.show_plate_box = show_plate_box
        self.show_confidence = show_confidence
        self.show_timestamp = show_timestamp
        self.save_snapshot_key = save_snapshot_key[:1] or "s"
        self.version = version
        self.git_commit = git_commit
        self.logger = logger or logging.getLogger(__name__)
        self.overlays_enabled = True
        self.quit_requested = False
        self._window_created = False
        if enabled and not self.enabled:
            self.logger.info("No graphical display detected. Running headless.")

    def create_window(self) -> None:
        """Create the OpenCV window if display mode is enabled."""
        if not self.enabled or self._window_created:
            return
        import cv2

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._window_created = True
        self._apply_fullscreen()

    def _apply_fullscreen(self) -> None:
        if not self.enabled or not self._window_created:
            return
        import cv2

        value = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, value)

    def draw_overlays(
        self,
        frame: Any,
        *,
        motion_state: str,
        fps: float,
        timestamp: datetime,
        plate_detection: Any | None = None,
    ) -> Any:
        """Return a frame copy with configured display overlays drawn."""
        import cv2

        output = frame.copy() if hasattr(frame, "copy") else frame
        if not self.overlays_enabled:
            return output
        lines = [f"GateKeeper AI v{self.version}", f"Git: {self.git_commit}"]
        if self.show_fps:
            lines.append(f"FPS: {fps:.2f}")
        if self.show_motion:
            lines.append(f"Motion: {motion_state}")
        if self.show_timestamp:
            lines.append(f"Timestamp: {timestamp.isoformat()}")
        for index, line in enumerate(lines):
            cv2.putText(output, line, (10, 25 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        if plate_detection is not None and self.show_plate_box:
            x, y, width, height = plate_detection.bounding_box
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 2)
            label = f"Box: ({x},{y},{width},{height})"
            if self.show_confidence:
                label += f" Confidence: {plate_detection.confidence:.3f}"
            cv2.putText(output, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        return output

    def update_frame(self, frame: Any, **overlay_kwargs: Any) -> Any:
        """Draw overlays, display the frame, and handle keyboard input."""
        displayed = self.draw_overlays(frame, **overlay_kwargs)
        if not self.enabled:
            return displayed
        self.create_window()
        import cv2

        import sys
        camera_manager = getattr(sys.modules.get("main") or sys.modules.get("__main__"), "CameraManager")
        image = camera_manager.apply_pipeline(displayed, "bgr") if len(displayed.shape) == 3 else displayed
        cv2.imshow(self.window_name, image)
        self.handle_key(cv2.waitKey(1) & 0xFF, displayed)
        return displayed

    def handle_key(self, key: int, frame: Any | None = None) -> None:
        """Apply keyboard shortcuts for quit, snapshot, fullscreen, and overlays."""
        if key in {-1, 255}:
            return
        if key == ord("q"):
            self.quit_requested = True
        elif key == ord(self.save_snapshot_key) and frame is not None:
            self.save_snapshot(frame)
        elif key == ord("f"):
            self.fullscreen = not self.fullscreen
            self._apply_fullscreen()
        elif key == ord("d"):
            self.overlays_enabled = not self.overlays_enabled

    def save_snapshot(self, frame: Any, timestamp: datetime | None = None) -> Path:
        """Save the currently displayed frame to snapshots/."""
        import sys

        camera_manager = getattr(sys.modules.get("main") or sys.modules.get("__main__"), "CameraManager")
        output_path = snapshot_path(timestamp)
        camera_manager.write_image(frame, output_path)
        self.logger.info("Snapshot saved: %s", output_path)
        return output_path

    def close(self) -> None:
        """Close the OpenCV window without disturbing headless runs."""
        if self.enabled and self._window_created:
            import cv2

            cv2.destroyWindow(self.window_name)
            self._window_created = False
