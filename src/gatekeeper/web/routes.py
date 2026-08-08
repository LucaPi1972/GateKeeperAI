"""Flask routes for the embedded GateKeeper AI HTTP preview."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def register_routes(app: Any, state: Any, stream_fps: int) -> None:
    """Register HTTP preview, MJPEG stream, health, and status routes."""
    from flask import Response, jsonify, render_template, request, send_from_directory

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/diagnostics")
    def diagnostics():
        return render_template("diagnostics.html")

    @app.get("/health")
    def health():
        return Response("OK\n", mimetype="text/plain")

    @app.get("/api/status")
    def status():
        return jsonify(state.snapshot())

    @app.get("/api/frame_info")
    def frame_info():
        return jsonify(state.frame_info())

    @app.get("/api/plate_calibration")
    def plate_calibration():
        return jsonify(state.plate_calibration_snapshot())

    @app.get("/plate_roi/<variant>.jpg")
    def plate_roi(variant: str):
        """Return the selected plate ROI in one of the diagnostic forms."""
        from src.gatekeeper.plate_roi import build_roi_variants, encode_variant

        if variant not in {"original", "rectified", "gray", "enhanced", "threshold"}:
            return Response("Unknown ROI variant\n", status=404, mimetype="text/plain")
        try:
            with state._lock:
                frame = state._frame
                bounding_box = state.plate_bounding_box
                if frame is None or bounding_box is None:
                    return Response("Plate ROI not available\n", status=404, mimetype="text/plain")
                image = build_roi_variants(frame, bounding_box)[variant]
            return Response(encode_variant(image), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})
        except (ValueError, RuntimeError) as exc:
            return Response(f"{exc}\n", status=409, mimetype="text/plain")

    @app.post("/api/plate_calibration/save")
    def save_plate_calibration():
        try:
            path = state.save_current_calibration_frame()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"path": str(path), "filename": path.name})

    @app.post("/api/snapshot")
    def snapshot():
        try:
            path = state.save_snapshot()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"path": str(path), "filename": path.name})

    @app.get("/api/diagnostics")
    def diagnostics_status():
        runtime_orientation = request.args.get("runtime_orientation", "false").lower() == "true"
        state.runtime_orientation = runtime_orientation
        if state.camera_manager is not None:
            try:
                state.diagnostics = state.camera_manager.generate_diagnostics(
                    Path(state.diagnostics_dir), runtime_orientation=runtime_orientation
                )
            except Exception as exc:
                return jsonify({"error": str(exc), **state.frame_info()}), 500
        return jsonify(state.frame_info())

    @app.get("/api/pipeline")
    def pipeline_status():
        return jsonify(state.pipeline_snapshot())

    @app.post("/api/camera")
    def update_camera():
        payload = request.get_json(silent=True) or {}
        updates: dict[str, Any] = {}
        for key in ("rotation", "flip_horizontal", "flip_vertical"):
            if key in payload:
                updates[key] = payload[key]
        if "controls" in payload and isinstance(payload["controls"], dict):
            updates["controls"] = payload["controls"]
        return jsonify(state.update_camera_config(updates))

    @app.get("/diagnostics/<path:filename>")
    def diagnostic_image(filename: str):
        return send_from_directory(Path(state.diagnostics_dir), filename)

    @app.get("/stream")
    def stream():
        return Response(
            _mjpeg_frames(state, stream_fps),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    for debug_name in ("gray", "edges", "contours", "candidates", "final"):
        app.add_url_rule(
            f"/debug/{debug_name}",
            endpoint=f"debug_{debug_name}",
            view_func=lambda name=debug_name: Response(
                _mjpeg_frames(state, stream_fps, debug_name=name),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            ),
        )


def _mjpeg_frames(state: Any, stream_fps: int, debug_name: str | None = None):
    interval = 1 / max(stream_fps, 1)
    while True:
        jpeg = state.debug_jpeg(debug_name) if debug_name else state.latest_frame_jpeg()
        if jpeg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(interval)
