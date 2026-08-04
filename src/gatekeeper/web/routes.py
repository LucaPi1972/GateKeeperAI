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

    @app.post("/api/snapshot")
    def snapshot():
        try:
            path = state.save_snapshot()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"path": str(path), "filename": path.name})

    @app.get("/api/diagnostics")
    def diagnostics_status():
        return jsonify(state.frame_info())

    @app.get("/api/pipeline")
    def pipeline_status():
        return jsonify(state.pipeline_snapshot())

    @app.post("/api/camera")
    def update_camera():
        payload = request.get_json(silent=True) or {}
        updates: dict[str, Any] = {}
        if "pipeline" in payload:
            updates["pipeline"] = payload["pipeline"]
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
