"""Flask routes for the embedded GateKeeper AI HTTP preview."""

from __future__ import annotations

import time
from typing import Any


def register_routes(app: Any, state: Any, stream_fps: int) -> None:
    """Register HTTP preview, MJPEG stream, health, and status routes."""
    from flask import Response, jsonify, render_template

    @app.get("/")
    def index():
        return render_template("index.html")

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
