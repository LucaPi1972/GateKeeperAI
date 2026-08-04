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

    @app.get("/stream")
    def stream():
        return Response(
            _mjpeg_frames(state, stream_fps),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )


def _mjpeg_frames(state: Any, stream_fps: int):
    interval = 1 / max(stream_fps, 1)
    while True:
        jpeg = state.latest_frame_jpeg()
        if jpeg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(interval)
