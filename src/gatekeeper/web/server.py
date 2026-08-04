"""Lightweight embedded Flask server for GateKeeper AI live preview."""

from __future__ import annotations

import logging
import socket
import threading
from pathlib import Path
from typing import Any

from .routes import register_routes


class LivePreviewServer:
    """Run the HTTP preview without opening or owning camera resources."""

    def __init__(
        self,
        *,
        state: Any,
        host: str = "0.0.0.0",
        port: int = 8080,
        stream_fps: int = 5,
        logger: logging.Logger | None = None,
    ) -> None:
        self.state = state
        self.host = host
        self.port = port
        self.stream_fps = stream_fps
        self.logger = logger or logging.getLogger(__name__)
        self._thread: threading.Thread | None = None
        self._app: Any | None = None

    def start(self) -> bool:
        """Start Flask in a daemon thread."""
        try:
            from flask import Flask
        except ImportError:
            app = self._create_wsgi_fallback()
            self._app = app
            self._thread = threading.Thread(
                target=self._run_wsgi_fallback, args=(app,), daemon=True
            )
        else:
            web_dir = Path(__file__).resolve().parent
            app = Flask(
                __name__,
                template_folder=str(web_dir / "templates"),
                static_folder=str(web_dir / "static"),
            )
            register_routes(app, self.state, self.stream_fps)
            self._app = app
            self._thread = threading.Thread(
                target=app.run,
                kwargs={
                    "host": self.host,
                    "port": self.port,
                    "threaded": True,
                    "use_reloader": False,
                },
                daemon=True,
            )
        self._thread.start()
        self.logger.info("HTTP server started")
        self.logger.info("Listening on:")
        self.logger.info("http://127.0.0.1:%s", self.port)
        self.logger.info("http://%s:%s", self._raspberry_ip(), self.port)
        return True

    def _create_wsgi_fallback(self):
        """Create a tiny stdlib fallback for environments without Flask installed."""
        import json

        web_dir = Path(__file__).resolve().parent
        index = (web_dir / "templates" / "index.html").read_text(encoding="utf-8")
        css = (web_dir / "static" / "style.css").read_text(encoding="utf-8")

        def app(environ, start_response):
            path = environ.get("PATH_INFO", "/")
            if path == "/health":
                start_response("200 OK", [("Content-Type", "text/plain")])
                return [b"OK\n"]
            if path == "/api/status":
                payload = json.dumps(self.state.snapshot()).encode("utf-8")
                start_response("200 OK", [("Content-Type", "application/json")])
                return [payload]
            if path == "/api/frame_info":
                payload = json.dumps(self.state.frame_info()).encode("utf-8")
                start_response("200 OK", [("Content-Type", "application/json")])
                return [payload]
            if path == "/api/snapshot" and environ.get("REQUEST_METHOD") == "POST":
                try:
                    snapshot = self.state.save_snapshot()
                    payload = json.dumps({"path": str(snapshot), "filename": snapshot.name}).encode("utf-8")
                    start_response("200 OK", [("Content-Type", "application/json")])
                except RuntimeError as exc:
                    payload = json.dumps({"error": str(exc)}).encode("utf-8")
                    start_response("409 Conflict", [("Content-Type", "application/json")])
                return [payload]
            if path == "/static/style.css":
                start_response("200 OK", [("Content-Type", "text/css")])
                return [css.encode("utf-8")]
            if path.startswith("/debug/"):
                start_response("200 OK", [("Content-Type", "multipart/x-mixed-replace; boundary=frame")])
                return self._mjpeg_frames(path.rsplit("/", 1)[-1])
            if path == "/stream":
                start_response(
                    "200 OK",
                    [("Content-Type", "multipart/x-mixed-replace; boundary=frame")],
                )
                return self._mjpeg_frames()
            if path == "/":
                start_response("200 OK", [("Content-Type", "text/html")])
                return [index.encode("utf-8")]
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Not Found\n"]

        return app

    def _run_wsgi_fallback(self, app) -> None:
        from wsgiref.simple_server import make_server

        with make_server(self.host, self.port, app) as httpd:
            httpd.serve_forever()

    def _mjpeg_frames(self, debug_name: str | None = None):
        import time

        interval = 1 / max(self.stream_fps, 1)
        while True:
            jpeg = self.state.debug_jpeg(debug_name) if debug_name else self.state.latest_frame_jpeg()
            if jpeg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(interval)

    @staticmethod
    def _raspberry_ip() -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except OSError:
            return "<raspberry-ip>"
