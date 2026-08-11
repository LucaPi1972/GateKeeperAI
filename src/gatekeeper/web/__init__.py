"""Embedded HTTP preview package."""

from __future__ import annotations

from .server import LivePreviewServer as _LivePreviewServer
from . import server as _server_module


class LivePreviewServer(_LivePreviewServer):
    """Live preview server with the optional Raspberry Pi gate-test output."""

    def start(self) -> bool:
        from flask import jsonify
        from src.gatekeeper.gate_output import GateOutput, start_gate_watch
        from .routes import _refresh_ocr

        original_register_routes = _server_module.register_routes
        gate_output = GateOutput()

        def register_with_gate(app, state, stream_fps):
            original_register_routes(app, state, stream_fps)

            @app.get("/api/gate")
            def gate_status():
                return jsonify(gate_output.snapshot())

            @app.post("/api/gate/trigger")
            def gate_trigger():
                return jsonify({"triggered": gate_output.trigger("MANUAL_TEST"), **gate_output.snapshot()})

        _server_module.register_routes = register_with_gate
        try:
            started = super().start()
        finally:
            _server_module.register_routes = original_register_routes

        if started:
            self.gate_output = gate_output
            self.gate_watch = start_gate_watch(self.state, gate_output, _refresh_ocr)
        return started


__all__ = ["LivePreviewServer"]
