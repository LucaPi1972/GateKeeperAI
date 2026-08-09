"""Flask routes for the embedded GateKeeper AI HTTP preview."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_OCR_CACHE: dict[int, dict[str, Any]] = {}
_OCR_CACHE_LOCK = threading.RLock()
_OCR_TTL = 2.5
_OCR_INTERVAL = 1.0
_OCR_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gatekeeper-ocr")


def _cache_for(state: Any) -> dict[str, Any]:
    key = id(state)
    with _OCR_CACHE_LOCK:
        return _OCR_CACHE.setdefault(key, {
            "status": "NO_PLATE", "available": False,
            "error": "Plate ROI not available", "outputs": {},
            "consensus": "", "agreement": 0.0, "frame_id": None,
            "box": None, "updated_at": 0.0, "last_seen": 0.0,
            "last_requested_frame": None, "last_run": 0.0,
            "running": False, "frame": None, "threshold_attempted": False,
        })


def _run_tesseract(image: Any) -> str:
    """Run system Tesseract on one prepared ROI image."""
    import cv2
    if shutil.which("tesseract") is None:
        raise FileNotFoundError("Tesseract is not installed")
    with tempfile.TemporaryDirectory(prefix="gatekeeper_ocr_") as temp_dir:
        image_path = Path(temp_dir) / "roi.png"
        if not cv2.imwrite(str(image_path), image):
            return ""
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "--psm", "7",
             "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"],
            capture_output=True, text=True, timeout=2.0, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""


def _ocr_one(name: str, image: Any) -> tuple[str, str]:
    """Run one OCR pass and return raw text plus normalized text."""
    from src.plate_ocr import clean_ocr_text
    raw = _run_tesseract(image)
    return raw, clean_ocr_text(raw)


def _ocr_worker(state: Any, frame: Any, box: tuple[int, int, int, int], frame_id: int | None) -> None:
    from src.gatekeeper.plate_roi import build_roi_variants
    cache = _cache_for(state)
    try:
        variants = build_roi_variants(frame, box)
        outputs: dict[str, dict[str, str]] = {}
        # Rectified and Enhanced are the fast primary OCR paths. They are
        # independent, so execute them concurrently to reduce wall-clock
        # latency without adding more Tesseract processes than CPU workers.
        futures = {
            _OCR_EXECUTOR.submit(_ocr_one, name, variants[name]): name
            for name in ("rectified", "enhanced")
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                raw, text = future.result()
                outputs[name] = {"raw": raw, "text": text}
            except FileNotFoundError:
                with _OCR_CACHE_LOCK:
                    cache.update({"status": "UNAVAILABLE", "available": False,
                                  "error": "Tesseract is not installed", "outputs": outputs})
                return
            except subprocess.TimeoutExpired:
                outputs[name] = {"raw": "", "text": ""}

        primary_values = [outputs[name]["text"] for name in ("rectified", "enhanced") if outputs.get(name, {}).get("text")]
        # Threshold is deliberately a fallback: it is useful when the two
        # primary variants disagree, but running it on every frame costs an
        # additional Tesseract process and was the main source of latency.
        threshold_attempted = False
        if not primary_values or len(set(primary_values)) > 1:
            threshold_attempted = True
            try:
                raw, text = _ocr_one("threshold", variants["threshold"])
                outputs["threshold"] = {"raw": raw, "text": text}
            except FileNotFoundError:
                with _OCR_CACHE_LOCK:
                    cache.update({"status": "UNAVAILABLE", "available": False,
                                  "error": "Tesseract is not installed", "outputs": outputs})
                return
            except subprocess.TimeoutExpired:
                outputs["threshold"] = {"raw": "", "text": ""}

        values = [item["text"] for item in outputs.values() if item.get("text")]
        counts = {value: values.count(value) for value in set(values)}
        consensus = max(counts, key=counts.get) if counts else ""
        agreement = counts[consensus] / len(values) if values and consensus else 0.0
        with _OCR_CACHE_LOCK:
            cache.update({"status": "LIVE", "available": True, "error": "",
                          "outputs": outputs, "consensus": consensus,
                          "agreement": agreement, "frame_id": frame_id,
                          "box": box, "updated_at": time.monotonic(),
                          "threshold_attempted": threshold_attempted})
    except Exception as exc:
        logging.getLogger("gatekeeper").exception("Live OCR worker failed")
        with _OCR_CACHE_LOCK:
            cache.update({"status": "ERROR", "available": False, "error": str(exc)})
    finally:
        with _OCR_CACHE_LOCK:
            cache["running"] = False


def _refresh_ocr(state: Any) -> dict[str, Any]:
    """Return cached OCR immediately and schedule a low-rate background refresh."""
    cache = _cache_for(state)
    now = time.monotonic()
    with state._lock:
        frame = state._frame
        box = state.plate_bounding_box
        frame_id = state._frame_master.frame_id if state._frame_master is not None else None
        frame_copy = frame.copy() if frame is not None and box is not None else None
    with _OCR_CACHE_LOCK:
        if box is not None and frame_copy is not None:
            current_box = tuple(box)
            cache["last_seen"] = now
            cache["box"] = current_box
            cache["frame"] = frame_copy
            should_run = (
                not cache["running"]
                and now - cache["last_run"] >= _OCR_INTERVAL
                and (
                    frame_id != cache["last_requested_frame"]
                    or current_box != cache.get("requested_box")
                )
            )
            if should_run:
                cache["running"] = True
                cache["last_requested_frame"] = frame_id
                cache["requested_box"] = current_box
                cache["last_run"] = now
                threading.Thread(
                    target=_ocr_worker,
                    args=(state, frame_copy, current_box, frame_id),
                    daemon=True,
                ).start()
            if cache["status"] in {"NO_PLATE", "STALE"}:
                cache["status"] = "RUNNING"
        elif now - cache["last_seen"] > _OCR_TTL:
            cache.update({"status": "NO_PLATE", "available": False,
                          "error": "Plate ROI not available", "outputs": {},
                          "consensus": "", "agreement": 0.0, "box": None,
                          "frame": None, "threshold_attempted": False})
        elif cache["status"] == "LIVE":
            cache["status"] = "STALE"
        result = {k: v for k, v in cache.items() if k != "frame"}
        if result["updated_at"]:
            result["age"] = round(max(0.0, now - result["updated_at"]), 2)
        else:
            result["age"] = None
    return result


def register_routes(app: Any, state: Any, stream_fps: int) -> None:
    """Register HTTP preview, MJPEG stream, health, and status routes."""
    from flask import Response, jsonify, render_template, request, send_from_directory

    @app.get("/")
    def index(): return render_template("index.html")

    @app.get("/diagnostics")
    def diagnostics(): return render_template("diagnostics.html")

    @app.get("/health")
    def health(): return Response("OK\n", mimetype="text/plain")

    @app.get("/api/status")
    def status(): return jsonify(state.snapshot())

    @app.get("/api/frame_info")
    def frame_info(): return jsonify(state.frame_info())

    @app.get("/api/plate_calibration")
    def plate_calibration(): return jsonify(state.plate_calibration_snapshot())

    @app.get("/api/plate_ocr")
    def plate_ocr(): return jsonify(_refresh_ocr(state))

    @app.get("/plate_roi/<variant>.jpg")
    def plate_roi(variant: str):
        from src.gatekeeper.plate_roi import build_roi_variants, encode_variant
        if variant not in {"original", "rectified", "gray", "enhanced", "threshold"}:
            return Response("Unknown ROI variant\n", status=404, mimetype="text/plain")
        cache = _cache_for(state)
        with state._lock:
            frame = state._frame
            box = state.plate_bounding_box
        with _OCR_CACHE_LOCK:
            if frame is None or box is None:
                if time.monotonic() - cache["last_seen"] <= _OCR_TTL:
                    frame, box = cache.get("frame"), cache.get("box")
        if frame is None or box is None:
            return Response("Plate ROI not available\n", status=404, mimetype="text/plain")
        try:
            image = build_roi_variants(frame, tuple(box))[variant]
            return Response(encode_variant(image), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})
        except (ValueError, RuntimeError) as exc:
            return Response(f"{exc}\n", status=409, mimetype="text/plain")

    @app.post("/api/plate_calibration/save")
    def save_plate_calibration():
        try: path = state.save_current_calibration_frame()
        except RuntimeError as exc: return jsonify({"error": str(exc)}), 409
        return jsonify({"path": str(path), "filename": path.name})

    @app.post("/api/snapshot")
    def snapshot():
        try: path = state.save_snapshot()
        except RuntimeError as exc: return jsonify({"error": str(exc)}), 409
        return jsonify({"path": str(path), "filename": path.name})

    @app.get("/api/diagnostics")
    def diagnostics_status():
        runtime_orientation = request.args.get("runtime_orientation", "false").lower() == "true"
        state.runtime_orientation = runtime_orientation
        if state.camera_manager is not None:
            try: state.diagnostics = state.camera_manager.generate_diagnostics(Path(state.diagnostics_dir), runtime_orientation=runtime_orientation)
            except Exception as exc: return jsonify({"error": str(exc), **state.frame_info()}), 500
        return jsonify(state.frame_info())

    @app.get("/api/pipeline")
    def pipeline_status(): return jsonify(state.pipeline_snapshot())

    @app.post("/api/camera")
    def update_camera():
        payload = request.get_json(silent=True) or {}
        updates: dict[str, Any] = {}
        for key in ("rotation", "flip_horizontal", "flip_vertical"):
            if key in payload: updates[key] = payload[key]
        if "controls" in payload and isinstance(payload["controls"], dict): updates["controls"] = payload["controls"]
        return jsonify(state.update_camera_config(updates))

    @app.get("/diagnostics/<path:filename>")
    def diagnostic_image(filename: str): return send_from_directory(Path(state.diagnostics_dir), filename)

    @app.get("/stream")
    def stream(): return Response(_mjpeg_frames(state, stream_fps), mimetype="multipart/x-mixed-replace; boundary=frame")

    for debug_name in ("gray", "edges", "contours", "candidates", "final"):
        app.add_url_rule(f"/debug/{debug_name}", endpoint=f"debug_{debug_name}",
                         view_func=lambda name=debug_name: Response(_mjpeg_frames(state, stream_fps, debug_name=name), mimetype="multipart/x-mixed-replace; boundary=frame"))


def _mjpeg_frames(state: Any, stream_fps: int, debug_name: str | None = None):
    interval = 1 / max(stream_fps, 1)
    while True:
        jpeg = state.debug_jpeg(debug_name) if debug_name else state.latest_frame_jpeg()
        if jpeg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(interval)
