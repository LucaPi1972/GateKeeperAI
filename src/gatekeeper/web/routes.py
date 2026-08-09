"""Flask routes for the embedded GateKeeper AI HTTP preview."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

_OCR_CACHE: dict[int, dict[str, Any]] = {}
_OCR_CACHE_LOCK = threading.RLock()
_OCR_TTL = 2.5
_OCR_INTERVAL = 0.8
_OCR_HISTORY_SIZE = 8
_OCR_FREEZE_TTL = 8.0
_OCR_FREEZE_RETRIES = 3
_OCR_MAX_ANGLE = 30.0


def _cache_for(state: Any) -> dict[str, Any]:
    key = id(state)
    with _OCR_CACHE_LOCK:
        return _OCR_CACHE.setdefault(key, {
            "status": "NO_PLATE", "available": False, "error": "Plate ROI not available",
            "outputs": {}, "consensus": "", "agreement": 0.0, "confidence": 0.0,
            "format_score": 0.0, "samples": 0, "frame_id": None, "box": None,
            "updated_at": 0.0, "last_seen": 0.0, "last_run": 0.0,
            "running": False, "duration": None, "history": [],
            "ocr_runs": 0, "valid_runs": 0, "stable": False,
            "frozen_frame": None, "frozen_frame_id": None, "frozen_box": None,
            "freeze_started": 0.0, "freeze_attempts": 0, "plate_angle": None,
        })


def _box_iou(first: tuple[int, int, int, int] | None, second: tuple[int, int, int, int] | None) -> float:
    if first is None or second is None:
        return 0.0
    ax, ay, aw, ah = first; bx, by, bw, bh = second
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = float(iw * ih)
    union = float(aw * ah + bw * bh) - intersection
    return intersection / union if union > 0 else 0.0


def _selected_angle(state: Any, box: tuple[int, int, int, int] | None) -> float | None:
    if box is None:
        return None
    with state._lock:
        candidates = list(state.candidates or [])
    for candidate in candidates:
        if tuple(getattr(candidate, "bounding_box", ())) == tuple(box) and getattr(candidate, "selected", False):
            return abs(float(getattr(candidate, "rotation", 0.0)))
    for candidate in candidates:
        if tuple(getattr(candidate, "bounding_box", ())) == tuple(box):
            return abs(float(getattr(candidate, "rotation", 0.0)))
    return None


def _run_tesseract(image: Any) -> dict[str, Any]:
    """Run one local Tesseract pass on the Enhanced ROI."""
    import cv2
    if shutil.which("tesseract") is None:
        raise FileNotFoundError("Tesseract is not installed")
    with tempfile.TemporaryDirectory(prefix="gatekeeper_ocr_") as temp_dir:
        image_path = Path(temp_dir) / "roi.png"
        if not cv2.imwrite(str(image_path), image):
            return {"raw": "", "text": "", "engine_confidence": None}
        started = time.monotonic()
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "--psm", "7", "--oem", "1",
             "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
             "-c", "preserve_interword_spaces=0", "tsv"],
            capture_output=True, text=True, timeout=3.0, check=False,
        )
        logging.getLogger("gatekeeper").debug("Tesseract Enhanced OCR completed in %.2fs rc=%s", time.monotonic() - started, result.returncode)
        if result.returncode != 0:
            return {"raw": "", "text": "", "engine_confidence": None}
        texts: list[str] = []; confidences: list[float] = []
        for line in result.stdout.splitlines()[1:]:
            fields = line.split("\t")
            if len(fields) < 12:
                continue
            try: confidence = float(fields[10])
            except ValueError: continue
            token = fields[11].strip()
            if token:
                texts.append(token)
                if confidence >= 0: confidences.append(confidence)
        raw = "".join(texts).strip()
        return {"raw": raw, "text": raw, "engine_confidence": round(sum(confidences) / len(confidences), 1) if confidences else None}


def _ocr_enhanced(image: Any) -> dict[str, Any]:
    from src.plate_ocr import score_ocr_candidate
    result = _run_tesseract(image)
    score = score_ocr_candidate(result.get("text", ""), engine_confidence=result.get("engine_confidence"))
    result.update({"name": "enhanced", "corrected": score["corrected"], "format_score": score["format_score"], "score": score["score"]})
    return result


def _fuse_history(cache: dict[str, Any]) -> dict[str, Any]:
    from src.plate_ocr import fuse_temporal_results
    return fuse_temporal_results(list(cache.get("history", [])), max_items=_OCR_HISTORY_SIZE)


def _ocr_worker(state: Any, frame: Any, box: tuple[int, int, int, int], frame_id: int | None) -> None:
    from src.gatekeeper.plate_roi import build_roi_variants
    cache = _cache_for(state); started = time.monotonic()
    try:
        variants = build_roi_variants(frame, box, margin_ratio=0.05)
        result = _ocr_enhanced(variants["enhanced"])
        history_item = {"text": result.get("corrected", ""), "score": result.get("score", 0.0), "engine_confidence": result.get("engine_confidence"), "format_score": result.get("format_score", 0.0), "frame_id": frame_id}
        with _OCR_CACHE_LOCK:
            cache["ocr_runs"] = int(cache.get("ocr_runs", 0)) + 1
            if result.get("corrected"):
                cache["valid_runs"] = int(cache.get("valid_runs", 0)) + 1
            history = cache.setdefault("history", [])
            if history and history[-1].get("frame_id") == frame_id:
                history[-1] = history_item
            else:
                history.append(history_item); del history[:-_OCR_HISTORY_SIZE]
            fused = _fuse_history(cache); text = fused.get("text", "")
            stable = bool(fused.get("samples", 0) >= 3 and fused.get("agreement", 0.0) >= 0.67 and fused.get("confidence", 0.0) >= 70.0)
            cache.update({"status": "LIVE" if text else "NO_TEXT", "available": True, "error": "", "outputs": {"enhanced": result}, "consensus": text, "agreement": fused.get("agreement", 0.0), "confidence": fused.get("confidence", 0.0), "format_score": result.get("format_score", 0.0), "samples": fused.get("samples", 0), "frame_id": frame_id, "box": box, "updated_at": time.monotonic(), "duration": round(time.monotonic() - started, 2), "stable": stable})
    except FileNotFoundError:
        with _OCR_CACHE_LOCK: cache.update({"status": "UNAVAILABLE", "available": False, "error": "Tesseract is not installed"})
    except subprocess.TimeoutExpired:
        with _OCR_CACHE_LOCK:
            cache["ocr_runs"] = int(cache.get("ocr_runs", 0)) + 1
            cache.update({"status": "NO_TEXT", "available": True, "error": "Tesseract timeout", "duration": round(time.monotonic() - started, 2), "stable": False})
    except Exception as exc:
        logging.getLogger("gatekeeper").exception("Live OCR worker failed")
        with _OCR_CACHE_LOCK: cache.update({"status": "ERROR", "available": False, "error": str(exc), "duration": round(time.monotonic() - started, 2), "stable": False})
    finally:
        with _OCR_CACHE_LOCK: cache["running"] = False


def _start_ocr_worker(state: Any, frame: Any, box: tuple[int, int, int, int], frame_id: int | None) -> None:
    threading.Thread(target=_ocr_worker, args=(state, frame, box, frame_id), daemon=True, name="gatekeeper-ocr").start()


def _clear_no_plate(cache: dict[str, Any]) -> None:
    cache.update({"status": "NO_PLATE", "available": False, "error": "Plate ROI not available", "outputs": {}, "consensus": "", "agreement": 0.0, "confidence": 0.0, "format_score": 0.0, "samples": 0, "box": None, "history": [], "ocr_runs": 0, "valid_runs": 0, "stable": False, "duration": None, "frozen_frame": None, "frozen_frame_id": None, "frozen_box": None, "freeze_started": 0.0, "freeze_attempts": 0, "plate_angle": None})


def _reset_track(cache: dict[str, Any]) -> None:
    cache["history"] = []; cache["samples"] = 0; cache["ocr_runs"] = 0; cache["valid_runs"] = 0; cache["consensus"] = ""; cache["agreement"] = 0.0; cache["confidence"] = 0.0; cache["stable"] = False


def _release_freeze(cache: dict[str, Any]) -> None:
    cache["frozen_frame"] = None; cache["frozen_frame_id"] = None; cache["frozen_box"] = None; cache["freeze_started"] = 0.0; cache["freeze_attempts"] = 0


def _refresh_ocr(state: Any) -> dict[str, Any]:
    cache = _cache_for(state); now = time.monotonic()
    with state._lock:
        live_frame = state._frame; live_box = state.plate_bounding_box; live_frame_id = state._frame_master.frame_id if state._frame_master is not None else None
        live_copy = live_frame.copy() if live_frame is not None and live_box is not None else None
    with _OCR_CACHE_LOCK:
        if live_box is not None and live_copy is not None:
            current_box = tuple(live_box); angle = _selected_angle(state, current_box); cache["last_seen"] = now; cache["plate_angle"] = angle
            if angle is not None and angle > _OCR_MAX_ANGLE:
                _release_freeze(cache); _reset_track(cache); cache.update({"status": "ANGLE_REJECTED", "available": True, "error": f"Plate angle {angle:.1f}° exceeds ±{_OCR_MAX_ANGLE:.0f}°", "box": current_box, "updated_at": now})
            else:
                previous_box = cache.get("box")
                if previous_box is not None and _box_iou(tuple(previous_box), current_box) < 0.25 and not cache.get("running"):
                    _reset_track(cache); _release_freeze(cache)
                if cache["frozen_frame"] is None:
                    cache["frozen_frame"] = live_copy; cache["frozen_frame_id"] = live_frame_id; cache["frozen_box"] = current_box; cache["freeze_started"] = now; cache["freeze_attempts"] = 0
                if now - float(cache.get("freeze_started", 0.0)) > _OCR_FREEZE_TTL:
                    _release_freeze(cache); cache["frozen_frame"] = live_copy; cache["frozen_frame_id"] = live_frame_id; cache["frozen_box"] = current_box; cache["freeze_started"] = now; cache["freeze_attempts"] = 0
                frozen_frame = cache["frozen_frame"]; frozen_id = cache["frozen_frame_id"]; frozen_box = tuple(cache["frozen_box"] or current_box); attempts = int(cache.get("freeze_attempts", 0))
                should_run = (not cache["running"] and attempts < _OCR_FREEZE_RETRIES and now - cache["last_run"] >= _OCR_INTERVAL)
                if should_run and frozen_frame is not None:
                    cache["running"] = True; cache["last_run"] = now; cache["freeze_attempts"] = attempts + 1; _start_ocr_worker(state, frozen_frame.copy(), frozen_box, frozen_id)
                if cache.get("updated_at", 0.0) and not cache.get("running") and cache.get("consensus"):
                    _release_freeze(cache)
                elif attempts >= _OCR_FREEZE_RETRIES and not cache.get("running"):
                    _release_freeze(cache)
                cache["status"] = "LIVE" if cache.get("consensus") else ("RUNNING" if cache.get("running") else "NO_TEXT")
        elif now - cache["last_seen"] > _OCR_TTL:
            _clear_no_plate(cache)
        elif cache["status"] in {"LIVE", "NO_TEXT"}:
            cache["status"] = "STALE"
        result = {k: v for k, v in cache.items() if k not in {"frame", "frozen_frame"}}
        result["age"] = round(max(0.0, now - result["updated_at"]), 2) if result["updated_at"] else None
        result["valid_rate"] = round((result["valid_runs"] / result["ocr_runs"]) * 100.0, 1) if result["ocr_runs"] else 0.0
        result["frozen"] = bool(cache.get("frozen_frame") is not None); result["freeze_attempts"] = int(cache.get("freeze_attempts", 0)); result["angle_limit"] = _OCR_MAX_ANGLE
    return result


def register_routes(app: Any, state: Any, stream_fps: int) -> None:
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
        if variant not in {"original", "enhanced", "rectified", "gray", "threshold"}: return Response("Unknown ROI variant\n", status=404, mimetype="text/plain")
        cache = _cache_for(state)
        with state._lock: frame = state._frame; box = state.plate_bounding_box
        with _OCR_CACHE_LOCK:
            if frame is None or box is None:
                if time.monotonic() - cache["last_seen"] <= _OCR_TTL: frame, box = cache.get("frame"), cache.get("box")
            if frame is None or box is None: frame, box = cache.get("frozen_frame"), cache.get("frozen_box")
        if frame is None or box is None: return Response("Plate ROI not available\n", status=404, mimetype="text/plain")
        try: return Response(encode_variant(build_roi_variants(frame, tuple(box))[variant]), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})
        except (ValueError, RuntimeError) as exc: return Response(f"{exc}\n", status=409, mimetype="text/plain")
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
        runtime_orientation = request.args.get("runtime_orientation", "false").lower() == "true"; state.runtime_orientation = runtime_orientation
        if state.camera_manager is not None:
            try: state.diagnostics = state.camera_manager.generate_diagnostics(Path(state.diagnostics_dir), runtime_orientation=runtime_orientation)
            except Exception as exc: return jsonify({"error": str(exc), **state.frame_info()}), 500
        return jsonify(state.frame_info())
    @app.get("/api/pipeline")
    def pipeline_status(): return jsonify(state.pipeline_snapshot())
    @app.post("/api/camera")
    def update_camera():
        payload = request.get_json(silent=True) or {}; updates: dict[str, Any] = {}
        for key in ("rotation", "flip_horizontal", "flip_vertical"):
            if key in payload: updates[key] = payload[key]
        if "controls" in payload and isinstance(payload["controls"], dict): updates["controls"] = payload["controls"]
        return jsonify(state.update_camera_config(updates))
    @app.get("/diagnostics/<path:filename>")
    def diagnostic_image(filename: str): return send_from_directory(Path(state.diagnostics_dir), filename)
    @app.get("/stream")
    def stream(): return Response(_mjpeg_frames(state, stream_fps), mimetype="multipart/x-mixed-replace; boundary=frame")
    for debug_name in ("gray", "edges", "contours", "candidates", "final"):
        app.add_url_rule(f"/debug/{debug_name}", endpoint=f"debug_{debug_name}", view_func=lambda name=debug_name: Response(_mjpeg_frames(state, stream_fps, debug_name=name), mimetype="multipart/x-mixed-replace; boundary=frame"))


def _mjpeg_frames(state: Any, stream_fps: int, debug_name: str | None = None):
    interval = 1 / max(stream_fps, 1)
    while True:
        jpeg = state.debug_jpeg(debug_name) if debug_name else state.latest_frame_jpeg()
        if jpeg is not None: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(interval)
