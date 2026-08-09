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
_OCR_INTERVAL = 0.8
_OCR_FALLBACK_INTERVAL = 1.5
_OCR_HISTORY_SIZE = 8


def _cache_for(state: Any) -> dict[str, Any]:
    key = id(state)
    with _OCR_CACHE_LOCK:
        return _OCR_CACHE.setdefault(key, {
            "status": "NO_PLATE", "available": False, "error": "Plate ROI not available",
            "outputs": {}, "consensus": "", "agreement": 0.0, "confidence": 0.0,
            "format_score": 0.0, "samples": 0, "frame_id": None, "box": None,
            "updated_at": 0.0, "last_seen": 0.0, "last_requested_frame": None,
            "last_run": 0.0, "last_fallback_run": 0.0, "running": False, "frame": None,
            "threshold_attempted": False, "duration": None, "history": [],
            "ocr_runs": 0, "valid_runs": 0, "stable": False,
        })


def _run_tesseract(image: Any) -> dict[str, Any]:
    """Run one fast Tesseract pass and return text plus TSV confidence."""
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
        elapsed = time.monotonic() - started
        logging.getLogger("gatekeeper").debug(
            "Tesseract OCR completed in %.2fs rc=%s", elapsed, result.returncode
        )
        if result.returncode != 0:
            return {"raw": "", "text": "", "engine_confidence": None}
        texts: list[str] = []
        confidences: list[float] = []
        for line in result.stdout.splitlines()[1:]:
            fields = line.split("\t")
            if len(fields) < 12:
                continue
            try:
                confidence = float(fields[10])
            except ValueError:
                continue
            token = fields[11].strip()
            if token:
                texts.append(token)
                if confidence >= 0:
                    confidences.append(confidence)
        raw = "".join(texts).strip()
        return {
            "raw": raw,
            "text": raw,
            "engine_confidence": round(sum(confidences) / len(confidences), 1) if confidences else None,
        }


def _ocr_one(name: str, image: Any) -> dict[str, Any]:
    from src.plate_ocr import score_ocr_candidate
    result = _run_tesseract(image)
    score = score_ocr_candidate(result.get("text", ""), engine_confidence=result.get("engine_confidence"))
    result.update({"name": name, "corrected": score["corrected"], "format_score": score["format_score"], "score": score["score"]})
    return result


def _ocr_one_safe(name: str, image: Any) -> dict[str, Any]:
    """Run a fallback OCR pass without allowing one failed variant to kill the cycle."""
    try:
        return _ocr_one(name, image)
    except subprocess.TimeoutExpired:
        return {"name": name, "raw": "", "text": "", "corrected": "", "engine_confidence": None, "format_score": 0.0, "score": 0.0, "error": "timeout"}
    except Exception as exc:
        logging.getLogger("gatekeeper").debug("OCR fallback %s failed: %s", name, exc)
        return {"name": name, "raw": "", "text": "", "corrected": "", "engine_confidence": None, "format_score": 0.0, "score": 0.0, "error": str(exc)}


def _fuse_history(cache: dict[str, Any]) -> dict[str, Any]:
    from src.plate_ocr import fuse_temporal_results
    return fuse_temporal_results(list(cache.get("history", [])), max_items=_OCR_HISTORY_SIZE)


def _ocr_worker(state: Any, frame: Any, box: tuple[int, int, int, int], frame_id: int | None) -> None:
    from src.gatekeeper.plate_roi import build_roi_variants
    cache = _cache_for(state)
    started = time.monotonic()
    try:
        variants = build_roi_variants(frame, box, margin_ratio=0.05)
        outputs: dict[str, dict[str, Any]] = {}

        # Fast path: one enhanced OCR pass for normal field operation.
        primary = _ocr_one("enhanced_5", variants["enhanced"])
        outputs["enhanced"] = primary
        best = primary
        threshold_attempted = False

        # If the fast result is weak, run the two complementary variants in
        # parallel. This keeps the fallback latency close to one OCR pass while
        # giving the temporal fusion two independent preprocessing paths.
        now = time.monotonic()
        fallback_due = now - float(cache.get("last_fallback_run", 0.0)) >= _OCR_FALLBACK_INTERVAL
        if primary["score"] < 0.68 and fallback_due:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="gatekeeper-ocr-fallback") as executor:
                futures = {
                    executor.submit(_ocr_one_safe, "rectified", variants["rectified"]): "rectified",
                    executor.submit(_ocr_one_safe, "threshold", variants["threshold"]): "threshold",
                }
                for future in as_completed(futures):
                    name = futures[future]
                    result = future.result()
                    outputs[name] = result
                    if result.get("score", 0.0) > best.get("score", 0.0):
                        best = result
            threshold_attempted = "threshold" in outputs
            with _OCR_CACHE_LOCK:
                cache["last_fallback_run"] = now

        history_item = {
            "text": best.get("corrected", ""),
            "score": best.get("score", 0.0),
            "engine_confidence": best.get("engine_confidence"),
            "format_score": best.get("format_score", 0.0),
            "frame_id": frame_id,
        }
        with _OCR_CACHE_LOCK:
            history = cache.setdefault("history", [])
            history.append(history_item)
            del history[:-_OCR_HISTORY_SIZE]
            fused = _fuse_history(cache)
            status = "LIVE" if fused.get("text") else "NO_TEXT"
            cache["ocr_runs"] = int(cache.get("ocr_runs", 0)) + 1
            if fused.get("text"):
                cache["valid_runs"] = int(cache.get("valid_runs", 0)) + 1
            stable = bool(
                fused.get("samples", 0) >= 3
                and fused.get("agreement", 0.0) >= 0.67
                and fused.get("confidence", 0.0) >= 70.0
            )
            cache.update({
                "status": status, "available": True, "error": "", "outputs": outputs,
                "consensus": fused.get("text", ""), "agreement": fused.get("agreement", 0.0),
                "confidence": fused.get("confidence", 0.0),
                "format_score": best.get("format_score", 0.0),
                "samples": fused.get("samples", 0), "frame_id": frame_id, "box": box,
                "updated_at": time.monotonic(), "threshold_attempted": threshold_attempted,
                "duration": round(time.monotonic() - started, 2), "stable": stable,
            })
    except FileNotFoundError:
        with _OCR_CACHE_LOCK:
            cache.update({"status": "UNAVAILABLE", "available": False, "error": "Tesseract is not installed"})
    except subprocess.TimeoutExpired:
        with _OCR_CACHE_LOCK:
            cache["ocr_runs"] = int(cache.get("ocr_runs", 0)) + 1
            cache.update({"status": "NO_TEXT", "available": True, "error": "Tesseract timeout", "duration": round(time.monotonic() - started, 2), "stable": False})
    except Exception as exc:
        logging.getLogger("gatekeeper").exception("Live OCR worker failed")
        with _OCR_CACHE_LOCK:
            cache.update({"status": "ERROR", "available": False, "error": str(exc), "duration": round(time.monotonic() - started, 2), "stable": False})
    finally:
        with _OCR_CACHE_LOCK:
            cache["running"] = False


def _start_ocr_worker(state: Any, frame: Any, box: tuple[int, int, int, int], frame_id: int | None) -> None:
    threading.Thread(target=_ocr_worker, args=(state, frame, box, frame_id), daemon=True, name="gatekeeper-ocr").start()


def _clear_no_plate(cache: dict[str, Any]) -> None:
    cache.update({
        "status": "NO_PLATE", "available": False, "error": "Plate ROI not available",
        "outputs": {}, "consensus": "", "agreement": 0.0, "confidence": 0.0,
        "format_score": 0.0, "samples": 0, "box": None, "frame": None,
        "history": [], "ocr_runs": 0, "valid_runs": 0, "stable": False,
        "duration": None, "threshold_attempted": False,
    })


def _refresh_ocr(state: Any) -> dict[str, Any]:
    """Return cached OCR immediately and schedule one low-rate background pass."""
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
                not cache["running"] and now - cache["last_run"] >= _OCR_INTERVAL
                and (frame_id != cache["last_requested_frame"] or current_box != cache.get("requested_box"))
            )
            if should_run:
                cache["running"] = True
                cache["last_requested_frame"] = frame_id
                cache["requested_box"] = current_box
                cache["last_run"] = now
                _start_ocr_worker(state, frame_copy, current_box, frame_id)
            if cache["status"] in {"NO_PLATE", "STALE", "NO_TEXT"}:
                cache["status"] = "RUNNING"
        elif now - cache["last_seen"] > _OCR_TTL:
            _clear_no_plate(cache)
        elif cache["status"] in {"LIVE", "NO_TEXT"}:
            cache["status"] = "STALE"
        result = {k: v for k, v in cache.items() if k != "frame"}
        result["age"] = round(max(0.0, now - result["updated_at"]), 2) if result["updated_at"] else None
        result["valid_rate"] = round((result["valid_runs"] / result["ocr_runs"]) * 100.0, 1) if result["ocr_runs"] else 0.0
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
        app.add_url_rule(f"/debug/{debug_name}", endpoint=f"debug_{debug_name}", view_func=lambda name=debug_name: Response(_mjpeg_frames(state, stream_fps, debug_name=name), mimetype="multipart/x-mixed-replace; boundary=frame"))


def _mjpeg_frames(state: Any, stream_fps: int, debug_name: str | None = None):
    interval = 1 / max(stream_fps, 1)
    while True:
        jpeg = state.debug_jpeg(debug_name) if debug_name else state.latest_frame_jpeg()
        if jpeg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(interval)
