# Changelog

## 0.8.7

- Reduce normal live OCR to one Tesseract pass per cycle instead of chaining multiple OCR passes.
- Keep rectified and threshold OCR as rate-limited fallbacks only when the primary result is weak.
- Reduce the Tesseract hard timeout from 6 seconds to 3 seconds so a bad OCR run cannot hold the gate workflow for too long.
- Keep OCR asynchronous so the live preview and plate detector remain responsive.
- Reset OCR history and counters after the plate has been absent beyond the retention timeout, avoiding stale sample counts such as `21/21` while showing `NO PLATE`.
- Keep detector, temporal plate selection, Reading Zone, camera pipeline, BGR Live Preview, FRAME_MASTER and ROI generation unchanged.

## 0.8.6

- Add field-test OCR stability metrics without changing detector or ROI behavior.
- Show valid OCR sample count, dominant-result stability, and latest OCR duration in the single-screen Live Preview.
- Distinguish temporal stability from OCR engine confidence so a small number of consistent samples can be evaluated correctly.
- Keep camera controls unchanged; brighter illumination remains an external test condition rather than an automatic software correction.
- Keep PlateDetector, temporal selection, Reading Zone, camera color handling, BGR Live Preview, FRAME_MASTER, Motion Detection, and ROI generation unchanged.

## 0.8.5

- Add OCR accuracy scoring based on Tesseract TSV confidence, plate-format compatibility, and normalized candidate quality.
- Add conservative context handling for common ambiguous plate characters such as `O/0`, `I/1`, `S/5`, and `B/8` according to character position.
- Add temporal OCR fusion across recent frames so isolated OCR errors do not immediately replace a repeated high-quality reading.
- Test a 5% ROI margin first and broaden to 0%/10% enhanced variants only when the first result is weak; keep rectified and threshold passes available for confirmation.
- Keep OCR asynchronous and throttled, with explicit `LIVE`, `STALE`, `NO_TEXT`, `NO_PLATE`, `UNAVAILABLE`, and `ERROR` states.
- Expose OCR confidence, format score, temporal agreement, sample count, and per-variant diagnostics in the single-screen Live Preview.
- Keep PlateDetector, temporal plate selection, Reading Zone, camera color handling, BGR Live Preview, FRAME_MASTER, Motion Detection, and ROI generation unchanged.

## 0.8.3

- Reduce OCR latency by running the two primary OCR passes (`rectified` and `enhanced`) concurrently.
- Run adaptive-threshold OCR only as a fallback when the primary passes are empty or disagree.
- Increase the minimum OCR interval to 1 second to avoid unnecessary Tesseract launches on a Raspberry Pi.
- Keep OCR work outside the HTTP request path and preserve the explicit `NO_PLATE`, `RUNNING`, `LIVE`, `STALE`, `UNAVAILABLE`, and `ERROR` states.
- Keep the grayscale ROI available for visual diagnostics but do not launch a redundant Tesseract pass for it.
- Keep PlateDetector, temporal selection, Reading Zone, camera color handling, BGR Live Preview, FRAME_MASTER, Motion Detection, and ROI generation unchanged.
- Keep the single-screen Live Preview and show the OCR optimization state clearly.

## 0.8.2

- Make Live OCR asynchronous and throttled instead of running four Tesseract jobs inside the HTTP request.
- Refresh OCR approximately every 0.5 seconds when a new selected frame is available.
- Expose explicit OCR states: `NO_PLATE`, `RUNNING`, `LIVE`, `STALE`, `UNAVAILABLE`, and `ERROR`.
- Show OCR result age so an old result cannot be mistaken for a current reading.
- Retain the last valid ROI briefly when the detector temporarily loses the plate, while clearly marking the OCR state as `STALE`.
- Clear OCR results after the retention timeout when no valid plate is available.
- Keep PlateDetector, temporal selection, Reading Zone, camera color handling, BGR Live Preview, FRAME_MASTER, Motion Detection, and ROI generation unchanged.
- Update the single-screen HTTP Live Preview to refresh OCR every second and display the current OCR state.

## 0.8.1

- Add an optional live OCR test stage after Plate ROI preparation.
- Run the system Tesseract engine independently against rectified, grayscale, enhanced, and adaptive-threshold ROI variants.
- Show each OCR result and a simple consensus/agreement value in the single-screen HTTP Live Preview.
- Keep OCR optional: if Tesseract is unavailable, the detector and ROI preview continue to operate normally.
- Do not change PlateDetector thresholds, temporal selection, Reading Zone, camera color handling, BGR Live Preview, FRAME_MASTER, Motion Detection, or ROI generation.
- This release is a test/measurement stage; no OCR result is used to make plate-selection decisions.

## 0.8.0

- Keep the proven 0.7.9 plate detector and temporal selection behavior unchanged.
- Add a five-stage Plate ROI preview: original, rectified, grayscale, enhanced, and adaptive threshold.
- Add lightweight four-corner plate rectification inside the selected ROI so the next OCR stage can be evaluated on a more level image.
- Keep ROI processing derived only from the selected detector bounding box; no OCR or recognition rule is changed in this release.
- Show the rectified ROI directly in the single-screen HTTP Live Preview.
- Keep camera color handling, Live Preview BGR behavior, FRAME_MASTER, Motion Detection, snapshots, diagnostics, and detector thresholds unchanged.

## 0.7.8

- Compact the HTTP Live Preview into a single-screen calibration layout.
- Keep live preview, selection, plate metrics, and detector thresholds visible together without a second calibration page.
- Reduce typography, spacing, card padding, and control size so the diagnostic information fits below the live image on desktop screens.
- Keep the detector, camera color pipeline, BGR Live Preview behavior, FRAME_MASTER, Motion Detection, snapshots, diagnostics, and OCR scope unchanged.
- Correct the visible HTTP page title/header to `GateKeeper AI 0.7.8`.

## 0.7.7

- Keep the 0.7.6 detector thresholds, soft Reading Zone ranking, and temporal selection behavior unchanged.
- Add measurement-only PlateDetector diagnostics for field calibration.
- Expose selection margin, selection mode, and stable selection metadata.
- Log a compact `PLATE DIAGNOSTICS` summary every 10 seconds without changing detection decisions.

## 0.7.6

- Keep the proven 0.7.5 full-frame detector and soft Reading Zone behavior unchanged as the baseline.
- Add conservative temporal selection stability.
- Keep detector thresholds unchanged.

## Current Release

See `VERSION` for the current application version.
