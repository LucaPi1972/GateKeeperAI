# Changelog

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
- Keep the detector, camera color pipeline, BGR Live Preview behavior, FRAME_MASTER, Motion Detection, snapshots, and OCR scope unchanged.
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
