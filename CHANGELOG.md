# Changelog

## 0.7.6

- Keep the proven 0.7.5 full-frame detector and soft Reading Zone behavior unchanged as the baseline.
- Add conservative temporal selection stability: when the previously selected plate remains competitive, prefer it instead of jumping between similarly scored candidates.
- Add `stability_score`, `selection_margin`, and `selection_mode` metadata so calibration can distinguish strict selection from the soft fallback path and quantify how clearly the winner beats the next candidate.
- Keep the detector thresholds unchanged: `min_area: 2500`, `max_area: 150000`, aspect ratio `3.5-6.5`, rectangularity `0.80`, rotation `15`, and confidence threshold `0.70`.
- Keep the Reading Zone weight at 15% and the soft-selection threshold at 0.42.
- Do not change camera color handling, Live Preview BGR behavior, FRAME_MASTER, Motion Detection, diagnostics, snapshots, crop generation, or OCR scope.

## 0.7.5

- Keep the proven 0.7.4 full-frame contour detector; the Reading Zone is now a soft ranking signal instead of a hard filter.
- Add a small central Reading Zone preference with a conservative 15% ranking weight so candidates outside the guide are never discarded solely for position.
- Add a near-valid fallback selection path when strict thresholds produce no selected candidate, while still rejecting grossly small, oversized, badly proportioned, or strongly rotated candidates.
- Expose `zone_score` and `selection_score` for every candidate and show both metrics in the HTTP Plate Calibration panel.
- Keep the 0.7.4 detector thresholds unchanged: `min_area: 2500`, `max_area: 150000`, aspect ratio `3.5-6.5`, rectangularity `0.80`, rotation `15`, and confidence threshold `0.70`.
- Keep camera color handling, Live Preview BGR behavior, FRAME_MASTER, Motion Detection, diagnostics, snapshots, and OCR scope unchanged.

## 0.7.4

- Restore the proven 0.7.2 PlateDetector contour pipeline after the 0.7.3 isolated Reading Zone test proved too restrictive in real use.
- Keep the compact 328x70 px central Reading Zone as a visual guide only; it no longer gates full-frame candidate detection.
- Keep `max_area: 150000` and all other 0.7.2 detector thresholds unchanged.
- Keep Live Preview color handling, FRAME_MASTER, Motion Detection, diagnostics, snapshots, and OCR scope unchanged.
- This release is intended to establish a stable detector baseline before introducing a softer ROI strategy for false-positive reduction.

## 0.7.2

- Shrink the centered plate-ideal/reading guide to a compact 328x70 px zone on the 1640x1232 live frame, preserving an approximately 4.7:1 plate proportion.
- Keep the guide centered at x=656, y=581 so the printed plate can be positioned deliberately inside a small reading area and false candidates outside the area remain visually obvious during calibration.
- Disable continuous automatic calibration-frame saving; calibration frames remain manual through the existing Save Calibration Frame button.
- Keep the PlateDetector thresholds from 0.7.1 unchanged, including `max_area: 150000`.
- Keep Live Preview color handling, FRAME_MASTER, Motion Detection, diagnostics, snapshots, and OCR scope unchanged.

## 0.7.1

- Increase only the PlateDetector `max_area` default and configured value from `70000` to `150000`; all other detector thresholds remain unchanged.
- Add live Plate Calibration overlays to the existing HTTP Live Preview using the same current FRAME_MASTER-derived frame already evaluated by PlateDetector, without reopening the camera or creating another stream.
- Draw selected, valid-not-selected, and rejected candidates with metrics, rejection reasons, selected-plate emphasis, and runtime threshold summaries.
- Update the Plate Calibration web panel and save action so calibration state refreshes continuously and saved calibration frames use the current annotated preview frame.

## 0.7.0

- Start the license-plate calibration phase for observing the existing PlateDetector with printed Italian plates.
- Add configurable `plate_calibration` mode, candidate metadata, explicit rejection reasons, selected-candidate logging, and annotated `images/calibration_*.jpg` saves while preserving existing `images/plate_*.jpg` crops.
- Add `/api/plate_calibration`, manual calibration-frame saving, and a Plate Calibration web panel with read-only existing detector thresholds.
- Keep OCR, whitelist, GPIO/access control, Live Preview color handling, motion detection, and SQLite schema unchanged.

## Current Release

See `VERSION` for the current application version.
