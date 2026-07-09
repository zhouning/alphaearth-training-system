# Paper12 ArcGIS Validation Packet Implementation Plan

**Goal:** Add an offline packet builder, Paper12 prediction exporter, and finalizer
that prepare real Linhe samples for manual ArcGIS-vs-Paper12 validation without
fabricating evidence.

## Task 1: Packet Builder Tests

- [x] Add tests with toy CSV index, RGB patches, and Esri masks.
- [x] Verify tests fail because the packet builder script is missing.

## Task 2: Packet Builder Script

- [x] Implement CSV/parquet index loading.
- [x] Implement class-fraction sampling with critical-class coverage.
- [x] Export RGB arrays, Esri masks, preview PNGs, manifest stub, README, and
  summary JSON.
- [x] Keep manual and Paper12 fields blank and mark `evaluator_ready: false`.

## Task 3: Protocol Wiring

- [x] Add protocol links to `scripts/prepare_arcgis_replacement_validation_packet.py`.
- [x] Keep ArcGIS replacement status conservative.

## Task 4: Local Auto-Discovery

- [x] Add `--index auto` and `--patch-root` support to scan local Linhe
  `p_*.npz` / `lulc_<year>_p_*.npz` pairs without parquet dependencies.
- [x] Make sampling load only masks during candidate ranking and defer RGB
  loading until selected samples are exported.
- [x] Smoke-run a six-sample packet from `data/linhe_patches` to `D:\tmp` and
  confirm the evaluator remains `not_validated` until manual and Paper12 masks
  are filled.

## Task 5: Packet Finalizer

- [x] Add tests for missing manual/Paper12 masks, ready evaluator manifest output,
  shape mismatch rejection, and CLI execution.
- [x] Implement `scripts/finalize_arcgis_replacement_validation_packet.py` to
  wire existing masks into an evaluator-ready manifest without creating labels
  or predictions.
- [x] Update protocol, validation template, and generated packet README to route
  completed packets through the finalizer before the evaluator.

## Task 6: Paper12 Prediction Exporter

- [x] Add tests for checkpoint-backed mask export, resume-safe skipping,
  overwrite behavior, shape mismatch rejection, and missing-checkpoint CLI
  failure.
- [x] Implement `scripts/export_paper12_packet_predictions.py` to write
  `paper12_masks/<sample_id>.npy` from an existing Paper12 LULC checkpoint.
- [x] Keep the exporter bounded: it does not create manual masks, modify ArcGIS
  masks, or update replacement status.
- [x] Update protocol, validation template, generated packet README, and review
  audit to include the exporter before finalization/evaluation.

## Task 7: Verification

- [x] Run focused packet, exporter, finalizer, evaluator, and audit tests.
- [x] Run full pytest.
- [x] Run `git diff --check`.
- [ ] Commit and push to `origin/master`.