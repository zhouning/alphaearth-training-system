# Paper12 ArcGIS Validation Packet Implementation Plan

**Goal:** Add an offline packet builder that prepares real Linhe samples for
manual ArcGIS-vs-Paper12 validation without fabricating evidence.

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

## Task 4: Verification

- [ ] Run focused packet and evaluator tests.
- [ ] Run full pytest.
- [ ] Run `git diff --check`.
- [ ] Commit and push to `origin/master`.
