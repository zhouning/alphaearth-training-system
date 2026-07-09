# Paper12 ArcGIS Replacement Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a testable evaluator and manifest schema for future Linhe ArcGIS-vs-Paper12 replacement validation.

**Architecture:** Keep the manual validation protocol as static JSON/CSV artifacts, implement metrics and status decisions in one focused script, and test with tiny synthetic masks. The existing `arcgis_replacement_validation_template.json` remains conservative until real paired evidence is supplied.

**Tech Stack:** Python standard library, NumPy, optional rasterio for raster mask loading, pytest.

---

### Task 1: Protocol Artifacts

**Files:**
- Create: `paper12_results/linhe_manual_validation_protocol.json`
- Create: `paper12_results/linhe_manual_validation_manifest_template.csv`
- Mirror both files under `submission/paper12_isprs_jprs_20260606/06_supplementary_material/paper12_results/`
- Test: `tests/test_arcgis_replacement_evaluator.py`

- [ ] **Step 1: Write a failing test that loads the protocol JSON and CSV header.**
- [ ] **Step 2: Run the test and verify it fails because the artifacts do not exist.**
- [ ] **Step 3: Add the protocol JSON and header-only CSV template.**
- [ ] **Step 4: Run the test and verify it passes.**

### Task 2: Evaluator Metrics

**Files:**
- Create: `scripts/evaluate_arcgis_replacement.py`
- Test: `tests/test_arcgis_replacement_evaluator.py`

- [ ] **Step 1: Write failing tests for tiny mask metrics.**
- [ ] **Step 2: Run the tests and verify imports/functions are missing.**
- [ ] **Step 3: Implement mask loading, confusion matrix, IoU, OA, macro F1, and paired deltas.**
- [ ] **Step 4: Run the tests and verify metrics pass.**

### Task 3: Decision Status and CLI

**Files:**
- Modify: `scripts/evaluate_arcgis_replacement.py`
- Test: `tests/test_arcgis_replacement_evaluator.py`

- [ ] **Step 1: Write failing tests for `replacement_candidate`, `partial`, and `not_validated` decisions.**
- [ ] **Step 2: Run the tests and verify the decision logic is missing.**
- [ ] **Step 3: Implement decision logic and CLI output JSON writing.**
- [ ] **Step 4: Run focused tests, then full tests.**

### Task 4: Verification and Commit

**Files:**
- All files above.

- [ ] **Step 1: Run `python -m pytest tests/test_arcgis_replacement_evaluator.py -q`.**
- [ ] **Step 2: Run `python -m pytest -q`.**
- [ ] **Step 3: Run `git diff --check`.**
- [ ] **Step 4: Commit with `feat: add paper12 ArcGIS replacement evaluator`.**
- [ ] **Step 5: Push to `origin master`.**
