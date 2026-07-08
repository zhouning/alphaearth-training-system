# Paper12 ArcGIS Replacement Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative ArcGIS replacement validation boundary for Paper12 without inventing unavailable validation results.

**Architecture:** Store the replacement decision protocol as a JSON artifact, derive audit fields from it, expose the status through backend summaries, and guard manuscript wording with tests. The implementation follows existing Paper12 audit and supplementary mirror patterns.

**Tech Stack:** Python, pytest, JSON artifacts, LaTeX manuscript text, FastAPI service helpers.

---

### Task 1: Validation Template

**Files:**
- Create: `paper12_results/arcgis_replacement_validation_template.json`
- Create mirror: `submission/paper12_isprs_jprs_20260606/06_supplementary_material/paper12_results/arcgis_replacement_validation_template.json`
- Test: `tests/test_paper12_public_dataset_results.py`

- [ ] **Step 1: Write the failing test**

Add a test requiring the template and supplementary mirror to exist, with `manual_ground_truth_available`, `arcgis_reference_available`, and `paper12_model_checkpoint_available` all false, plus final status `not_validated`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paper12_public_dataset_results.py::test_arcgis_replacement_validation_template_is_conservative_and_mirrored -q`

- [ ] **Step 3: Add the JSON template**

Create a deterministic JSON object with schema `paper12.arcgis_replacement_validation.v1`, decision statuses, required evidence flags, empty metrics, and next actions.

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest target.

### Task 2: Review Audit

**Files:**
- Modify: `geoadapter/bench/paper12_audit.py`
- Modify: `tests/test_paper12_review_audit.py`
- Regenerate: `paper12_results/review_audit_summary.json`
- Regenerate mirror: `submission/paper12_isprs_jprs_20260606/06_supplementary_material/paper12_results/review_audit_summary.json`

- [ ] **Step 1: Write failing audit tests**

Require `arcgis_replacement_audit` to report status `not_validated`, evidence level `weak_supervision_evidence`, no independent manual ground truth, and no universal replacement claim.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paper12_review_audit.py::test_review_audit_records_arcgis_replacement_boundary -q`

- [ ] **Step 3: Implement audit derivation**

Read the template from `SOURCE_FILES`, derive flags and decision status, and include the new section in `build_review_audit`.

- [ ] **Step 4: Regenerate audit JSON and mirror**

Run: `python -m geoadapter.bench.paper12_audit --repo-root . --output paper12_results/review_audit_summary.json`

- [ ] **Step 5: Run test to verify it passes**

Run the same pytest target.

### Task 3: Backend Summaries

**Files:**
- Modify: `ae_backend/app/services/paper12_summary.py`
- Modify: `ae_backend/app/services/system_capabilities.py`
- Modify: `tests/test_model_hub_api.py`

- [ ] **Step 1: Write failing backend tests**

Require `/api/ae/model-hub/paper12-summary` to include `arcgis_replacement_validation` with status `not_validated`, and require system evidence sources to include the template.

- [ ] **Step 2: Run tests to verify they fail**

Run the specific pytest targets in `tests/test_model_hub_api.py`.

- [ ] **Step 3: Implement backend fields**

Load the template in `paper12_summary.py` and add it to the returned payload. Add the template to optional Paper12 evidence sources in `system_capabilities.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run the same pytest targets.

### Task 4: Manuscript and Submission Guardrails

**Files:**
- Modify: `paper12/sections/linhe_validation.tex`
- Modify: `paper12/sections/discussion.tex`
- Modify: `paper12/sections/conclusion.tex`
- Modify mirrored submission section files.
- Modify: `submission/paper12_isprs_jprs_20260606/REQUIRED_EXPERIMENTS_ISPRS.md`
- Modify: `submission/paper12_isprs_jprs_20260606/06_supplementary_material/README_supplementary.md`
- Test: `tests/test_paper12_public_dataset_results.py`

- [ ] **Step 1: Write failing manuscript tests**

Require wording that Paper12 is not a validated ArcGIS replacement, and require the phrase `arcgis_replacement_validation_template.json` in manuscript/supporting material.

- [ ] **Step 2: Run test to verify it fails**

Run the selected test target.

- [ ] **Step 3: Patch text**

Add a concise paragraph to Linhe validation and propagate the boundary sentence into discussion/conclusion and submission docs.

- [ ] **Step 4: Run test to verify it passes**

Run the selected test target.

### Task 5: Verification, Compile, Commit, Push

**Files:**
- Generated: submission LaTeX PDF if compilation changes output.

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest tests/test_paper12_review_audit.py tests/test_paper12_public_dataset_results.py tests/test_model_hub_api.py -q`

- [ ] **Step 2: Run full tests**

Run: `python -m pytest -q`

- [ ] **Step 3: Compile LaTeX**

Run `pdflatex -interaction=nonstopmode main.tex` twice in `paper12`, and `pdflatex -interaction=nonstopmode main_isprs_jprs.tex` twice in the submission LaTeX folder.

- [ ] **Step 4: Check whitespace and status**

Run `git diff --check`, `git status --short`, and inspect the staged diff.

- [ ] **Step 5: Commit and push**

Commit with `feat: add paper12 ArcGIS replacement validation boundary` and push to `origin master`.
