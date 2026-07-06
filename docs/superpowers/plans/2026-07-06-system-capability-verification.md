# System Capability Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic system verification layer and UI surface that turns AlphaEarth System capability claims into auditable checks, warnings, and next actions.

**Architecture:** Keep verification as a read-only service beside the existing system capability service. The backend reuses `build_system_capabilities()` and the model registry so readiness semantics stay centralized, then the frontend displays the verification summary inside the existing Model Hub workbench.

**Tech Stack:** FastAPI, pytest, Vue 3 via the existing single-file `ae_frontend/index.html`, local JSON registry metadata, no model loading or network access.

---

## Scope

This plan implements only the system-first verification slice approved in `docs/superpowers/specs/2026-07-06-system-capability-verification-design.md`.

Do not edit Paper12 manuscript, submission, generated PDF, or Paper12 audit files while executing this plan. The current worktree has unrelated Paper12 changes; leave them unstaged and untouched.

## File Structure

- Create `ae_backend/app/services/system_verification.py`
  - Builds verification checks from the existing capability payload.
  - Owns status aggregation, check records, blocking issues, and next actions.
  - Does not load PyTorch, rasterio datasets, checkpoints, or external services.
- Modify `ae_backend/app/api/system.py`
  - Adds `GET /api/ae/system/verification`.
  - Reuses `get_model_registry()` like `/capabilities`.
- Modify `tests/test_model_hub_api.py`
  - Adds backend contract tests for endpoint shape, conservative ArcGIS guard, planned-model checkpoint semantics, and missing optional evidence warnings.
- Modify `tests/test_model_hub_frontend_entry.py`
  - Adds stable ASCII anchor tests for verification state, API route, helper functions, next actions, checks, and raw JSON detail hooks.
- Modify `ae_frontend/index.html`
  - Adds verification state and fetch method.
  - Extends the existing System Capability Workbench inside the Model Hub tab.
  - Adds per-model verification chips and next actions without changing existing job controls.

## Task 1: Backend Verification API Tests

**Files:**
- Modify: `tests/test_model_hub_api.py`
- Later create: `ae_backend/app/services/system_verification.py`
- Later modify: `ae_backend/app/api/system.py`

- [ ] **Step 1: Add failing endpoint contract test**

Append this test near the existing system capability endpoint tests in `tests/test_model_hub_api.py`:

```python
def test_system_verification_endpoint_reports_contract_checks():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/system/verification")

    assert response.status_code == 200
    body = response.json()
    assert body["system"] == "AlphaEarth System"
    assert body["overall_status"] in {"pass", "warning", "fail"}
    assert set(body["summary"]) == {"pass", "warning", "fail", "not_applicable"}
    assert set(body) >= {"generated_at", "capabilities", "checks", "notes"}
    assert body["checks"]
    assert body["notes"]

    capabilities = {item["id"]: item for item in body["capabilities"]}
    lulc = capabilities["lulc_6class_prithvi_houlsby"]
    assert lulc["overall_status"] in {"pass", "warning"}
    assert lulc["blocking_issues"] == []
    assert any(check_id.endswith(":checkpoint_configured") for check_id in lulc["checks"])

    crop_id = "prithvi_crop_classification_arcgis_style"
    crop = capabilities[crop_id]
    assert crop["overall_status"] in {"pass", "warning"}
    assert crop["blocking_issues"] == []
    assert any(check_id.endswith(":arcgis_replacement_guard") for check_id in crop["checks"])

    crop_checks = {
        check["id"]: check
        for check in body["checks"]
        if check["capability_id"] == crop_id
    }
    guard = crop_checks[f"{crop_id}:arcgis_replacement_guard"]
    assert guard["category"] == "replacement_boundary"
    assert guard["status"] == "pass"
    assert "not a validated ArcGIS replacement" in guard["detail"]
    assert "validated Prithvi crop checkpoint" in " ".join(crop["next_actions"])
```

- [ ] **Step 2: Run the contract test and verify it fails**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_verification_endpoint_reports_contract_checks -q
```

Expected: FAIL with HTTP 404 for `/api/ae/system/verification` or an import error for the missing verification service.

- [ ] **Step 3: Add failing planned-model checkpoint semantics test**

Append this test to `tests/test_model_hub_api.py`:

```python
def test_system_verification_does_not_fail_planned_models_for_missing_checkpoints():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/system/verification")

    assert response.status_code == 200
    body = response.json()
    capabilities = {item["id"]: item for item in body["capabilities"]}
    planned = capabilities["building_extraction_prithvi"]
    assert planned["overall_status"] in {"not_applicable", "warning"}
    assert all("checkpoint" not in issue.lower() for issue in planned["blocking_issues"])

    planned_checks = [
        check
        for check in body["checks"]
        if check["capability_id"] == "building_extraction_prithvi"
    ]
    checkpoint_checks = [
        check for check in planned_checks if check["category"] == "checkpoint_configuration"
    ]
    assert checkpoint_checks
    assert all(check["status"] == "not_applicable" for check in checkpoint_checks)
```

- [ ] **Step 4: Run the planned-model test and verify it fails**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_verification_does_not_fail_planned_models_for_missing_checkpoints -q
```

Expected: FAIL because the verification endpoint does not exist yet.

- [ ] **Step 5: Add failing missing optional evidence test**

Append this test to `tests/test_model_hub_api.py`:

```python
def test_system_verification_reports_missing_optional_evidence_as_warnings(
    monkeypatch,
    tmp_path: Path,
):
    from app.main import app
    import app.services.system_capabilities as system_capabilities

    monkeypatch.setattr(system_capabilities, "PAPER12_RESULTS_DIR", tmp_path)

    client = TestClient(app)
    response = client.get("/api/ae/system/verification")

    assert response.status_code == 200
    body = response.json()
    evidence_checks = [
        check
        for check in body["checks"]
        if check["category"] == "evidence_source"
    ]
    assert evidence_checks
    assert any(check["status"] == "warning" for check in evidence_checks)
    assert all(check["status"] != "fail" for check in evidence_checks)
    assert any("Missing optional evidence file" in check["detail"] for check in evidence_checks)
```

- [ ] **Step 6: Run the missing evidence test and verify it fails**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_verification_reports_missing_optional_evidence_as_warnings -q
```

Expected: FAIL because the verification endpoint does not exist yet.

- [ ] **Step 7: Commit failing backend tests**

Stage only the API test file:

```powershell
git add tests/test_model_hub_api.py
git commit -m "test: cover system capability verification API"
```

## Task 2: Backend Verification Service and Route

**Files:**
- Create: `ae_backend/app/services/system_verification.py`
- Modify: `ae_backend/app/api/system.py`
- Test: `tests/test_model_hub_api.py`

- [ ] **Step 1: Create service module**

Create `ae_backend/app/services/system_verification.py` with this structure:

```python
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.services.model_hub_registry import VALID_STATUSES, ModelHubRegistry
from app.services import system_capabilities


STATUS_ORDER = {
    "fail": 3,
    "warning": 2,
    "pass": 1,
    "not_applicable": 0,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _check(
    *,
    capability_id: str,
    category: str,
    status: str,
    severity: str,
    title: str,
    detail: str,
    evidence_refs: list[str],
    remediation: str | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{capability_id}:{category}",
        "capability_id": capability_id,
        "category": category,
        "status": status,
        "severity": severity,
        "title": title,
        "detail": detail,
        "evidence_refs": evidence_refs,
        "remediation": remediation,
    }


def _aggregate_status(statuses: list[str]) -> str:
    if not statuses:
        return "not_applicable"
    return max(statuses, key=lambda item: STATUS_ORDER[item])


def _registry_status_check(capability: dict[str, Any]) -> dict[str, Any]:
    readiness = capability["readiness"]
    if readiness in VALID_STATUSES:
        return _check(
            capability_id=capability["id"],
            category="registry_status",
            status="pass",
            severity="info",
            title="Registry status is valid",
            detail=f"Capability status {readiness!r} is registered and recognized.",
            evidence_refs=["model_hub_registry"],
        )
    return _check(
        capability_id=capability["id"],
        category="registry_status",
        status="fail",
        severity="error",
        title="Registry status is unknown",
        detail=f"Capability status {readiness!r} is not recognized.",
        evidence_refs=["model_hub_registry"],
        remediation="Use one of ready, demo_only, planned, or not_configured.",
    )


def _runtime_mode_check(capability: dict[str, Any]) -> dict[str, Any]:
    if capability["readiness"] in {"planned", "not_configured"}:
        return _check(
            capability_id=capability["id"],
            category="runtime_mode",
            status="not_applicable",
            severity="info",
            title="Runtime mode is not required for planned capability",
            detail="Planned capabilities are tracked in the registry before executable runtime modes are configured.",
            evidence_refs=["system_capabilities"],
        )
    if capability["runtime_modes"]:
        return _check(
            capability_id=capability["id"],
            category="runtime_mode",
            status="pass",
            severity="info",
            title="Runtime mode is declared",
            detail=f"Declared runtime modes: {', '.join(capability['runtime_modes'])}.",
            evidence_refs=["model_hub_registry", "system_capabilities"],
        )
    return _check(
        capability_id=capability["id"],
        category="runtime_mode",
        status="fail",
        severity="error",
        title="Executable capability has no runtime mode",
        detail="Ready and demo capabilities must expose at least one runtime mode.",
        evidence_refs=["model_hub_registry", "system_capabilities"],
        remediation="Add a supported runtime mode or downgrade the capability readiness.",
    )


def _checkpoint_check(capability: dict[str, Any]) -> dict[str, Any]:
    if capability["readiness"] in {"planned", "not_configured"}:
        return _check(
            capability_id=capability["id"],
            category="checkpoint_configuration",
            status="not_applicable",
            severity="info",
            title="Checkpoint is not required for planned capability",
            detail="Missing checkpoints do not fail planned capabilities.",
            evidence_refs=["system_capabilities"],
        )
    if capability["checkpoint"]["configured"]:
        return _check(
            capability_id=capability["id"],
            category="checkpoint_configuration",
            status="pass",
            severity="info",
            title="Checkpoint is configured",
            detail=f"Checkpoint path is configured: {capability['checkpoint']['path']}.",
            evidence_refs=["model_hub_registry"],
        )
    if capability["workflow_level"] in {"demo", "contract_demo"}:
        return _check(
            capability_id=capability["id"],
            category="checkpoint_configuration",
            status="pass",
            severity="info",
            title="Demo workflow does not require a checkpoint",
            detail="This capability is explicitly marked as a demo or contract demo, so missing checkpoint weights do not block the demo contract.",
            evidence_refs=["model_hub_registry", "system_capabilities"],
        )
    return _check(
        capability_id=capability["id"],
        category="checkpoint_configuration",
        status="fail",
        severity="error",
        title="Ready capability is missing a checkpoint",
        detail="Runnable ready capabilities must declare a checkpoint path.",
        evidence_refs=["model_hub_registry"],
        remediation="Configure a checkpoint path or downgrade the capability readiness.",
    )


def _replacement_boundary_check(capability: dict[str, Any]) -> dict[str, Any]:
    if capability["id"] != "prithvi_crop_classification_arcgis_style":
        return _check(
            capability_id=capability["id"],
            category="replacement_boundary",
            status="not_applicable",
            severity="info",
            title="ArcGIS replacement guard is not applicable",
            detail="This capability is not the ArcGIS-style Prithvi crop workflow.",
            evidence_refs=["system_capabilities"],
        )
    replacement = capability["arcgis_replacement"]
    if replacement["status"] == "not_ready":
        return _check(
            capability_id=capability["id"],
            category="replacement_boundary",
            status="pass",
            severity="info",
            title="ArcGIS replacement guard is conservative",
            detail="The crop capability is marked as demo-only and not a validated ArcGIS replacement.",
            evidence_refs=["model_hub_registry", "system_capabilities"],
            remediation="Attach a validated Prithvi crop checkpoint before claiming replacement readiness.",
        )
    return _check(
        capability_id=capability["id"],
        category="replacement_boundary",
        status="fail",
        severity="error",
        title="ArcGIS replacement guard is overclaiming",
        detail=f"Replacement status {replacement['status']!r} is not allowed without a validated checkpoint-backed workflow.",
        evidence_refs=["model_hub_registry", "system_capabilities"],
        remediation="Reset replacement status to not_ready until validation evidence exists.",
    )


def _evidence_source_checks(evidence_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for source in evidence_sources:
        label = str(source["label"])
        source_id = label.lower().replace(" ", "_").replace(".", "")
        if source.get("available"):
            checks.append(
                _check(
                    capability_id="system",
                    category=f"evidence_source:{source_id}",
                    status="pass",
                    severity="info",
                    title=f"{label} evidence file is available",
                    detail=f"Optional evidence file exists at {source['path']}.",
                    evidence_refs=[source["path"]],
                )
            )
        else:
            checks.append(
                _check(
                    capability_id="system",
                    category=f"evidence_source:{source_id}",
                    status="warning",
                    severity="warning",
                    title=f"{label} evidence file is missing",
                    detail=f"Missing optional evidence file: {source['path']}.",
                    evidence_refs=[source["path"]],
                    remediation="Regenerate or attach the optional Paper12 evidence artifact when available.",
                )
            )
    return checks


def _capability_verification(
    capability: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = [check["status"] for check in checks]
    overall_status = _aggregate_status(statuses)
    blocking_issues = [
        check["detail"]
        for check in checks
        if check["status"] == "fail"
    ]
    next_actions = [
        check["remediation"]
        for check in checks
        if check["status"] in {"warning", "fail"} and check.get("remediation")
    ]
    if capability["id"] == "prithvi_crop_classification_arcgis_style":
        action = "Attach a validated Prithvi crop checkpoint before claiming replacement readiness."
        if action not in next_actions:
            next_actions.append(action)
    for action in capability.get("next_steps", []):
        if action not in next_actions and overall_status in {"warning", "fail"}:
            next_actions.append(action)
    return {
        "id": capability["id"],
        "overall_status": overall_status,
        "checks": [check["id"] for check in checks],
        "blocking_issues": blocking_issues,
        "next_actions": next_actions,
    }


def build_system_verification(registry: ModelHubRegistry) -> dict[str, Any]:
    capability_payload = system_capabilities.build_system_capabilities(registry)
    all_checks: list[dict[str, Any]] = []
    capability_summaries = []

    for capability in capability_payload["capabilities"]:
        capability_checks = [
            _registry_status_check(capability),
            _runtime_mode_check(capability),
            _checkpoint_check(capability),
            _replacement_boundary_check(capability),
        ]
        all_checks.extend(capability_checks)
        capability_summaries.append(_capability_verification(capability, capability_checks))

    evidence_checks = _evidence_source_checks(capability_payload.get("evidence_sources", []))
    all_checks.extend(evidence_checks)

    summary = dict(Counter(check["status"] for check in all_checks))
    for status in ["pass", "warning", "fail", "not_applicable"]:
        summary.setdefault(status, 0)

    return {
        "system": "AlphaEarth System",
        "generated_at": _utc_now(),
        "overall_status": _aggregate_status([check["status"] for check in all_checks]),
        "summary": {
            "pass": summary["pass"],
            "warning": summary["warning"],
            "fail": summary["fail"],
            "not_applicable": summary["not_applicable"],
        },
        "capabilities": capability_summaries,
        "checks": all_checks,
        "notes": [
            "Verification is deterministic and does not load model weights.",
            "A pass means the declared system contract is internally consistent, not that global production accuracy is proven.",
            "ArcGIS replacement readiness remains conservative until checkpoint-backed validation evidence is attached.",
        ],
    }
```

- [ ] **Step 2: Add route**

Modify `ae_backend/app/api/system.py` to import the service and add the route:

```python
from app.services.system_capabilities import build_system_capabilities
from app.services.system_verification import build_system_verification
```

Add below `get_system_capabilities()`:

```python
@router.get("/verification")
def get_system_verification():
    return build_system_verification(get_model_registry())
```

- [ ] **Step 3: Run backend verification tests**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_verification_endpoint_reports_contract_checks tests/test_model_hub_api.py::test_system_verification_does_not_fail_planned_models_for_missing_checkpoints tests/test_model_hub_api.py::test_system_verification_reports_missing_optional_evidence_as_warnings -q
```

Expected: PASS.

- [ ] **Step 4: Run existing capability endpoint tests**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_capabilities_endpoint_reports_operational_readiness tests/test_model_hub_api.py::test_system_capabilities_tolerates_missing_optional_evidence -q
```

Expected: PASS.

- [ ] **Step 5: Commit backend implementation**

Stage only relevant files:

```powershell
git add ae_backend/app/services/system_verification.py ae_backend/app/api/system.py
git commit -m "feat: add system capability verification API"
```

## Task 3: Frontend Verification Contract Tests

**Files:**
- Modify: `tests/test_model_hub_frontend_entry.py`
- Later modify: `ae_frontend/index.html`

- [ ] **Step 1: Add failing frontend anchor test**

Append this test to `tests/test_model_hub_frontend_entry.py`:

```python
def test_frontend_exposes_system_verification_workbench_hooks():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "/api/ae/system/verification" in html
    assert "systemVerification" in html
    assert "loadingSystemVerification" in html
    assert "systemVerificationError" in html
    assert "fetchSystemVerification" in html
    assert "verificationForCapability" in html
    assert "verificationStatusClass" in html
    assert "overall_status" in html
    assert "next_actions" in html
    assert "checks" in html
    assert "systemVerificationRawJson" in html
```

- [ ] **Step 2: Add regression test for existing Model Hub controls**

Append this test to `tests/test_model_hub_frontend_entry.py`:

```python
def test_frontend_keeps_model_hub_job_controls_with_system_verification():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "runModelHubDemo" in html
    assert "runModelHubRasterDemo" in html
    assert "modelHubJob.artifacts" in html
    assert "modelHubJob.logs" in html
    assert "fetchSystemCapabilities" in html
    assert "fetchSystemVerification" in html
```

- [ ] **Step 3: Run frontend anchor tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py::test_frontend_exposes_system_verification_workbench_hooks tests/test_model_hub_frontend_entry.py::test_frontend_keeps_model_hub_job_controls_with_system_verification -q
```

Expected: first test FAILS because verification hooks are not in `index.html`; second may fail until `fetchSystemVerification` is added.

- [ ] **Step 4: Commit failing frontend tests**

Stage only the frontend test file:

```powershell
git add tests/test_model_hub_frontend_entry.py
git commit -m "test: cover system verification frontend"
```

## Task 4: Frontend Verification UI

**Files:**
- Modify: `ae_frontend/index.html`
- Test: `tests/test_model_hub_frontend_entry.py`

- [ ] **Step 1: Add verification state**

In the Vue setup block near existing `systemCapabilities` refs, add:

```javascript
const systemVerification = ref(null);
const loadingSystemVerification = ref(false);
const systemVerificationError = ref(null);
```

- [ ] **Step 2: Add status helpers**

Near `systemCapabilityFor` and related helpers, add:

```javascript
const verificationStatusClass = (status) => {
    if (status === 'pass') return 'bg-emerald-50 text-emerald-700 border-emerald-200';
    if (status === 'warning') return 'bg-amber-50 text-amber-700 border-amber-200';
    if (status === 'fail') return 'bg-red-50 text-red-700 border-red-200';
    if (status === 'not_applicable') return 'bg-gray-50 text-gray-600 border-gray-200';
    return 'bg-gray-50 text-gray-700 border-gray-200';
};

const verificationForCapability = (model) => {
    const capabilities = systemVerification.value?.capabilities || [];
    return capabilities.find(item => item.id === model.model_id) || null;
};

const systemVerificationRawJson = computed(() => {
    if (!systemVerification.value) return '';
    return JSON.stringify(systemVerification.value, null, 2);
});
```

- [ ] **Step 3: Add verification fetch method**

Near `fetchSystemCapabilities`, add:

```javascript
const fetchSystemVerification = async () => {
    loadingSystemVerification.value = true;
    systemVerificationError.value = null;
    try {
        const res = await fetch(`${API_BASE}/api/ae/system/verification`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        systemVerification.value = await res.json();
    } catch (e) {
        systemVerificationError.value = e?.message || 'unknown error';
    } finally {
        loadingSystemVerification.value = false;
    }
};
```

- [ ] **Step 4: Fetch verification with Model Hub data**

In `fetchModelHubModels`, after the existing `fetchSystemCapabilities()` call, add:

```javascript
if (!systemVerification.value) await fetchSystemVerification();
```

In the tab watcher for `modelHub`, add:

```javascript
if (!systemVerification.value) fetchSystemVerification();
```

- [ ] **Step 5: Add verification summary UI**

Inside the existing System Capability Workbench block, below the capability summary and before per-model cards, add markup with these anchors:

```html
<div class="glass-card p-4 space-y-3">
    <div class="flex flex-wrap items-center justify-between gap-3">
        <div>
            <div class="text-sm font-semibold text-gray-800">System verification</div>
            <div class="text-xs text-gray-500">Deterministic checks for registry, runtime, checkpoint, evidence, and replacement boundary.</div>
        </div>
        <button type="button" class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded-md text-sm hover:bg-gray-50 transition-colors cursor-pointer disabled:opacity-50" :disabled="loadingSystemVerification" @click="fetchSystemVerification">
            {{ loadingSystemVerification ? 'Checking...' : 'Refresh verification' }}
        </button>
    </div>
    <div v-if="systemVerification" class="space-y-3">
        <div class="flex flex-wrap items-center gap-2 text-xs">
            <span class="px-2 py-1 rounded border" :class="verificationStatusClass(systemVerification.overall_status)">
                overall_status: {{ systemVerification.overall_status }}
            </span>
            <span v-for="(count, status) in systemVerification.summary" :key="status" class="px-2 py-1 rounded border" :class="verificationStatusClass(status)">
                {{ status }}: {{ count }}
            </span>
        </div>
        <details class="text-xs">
            <summary class="cursor-pointer text-gray-600">Raw verification JSON</summary>
            <pre class="mt-2 max-h-64 overflow-auto bg-gray-900 text-gray-100 p-3 rounded">{{ systemVerificationRawJson }}</pre>
        </details>
    </div>
    <div v-else-if="systemVerificationError" class="text-sm text-red-600">System verification failed: {{ systemVerificationError }}</div>
</div>
```

When translating labels to Chinese during implementation, keep these ASCII anchors unchanged: `System verification`, `overall_status`, and `Raw verification JSON`.

- [ ] **Step 6: Add per-card verification chip and next actions**

Inside each Model Hub model card near existing system capability chips, add:

```html
<span v-if="verificationForCapability(model)" class="px-2 py-1 rounded border" :class="verificationStatusClass(verificationForCapability(model).overall_status)">
    verification: {{ verificationForCapability(model).overall_status }}
</span>
```

Inside the existing system capability detail area or just below it, add:

```html
<div v-if="verificationForCapability(model)" class="text-xs text-gray-500 bg-white border border-gray-200 rounded p-2 space-y-2">
    <div class="flex flex-wrap gap-2">
        <span v-for="checkId in verificationForCapability(model).checks" :key="checkId" class="px-2 py-1 rounded border bg-gray-50 border-gray-200 break-all">
            {{ checkId }}
        </span>
    </div>
    <div v-if="verificationForCapability(model).blocking_issues && verificationForCapability(model).blocking_issues.length">
        <div class="font-medium text-red-700">Blocking issues</div>
        <ul class="list-disc pl-4">
            <li v-for="issue in verificationForCapability(model).blocking_issues" :key="issue">{{ issue }}</li>
        </ul>
    </div>
    <div v-if="verificationForCapability(model).next_actions && verificationForCapability(model).next_actions.length">
        <div class="font-medium text-gray-700">next_actions</div>
        <ul class="list-disc pl-4">
            <li v-for="action in verificationForCapability(model).next_actions" :key="action">{{ action }}</li>
        </ul>
    </div>
</div>
```

- [ ] **Step 7: Return new refs and helpers from setup**

In the setup return object, include:

```javascript
systemVerification, loadingSystemVerification, systemVerificationError, fetchSystemVerification,
verificationForCapability, verificationStatusClass, systemVerificationRawJson,
```

- [ ] **Step 8: Run frontend tests**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py::test_frontend_exposes_system_verification_workbench_hooks tests/test_model_hub_frontend_entry.py::test_frontend_keeps_model_hub_job_controls_with_system_verification -q
```

Expected: PASS.

- [ ] **Step 9: Run all frontend entry tests**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit frontend implementation**

Stage only relevant files:

```powershell
git add ae_frontend/index.html
git commit -m "feat: add system verification workbench UI"
```

## Task 5: Focused Integration Verification

**Files:**
- Verify only; no planned edits.

- [ ] **Step 1: Run focused Model Hub API and frontend tests**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py tests/test_model_hub_frontend_entry.py -q
```

Expected: PASS. If pytest prints a Windows native-library access-violation stack after the summary but exits with code 0, record it as an environment warning, not a test failure.

- [ ] **Step 2: Run adjacent regression tests**

Run:

```powershell
python -m pytest tests/test_model_hub_registry.py tests/test_inference_api.py tests/test_inference_service.py -q
```

Expected: PASS.

- [ ] **Step 3: Check whitespace**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors in files touched by this plan. Pre-existing CRLF warnings in unrelated Paper12 files may remain; do not edit those files here.

- [ ] **Step 4: Optional local endpoint smoke test**

If a dev server is already running, test the endpoint:

```powershell
curl.exe -s http://127.0.0.1:61530/api/ae/system/verification
```

Expected: JSON containing `"system":"AlphaEarth System"` and `"overall_status"`.

If no server is running, skip this step and rely on `TestClient` coverage.

- [ ] **Step 5: Commit any verification-only test adjustment**

Only if Task 5 reveals a necessary test-only adjustment, stage that file and commit:

```powershell
git add tests/test_model_hub_api.py tests/test_model_hub_frontend_entry.py
git commit -m "test: stabilize system verification checks"
```

If no adjustment is needed, do not create an empty commit.

## Self-Review

- Spec coverage:
  - `GET /api/ae/system/verification`: Task 2.
  - Response fields `system`, `generated_at`, `overall_status`, `summary`, `capabilities`, `checks`, `notes`: Task 1 and Task 2.
  - Registry, runtime, checkpoint, evidence, and replacement-boundary checks: Task 2.
  - LULC pass or non-blocking operational check: Task 1.
  - Crop ArcGIS replacement guard: Task 1 and Task 2.
  - Planned models do not fail solely for missing checkpoints: Task 1 and Task 2.
  - Missing optional evidence files become warnings: Task 1 and Task 2.
  - Frontend summary, per-card status, next actions, and raw JSON detail: Task 3 and Task 4.
  - Existing job controls preserved: Task 3 and Task 4.
- Placeholder scan:
  - No placeholder instructions are present.
  - Each code-changing task includes concrete snippets and exact commands.
- Type consistency:
  - Backend uses `overall_status`, `summary`, `capabilities`, `checks`, `blocking_issues`, and `next_actions`.
  - Frontend tests and UI use the same keys.
  - Check categories use `registry_status`, `runtime_mode`, `checkpoint_configuration`, `replacement_boundary`, and evidence categories prefixed by `evidence_source`.


