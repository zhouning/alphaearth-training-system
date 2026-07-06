# System Capability Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a system-first AlphaEarth capability workbench API and UI that reports runnable, evaluable, demo-only, planned, and not-configured capabilities without making Paper 12 the primary product surface.

**Architecture:** Add a lightweight backend capability service and `/api/ae/system/capabilities` route that derive product status from the existing model registry, runtime-mode metadata, and local evidence files. Update the existing static Vue Model Hub tab to consume this endpoint while preserving current model-hub job flows and the legacy Paper 12 summary endpoint.

**Tech Stack:** FastAPI, Pydantic-free dict responses, existing JSON model registry, static Vue 3 composition API in `ae_frontend/index.html`, pytest with FastAPI `TestClient`.

---

## File Structure

- Create `ae_backend/app/services/system_capabilities.py`
  - Build the system capability response.
  - Keep it lightweight: registry and JSON file reads only; no PyTorch, rasterio, or model loading imports.
  - Export `build_system_capabilities(registry: ModelHubRegistry) -> dict[str, Any]`.

- Create `ae_backend/app/api/system.py`
  - Define a FastAPI router with `GET /capabilities`.
  - Reuse `get_model_registry()` from `app.api.model_hub` so the endpoint sees the same registry cache.

- Modify `ae_backend/app/main.py`
  - Import `system`.
  - Mount `system.router` under `/api/ae/system`.

- Modify `tests/test_model_hub_api.py`
  - Add backend contract tests for `/api/ae/system/capabilities`.
  - Keep existing `/api/ae/model-hub/paper12-summary` tests for compatibility.

- Modify `tests/test_model_hub_frontend_entry.py`
  - Replace the Paper 12-first frontend test with system-first assertions.
  - Keep existing model-hub runtime and job-result assertions.

- Modify `ae_frontend/index.html`
  - Add `systemCapabilities`, loading/error state, and fetch helpers.
  - Replace the Paper 12-first summary section with a System Capability Workbench section.
  - Keep the existing Paper 12 summary method for backward-compatible evidence loading if useful, but do not make it the primary page headline.
  - Keep existing model-hub list, demo job, raster demo, summary, artifacts, logs, and raw JSON flows.

---

### Task 1: Backend Contract Tests

**Files:**
- Modify: `tests/test_model_hub_api.py`
- Do not modify implementation files in this task.

- [ ] **Step 1: Add failing system capability endpoint test**

Append these tests near the existing Paper 12 summary tests in `tests/test_model_hub_api.py`:

```python
def test_system_capabilities_endpoint_reports_operational_readiness():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/system/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["system"] == "AlphaEarth System"
    assert set(body) >= {
        "generated_at",
        "readiness_counts",
        "summary",
        "capabilities",
        "evidence_sources",
    }
    assert body["readiness_counts"]["ready"] >= 1
    assert body["readiness_counts"]["demo_only"] >= 1
    assert body["readiness_counts"]["planned"] >= 1
    assert body["summary"]["runnable_models"] >= 1
    assert body["summary"]["demo_workflows"] >= 1
    assert body["summary"]["arcgis_replacement_ready"] is False

    capabilities = {item["id"]: item for item in body["capabilities"]}
    lulc = capabilities["lulc_6class_prithvi_houlsby"]
    assert lulc["readiness"] == "ready"
    assert lulc["workflow_level"] == "runnable_and_evaluable"
    assert lulc["checkpoint"]["configured"] is True
    assert "demo_patch" in lulc["runtime_modes"]
    assert any(item["label"] == "mIoU" for item in lulc["evidence"])

    crop = capabilities["prithvi_crop_classification_arcgis_style"]
    assert crop["readiness"] == "demo_only"
    assert crop["workflow_level"] == "contract_demo"
    assert crop["checkpoint"]["configured"] is False
    assert crop["arcgis_replacement"]["status"] == "not_ready"
    assert "No validated crop checkpoint" in crop["arcgis_replacement"]["reason"]
    assert "upload_raster_demo" in crop["runtime_modes"]
```

- [ ] **Step 2: Add missing evidence tolerance test**

Append this test after the first new test:

```python
def test_system_capabilities_tolerates_missing_optional_evidence(monkeypatch, tmp_path: Path):
    from app.main import app
    import app.services.system_capabilities as system_capabilities

    monkeypatch.setattr(system_capabilities, "PAPER12_RESULTS_DIR", tmp_path)

    client = TestClient(app)
    response = client.get("/api/ae/system/capabilities")

    assert response.status_code == 200
    body = response.json()
    missing_sources = [
        item
        for item in body["evidence_sources"]
        if item["kind"] == "paper12_benchmark" and item["available"] is False
    ]
    assert missing_sources
    assert all("missing" in item["note"].lower() for item in missing_sources)
```

- [ ] **Step 3: Run tests and verify expected failure**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_capabilities_endpoint_reports_operational_readiness tests/test_model_hub_api.py::test_system_capabilities_tolerates_missing_optional_evidence -q
```

Expected: both tests fail because `/api/ae/system/capabilities` and `app.services.system_capabilities` do not exist yet.

- [ ] **Step 4: Commit tests**

Run:

```powershell
git add tests/test_model_hub_api.py
git commit -m "test: cover system capability endpoint"
```

Expected: commit contains only `tests/test_model_hub_api.py`.

---

### Task 2: Backend Service And Route

**Files:**
- Create: `ae_backend/app/services/system_capabilities.py`
- Create: `ae_backend/app/api/system.py`
- Modify: `ae_backend/app/main.py`
- Test: `tests/test_model_hub_api.py`

- [ ] **Step 1: Create the service module**

Create `ae_backend/app/services/system_capabilities.py` with:

```python
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.services.model_hub_registry import ModelHubRegistry


PAPER12_RESULTS_DIR = Path(PROJECT_ROOT) / "paper12_results"


def _runtime_modes_for_model(model: dict[str, Any]) -> list[str]:
    modes = set(model.get("package_profile", {}).get("runtime_modes", []))
    default_mode = model.get("input_spec", {}).get("default_demo_input_mode")
    if default_mode:
        modes.add(str(default_mode))
    for mode in model.get("input_spec", {}).get("supported_job_input_modes", []):
        modes.add(str(mode))
    if model.get("status") == "demo_only":
        modes.add("cached_demo")
    if model.get("model_id") == "lulc_6class_prithvi_houlsby":
        modes.add("demo_patch")
    return sorted(modes)


def _workflow_level(model: dict[str, Any], runtime_modes: list[str]) -> str:
    if model["model_id"] == "prithvi_crop_classification_arcgis_style":
        return "contract_demo"
    if model["status"] == "ready" and runtime_modes:
        return "runnable_and_evaluable"
    if model["status"] == "ready":
        return "registered_ready"
    if model["status"] == "demo_only":
        return "demo"
    if model["status"] == "planned":
        return "planned"
    return "not_configured"


def _checkpoint(model: dict[str, Any]) -> dict[str, Any]:
    path = model.get("checkpoint_path")
    return {
        "configured": bool(path),
        "path": path,
    }


def _metric_evidence(model: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = model.get("metrics", {})
    evidence = []
    for key, value in metrics.items():
        if key in {"source", "label_source", "readiness"}:
            continue
        if value is None:
            continue
        evidence.append(
            {
                "kind": "metric",
                "label": key,
                "value": value,
                "source": metrics.get("source", "model_hub_registry"),
            }
        )
    if "label_source" in metrics:
        evidence.append(
            {
                "kind": "label_source",
                "label": "label_source",
                "value": metrics["label_source"],
                "source": "model_hub_registry",
            }
        )
    return evidence


def _limitations(model: dict[str, Any]) -> list[str]:
    package_profile = model.get("package_profile", {})
    applicability = package_profile.get("applicability", {})
    limitations = applicability.get("limitations", [])
    if isinstance(limitations, list):
        return [str(item) for item in limitations]
    return []


def _next_steps(model: dict[str, Any]) -> list[str]:
    package_profile = model.get("package_profile", {})
    model_card = package_profile.get("model_card", {})
    next_step = model_card.get("next_step")
    if model["model_id"] == "prithvi_crop_classification_arcgis_style":
        steps = [
            "Attach a validated Prithvi crop head and HLS preprocessing pipeline.",
            "Run independent crop validation before claiming replacement readiness.",
        ]
        if next_step and next_step not in steps:
            steps.insert(0, str(next_step))
        return steps
    if model["status"] in {"planned", "not_configured"}:
        return ["Configure checkpoint, preprocessing, runtime, and validation metrics."]
    if next_step:
        return [str(next_step)]
    return []


def _arcgis_replacement(model: dict[str, Any], workflow_level: str) -> dict[str, Any]:
    if model["model_id"] == "prithvi_crop_classification_arcgis_style":
        return {
            "status": "not_ready",
            "reason": "No validated crop checkpoint or ArcGIS-compatible HLS preprocessing contract is configured.",
        }
    if workflow_level == "runnable_and_evaluable":
        return {
            "status": "partial",
            "reason": "Runnable inside AlphaEarth System, but not packaged as an ArcGIS .dlpk replacement.",
        }
    if workflow_level in {"demo", "contract_demo"}:
        return {
            "status": "demo_only",
            "reason": "Capability is available as a demo workflow, not a validated ArcGIS replacement.",
        }
    return {
        "status": "not_ready",
        "reason": "Capability is registered but lacks a validated runnable replacement workflow.",
    }


def _capability(model: dict[str, Any]) -> dict[str, Any]:
    runtime_modes = _runtime_modes_for_model(model)
    workflow_level = _workflow_level(model, runtime_modes)
    return {
        "id": model["model_id"],
        "display_name": model["display_name"],
        "task_type": model["task_type"],
        "readiness": model["status"],
        "workflow_level": workflow_level,
        "runtime_modes": runtime_modes,
        "input_spec": model.get("input_spec", {}),
        "output_spec": model.get("output_spec", {}),
        "supported_sensors": model.get("supported_sensors", []),
        "trained_region": model.get("trained_region"),
        "checkpoint": _checkpoint(model),
        "evidence": _metric_evidence(model),
        "limitations": _limitations(model),
        "next_steps": _next_steps(model),
        "arcgis_replacement": _arcgis_replacement(model, workflow_level),
    }


def _evidence_source(path: Path, label: str) -> dict[str, Any]:
    available = path.exists()
    payload = {
        "kind": "paper12_benchmark",
        "label": label,
        "path": f"paper12_results/{path.name}",
        "available": available,
    }
    if not available:
        payload["note"] = f"Missing local evidence file: paper12_results/{path.name}"
    return payload


def _evidence_sources() -> list[dict[str, Any]]:
    return [
        _evidence_source(
            PAPER12_RESULTS_DIR / "eurosat_channel_bridge_summary.json",
            "EuroSAT channel bridge",
        ),
        _evidence_source(
            PAPER12_RESULTS_DIR / "landcoverai_segmentation.json",
            "LandCover.ai segmentation",
        ),
        _evidence_source(
            PAPER12_RESULTS_DIR / "loveda_full_finetune_summary.json",
            "LoveDA full fine-tuning",
        ),
    ]


def build_system_capabilities(registry: ModelHubRegistry) -> dict[str, Any]:
    models = [model.to_dict() for model in registry.models]
    capabilities = [_capability(model) for model in models]

    readiness_counts = dict(Counter(model["status"] for model in models))
    for status in ["ready", "evaluable", "demo_only", "planned", "not_configured"]:
        readiness_counts.setdefault(status, 0)
    readiness_counts["evaluable"] = sum(
        1 for item in capabilities if item["workflow_level"] == "runnable_and_evaluable"
    )

    runnable_models = sum(
        1
        for item in capabilities
        if item["workflow_level"] in {"runnable_and_evaluable", "registered_ready"}
    )
    demo_workflows = sum(
        1 for item in capabilities if item["workflow_level"] in {"demo", "contract_demo"}
    )
    planned_workflows = readiness_counts["planned"] + readiness_counts["not_configured"]

    return {
        "system": "AlphaEarth System",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "readiness_counts": readiness_counts,
        "summary": {
            "runnable_models": runnable_models,
            "demo_workflows": demo_workflows,
            "planned_workflows": planned_workflows,
            "arcgis_replacement_ready": False,
        },
        "capabilities": capabilities,
        "evidence_sources": _evidence_sources(),
        "notes": [
            "Capability status is derived from the local registry, runtime declarations, and committed evidence files.",
            "Paper 12 artifacts are supporting evidence only; system readiness is the primary product surface.",
            "ArcGIS replacement readiness is conservative and does not imply .dlpk compatibility.",
        ],
    }
```

- [ ] **Step 2: Create the system API router**

Create `ae_backend/app/api/system.py` with:

```python
from __future__ import annotations

from fastapi import APIRouter

from app.api.model_hub import get_model_registry
from app.services.system_capabilities import build_system_capabilities


router = APIRouter()


@router.get("/capabilities")
def get_system_capabilities():
    return build_system_capabilities(get_model_registry())
```

- [ ] **Step 3: Mount the router**

Modify `ae_backend/app/main.py`.

Change the import line from:

```python
from app.api import pipeline, training, satellites, areas, models, results, inference, model_hub
```

to:

```python
from app.api import pipeline, training, satellites, areas, models, results, inference, model_hub, system
```

Add this router block after the model-hub router block:

```python
app.include_router(
    system.router,
    prefix=f"{settings.API_V1_STR}/system",
    tags=["system"]
)
```

- [ ] **Step 4: Run backend tests**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_capabilities_endpoint_reports_operational_readiness tests/test_model_hub_api.py::test_system_capabilities_tolerates_missing_optional_evidence -q
```

Expected: both tests pass.

- [ ] **Step 5: Run model-hub compatibility tests**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_model_hub_returns_paper12_summary tests/test_model_hub_api.py::test_model_hub_lists_phase1_models -q
```

Expected: both tests pass; the legacy Paper 12 summary endpoint remains available.

- [ ] **Step 6: Commit backend implementation**

Run:

```powershell
git add ae_backend/app/services/system_capabilities.py ae_backend/app/api/system.py ae_backend/app/main.py
git commit -m "feat: add system capability API"
```

Expected: commit contains only the new service, new router, and main router mount.

---

### Task 3: Frontend Contract Tests

**Files:**
- Modify: `tests/test_model_hub_frontend_entry.py`
- Do not modify `ae_frontend/index.html` in this task.

- [ ] **Step 1: Replace Paper 12-first frontend test**

Replace `test_frontend_exposes_paper12_summary_panel` in `tests/test_model_hub_frontend_entry.py` with:

```python
def test_frontend_exposes_system_capability_workbench():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "systemCapabilities" in html
    assert "fetchSystemCapabilities" in html
    assert "/api/ae/system/capabilities" in html
    assert "绯荤粺鑳藉姏宸ヤ綔鍙? in html
    assert "AlphaEarth System" in html
    assert "readiness_counts" in html
    assert "evidence_sources" in html
    assert "arcgis_replacement" in html
    assert "Paper12 鑳藉姏鎬昏" not in html
```

- [ ] **Step 2: Add capability-card frontend hooks test**

Append this test after the workbench test:

```python
def test_frontend_exposes_system_capability_card_hooks():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "systemCapabilityFor" in html
    assert "capabilityWorkflowLabel" in html
    assert "formatEvidenceValue" in html
    assert "systemCapabilityFor(model).workflow_level" in html
    assert "systemCapabilityFor(model).checkpoint" in html
    assert "systemCapabilityFor(model).limitations" in html
    assert "systemCapabilityFor(model).next_steps" in html
```

- [ ] **Step 3: Run frontend tests and verify expected failure**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py::test_frontend_exposes_system_capability_workbench tests/test_model_hub_frontend_entry.py::test_frontend_exposes_system_capability_card_hooks -q
```

Expected: both tests fail because the frontend still exposes `paper12Summary` as the primary panel.

- [ ] **Step 4: Commit frontend tests**

Run:

```powershell
git add tests/test_model_hub_frontend_entry.py
git commit -m "test: cover system capability frontend"
```

Expected: commit contains only `tests/test_model_hub_frontend_entry.py`.

---

### Task 4: Frontend System Workbench Integration

**Files:**
- Modify: `ae_frontend/index.html`
- Test: `tests/test_model_hub_frontend_entry.py`

- [ ] **Step 1: Add frontend state**

In the Vue setup section, replace:

```javascript
const paper12Summary = ref(null);
const loadingPaper12Summary = ref(false);
```

with:

```javascript
const systemCapabilities = ref(null);
const loadingSystemCapabilities = ref(false);
const systemCapabilitiesError = ref(null);
const paper12Summary = ref(null);
const loadingPaper12Summary = ref(false);
```

- [ ] **Step 2: Add capability helper functions**

Replace the existing `paper12CapabilityFor` function block with:

```javascript
const systemCapabilityFor = (model) => {
    const capabilities = systemCapabilities.value?.capabilities || [];
    return capabilities.find(item => item.id === model?.model_id) || null;
};

const paper12CapabilityFor = (model) => {
    const capabilities = paper12Summary.value?.capabilities || [];
    return capabilities.find(item => item.model_id === model?.model_id) || null;
};

const capabilityWorkflowLabel = (level) => {
    const labels = {
        runnable_and_evaluable: '鍙繍琛?/ 鍙瘎浼?,
        registered_ready: '宸叉敞鍐屽彲鐢?,
        contract_demo: '濂戠害婕旂ず',
        demo: '婕旂ず宸ヤ綔娴?,
        planned: '璁″垝涓?,
        not_configured: '鏈厤缃?,
    };
    return labels[level] || level || '鏈煡';
};

const formatEvidenceValue = (value) => {
    if (value === null || value === undefined) return 'n/a';
    if (typeof value === 'number') return formatMetric(value);
    return String(value);
};
```

- [ ] **Step 3: Add system capability fetch method**

Insert this method before `fetchPaper12Summary`:

```javascript
const fetchSystemCapabilities = async () => {
    if (loadingSystemCapabilities.value) return;
    loadingSystemCapabilities.value = true;
    systemCapabilitiesError.value = null;
    try {
        const res = await fetch('/api/ae/system/capabilities');
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
        systemCapabilities.value = await res.json();
    } catch (e) {
        console.error('Failed to load AlphaEarth System capabilities', e);
        systemCapabilitiesError.value = e?.message || 'unknown error';
        setModelHubStatus('error', '鍔犺浇绯荤粺鑳藉姏宸ヤ綔鍙板け璐ワ紝璇风‘璁ゅ悗绔?/api/ae/system/capabilities 鍙闂€?);
    } finally {
        loadingSystemCapabilities.value = false;
    }
};
```

- [ ] **Step 4: Update model-hub fetch flow**

In `fetchModelHubModels`, replace:

```javascript
if (!paper12Summary.value) await fetchPaper12Summary();
```

with:

```javascript
if (!systemCapabilities.value) await fetchSystemCapabilities();
```

In the `watch(currentTab, ...)` `modelHub` branch, replace:

```javascript
if (!paper12Summary.value) fetchPaper12Summary();
```

with:

```javascript
if (!systemCapabilities.value) fetchSystemCapabilities();
```

In `onMounted`, keep `fetchModelHubModels();`. That call will load system capabilities through `fetchModelHubModels`.

- [ ] **Step 5: Replace the Paper 12-first summary panel**

Replace the `<section class="glass-card p-5 space-y-4">` that starts with:

```html
<h3 class="text-lg font-semibold text-gray-900">Paper12 鑳藉姏鎬昏</h3>
```

and ends just before:

```html
<div v-if="loadingModelHubModels" class="p-12 text-center text-gray-500">
```

with:

```html
<section class="glass-card p-5 space-y-4">
    <div class="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
            <h3 class="text-lg font-semibold text-gray-900">绯荤粺鑳藉姏宸ヤ綔鍙?/h3>
            <p class="text-sm text-gray-500 mt-1">AlphaEarth System 褰撳墠鍙繍琛屻€佸彲璇勪及銆佹紨绀哄拰璁″垝涓殑閬ユ劅鑳藉姏鎬昏銆?/p>
        </div>
        <button type="button" class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded-md text-sm hover:bg-gray-50 transition-colors cursor-pointer disabled:opacity-50" :disabled="loadingSystemCapabilities" @click="fetchSystemCapabilities">
            {{ loadingSystemCapabilities ? '鍒锋柊涓?..' : '鍒锋柊绯荤粺鑳藉姏' }}
        </button>
    </div>
    <div v-if="systemCapabilities" class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div class="border border-gray-200 rounded-md p-3 bg-gray-50">
            <div class="text-xs text-gray-500 font-semibold uppercase tracking-wider">AlphaEarth System</div>
            <div class="mt-2 text-sm text-gray-700">
                鍙繍琛?{{ systemCapabilities.summary.runnable_models }} 涓紝婕旂ず {{ systemCapabilities.summary.demo_workflows }} 涓紝璁″垝 {{ systemCapabilities.summary.planned_workflows }} 涓?            </div>
            <div class="mt-2 flex flex-wrap gap-2 text-xs">
                <span v-for="(count, status) in systemCapabilities.readiness_counts" :key="status" class="px-2 py-1 rounded border" :class="statusClass(status)">{{ status }}: {{ count }}</span>
            </div>
        </div>
        <div class="border border-gray-200 rounded-md p-3 bg-gray-50">
            <div class="text-xs text-gray-500 font-semibold uppercase tracking-wider">鏇夸唬杈圭晫</div>
            <p class="mt-2 text-sm text-gray-700">
                ArcGIS/Prithvi 鏇夸唬灏辩华锛歿{ systemCapabilities.summary.arcgis_replacement_ready ? '鏄? : '鍚? }}
            </p>
            <p class="mt-1 text-xs text-gray-500">鏈湴 crop 鑳藉姏浠嶆寜濂戠害婕旂ず澶勭悊锛屾湭澹版槑鍙浛浠?ArcGIS 棰勮缁冩ā鍨嬨€?/p>
        </div>
        <div class="border border-gray-200 rounded-md p-3 bg-gray-50">
            <div class="text-xs text-gray-500 font-semibold uppercase tracking-wider">璇佹嵁鏉ユ簮</div>
            <ul class="mt-2 space-y-1 text-xs text-gray-600">
                <li v-for="source in systemCapabilities.evidence_sources" :key="source.path" class="break-all">
                    <span :class="source.available ? 'text-green-700' : 'text-amber-700'">{{ source.available ? 'available' : 'missing' }}</span>
                    路 {{ source.label }} 路 {{ source.path }}
                </li>
            </ul>
        </div>
    </div>
    <div v-if="systemCapabilities && systemCapabilities.notes && systemCapabilities.notes.length" class="text-xs text-gray-500 space-y-1">
        <div v-for="note in systemCapabilities.notes" :key="note">{{ note }}</div>
    </div>
    <div v-else-if="loadingSystemCapabilities" class="text-sm text-gray-500">姝ｅ湪鍔犺浇绯荤粺鑳藉姏宸ヤ綔鍙?..</div>
    <div v-else-if="systemCapabilitiesError" class="text-sm text-red-600">绯荤粺鑳藉姏鍔犺浇澶辫触锛歿{ systemCapabilitiesError }}</div>
    <div v-else class="text-sm text-gray-500">绯荤粺鑳藉姏宸ヤ綔鍙板皻鏈姞杞姐€?/div>
</section>
```

- [ ] **Step 6: Update the model card status and evidence**

In the model-card status chip block, replace:

```html
<span v-if="paper12CapabilityFor(model)" class="px-2 py-1 rounded border bg-gray-50 text-gray-700 border-gray-200">
    {{ paper12CapabilityFor(model).arcgis_replacement_status }}
</span>
```

with:

```html
<span v-if="systemCapabilityFor(model)" class="px-2 py-1 rounded border bg-gray-50 text-gray-700 border-gray-200">
    {{ capabilityWorkflowLabel(systemCapabilityFor(model).workflow_level) }}
</span>
<span v-if="systemCapabilityFor(model)?.arcgis_replacement" class="px-2 py-1 rounded border bg-gray-50 text-gray-700 border-gray-200">
    {{ systemCapabilityFor(model).arcgis_replacement.status }}
</span>
```

Replace the existing Paper 12 capability note block:

```html
<div v-if="paper12CapabilityFor(model)" class="text-xs text-gray-500 bg-gray-50 border border-gray-200 rounded p-2">
    <div>{{ paper12CapabilityFor(model).reason }}</div>
    <div class="mt-1 text-gray-600">涓嬩竴姝ワ細{{ paper12CapabilityFor(model).next_step }}</div>
</div>
```

with:

```html
<div v-if="systemCapabilityFor(model)" class="text-xs text-gray-500 bg-gray-50 border border-gray-200 rounded p-2 space-y-2">
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <div><span class="text-gray-500">宸ヤ綔娴?</span> <span class="text-gray-800">{{ capabilityWorkflowLabel(systemCapabilityFor(model).workflow_level) }}</span></div>
        <div><span class="text-gray-500">Checkpoint:</span> <span class="text-gray-800">{{ systemCapabilityFor(model).checkpoint.configured ? systemCapabilityFor(model).checkpoint.path : '鏈厤缃? }}</span></div>
    </div>
    <div v-if="systemCapabilityFor(model).evidence && systemCapabilityFor(model).evidence.length">
        <div class="text-gray-600 font-medium mb-1">璇佹嵁</div>
        <div class="flex flex-wrap gap-2">
            <span v-for="evidence in systemCapabilityFor(model).evidence" :key="evidence.kind + evidence.label" class="px-2 py-1 rounded border bg-white border-gray-200">
                {{ evidence.label }}: {{ formatEvidenceValue(evidence.value) }}
            </span>
        </div>
    </div>
    <div v-if="systemCapabilityFor(model).limitations && systemCapabilityFor(model).limitations.length">
        <div class="text-gray-600 font-medium mb-1">闄愬埗</div>
        <ul class="list-disc pl-4 space-y-1">
            <li v-for="limitation in systemCapabilityFor(model).limitations" :key="limitation">{{ limitation }}</li>
        </ul>
    </div>
    <div v-if="systemCapabilityFor(model).next_steps && systemCapabilityFor(model).next_steps.length">
        <div class="text-gray-600 font-medium mb-1">涓嬩竴姝?/div>
        <ul class="list-disc pl-4 space-y-1">
            <li v-for="step in systemCapabilityFor(model).next_steps" :key="step">{{ step }}</li>
        </ul>
    </div>
</div>
```

- [ ] **Step 7: Update returned Vue bindings**

In the `return { ... }` block, replace:

```javascript
paper12Summary, loadingPaper12Summary, fetchPaper12Summary,
```

with:

```javascript
systemCapabilities, loadingSystemCapabilities, systemCapabilitiesError, fetchSystemCapabilities,
paper12Summary, loadingPaper12Summary, fetchPaper12Summary,
```

Replace:

```javascript
formatMetric, statusClass, paper12CapabilityFor, summarizeModelHubJob,
```

with:

```javascript
formatMetric, statusClass, systemCapabilityFor, paper12CapabilityFor, capabilityWorkflowLabel, formatEvidenceValue, summarizeModelHubJob,
```

- [ ] **Step 8: Run frontend contract tests**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py -q
```

Expected: all frontend model-hub contract tests pass.

- [ ] **Step 9: Commit frontend implementation**

Run:

```powershell
git add ae_frontend/index.html
git commit -m "feat: add system capability workbench UI"
```

Expected: commit contains only `ae_frontend/index.html`.

---

### Task 5: Focused Verification

**Files:**
- No code edits expected.

- [ ] **Step 1: Run backend and frontend focused suite**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py tests/test_model_hub_registry.py tests/test_model_hub_frontend_entry.py tests/test_lulc_frontend_entry.py tests/test_inference_api.py tests/test_inference_service.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 3: Inspect final diff scope**

Run:

```powershell
git status --short
git diff --stat HEAD
```

Expected:

- System work is committed in the planned test/implementation commits.
- Existing unrelated Paper 12 dirty files may still appear in `git status --short`; do not revert them.
- No uncommitted changes remain in system files unless they are intentional and reviewed.

- [ ] **Step 4: Start local server for user verification**

Run:

```powershell
python -m uvicorn app.main:app --app-dir ae_backend --host 127.0.0.1 --port 61529
```

Expected:

- Server starts and serves the app at `http://127.0.0.1:61529/`.
- If port `61529` is occupied, use `61530` and report the actual URL.
- Do not leave multiple duplicate server processes running.

---

## Self-Review

Spec coverage:

- System-first endpoint: Task 1 and Task 2.
- Readiness and workflow levels: Task 1 and Task 2.
- Conservative ArcGIS replacement status: Task 1 and Task 2.
- Evidence sources with missing-file tolerance: Task 1 and Task 2.
- Frontend system workbench replacing Paper 12-first summary: Task 3 and Task 4.
- Existing model-hub/LULC workflow preservation: Task 4 and Task 5.
- Focused verification and local server handoff: Task 5.

Placeholder scan:

- No `TBD`, `TODO`, `implement later`, or placeholder steps are used.
- All code-changing steps include concrete code blocks or exact replacement text.

Type consistency:

- Backend response uses `id` for system capability records.
- Frontend `systemCapabilityFor(model)` looks up `item.id === model.model_id`.
- Frontend tests assert `systemCapabilityFor(model).workflow_level`, `systemCapabilityFor(model).checkpoint`, `systemCapabilityFor(model).limitations`, and `systemCapabilityFor(model).next_steps`, matching the Task 4 template aliases.
- Legacy Paper 12 summary uses `model_id` and remains supported through `paper12CapabilityFor`.

