# Paper 12 Model-Hub UI Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Paper 12 capability summary API and enhance the existing AlphaEarth System model-center UI so users can inspect model readiness and run supported Paper 12 / Prithvi demos.

**Architecture:** Keep the current FastAPI model-hub router and static Vue 3 frontend. Add one lightweight backend summary service that reads committed JSON result files and registry metadata, then expose it through `/api/ae/model-hub/paper12-summary`; enhance `ae_frontend/index.html` with a summary band, clearer model cards, and job result summaries.

**Tech Stack:** Python, FastAPI, pytest, static Vue 3 in `ae_frontend/index.html`, Tailwind utility classes already loaded by the page.

---

## File Map

- Create `ae_backend/app/services/paper12_summary.py`
  - Pure-Python summary builder.
  - Reads `paper12_results/*.json` and `model_hub_models.json` through the existing registry object.
  - Does not import PyTorch, rasterio, or ML code.

- Modify `ae_backend/app/api/model_hub.py`
  - Add `GET /paper12-summary`.
  - Reuse `get_model_registry()` and the new service.

- Modify `tests/test_model_hub_api.py`
  - Add API-level tests for summary endpoint and missing optional file handling.

- Modify `tests/test_model_hub_frontend_entry.py`
  - Add frontend contract checks for the summary UI and result-summary UI.

- Modify `ae_frontend/index.html`
  - Add Paper 12 summary state and API load method.
  - Add summary band and clearer model cards in the existing `modelHub` tab.
  - Add structured job result summary above raw JSON.

---

### Task 1: Add Failing API Tests for Paper 12 Summary

**Files:**
- Modify: `tests/test_model_hub_api.py`

- [ ] **Step 1: Add the missing-file helper and endpoint tests**

Append these tests near the existing model-hub tests in `tests/test_model_hub_api.py`:

```python
def test_model_hub_returns_paper12_summary():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/model-hub/paper12-summary")

    assert response.status_code == 200
    body = response.json()
    assert body["paper"] == "paper12"
    assert body["readiness_counts"]["ready"] >= 1
    assert body["readiness_counts"]["demo_only"] >= 1
    assert body["readiness_counts"]["planned"] >= 1

    benchmarks = {item["id"]: item for item in body["benchmarks"]}
    assert benchmarks["eurosat_channel_bridge"]["best_method"] == "learned_bridge_houlsby"
    assert benchmarks["eurosat_channel_bridge"]["metric"] == "overall_accuracy"
    assert benchmarks["eurosat_channel_bridge"]["best_value"] > 0.9
    assert benchmarks["landcoverai_segmentation"]["best_method"] == "houlsby"
    assert benchmarks["landcoverai_segmentation"]["metric"] == "mIoU"
    assert benchmarks["landcoverai_segmentation"]["best_value"] > 0.64

    crop = {
        item["model_id"]: item
        for item in body["capabilities"]
    }["prithvi_crop_classification_arcgis_style"]
    assert crop["readiness"] == "demo_only"
    assert crop["arcgis_replacement_status"] == "not_yet"
    assert "No validated crop checkpoint" in crop["reason"]
```

Add this second test to verify graceful degradation. It monkeypatches the service root after the service exists in Task 2:

```python
def test_model_hub_paper12_summary_reports_missing_optional_results(monkeypatch, tmp_path: Path):
    from app.main import app
    import app.services.paper12_summary as paper12_summary

    monkeypatch.setattr(paper12_summary, "PAPER12_RESULTS_DIR", tmp_path)

    client = TestClient(app)
    response = client.get("/api/ae/model-hub/paper12-summary")

    assert response.status_code == 200
    body = response.json()
    missing = [item for item in body["benchmarks"] if item.get("status") == "missing"]
    assert missing
    assert any("missing" in item["note"].lower() for item in missing)
```

- [ ] **Step 2: Run tests and verify they fail for missing route/module**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_model_hub_returns_paper12_summary tests/test_model_hub_api.py::test_model_hub_paper12_summary_reports_missing_optional_results -q
```

Expected: FAIL because `/api/ae/model-hub/paper12-summary` does not exist or `app.services.paper12_summary` cannot be imported.

- [ ] **Step 3: Commit the failing tests only**

```powershell
git add tests/test_model_hub_api.py
git commit -m "test: cover paper12 model hub summary api"
```

---

### Task 2: Implement Paper 12 Summary Service and Route

**Files:**
- Create: `ae_backend/app/services/paper12_summary.py`
- Modify: `ae_backend/app/api/model_hub.py`
- Test: `tests/test_model_hub_api.py`

- [ ] **Step 1: Create `paper12_summary.py`**

Create `ae_backend/app/services/paper12_summary.py` with this implementation:

```python
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from app.core.config import PROJECT_ROOT
from app.services.model_hub_registry import ModelHubRegistry


PAPER12_RESULTS_DIR = Path(PROJECT_ROOT) / "paper12_results"


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _missing_benchmark(benchmark_id: str, label: str, source: str) -> dict[str, Any]:
    return {
        "id": benchmark_id,
        "label": label,
        "metric": None,
        "best_method": None,
        "best_value": None,
        "source": source,
        "status": "missing",
        "note": f"Missing local result file: {source}",
    }


def _best_from_summary(
    *,
    benchmark_id: str,
    label: str,
    source_name: str,
    metric_field: str,
    display_metric: str,
) -> dict[str, Any]:
    path = PAPER12_RESULTS_DIR / source_name
    data = _load_json(path)
    if not isinstance(data, dict):
        return _missing_benchmark(benchmark_id, label, f"paper12_results/{source_name}")

    candidates = []
    for method, values in data.items():
        if isinstance(values, dict) and metric_field in values:
            candidates.append((method, float(values[metric_field])))

    if not candidates:
        return {
            "id": benchmark_id,
            "label": label,
            "metric": display_metric,
            "best_method": None,
            "best_value": None,
            "source": f"paper12_results/{source_name}",
            "status": "missing",
            "note": f"No metric field {metric_field!r} found in {source_name}",
        }

    best_method, best_value = max(candidates, key=lambda item: item[1])
    return {
        "id": benchmark_id,
        "label": label,
        "metric": display_metric,
        "best_method": best_method,
        "best_value": best_value,
        "source": f"paper12_results/{source_name}",
        "status": "available",
    }


def _landcoverai_segmentation() -> dict[str, Any]:
    source_name = "landcoverai_segmentation.json"
    path = PAPER12_RESULTS_DIR / source_name
    rows = _load_json(path)
    if not isinstance(rows, list):
        return _missing_benchmark(
            "landcoverai_segmentation",
            "LandCover.ai semantic segmentation",
            f"paper12_results/{source_name}",
        )

    by_method: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if isinstance(row, dict) and "method" in row and "mIoU" in row:
            by_method[str(row["method"])].append(float(row["mIoU"]))

    if not by_method:
        return {
            "id": "landcoverai_segmentation",
            "label": "LandCover.ai semantic segmentation",
            "metric": "mIoU",
            "best_method": None,
            "best_value": None,
            "source": f"paper12_results/{source_name}",
            "status": "missing",
            "note": "No mIoU rows found in landcoverai_segmentation.json",
        }

    means = {method: mean(values) for method, values in by_method.items()}
    best_method = max(means, key=means.get)
    return {
        "id": "landcoverai_segmentation",
        "label": "LandCover.ai semantic segmentation",
        "metric": "mIoU",
        "best_method": best_method,
        "best_value": means[best_method],
        "source": f"paper12_results/{source_name}",
        "status": "available",
    }


def _loveda_full_finetune() -> dict[str, Any]:
    source_name = "loveda_full_finetune_summary.json"
    data = _load_json(PAPER12_RESULTS_DIR / source_name)
    if not isinstance(data, dict):
        return _missing_benchmark(
            "loveda_full_finetune",
            "LoveDA full fine-tuning baseline",
            f"paper12_results/{source_name}",
        )

    candidates = []
    for direction, values in data.items():
        if isinstance(values, dict) and "mIoU_mean" in values:
            candidates.append((str(direction), float(values["mIoU_mean"])))

    if not candidates:
        return {
            "id": "loveda_full_finetune",
            "label": "LoveDA full fine-tuning baseline",
            "metric": "mIoU",
            "best_method": None,
            "best_value": None,
            "source": f"paper12_results/{source_name}",
            "status": "missing",
            "note": "No mIoU_mean values found in loveda_full_finetune_summary.json",
        }

    best_method, best_value = max(candidates, key=lambda item: item[1])
    return {
        "id": "loveda_full_finetune",
        "label": "LoveDA full fine-tuning baseline",
        "metric": "mIoU",
        "best_method": best_method,
        "best_value": best_value,
        "source": f"paper12_results/{source_name}",
        "status": "available",
    }


def _arcgis_replacement_status(model: dict[str, Any]) -> tuple[str, str, str]:
    model_id = model["model_id"]
    status = model["status"]

    if model_id == "prithvi_crop_classification_arcgis_style":
        return (
            "not_yet",
            "No validated crop checkpoint is configured; current runtime is a deterministic contract demo.",
            "Attach a validated Prithvi crop head and HLS preprocessing pipeline.",
        )
    if model_id == "water_flood_prithvi":
        return (
            "not_yet",
            "Flood segmentation is registered as planned and has no configured checkpoint.",
            "Attach a validated flood segmentation checkpoint and 6-band HLS/Sen1Floods11 preprocessing.",
        )
    if status == "ready":
        return (
            "partial",
            "Local model is runnable in AlphaEarth System but is not packaged as an ArcGIS .dlpk.",
            "Validate against target production geographies and package deployment artifacts if ArcGIS compatibility is required.",
        )
    if status == "demo_only":
        return (
            "demo_only",
            "Capability is available only as a cached or deterministic demo.",
            "Replace the demo runtime with a validated checkpoint-backed inference path.",
        )
    return (
        "planned",
        "Capability is registered but not executable in the local model hub.",
        "Configure checkpoint, preprocessing, runtime, and validation metrics.",
    )


def build_paper12_summary(registry: ModelHubRegistry) -> dict[str, Any]:
    models = [model.to_dict() for model in registry.models]
    readiness_counts = dict(Counter(model["status"] for model in models))
    for status in ["ready", "demo_only", "planned", "not_configured"]:
        readiness_counts.setdefault(status, 0)

    benchmarks = [
        _best_from_summary(
            benchmark_id="eurosat_channel_bridge",
            label="EuroSAT channel bridge",
            source_name="eurosat_channel_bridge_summary.json",
            metric_field="overall_accuracy_mean",
            display_metric="overall_accuracy",
        ),
        _best_from_summary(
            benchmark_id="peft_capacity_sweep",
            label="EuroSAT PEFT capacity sweep",
            source_name="peft_capacity_sweep_summary.json",
            metric_field="overall_accuracy_mean",
            display_metric="overall_accuracy",
        ),
        _landcoverai_segmentation(),
        _loveda_full_finetune(),
    ]

    capabilities = []
    for model in models:
        replacement_status, reason, next_step = _arcgis_replacement_status(model)
        capabilities.append(
            {
                "model_id": model["model_id"],
                "display_name": model["display_name"],
                "task_type": model["task_type"],
                "readiness": model["status"],
                "arcgis_replacement_status": replacement_status,
                "reason": reason,
                "next_step": next_step,
                "runtime_modes": model.get("package_profile", {}).get("runtime_modes", []),
            }
        )

    return {
        "paper": "paper12",
        "readiness_counts": readiness_counts,
        "benchmarks": benchmarks,
        "capabilities": capabilities,
        "notes": [
            "Paper 12 results are local benchmark evidence for AlphaEarth System model-hub integration.",
            "ArcGIS replacement status is conservative and does not imply .dlpk compatibility.",
        ],
    }
```

- [ ] **Step 2: Add the API route**

In `ae_backend/app/api/model_hub.py`, add this import:

```python
from app.services.paper12_summary import build_paper12_summary
```

Add this route after `list_models()`:

```python
@router.get("/paper12-summary")
def get_paper12_summary():
    return build_paper12_summary(get_model_registry())
```

Keep `/models/{model_id}` below this route so `paper12-summary` is not interpreted as a model id.

- [ ] **Step 3: Run the focused API tests**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_model_hub_returns_paper12_summary tests/test_model_hub_api.py::test_model_hub_paper12_summary_reports_missing_optional_results -q
```

Expected: PASS.

- [ ] **Step 4: Commit backend implementation**

```powershell
git add ae_backend/app/services/paper12_summary.py ae_backend/app/api/model_hub.py
git commit -m "feat: add paper12 model hub summary api"
```

---

### Task 3: Add Failing Frontend Contract Tests

**Files:**
- Modify: `tests/test_model_hub_frontend_entry.py`

- [ ] **Step 1: Add frontend contract tests**

Append these tests to `tests/test_model_hub_frontend_entry.py`:

```python
def test_frontend_exposes_paper12_summary_panel():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "paper12Summary" in html
    assert "fetchPaper12Summary" in html
    assert "/api/ae/model-hub/paper12-summary" in html
    assert "Paper12 能力总览" in html
    assert "readiness_counts" in html
    assert "benchmarks" in html
    assert "arcgis_replacement_status" in html


def test_frontend_exposes_model_hub_job_summary_sections():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "summarizeModelHubJob" in html
    assert "任务摘要" in html
    assert "输出制品" in html
    assert "运行日志" in html
    assert "原始 JSON" in html
    assert "modelHubJob.artifacts" in html
    assert "modelHubJob.logs" in html
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py::test_frontend_exposes_paper12_summary_panel tests/test_model_hub_frontend_entry.py::test_frontend_exposes_model_hub_job_summary_sections -q
```

Expected: FAIL because the frontend has not yet added these strings or methods.

- [ ] **Step 3: Commit the failing frontend tests**

```powershell
git add tests/test_model_hub_frontend_entry.py
git commit -m "test: cover paper12 model hub frontend"
```

---

### Task 4: Enhance Existing Model-Hub UI

**Files:**
- Modify: `ae_frontend/index.html`
- Test: `tests/test_model_hub_frontend_entry.py`

- [ ] **Step 1: Add Paper 12 state variables**

Inside the Vue `setup()` block near existing model-hub refs, add:

```javascript
const paper12Summary = ref(null);
const loadingPaper12Summary = ref(false);
```

- [ ] **Step 2: Add helpers and fetch method**

Add these functions near the existing model-hub functions:

```javascript
const formatMetric = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
    const numeric = Number(value);
    if (Math.abs(numeric) >= 1) return numeric.toFixed(3);
    return numeric.toFixed(4);
};

const statusClass = (status) => {
    if (status === 'ready' || status === 'succeeded') return 'bg-green-50 text-green-700 border-green-200';
    if (status === 'demo_only' || status === 'running') return 'bg-blue-50 text-blue-700 border-blue-200';
    if (status === 'planned' || status === 'pending') return 'bg-amber-50 text-amber-700 border-amber-200';
    if (status === 'failed') return 'bg-red-50 text-red-700 border-red-200';
    return 'bg-gray-50 text-gray-700 border-gray-200';
};

const fetchPaper12Summary = async () => {
    loadingPaper12Summary.value = true;
    try {
        const res = await fetch('/api/ae/model-hub/paper12-summary');
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
        paper12Summary.value = await res.json();
    } catch (e) {
        console.error('Failed to load Paper12 model hub summary', e);
        setModelHubStatus('error', '加载 Paper12 能力摘要失败，请确认后端 /api/ae/model-hub/paper12-summary 可访问。');
    } finally {
        loadingPaper12Summary.value = false;
    }
};

const summarizeModelHubJob = (job) => {
    if (!job || !job.result) return [];
    const summary = job.result.summary || {};
    const rows = [];
    if (job.result.task) rows.push(['任务', job.result.task]);
    if (job.result.model_id) rows.push(['模型', job.result.model_id]);
    if (job.result.input_mode) rows.push(['输入模式', job.result.input_mode]);
    if (summary.dominant_class) rows.push(['主导类别', summary.dominant_class]);
    if (summary.tile_count !== undefined) rows.push(['切片数', summary.tile_count]);
    if (summary.mask_shape) rows.push(['掩膜尺寸', Array.isArray(summary.mask_shape) ? summary.mask_shape.join(' x ') : summary.mask_shape]);
    if (summary.class_area_fraction) {
        const topClass = Object.entries(summary.class_area_fraction)
            .sort((a, b) => Number(b[1]) - Number(a[1]))[0];
        if (topClass) rows.push(['最大面积占比', `${topClass[0]} ${formatMetric(topClass[1])}`]);
    }
    return rows;
};
```

- [ ] **Step 3: Update model-hub fetch flow**

In `fetchModelHubModels`, after setting `modelHubModels.value`, call:

```javascript
if (!paper12Summary.value) await fetchPaper12Summary();
```

In the tab watcher for `modelHub`, ensure both loaders run:

```javascript
if (modelHubModels.value.length === 0) fetchModelHubModels();
if (!paper12Summary.value) fetchPaper12Summary();
```

- [ ] **Step 4: Add Paper 12 summary band markup**

Inside `<div v-show="currentTab === 'modelHub'" ...>`, add this block above the model cards:

```html
<div class="glass-card p-5">
    <div class="flex items-start justify-between gap-4">
        <div>
            <h3 class="text-lg font-semibold text-gray-900">Paper12 能力总览</h3>
            <p class="text-sm text-gray-500 mt-1">展示本地 Paper12/Prithvi 能力证据、模型就绪状态与 ArcGIS 替代边界。</p>
        </div>
        <button type="button" class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded-md text-sm hover:bg-gray-50 transition-colors cursor-pointer disabled:opacity-50" :disabled="loadingPaper12Summary" @click="fetchPaper12Summary">
            {{ loadingPaper12Summary ? '刷新中...' : '刷新摘要' }}
        </button>
    </div>
    <div v-if="paper12Summary" class="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div class="border border-gray-200 rounded-md p-3 bg-gray-50">
            <div class="text-xs text-gray-500 font-semibold uppercase tracking-wider">模型状态</div>
            <div class="mt-2 flex flex-wrap gap-2 text-xs">
                <span v-for="(count, status) in paper12Summary.readiness_counts" :key="status" class="px-2 py-1 rounded border" :class="statusClass(status)">{{ status }}: {{ count }}</span>
            </div>
        </div>
        <div class="border border-gray-200 rounded-md p-3 bg-gray-50 lg:col-span-2">
            <div class="text-xs text-gray-500 font-semibold uppercase tracking-wider">Paper12 Benchmarks</div>
            <div class="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2">
                <div v-for="benchmark in paper12Summary.benchmarks" :key="benchmark.id" class="bg-white border border-gray-200 rounded p-2">
                    <div class="text-sm font-medium text-gray-800">{{ benchmark.label }}</div>
                    <div class="text-xs text-gray-500 mt-1">
                        <span v-if="benchmark.status === 'available'">{{ benchmark.best_method }} · {{ benchmark.metric }} {{ formatMetric(benchmark.best_value) }}</span>
                        <span v-else>{{ benchmark.note }}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div v-else class="mt-4 text-sm text-gray-500">Paper12 能力摘要尚未加载。</div>
</div>
```

- [ ] **Step 5: Improve model cards**

Inside each model card, add status and replacement chips:

```html
<div class="flex flex-wrap gap-2 text-xs">
    <span class="px-2 py-1 rounded border" :class="statusClass(model.status)">{{ model.status }}</span>
    <span v-if="paper12Summary" class="px-2 py-1 rounded border bg-gray-50 text-gray-700 border-gray-200">
        {{ (paper12Summary.capabilities || []).find(item => item.model_id === model.model_id)?.arcgis_replacement_status || 'unmapped' }}
    </span>
</div>
```

Add limitation text below input/output metadata:

```html
<div v-if="paper12Summary && (paper12Summary.capabilities || []).find(item => item.model_id === model.model_id)" class="text-xs text-gray-500 bg-gray-50 border border-gray-200 rounded p-2">
    {{ (paper12Summary.capabilities || []).find(item => item.model_id === model.model_id).reason }}
</div>
```

- [ ] **Step 6: Replace raw-only job display**

In the `modelHubJob` panel, add these sections before the raw JSON:

```html
<div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
    <div class="border border-gray-200 rounded-md p-3 bg-gray-50 lg:col-span-2">
        <h4 class="text-sm font-semibold text-gray-700 mb-2">任务摘要</h4>
        <dl class="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
            <div v-for="row in summarizeModelHubJob(modelHubJob)" :key="row[0]" class="flex justify-between gap-3 border-b border-gray-200 pb-1">
                <dt class="text-gray-500">{{ row[0] }}</dt>
                <dd class="text-gray-900 font-medium text-right break-all">{{ row[1] }}</dd>
            </div>
        </dl>
        <p v-if="summarizeModelHubJob(modelHubJob).length === 0" class="text-sm text-gray-500">暂无结构化结果摘要。</p>
    </div>
    <div class="border border-gray-200 rounded-md p-3 bg-gray-50">
        <h4 class="text-sm font-semibold text-gray-700 mb-2">输出制品</h4>
        <ul v-if="modelHubJob.artifacts && modelHubJob.artifacts.length" class="space-y-1 text-xs">
            <li v-for="artifact in modelHubJob.artifacts" :key="artifact.kind + artifact.path" class="break-all">
                <span class="font-medium text-gray-700">{{ artifact.kind }}</span>
                <span class="text-gray-500"> · {{ artifact.path }}</span>
            </li>
        </ul>
        <p v-else class="text-sm text-gray-500">暂无输出制品。</p>
    </div>
</div>
<div class="border border-gray-200 rounded-md p-3 bg-gray-50 mb-4">
    <h4 class="text-sm font-semibold text-gray-700 mb-2">运行日志</h4>
    <pre class="text-xs bg-gray-900 text-green-300 rounded p-3 overflow-auto max-h-40 data-font">{{ (modelHubJob.logs || []).join('\n') || '暂无日志' }}</pre>
</div>
<details open>
    <summary class="cursor-pointer text-sm font-semibold text-gray-700 mb-2">原始 JSON</summary>
    <pre class="text-xs bg-gray-900 text-green-300 rounded p-3 overflow-auto max-h-80 data-font">{{ JSON.stringify(modelHubJob, null, 2) }}</pre>
</details>
```

- [ ] **Step 7: Return new state and helpers**

In the `return { ... }` object, add:

```javascript
paper12Summary, loadingPaper12Summary, fetchPaper12Summary,
formatMetric, statusClass, summarizeModelHubJob,
```

- [ ] **Step 8: Run frontend contract tests**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit frontend implementation**

```powershell
git add ae_frontend/index.html
git commit -m "feat: enhance paper12 model hub ui"
```

---

### Task 5: Final Verification

**Files:**
- No code edits expected.

- [ ] **Step 1: Run model hub focused suite**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py tests/test_model_hub_registry.py tests/test_model_hub_frontend_entry.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run Paper 12 result contract suite**

Run:

```powershell
python -m pytest tests/test_paper12_public_dataset_results.py -q
```

Expected: pass, confirming result files used by the summary still match manuscript contracts.

- [ ] **Step 3: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 4: Inspect final diff**

Run:

```powershell
git status --short
git diff --stat HEAD~4..HEAD
```

Expected: only intended tests, backend summary API, frontend UI, and plan/spec commits are present.

---

## Self-Review

- Spec coverage: Tasks cover backend summary endpoint, frontend summary band, readiness/status display, job controls preservation, job result summary, tests, and verification.
- Scope control: The plan does not add real crop/flood/burn/weather checkpoints, `.dlpk` compatibility, new frontend framework, manuscript edits, or training changes.
- Type consistency: `paper12Summary`, `fetchPaper12Summary`, `statusClass`, `formatMetric`, and `summarizeModelHubJob` are introduced in Task 4 and returned from Vue setup before use in markup.
- TDD sequence: Tasks 1 and 3 add failing tests before implementation; Tasks 2 and 4 implement the minimal backend/frontend code to satisfy them.
