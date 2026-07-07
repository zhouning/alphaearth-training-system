# System Evidence Drill-down Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, read-only evidence drill-down API and Model Hub UI so users can inspect the artifacts behind AlphaEarth System verification checks.

**Architecture:** Keep evidence inspection as a focused backend service beside `system_verification.py`, using verification checks as the source of truth. The service resolves registry refs and local artifact refs under a small allowlist, returns bounded previews for safe text-like files, and the existing Model Hub workbench renders the result without running inference or editing Paper12 outputs.

**Tech Stack:** FastAPI, pytest, Vue 3 in the existing single-file `ae_frontend/index.html`, local filesystem metadata, JSON/CSV/text parsing from the Python standard library, no model loading and no network access.

---

## Scope

This plan implements only the evidence drill-down slice approved in `docs/superpowers/specs/2026-07-07-system-evidence-drilldown-design.md`.

Do not edit Paper12 manuscript, submission, PDF, audit, or generated result files while executing this plan. The current worktree contains unrelated Paper12 changes; leave them unstaged and untouched.

## File Structure

- Create `ae_backend/app/services/system_evidence.py`
  - Calls `system_verification.build_system_verification(registry)`.
  - Resolves each check's `evidence_refs`.
  - Classifies refs as registry, available local artifact, missing local artifact, blocked unsafe ref, or not applicable.
  - Returns file metadata and small previews for JSON, CSV, text, Markdown, and log files.
  - Blocks absolute paths outside allowed roots and `..` traversal, and never exposes full absolute paths.
- Modify `ae_backend/app/api/system.py`
  - Adds `GET /api/ae/system/evidence`.
  - Reuses `get_model_registry()` like `/capabilities` and `/verification`.
- Modify `tests/test_model_hub_api.py`
  - Adds endpoint shape, missing evidence, registry ref, unsafe ref, and preview tests.
- Modify `tests/test_model_hub_frontend_entry.py`
  - Adds stable ASCII anchor tests for evidence state, fetch hooks, per-check evidence helpers, and preview formatting.
- Modify `ae_frontend/index.html`
  - Adds evidence state and fetch method.
  - Extends the existing System Capability Workbench inside the Model Hub tab.
  - Renders a compact Evidence drill-down summary and per-check evidence rows.
  - Keeps existing model job controls, capability summary, and verification summary intact.

## Task 1: Backend Evidence API Tests

**Files:**
- Modify: `tests/test_model_hub_api.py`
- Later create: `ae_backend/app/services/system_evidence.py`
- Later modify: `ae_backend/app/api/system.py`

- [ ] **Step 1: Add failing endpoint shape test**

Append this test near the existing system verification tests in `tests/test_model_hub_api.py`:

```python
def test_system_evidence_endpoint_reports_drilldown_payload():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/system/evidence")

    assert response.status_code == 200
    body = response.json()
    assert body["system"] == "AlphaEarth System"
    assert set(body) >= {"generated_at", "summary", "checks", "artifacts", "notes"}
    assert set(body["summary"]) == {"available", "missing", "previewable", "blocked"}
    assert body["checks"]
    assert body["artifacts"]
    assert body["notes"]

    evidence_items = [
        artifact
        for check in body["checks"]
        for artifact in check["evidence"]
    ]
    assert evidence_items
    assert any(item["status"] == "available" for item in evidence_items)
    assert any(item["ref"] == "model_hub_registry" and item["kind"] == "registry" for item in evidence_items)
    assert all(not Path(item.get("safe_path", "relative")).is_absolute() for item in evidence_items)
```

- [ ] **Step 2: Run endpoint shape test and verify it fails**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_evidence_endpoint_reports_drilldown_payload -q
```

Expected: FAIL with HTTP 404 for `/api/ae/system/evidence` or an import error for the missing evidence service.

- [ ] **Step 3: Add failing missing optional evidence test**

Append this test to `tests/test_model_hub_api.py`:

```python
def test_system_evidence_reports_missing_optional_artifacts(
    monkeypatch,
    tmp_path: Path,
):
    from app.main import app
    import app.services.system_capabilities as system_capabilities

    monkeypatch.setattr(system_capabilities, "PAPER12_RESULTS_DIR", tmp_path)

    client = TestClient(app)
    response = client.get("/api/ae/system/evidence")

    assert response.status_code == 200
    body = response.json()
    missing = [
        artifact
        for artifact in body["artifacts"]
        if artifact["status"] == "missing"
    ]
    assert missing
    assert body["summary"]["missing"] >= len(missing)
    assert all(item["safe_path"].startswith("paper12_results/") for item in missing)
    assert all("optional" in item["message"].lower() or "referenced" in item["message"].lower() for item in missing)
```

- [ ] **Step 4: Run missing optional evidence test and verify it fails**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_evidence_reports_missing_optional_artifacts -q
```

Expected: FAIL because the evidence endpoint does not exist yet.

- [ ] **Step 5: Add failing unsafe ref test**

Append this test to `tests/test_model_hub_api.py`:

```python
def test_system_evidence_blocks_unsafe_refs_without_leaking_absolute_paths(
    monkeypatch,
):
    import json

    from app.main import app
    import app.services.system_evidence as system_evidence

    absolute_ref = str(repo_root / "pyproject.toml")

    def fake_build_system_verification(registry):
        return {
            "system": "AlphaEarth System",
            "generated_at": "2026-07-07T00:00:00+00:00",
            "checks": [
                {
                    "id": "system:unsafe_ref",
                    "capability_id": "system",
                    "status": "warning",
                    "title": "Unsafe evidence ref",
                    "evidence_refs": ["../outside.txt", absolute_ref],
                    "remediation": "Remove unsafe evidence refs.",
                }
            ],
        }

    monkeypatch.setattr(
        system_evidence.system_verification,
        "build_system_verification",
        fake_build_system_verification,
    )

    client = TestClient(app)
    response = client.get("/api/ae/system/evidence")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["blocked"] == 2
    blocked = [artifact for artifact in body["artifacts"] if artifact["status"] == "blocked"]
    assert len(blocked) == 2
    assert all("safe_path" not in item for item in blocked)
    payload = json.dumps(body)
    assert str(repo_root) not in payload
    assert absolute_ref not in payload
```

- [ ] **Step 6: Run unsafe ref test and verify it fails**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_evidence_blocks_unsafe_refs_without_leaking_absolute_paths -q
```

Expected: FAIL because `app.services.system_evidence` is missing.

- [ ] **Step 7: Add failing bounded preview test**

Append this test to `tests/test_model_hub_api.py`:

```python
def test_system_evidence_previews_json_and_csv_from_allowed_roots(
    monkeypatch,
    tmp_path: Path,
):
    from app.main import app
    import app.services.system_evidence as system_evidence

    evidence_dir = tmp_path / "paper12_results"
    evidence_dir.mkdir()
    (evidence_dir / "preview.json").write_text(
        '{"metrics":{"overall_accuracy":0.91},"items":[1,2,3]}',
        encoding="utf-8",
    )
    (evidence_dir / "table.csv").write_text(
        "class,pixels,fraction\ncorn,64,0.64\nsoybean,36,0.36\n",
        encoding="utf-8",
    )

    def fake_build_system_verification(registry):
        return {
            "system": "AlphaEarth System",
            "generated_at": "2026-07-07T00:00:00+00:00",
            "checks": [
                {
                    "id": "system:preview_refs",
                    "capability_id": "system",
                    "status": "pass",
                    "title": "Preview refs",
                    "evidence_refs": [
                        "paper12_results/preview.json",
                        "paper12_results/table.csv",
                    ],
                    "remediation": None,
                }
            ],
        }

    monkeypatch.setattr(system_evidence, "PROJECT_ROOT_PATH", tmp_path.resolve())
    monkeypatch.setattr(
        system_evidence.system_verification,
        "build_system_verification",
        fake_build_system_verification,
    )

    client = TestClient(app)
    response = client.get("/api/ae/system/evidence")

    assert response.status_code == 200
    body = response.json()
    by_ref = {artifact["ref"]: artifact for artifact in body["artifacts"]}
    json_artifact = by_ref["paper12_results/preview.json"]
    csv_artifact = by_ref["paper12_results/table.csv"]

    assert json_artifact["previewable"] is True
    assert json_artifact["preview"]["type"] == "json"
    assert json_artifact["preview"]["content"]["metrics"]["overall_accuracy"] == 0.91
    assert csv_artifact["previewable"] is True
    assert csv_artifact["preview"]["type"] == "csv"
    assert csv_artifact["preview"]["header"] == ["class", "pixels", "fraction"]
    assert csv_artifact["preview"]["rows"][0] == ["corn", "64", "0.64"]
```

- [ ] **Step 8: Run bounded preview test and verify it fails**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_evidence_previews_json_and_csv_from_allowed_roots -q
```

Expected: FAIL because the evidence service is missing.

- [ ] **Step 9: Commit failing backend tests**

Stage only the API test file:

```powershell
git add tests/test_model_hub_api.py
git commit -m "test: cover system evidence drilldown API"
```

## Task 2: Backend Evidence Service and Route

**Files:**
- Create: `ae_backend/app/services/system_evidence.py`
- Modify: `ae_backend/app/api/system.py`
- Test: `tests/test_model_hub_api.py`

- [ ] **Step 1: Create evidence service module**

Create `ae_backend/app/services/system_evidence.py` with this content:

```python
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.services import system_verification
from app.services.model_hub_registry import ModelHubRegistry


PROJECT_ROOT_PATH = Path(PROJECT_ROOT).resolve()
ALLOWED_LOCAL_ROOTS = ("paper12_results", "results", "linhe_results")
ALLOWED_REGISTRY_REFS = {
    "model_hub_registry": "Model Hub registry",
    "system_capabilities": "System capability service",
}
PREVIEW_LIMIT_BYTES = 256 * 1024
JSON_ITEM_LIMIT = 20
JSON_DEPTH_LIMIT = 4
TEXT_LINE_LIMIT = 16
CSV_ROW_LIMIT = 8

JSON_EXTENSIONS = {".json", ".geojson"}
CSV_EXTENSIONS = {".csv"}
TEXT_EXTENSIONS = {".txt", ".md", ".log"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
TENSOR_EXTENSIONS = {".npz", ".pt", ".pth"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_posix(ref: str) -> str:
    return ref.strip().replace("\\", "/")


def _kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in JSON_EXTENSIONS:
        return "json"
    if suffix in CSV_EXTENSIONS:
        return "csv"
    if suffix == ".md":
        return "markdown"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in TENSOR_EXTENSIONS:
        return "tensor"
    return "binary"


def _safe_ref_for_output(ref: str) -> str:
    normalized = _to_posix(str(ref))
    if Path(normalized).is_absolute():
        return "<absolute path blocked>"
    return normalized


def _resolve_local_ref(ref: str) -> tuple[Path | None, str | None, str | None]:
    normalized = _to_posix(str(ref))
    if not normalized:
        return None, None, "Evidence ref is empty."
    if ".." in Path(normalized).parts:
        return None, None, "Evidence ref uses parent-directory traversal."

    root = PROJECT_ROOT_PATH.resolve()
    candidate = Path(normalized)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()

    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        return None, None, "Evidence ref is outside the project root."

    if not any(relative == allowed or relative.startswith(f"{allowed}/") for allowed in ALLOWED_LOCAL_ROOTS):
        return None, None, "Evidence ref is outside allowed evidence roots."
    return resolved, relative, None


def _compact_json(value: Any, *, depth: int = 0) -> Any:
    if depth >= JSON_DEPTH_LIMIT:
        return "<truncated>"
    if isinstance(value, dict):
        items = list(value.items())
        compact = {
            str(key): _compact_json(item, depth=depth + 1)
            for key, item in items[:JSON_ITEM_LIMIT]
        }
        if len(items) > JSON_ITEM_LIMIT:
            compact["__truncated__"] = True
        return compact
    if isinstance(value, list):
        compact_list = [_compact_json(item, depth=depth + 1) for item in value[:JSON_ITEM_LIMIT]]
        if len(value) > JSON_ITEM_LIMIT:
            compact_list.append({"__truncated__": True})
        return compact_list
    if isinstance(value, str) and len(value) > 240:
        return f"{value[:240]}..."
    return value


def _json_preview(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "type": "json",
        "truncated": False,
        "content": _compact_json(payload),
    }


def _csv_preview(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    rows = list(csv.reader(text.splitlines()))
    header = rows[0] if rows else []
    data_rows = rows[1 : CSV_ROW_LIMIT + 1]
    return {
        "type": "csv",
        "truncated": len(rows) > CSV_ROW_LIMIT + 1,
        "header": header,
        "rows": data_rows,
    }


def _text_preview(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "type": "text",
        "truncated": len(lines) > TEXT_LINE_LIMIT,
        "lines": lines[:TEXT_LINE_LIMIT],
    }


def _preview_for_path(path: Path, kind: str, size_bytes: int) -> tuple[bool, dict[str, Any] | None, str | None]:
    if size_bytes > PREVIEW_LIMIT_BYTES:
        return False, None, "Evidence artifact is available; inline preview skipped because the file is larger than 256 KB."
    try:
        if kind == "json":
            return True, _json_preview(path), None
        if kind == "csv":
            return True, _csv_preview(path), None
        if kind in {"text", "markdown"}:
            return True, _text_preview(path), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
        return False, None, f"Evidence artifact is available, but preview parsing failed: {exc.__class__.__name__}."
    return False, None, None


def _registry_artifact(ref: str) -> dict[str, Any]:
    return {
        "ref": ref,
        "kind": "registry",
        "status": "available",
        "source_id": ref,
        "label": ALLOWED_REGISTRY_REFS[ref],
        "previewable": False,
        "message": "Registry evidence source is recognized.",
    }


def _blocked_artifact(ref: str, message: str) -> dict[str, Any]:
    return {
        "ref": _safe_ref_for_output(ref),
        "kind": "blocked",
        "status": "blocked",
        "previewable": False,
        "message": message,
    }


def _local_artifact(ref: str) -> dict[str, Any]:
    path, safe_path, blocked_message = _resolve_local_ref(ref)
    if blocked_message or path is None or safe_path is None:
        return _blocked_artifact(ref, blocked_message or "Evidence ref cannot be resolved safely.")

    kind = _kind_for_path(path)
    if not path.exists():
        return {
            "ref": _to_posix(str(ref)),
            "kind": kind,
            "status": "missing",
            "safe_path": safe_path,
            "previewable": False,
            "message": "Referenced optional evidence artifact is missing.",
        }

    stat = path.stat()
    previewable, preview, preview_message = _preview_for_path(path, kind, stat.st_size)
    message = preview_message or "Evidence artifact is available."
    payload = {
        "ref": _to_posix(str(ref)),
        "kind": kind,
        "status": "available",
        "safe_path": safe_path,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
        "previewable": previewable,
        "message": message,
    }
    if preview is not None:
        payload["preview"] = preview
    return payload


def _artifact_for_ref(ref: str) -> dict[str, Any]:
    normalized = _to_posix(str(ref))
    if normalized in ALLOWED_REGISTRY_REFS:
        return _registry_artifact(normalized)
    if normalized:
        return _local_artifact(normalized)
    return {
        "ref": "",
        "kind": "not_applicable",
        "status": "not_applicable",
        "previewable": False,
        "message": "Evidence ref is empty.",
    }


def _artifact_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    summary_keys = [
        "ref",
        "kind",
        "status",
        "safe_path",
        "source_id",
        "label",
        "size_bytes",
        "modified_at",
        "previewable",
        "message",
        "preview",
    ]
    return {key: artifact[key] for key in summary_keys if key in artifact}


def build_system_evidence(registry: ModelHubRegistry) -> dict[str, Any]:
    verification = system_verification.build_system_verification(registry)
    checks: list[dict[str, Any]] = []
    artifacts_by_ref: dict[str, dict[str, Any]] = {}

    for check in verification.get("checks", []):
        evidence = []
        for ref in check.get("evidence_refs", []):
            artifact = _artifact_for_ref(str(ref))
            evidence.append(artifact)
            artifacts_by_ref.setdefault(artifact["ref"], _artifact_summary(artifact))
        checks.append(
            {
                "check_id": check["id"],
                "capability_id": check.get("capability_id"),
                "check_status": check.get("status"),
                "check_title": check.get("title"),
                "remediation": check.get("remediation"),
                "evidence": evidence,
            }
        )

    artifacts = list(artifacts_by_ref.values())
    status_counts = Counter(artifact["status"] for artifact in artifacts)

    return {
        "system": "AlphaEarth System",
        "generated_at": _utc_now(),
        "summary": {
            "available": status_counts.get("available", 0),
            "missing": status_counts.get("missing", 0),
            "previewable": sum(1 for artifact in artifacts if artifact.get("previewable")),
            "blocked": status_counts.get("blocked", 0),
        },
        "checks": checks,
        "artifacts": artifacts,
        "notes": [
            "Evidence drill-down reads local metadata and small previews only.",
            "It does not execute model inference, train models, or regenerate results.",
            "ArcGIS replacement readiness remains conservative until checkpoint-backed validation evidence is attached.",
        ],
    }
```

- [ ] **Step 2: Add route**

Modify `ae_backend/app/api/system.py` so imports include `build_system_evidence`:

```python
from app.services.system_capabilities import build_system_capabilities
from app.services.system_evidence import build_system_evidence
from app.services.system_verification import build_system_verification
```

Add this endpoint below `get_system_verification()`:

```python
@router.get("/evidence")
def get_system_evidence():
    return build_system_evidence(get_model_registry())
```

- [ ] **Step 3: Run backend evidence tests**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_evidence_endpoint_reports_drilldown_payload tests/test_model_hub_api.py::test_system_evidence_reports_missing_optional_artifacts tests/test_model_hub_api.py::test_system_evidence_blocks_unsafe_refs_without_leaking_absolute_paths tests/test_model_hub_api.py::test_system_evidence_previews_json_and_csv_from_allowed_roots -q
```

Expected: PASS.

- [ ] **Step 4: Run existing system API tests**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_system_capabilities_endpoint_reports_operational_readiness tests/test_model_hub_api.py::test_system_capabilities_tolerates_missing_optional_evidence tests/test_model_hub_api.py::test_system_verification_endpoint_reports_contract_checks tests/test_model_hub_api.py::test_system_verification_does_not_fail_planned_models_for_missing_checkpoints tests/test_model_hub_api.py::test_system_verification_reports_missing_optional_evidence_as_warnings -q
```

Expected: PASS.

- [ ] **Step 5: Commit backend implementation**

Stage only relevant files:

```powershell
git add ae_backend/app/services/system_evidence.py ae_backend/app/api/system.py
git commit -m "feat: add system evidence drilldown API"
```

## Task 3: Frontend Evidence Contract Tests

**Files:**
- Modify: `tests/test_model_hub_frontend_entry.py`
- Later modify: `ae_frontend/index.html`

- [ ] **Step 1: Add failing frontend evidence anchor test**

Append this test to `tests/test_model_hub_frontend_entry.py`:

```python
def test_frontend_exposes_system_evidence_drilldown_hooks():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "/api/ae/system/evidence" in html
    assert "systemEvidence" in html
    assert "loadingSystemEvidence" in html
    assert "systemEvidenceError" in html
    assert "fetchSystemEvidence" in html
    assert "Evidence drill-down" in html
    assert "evidenceForCheck" in html
    assert "evidenceStatusClass" in html
    assert "formatArtifactSize" in html
    assert "formatArtifactPreview" in html
    assert "preview" in html
```

- [ ] **Step 2: Add regression test for existing workbench controls**

Append this test to `tests/test_model_hub_frontend_entry.py`:

```python
def test_frontend_keeps_model_hub_controls_with_system_evidence():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "runModelHubDemo" in html
    assert "runModelHubRasterDemo" in html
    assert "fetchSystemCapabilities" in html
    assert "fetchSystemVerification" in html
    assert "fetchSystemEvidence" in html
    assert "systemVerificationRawJson" in html
    assert "modelHubJob.artifacts" in html
    assert "modelHubJob.logs" in html
```

- [ ] **Step 3: Run frontend evidence tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py::test_frontend_exposes_system_evidence_drilldown_hooks tests/test_model_hub_frontend_entry.py::test_frontend_keeps_model_hub_controls_with_system_evidence -q
```

Expected: first test FAILS because evidence hooks are not in `index.html`; second may fail until `fetchSystemEvidence` is added.

- [ ] **Step 4: Commit failing frontend tests**

Stage only the frontend test file:

```powershell
git add tests/test_model_hub_frontend_entry.py
git commit -m "test: cover system evidence frontend"
```

## Task 4: Frontend Evidence Drill-down UI

**Files:**
- Modify: `ae_frontend/index.html`
- Test: `tests/test_model_hub_frontend_entry.py`

- [ ] **Step 1: Add evidence state**

In the Vue setup block near the existing `systemCapabilities` and `systemVerification` refs, add:

```javascript
const systemEvidence = ref(null);
const loadingSystemEvidence = ref(false);
const systemEvidenceError = ref(null);
```

- [ ] **Step 2: Add evidence helper functions**

Near `verificationForCapability`, `verificationStatusClass`, and `systemVerificationRawJson`, add:

```javascript
const evidenceForCheck = (checkId) => {
    const checks = systemEvidence.value?.checks || [];
    return checks.find(item => item.check_id === checkId) || null;
};

const evidenceStatusClass = (status) => {
    if (status === 'available') return 'bg-emerald-50 text-emerald-700 border-emerald-200';
    if (status === 'missing') return 'bg-amber-50 text-amber-700 border-amber-200';
    if (status === 'blocked') return 'bg-red-50 text-red-700 border-red-200';
    if (status === 'not_applicable') return 'bg-gray-50 text-gray-600 border-gray-200';
    return 'bg-gray-50 text-gray-700 border-gray-200';
};

const formatArtifactSize = (bytes) => {
    if (bytes === undefined || bytes === null) return 'metadata only';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const formatArtifactPreview = (artifact) => {
    if (!artifact?.preview) return artifact?.message || 'No inline preview';
    if (artifact.preview.type === 'json') return JSON.stringify(artifact.preview.content, null, 2);
    if (artifact.preview.type === 'csv') {
        const rows = [artifact.preview.header, ...(artifact.preview.rows || [])]
            .filter(row => row && row.length)
            .map(row => row.join(', '));
        return rows.join('\n');
    }
    if (artifact.preview.type === 'text') return (artifact.preview.lines || []).join('\n');
    return JSON.stringify(artifact.preview, null, 2);
};

const systemEvidenceRawJson = computed(() => {
    if (!systemEvidence.value) return '';
    return JSON.stringify(systemEvidence.value, null, 2);
});
```

- [ ] **Step 3: Add evidence fetch method**

Near `fetchSystemVerification`, add:

```javascript
const fetchSystemEvidence = async () => {
    loadingSystemEvidence.value = true;
    systemEvidenceError.value = null;
    try {
        const res = await fetch(`${API_BASE}/api/ae/system/evidence`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        systemEvidence.value = await res.json();
    } catch (e) {
        systemEvidenceError.value = e?.message || 'unknown error';
    } finally {
        loadingSystemEvidence.value = false;
    }
};
```

- [ ] **Step 4: Fetch evidence with the Model Hub workbench**

In `fetchModelHubModels`, after the existing `fetchSystemVerification()` call, add:

```javascript
if (!systemEvidence.value) await fetchSystemEvidence();
```

In the tab watcher for `modelHub`, after the existing `fetchSystemVerification()` call, add:

```javascript
if (!systemEvidence.value) fetchSystemEvidence();
```

- [ ] **Step 5: Add Evidence drill-down summary block**

Inside the existing System Capability Workbench area, below the System verification summary and before per-model cards, add:

```html
<div class="glass-card p-4 space-y-3">
    <div class="flex flex-wrap items-center justify-between gap-3">
        <div>
            <div class="text-sm font-semibold text-gray-800">Evidence drill-down</div>
            <div class="text-xs text-gray-500">Inspect verification evidence refs, safe local artifacts, and bounded previews.</div>
        </div>
        <button type="button" class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded-md text-sm hover:bg-gray-50 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed" :disabled="loadingSystemEvidence" @click="fetchSystemEvidence">
            {{ loadingSystemEvidence ? 'Loading evidence...' : 'Refresh evidence' }}
        </button>
    </div>
    <div v-if="systemEvidence" class="space-y-3">
        <div class="flex flex-wrap items-center gap-2 text-xs">
            <span v-for="(count, status) in systemEvidence.summary" :key="status" class="px-2 py-1 rounded border" :class="evidenceStatusClass(status)">
                {{ status }}: {{ count }}
            </span>
        </div>
        <div v-if="systemEvidence.notes && systemEvidence.notes.length" class="text-xs text-gray-500 space-y-1">
            <div v-for="note in systemEvidence.notes" :key="note">{{ note }}</div>
        </div>
        <details class="text-xs">
            <summary class="cursor-pointer text-gray-600">Raw evidence JSON</summary>
            <pre class="mt-2 max-h-64 overflow-auto bg-gray-900 text-gray-100 p-3 rounded text-[11px] leading-relaxed">{{ systemEvidenceRawJson }}</pre>
        </details>
    </div>
    <div v-else-if="systemEvidenceError" class="text-sm text-red-600">System evidence failed: {{ systemEvidenceError }}</div>
</div>
```

- [ ] **Step 6: Add per-check evidence rows in each model card**

In the existing `v-if="verificationForCapability(model)"` block that renders check ids, replace the current check-id-only row with this structure:

```html
<div class="space-y-2">
    <div v-for="checkId in verificationForCapability(model).checks" :key="checkId" class="border border-gray-200 rounded p-2 bg-gray-50 space-y-2">
        <div class="flex flex-wrap items-center gap-2">
            <span class="px-2 py-1 rounded border bg-white border-gray-200 break-all">{{ checkId }}</span>
            <span v-if="evidenceForCheck(checkId)" class="px-2 py-1 rounded border" :class="verificationStatusClass(evidenceForCheck(checkId).check_status)">
                {{ evidenceForCheck(checkId).check_status }}
            </span>
        </div>
        <div v-if="evidenceForCheck(checkId)" class="space-y-2">
            <div v-for="artifact in evidenceForCheck(checkId).evidence" :key="artifact.ref" class="bg-white border border-gray-200 rounded p-2 space-y-1">
                <div class="flex flex-wrap items-center gap-2">
                    <span class="px-2 py-1 rounded border" :class="evidenceStatusClass(artifact.status)">{{ artifact.status }}</span>
                    <span class="text-gray-700">{{ artifact.kind }}</span>
                    <span class="text-gray-500 break-all">{{ artifact.safe_path || artifact.source_id || artifact.ref }}</span>
                    <span class="text-gray-400">{{ formatArtifactSize(artifact.size_bytes) }}</span>
                </div>
                <div class="text-gray-500">{{ artifact.message }}</div>
                <div v-if="artifact.modified_at" class="text-gray-400">modified_at: {{ artifact.modified_at }}</div>
                <details v-if="artifact.preview" class="text-xs">
                    <summary class="cursor-pointer text-gray-600">preview</summary>
                    <pre class="mt-2 max-h-48 overflow-auto bg-gray-900 text-gray-100 p-2 rounded text-[11px] leading-relaxed">{{ formatArtifactPreview(artifact) }}</pre>
                </details>
            </div>
        </div>
        <div v-else class="text-xs text-gray-400">No drill-down evidence loaded for this check.</div>
    </div>
</div>
```

- [ ] **Step 7: Return new state and helpers from setup**

In the setup return object, include:

```javascript
systemEvidence, loadingSystemEvidence, systemEvidenceError, fetchSystemEvidence,
evidenceForCheck, evidenceStatusClass, formatArtifactSize, formatArtifactPreview, systemEvidenceRawJson,
```

- [ ] **Step 8: Run frontend evidence tests**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py::test_frontend_exposes_system_evidence_drilldown_hooks tests/test_model_hub_frontend_entry.py::test_frontend_keeps_model_hub_controls_with_system_evidence -q
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
git commit -m "feat: add system evidence drilldown UI"
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

If the current-code dev server is running on `http://127.0.0.1:61531/`, test the endpoint with proxy bypass:

```powershell
curl.exe --noproxy "*" -s http://127.0.0.1:61531/api/ae/system/evidence
```

Expected: JSON containing `"system":"AlphaEarth System"` and `"summary"`.

If no current-code server is running, skip this smoke test and rely on `TestClient` coverage.

- [ ] **Step 5: Commit verification-only adjustments if required**

Only if Task 5 reveals a necessary test-only adjustment, stage that exact file and commit:

```powershell
git add tests/test_model_hub_api.py tests/test_model_hub_frontend_entry.py
git commit -m "test: stabilize system evidence drilldown checks"
```

If no adjustment is needed, do not create an empty commit.

## Self-Review

- Spec coverage:
  - `GET /api/ae/system/evidence`: Task 2.
  - Response fields `system`, `generated_at`, `summary`, `checks`, `artifacts`, and `notes`: Task 1 and Task 2.
  - Check-level drill-down with `check_id`, status, title, remediation, and evidence artifacts: Task 2 and Task 4.
  - Registry refs `model_hub_registry` and `system_capabilities` are recognized and not treated as missing files: Task 1 and Task 2.
  - Local allowlist roots `paper12_results/`, `results/`, and `linhe_results/`: Task 2.
  - Absolute path and parent traversal blocking with no absolute path leakage: Task 1 and Task 2.
  - JSON and CSV previews are structured and bounded: Task 1 and Task 2.
  - Text, Markdown, and log previews are bounded: Task 2.
  - Large and binary artifacts return metadata without inline preview: Task 2.
  - Frontend summary, refresh action, per-check rows, artifact metadata, and preview details: Task 3 and Task 4.
  - Existing Model Hub job controls are preserved: Task 3 and Task 4.
  - ArcGIS replacement readiness is not overclaimed: Task 2 notes and Task 4 display existing verification boundaries.
- Placeholder scan:
  - No placeholder markers or open-ended error handling instructions are present.
  - Each code-changing step includes concrete snippets and exact commands.
- Type consistency:
  - Backend returns `summary.available`, `summary.missing`, `summary.previewable`, and `summary.blocked`.
  - Backend check records use `check_id`, `capability_id`, `check_status`, `check_title`, `remediation`, and `evidence`.
  - Backend artifacts use `ref`, `kind`, `status`, `safe_path`, `source_id`, `size_bytes`, `modified_at`, `previewable`, `preview`, and `message`.
  - Frontend helpers and tests use the same keys: `systemEvidence`, `evidenceForCheck`, `evidenceStatusClass`, `formatArtifactSize`, `formatArtifactPreview`, and `systemEvidenceRawJson`.
