# Linhe Patch Labeling Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal in-app labeling workbench so the user can browse `linhe_patches`, assign one of seven land-cover labels, and persist labels to `data/linhe_patches/_labels.parquet` for later supervised training.

**Architecture:** Keep the scope narrow and local to the existing AlphaEarth-System. Add one new backend router for Linhe patch labeling, store labels in a parquet file adjacent to the dataset, and add one new frontend tab in `ae_frontend/index.html` that calls those APIs. Do not add polygon drawing, masks, collaboration, or training-time supervision in this plan.

**Tech Stack:** FastAPI, pandas, parquet, base64 PNG serving, Vue 3 (inline in `index.html`), pytest

---

## File Structure

- Create: `ae_backend/app/api/labels.py`
  - New API router for patch listing, image serving, label saving, and progress stats.
- Modify: `ae_backend/app/main.py`
  - Register the new labels router under `/api/ae/labels`.
- Modify: `ae_backend/app/services/trainer.py`
  - Reuse existing dataset resolution helper if needed, but no training changes in this plan.
- Modify: `ae_frontend/index.html`
  - Add a new `Linhe 标注` tab, local state, fetch methods, and category buttons.
- Create: `tests/test_labels_api.py`
  - Focused backend tests for listing, saving, and stats behavior.
- Modify: `docs/linhe_dataset_integration.md`
  - Document the new labeling workbench and `_labels.parquet` output.

---

### Task 1: Add backend helper functions for reading Linhe patch index and labels

**Files:**
- Create: `ae_backend/app/api/labels.py`
- Test: `tests/test_labels_api.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import pandas as pd

from app.api.labels import load_patch_index, load_labels


def test_load_patch_index_and_labels(tmp_path):
    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)

    pd.DataFrame([
        {"patch_path": "linhe_patches/scene_a/p_00000_00000.npz", "dataset_id": "linhe_patches"}
    ]).to_parquet(dataset_dir / "_index.parquet")

    pd.DataFrame([
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/p_00000_00000.npz", "label": "耕地", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:00"}
    ]).to_parquet(dataset_dir / "_labels.parquet")

    index_df = load_patch_index(dataset_dir)
    labels_df = load_labels(dataset_dir)

    assert len(index_df) == 1
    assert len(labels_df) == 1
    assert labels_df.iloc[0]["label"] == "耕地"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_labels_api.py::test_load_patch_index_and_labels -v`
Expected: FAIL with import error because `app.api.labels` or helper functions do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# ae_backend/app/api/labels.py
from pathlib import Path
import pandas as pd


LABEL_COLUMNS = ["dataset_id", "patch_path", "label", "reviewer", "labeled_at"]


def load_patch_index(dataset_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(dataset_dir / "_index.parquet")


def load_labels(dataset_dir: Path) -> pd.DataFrame:
    labels_path = dataset_dir / "_labels.parquet"
    if not labels_path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    return pd.read_parquet(labels_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_labels_api.py::test_load_patch_index_and_labels -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ae_backend/app/api/labels.py tests/test_labels_api.py
git commit -m "feat: add linhe label file helpers"
```

---

### Task 2: Add backend API for next unlabeled patch and progress stats

**Files:**
- Modify: `ae_backend/app/api/labels.py`
- Modify: `ae_backend/app/main.py`
- Test: `tests/test_labels_api.py`

- [ ] **Step 1: Write the failing tests**

```python
from fastapi.testclient import TestClient
from app.main import app


def test_labels_next_returns_first_unlabeled_patch(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame([
        {"patch_path": "linhe_patches/scene_a/p_00000_00000.npz"},
        {"patch_path": "linhe_patches/scene_a/p_00000_00128.npz"},
    ]).to_parquet(dataset_dir / "_index.parquet")
    pd.DataFrame([
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/p_00000_00000.npz", "label": "耕地", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:00"}
    ]).to_parquet(dataset_dir / "_labels.parquet")

    from app.core.config import settings
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/api/ae/labels/next?dataset_id=linhe_patches")
    data = resp.json()["data"]

    assert resp.status_code == 200
    assert data["patch_path"] == "linhe_patches/scene_a/p_00000_00128.npz"


def test_labels_stats_returns_counts(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame([
        {"patch_path": "linhe_patches/scene_a/a.npz"},
        {"patch_path": "linhe_patches/scene_a/b.npz"},
    ]).to_parquet(dataset_dir / "_index.parquet")
    pd.DataFrame([
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/a.npz", "label": "林地", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:00"}
    ]).to_parquet(dataset_dir / "_labels.parquet")

    from app.core.config import settings
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/api/ae/labels/stats?dataset_id=linhe_patches")
    data = resp.json()["data"]

    assert data["total"] == 2
    assert data["labeled"] == 1
    assert data["unlabeled"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_labels_api.py::test_labels_next_returns_first_unlabeled_patch tests/test_labels_api.py::test_labels_stats_returns_counts -v`
Expected: FAIL because router and endpoints are not wired yet.

- [ ] **Step 3: Write minimal implementation**

```python
# ae_backend/app/api/labels.py
from fastapi import APIRouter, HTTPException, Query
from app.core.config import settings

router = APIRouter()


def get_dataset_dir(dataset_id: str) -> Path:
    if dataset_id != "linhe_patches":
        raise HTTPException(status_code=400, detail="Only linhe_patches is supported in v1")
    return Path(settings.DATA_DIR) / dataset_id


@router.get("/next")
def get_next_patch(dataset_id: str = Query(...)):
    dataset_dir = get_dataset_dir(dataset_id)
    index_df = load_patch_index(dataset_dir)
    labels_df = load_labels(dataset_dir)
    labeled_paths = set(labels_df["patch_path"].tolist())
    unlabeled = index_df[~index_df["patch_path"].isin(labeled_paths)]
    if unlabeled.empty:
        return {"status": "success", "data": None}
    row = unlabeled.iloc[0]
    return {"status": "success", "data": {"dataset_id": dataset_id, "patch_path": row["patch_path"]}}


@router.get("/stats")
def get_label_stats(dataset_id: str = Query(...)):
    dataset_dir = get_dataset_dir(dataset_id)
    index_df = load_patch_index(dataset_dir)
    labels_df = load_labels(dataset_dir)
    return {
        "status": "success",
        "data": {
            "dataset_id": dataset_id,
            "total": int(len(index_df)),
            "labeled": int(len(labels_df)),
            "unlabeled": int(len(index_df) - len(labels_df)),
        },
    }
```

```python
# ae_backend/app/main.py
from app.api import pipeline, training, satellites, areas, models, labels

app.include_router(
    labels.router,
    prefix=f"{settings.API_V1_STR}/labels",
    tags=["labels"],
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_labels_api.py::test_labels_next_returns_first_unlabeled_patch tests/test_labels_api.py::test_labels_stats_returns_counts -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ae_backend/app/api/labels.py ae_backend/app/main.py tests/test_labels_api.py
git commit -m "feat: add label browsing endpoints"
```

---

### Task 3: Add backend API for patch image preview and label saving

**Files:**
- Modify: `ae_backend/app/api/labels.py`
- Test: `tests/test_labels_api.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from fastapi.testclient import TestClient
from app.main import app


def test_labels_preview_returns_base64_png(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "linhe_patches"
    scene_dir = dataset_dir / "scene_a"
    scene_dir.mkdir(parents=True)
    np.savez_compressed(scene_dir / "p_00000_00000.npz", rgb=np.ones((3, 32, 32), dtype=np.uint8) * 255)

    from app.core.config import settings
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/api/ae/labels/preview", params={"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/p_00000_00000.npz"})
    data = resp.json()["data"]

    assert resp.status_code == 200
    assert data["image_base64"].startswith("data:image/png;base64,")


def test_labels_save_persists_row(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame([{"patch_path": "linhe_patches/scene_a/p.npz"}]).to_parquet(dataset_dir / "_index.parquet")

    from app.core.config import settings
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.post("/api/ae/labels/save", json={
        "dataset_id": "linhe_patches",
        "patch_path": "linhe_patches/scene_a/p.npz",
        "label": "耕地",
        "reviewer": "tester",
    })

    saved = pd.read_parquet(dataset_dir / "_labels.parquet")
    assert resp.status_code == 200
    assert len(saved) == 1
    assert saved.iloc[0]["label"] == "耕地"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_labels_api.py::test_labels_preview_returns_base64_png tests/test_labels_api.py::test_labels_save_persists_row -v`
Expected: FAIL because preview/save endpoints do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# ae_backend/app/api/labels.py
import base64
from io import BytesIO
from datetime import datetime
import numpy as np
from PIL import Image
from pydantic import BaseModel


class SaveLabelRequest(BaseModel):
    dataset_id: str
    patch_path: str
    label: str
    reviewer: str


@router.get("/preview")
def get_patch_preview(dataset_id: str = Query(...), patch_path: str = Query(...)):
    dataset_dir = get_dataset_dir(dataset_id)
    patch_file = Path(settings.DATA_DIR) / patch_path
    arr = np.load(patch_file)["rgb"]
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    image = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"status": "success", "data": {"patch_path": patch_path, "image_base64": f"data:image/png;base64,{encoded}"}}


@router.post("/save")
def save_label(payload: SaveLabelRequest):
    dataset_dir = get_dataset_dir(payload.dataset_id)
    labels_df = load_labels(dataset_dir)
    labels_df = labels_df[labels_df["patch_path"] != payload.patch_path]
    labels_df.loc[len(labels_df)] = {
        "dataset_id": payload.dataset_id,
        "patch_path": payload.patch_path,
        "label": payload.label,
        "reviewer": payload.reviewer,
        "labeled_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    labels_df.to_parquet(dataset_dir / "_labels.parquet", index=False)
    return {"status": "success"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_labels_api.py::test_labels_preview_returns_base64_png tests/test_labels_api.py::test_labels_save_persists_row -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ae_backend/app/api/labels.py tests/test_labels_api.py
git commit -m "feat: add patch preview and label save endpoints"
```

---

### Task 4: Add frontend `Linhe 标注` tab and minimal labeling workflow

**Files:**
- Modify: `ae_frontend/index.html`
- Test: manual browser flow

- [ ] **Step 1: Write the failing manual check**

```text
Open the app and verify there is a navigation entry named “Linhe 标注”.
Expected now: it does not exist.
```

- [ ] **Step 2: Confirm the check fails**

Run: start the backend and open `/`
Expected: no `Linhe 标注` tab is present.

- [ ] **Step 3: Write minimal implementation**

```html
<!-- nav item -->
<li>
  <a href="#" @click.prevent="currentTab = 'labeling'" :class="currentTab === 'labeling' ? 'tab-active' : 'tab-inactive'" class="flex items-center space-x-3 px-3 py-2 rounded-md transition-colors duration-200 cursor-pointer">
    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path></svg>
    <span>Linhe 标注</span>
  </a>
</li>
```

```html
<!-- tab content -->
<div v-show="currentTab === 'labeling'" class="max-w-7xl mx-auto space-y-6">
  <div class="flex justify-between items-center">
    <div>
      <h2 class="text-2xl font-bold text-gray-900">Linhe Patch 标注</h2>
      <p class="text-sm text-gray-500 mt-1">对 linhe_patches 执行单标签地类分类标注</p>
    </div>
    <div class="text-sm text-gray-600">
      已标注 {{ labelingStats.labeled }} / {{ labelingStats.total }}，未标注 {{ labelingStats.unlabeled }}
    </div>
  </div>

  <div class="glass-card p-6 space-y-4">
    <div v-if="labelingCurrentPatch">
      <div class="text-xs text-gray-500 break-all mb-3">{{ labelingCurrentPatch.patch_path }}</div>
      <img :src="labelingPreview" class="w-96 h-96 object-contain border border-gray-200 rounded-md bg-gray-50" />
    </div>
    <div v-else class="text-gray-500">暂无未标注 patch。</div>

    <div class="grid grid-cols-4 gap-3">
      <button v-for="cls in labelingClasses" :key="cls" @click="savePatchLabel(cls)" class="px-4 py-2 bg-primary text-white rounded-md hover:bg-blue-600">{{ cls }}</button>
    </div>

    <div class="flex space-x-3">
      <button @click="fetchCurrentPatch()" class="px-4 py-2 bg-white border border-gray-300 rounded-md">下一张</button>
    </div>
  </div>
</div>
```

```javascript
const labelingClasses = ["耕地", "林地", "水体", "建筑", "道路", "裸地", "其他"];
const labelingCurrentPatch = ref(null);
const labelingPreview = ref("");
const labelingStats = ref({ total: 0, labeled: 0, unlabeled: 0 });

const fetchLabelingStats = async () => {
  const response = await fetch('/api/ae/labels/stats?dataset_id=linhe_patches');
  const json = await response.json();
  if (json.status === 'success') labelingStats.value = json.data;
};

const fetchCurrentPatch = async () => {
  const response = await fetch('/api/ae/labels/next?dataset_id=linhe_patches');
  const json = await response.json();
  if (json.status === 'success') {
    labelingCurrentPatch.value = json.data;
    if (json.data) {
      const previewResp = await fetch(`/api/ae/labels/preview?dataset_id=linhe_patches&patch_path=${encodeURIComponent(json.data.patch_path)}`);
      const previewJson = await previewResp.json();
      labelingPreview.value = previewJson.data.image_base64;
    } else {
      labelingPreview.value = "";
    }
  }
};

const savePatchLabel = async (label) => {
  if (!labelingCurrentPatch.value) return;
  await fetch('/api/ae/labels/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: 'linhe_patches',
      patch_path: labelingCurrentPatch.value.patch_path,
      label,
      reviewer: 'manual-user',
    }),
  });
  await fetchLabelingStats();
  await fetchCurrentPatch();
};
```

- [ ] **Step 4: Run manual check to verify it passes**

Run:
1. `uvicorn app.main:app --host 127.0.0.1 --port 8087` from `ae_backend`
2. Open `http://127.0.0.1:8087/`
3. Click `Linhe 标注`
4. Verify one patch image loads and label buttons are visible
5. Click one label and verify progress updates and next patch loads

Expected: the labeling tab works end-to-end for `linhe_patches`.

- [ ] **Step 5: Commit**

```bash
git add ae_frontend/index.html
git commit -m "feat: add linhe patch labeling tab"
```

---

### Task 5: Document the labeling workflow in the Linhe integration doc

**Files:**
- Modify: `docs/linhe_dataset_integration.md`
- Test: doc read-through

- [ ] **Step 1: Write the failing documentation check**

```text
Read docs/linhe_dataset_integration.md and verify it mentions:
- Linhe 标注 tab
- fixed seven classes
- output file data/linhe_patches/_labels.parquet
```

- [ ] **Step 2: Run the check to verify it fails**

Run: `D:/adk/.venv/Scripts/python.exe -c "from pathlib import Path; txt=Path('docs/linhe_dataset_integration.md').read_text(encoding='utf-8'); print(txt)"`
Expected: those details are missing.

- [ ] **Step 3: Write the minimal documentation update**

```markdown
### 6. 系统内标注能力
- 前端新增 `Linhe 标注` tab
- 第一版固定类别：耕地 / 林地 / 水体 / 建筑 / 道路 / 裸地 / 其他
- 标签结果保存到 `data/linhe_patches/_labels.parquet`
- 标注结果可直接与 `_index.parquet` 关联，供后续监督训练使用
```

- [ ] **Step 4: Run the check to verify it passes**

Run: `D:/adk/.venv/Scripts/python.exe -c "from pathlib import Path; txt=Path('docs/linhe_dataset_integration.md').read_text(encoding='utf-8'); assert 'Linhe 标注' in txt; assert '_labels.parquet' in txt; assert '林地' in txt; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add docs/linhe_dataset_integration.md
git commit -m "docs: describe linhe labeling workflow"
```

---

### Task 6: Run regression checks for the minimal labeling workbench

**Files:**
- Test: `tests/test_labels_api.py`
- Test: existing `tests/test_linhe_backend_integration.py`
- Test: existing `tests/test_linhe_scripts.py`
- Manual: frontend labeling flow

- [ ] **Step 1: Run backend labeling tests**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_labels_api.py -v`
Expected: PASS

- [ ] **Step 2: Run existing Linhe backend tests**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_linhe_backend_integration.py tests/test_linhe_scripts.py -v`
Expected: PASS

- [ ] **Step 3: Run the app and verify the labeling tab manually**

Run: `uvicorn app.main:app --host 127.0.0.1 --port 8087` from `ae_backend`
Expected: `Linhe 标注` tab loads and saves labels successfully.

- [ ] **Step 4: Verify labels file appears**

Run: `ls data/linhe_patches/_labels.parquet`
Expected: file exists after at least one saved label.

- [ ] **Step 5: Commit final verified state**

```bash
git add ae_backend/app/api/labels.py ae_backend/app/main.py ae_frontend/index.html tests/test_labels_api.py docs/linhe_dataset_integration.md
git commit -m "feat: add linhe patch labeling workbench"
```

---

## Self-Review

### Spec coverage
- In-app patch labeling tab: covered by Task 4
- Fixed seven classes: covered by Task 4 and Task 5
- `_labels.parquet` persistence: covered by Tasks 1 and 3
- Progress stats: covered by Task 2 and Task 4
- Minimal backend API only: covered by Tasks 1-3
- No polygon/mask/collaboration scope creep: enforced across all tasks

### Placeholder scan
- No `TBD`, `TODO`, or unspecified “add tests later” language remains.
- File paths, code snippets, commands, and assertions are concrete.

### Type consistency
- Dataset id is consistently `linhe_patches`
- Labels file is consistently `data/linhe_patches/_labels.parquet`
- Categories are consistently `耕地 / 林地 / 水体 / 建筑 / 道路 / 裸地 / 其他`
- API prefix is consistently `/api/ae/labels`

Plan complete and saved to `docs/superpowers/plans/2026-04-18-linhe-patch-labeling-workbench.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
