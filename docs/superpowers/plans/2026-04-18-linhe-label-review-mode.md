# Linhe Label Review Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing `Linhe 标注` page so the user can switch between unlabeled labeling and labeled review mode, filter by existing label, and overwrite old labels in place.

**Architecture:** Keep the review workflow inside the existing labeling tab. Extend the backend `GET /api/ae/labels/next` endpoint to support `mode` and `label` filters, return `current_label` for labeled items, and reuse the existing save endpoint to overwrite labels. The frontend stays in `ae_frontend/index.html`, with one small mode selector, one label filter, and reuse of the existing preview/save flow.

**Tech Stack:** FastAPI, pandas/parquet, Vue 3 in single-file `index.html`, pytest

---

## File Structure

- Modify: `ae_backend/app/api/labels.py`
  - Extend `get_next_patch` with `mode` and `label` query params.
  - Return `current_label` in labeled review mode.
- Modify: `tests/test_labels_api.py`
  - Add focused backend tests for labeled mode and label-filtered mode.
- Modify: `ae_frontend/index.html`
  - Add review mode selector, current-label display, label filter, and request wiring.
- Modify: `docs/linhe_dataset_integration.md`
  - Document the new review mode and category-focused cleanup workflow.

---

### Task 1: Extend backend `next` endpoint for review mode

**Files:**
- Modify: `ae_backend/app/api/labels.py`
- Test: `tests/test_labels_api.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_labels_next_returns_first_labeled_patch_in_review_mode(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.config import settings

    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame([
        {"patch_path": "linhe_patches/scene_a/a.npz"},
        {"patch_path": "linhe_patches/scene_a/b.npz"},
    ]).to_parquet(dataset_dir / "_index.parquet")
    pd.DataFrame([
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/a.npz", "label": "其他", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:00"},
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/b.npz", "label": "道路", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:01"},
    ]).to_parquet(dataset_dir / "_labels.parquet")

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    client = TestClient(app)
    resp = client.get("/api/ae/labels/next", params={"dataset_id": "linhe_patches", "mode": "labeled"})
    data = resp.json()["data"]

    assert resp.status_code == 200
    assert data["current_label"] in {"其他", "道路"}
    assert data["patch_path"] in {"linhe_patches/scene_a/a.npz", "linhe_patches/scene_a/b.npz"}


def test_labels_next_filters_by_existing_label(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.config import settings

    dataset_dir = tmp_path / "linhe_patches"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame([
        {"patch_path": "linhe_patches/scene_a/a.npz"},
        {"patch_path": "linhe_patches/scene_a/b.npz"},
    ]).to_parquet(dataset_dir / "_index.parquet")
    pd.DataFrame([
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/a.npz", "label": "其他", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:00"},
        {"dataset_id": "linhe_patches", "patch_path": "linhe_patches/scene_a/b.npz", "label": "道路", "reviewer": "tester", "labeled_at": "2026-04-18T12:00:01"},
    ]).to_parquet(dataset_dir / "_labels.parquet")

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    client = TestClient(app)
    resp = client.get("/api/ae/labels/next", params={"dataset_id": "linhe_patches", "mode": "labeled", "label": "道路"})
    data = resp.json()["data"]

    assert resp.status_code == 200
    assert data["patch_path"] == "linhe_patches/scene_a/b.npz"
    assert data["current_label"] == "道路"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_labels_api.py::test_labels_next_returns_first_labeled_patch_in_review_mode tests/test_labels_api.py::test_labels_next_filters_by_existing_label -v`
Expected: FAIL because `mode`, `label`, and `current_label` are not implemented yet.

- [ ] **Step 3: Write minimal implementation**

```python
@router.get("/next")
def get_next_patch(
    dataset_id: str = Query(...),
    mode: str = Query("unlabeled"),
    label: str | None = Query(None),
):
    dataset_dir = get_dataset_dir(dataset_id)
    index_df = load_patch_index(dataset_dir)
    labels_df = load_labels(dataset_dir)

    if mode == "labeled":
        labeled_df = labels_df.copy()
        if label:
            labeled_df = labeled_df[labeled_df["label"] == label]
        if labeled_df.empty:
            return {"status": "success", "data": None}
        row = labeled_df.iloc[0]
        return {
            "status": "success",
            "data": {
                "dataset_id": dataset_id,
                "patch_path": row["patch_path"],
                "current_label": row["label"],
            },
        }

    labeled_paths = set(labels_df["patch_path"].tolist())
    unlabeled = index_df[~index_df["patch_path"].isin(labeled_paths)]
    if unlabeled.empty:
        return {"status": "success", "data": None}
    row = unlabeled.iloc[0]
    return {"status": "success", "data": {"dataset_id": dataset_id, "patch_path": row["patch_path"], "current_label": None}}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_labels_api.py::test_labels_next_returns_first_labeled_patch_in_review_mode tests/test_labels_api.py::test_labels_next_filters_by_existing_label -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ae_backend/app/api/labels.py tests/test_labels_api.py
git commit -m "feat: add label review mode filtering"
```

---

### Task 2: Add review mode controls to the labeling page

**Files:**
- Modify: `ae_frontend/index.html`
- Test: manual browser check

- [ ] **Step 1: Write the failing manual check**

```text
Open the `Linhe 标注` tab.
Expected now: there is no mode switch, no current-label display, and no filter dropdown.
```

- [ ] **Step 2: Confirm the check fails**

Run: open `http://127.0.0.1:8087/` and inspect the labeling tab.
Expected: only unlabeled mode is available.

- [ ] **Step 3: Write minimal implementation**

```html
<div class="flex space-x-3 items-center">
  <select v-model="labelingMode" class="px-3 py-2 bg-white border border-gray-300 rounded-md text-sm">
    <option value="unlabeled">未标注模式</option>
    <option value="labeled">已标注复核模式</option>
  </select>
  <select v-if="labelingMode === 'labeled'" v-model="labelingFilter" class="px-3 py-2 bg-white border border-gray-300 rounded-md text-sm">
    <option value="">全部</option>
    <option v-for="cls in labelingClasses" :key="cls" :value="cls">{{ cls }}</option>
  </select>
</div>
```

```html
<div v-if="labelingCurrentPatch" class="text-sm text-gray-600 mb-2">
  当前标签：{{ labelingCurrentPatch.current_label || '未标注' }}
</div>
```

```javascript
const labelingMode = ref('unlabeled');
const labelingFilter = ref('');

const fetchCurrentPatch = async () => {
  const query = new URLSearchParams({
    dataset_id: 'linhe_patches',
    mode: labelingMode.value,
  });
  if (labelingMode.value === 'labeled' && labelingFilter.value) {
    query.set('label', labelingFilter.value);
  }
  const response = await fetch(`/api/ae/labels/next?${query.toString()}`);
  const json = await response.json();
  if (json.status === 'success') {
    labelingCurrentPatch.value = json.data;
    if (json.data) {
      const previewResp = await fetch(`/api/ae/labels/preview?dataset_id=linhe_patches&patch_path=${encodeURIComponent(json.data.patch_path)}`);
      const previewJson = await previewResp.json();
      labelingPreview.value = previewJson.data.image_base64;
    } else {
      labelingPreview.value = '';
    }
  }
};

watch([labelingMode, labelingFilter], async () => {
  if (currentTab.value === 'labeling') {
    await fetchCurrentPatch();
  }
});
```

- [ ] **Step 4: Run manual check to verify it passes**

Run:
1. Open `http://127.0.0.1:8087/`
2. Click `Linhe 标注`
3. Verify the mode switch exists
4. Switch to `已标注复核模式`
5. Verify the filter dropdown appears
6. Select `其他`
7. Verify a patch loads and the page shows `当前标签：其他`

Expected: review mode works in the same page.

- [ ] **Step 5: Commit**

```bash
git add ae_frontend/index.html
git commit -m "feat: add labeling review controls"
```

---

### Task 3: Reuse save endpoint for overwriting labels in review mode

**Files:**
- Modify: `ae_frontend/index.html`
- Test: manual browser check

- [ ] **Step 1: Write the failing manual check**

```text
In labeled review mode, change a patch labeled `其他` to `道路`.
Expected now: the page may save but does not clearly prove overwrite + next matching item flow.
```

- [ ] **Step 2: Confirm the check fails or is unclear**

Run: try relabeling one reviewed patch.
Expected: current behavior is not explicitly designed for filtered review flow.

- [ ] **Step 3: Write minimal implementation**

```javascript
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

The key requirement is that `fetchCurrentPatch()` must respect `labelingMode` and `labelingFilter`, so after overwrite it moves to the next matching patch under the current review scope.

- [ ] **Step 4: Run manual check to verify it passes**

Run:
1. Switch to `已标注复核模式`
2. Filter `其他`
3. Change one patch from `其他` to `道路`
4. Refresh stats or reopen the same filter
5. Verify that patch no longer appears in `其他` review mode

Expected: overwrite succeeds and filter-aware navigation works.

- [ ] **Step 5: Commit**

```bash
git add ae_frontend/index.html
git commit -m "feat: enable relabel flow in review mode"
```

---

### Task 4: Document the review workflow

**Files:**
- Modify: `docs/linhe_dataset_integration.md`
- Test: doc read-through

- [ ] **Step 1: Write the failing documentation check**

```text
Read docs/linhe_dataset_integration.md and verify it mentions:
- labeled review mode
- label filter for focused cleanup
- overwriting labels in `_labels.parquet`
```

- [ ] **Step 2: Run the check to verify it fails**

Run: `D:/adk/.venv/Scripts/python.exe -c "from pathlib import Path; print(Path('docs/linhe_dataset_integration.md').read_text(encoding='utf-8'))"`
Expected: the review-mode workflow is not fully described.

- [ ] **Step 3: Write the minimal documentation update**

```markdown
### 7. 复核 / 重标模式
- `Linhe 标注` 支持 `未标注模式` 与 `已标注复核模式`
- 在复核模式下可按现有标签筛选：耕地 / 林地 / 水体 / 建筑 / 道路 / 裸地 / 其他
- 页面显示 `当前标签`
- 点击新标签会覆盖更新 `_labels.parquet` 中该 patch 的旧标签
- 推荐优先复核：其他 / 道路 / 水体 / 林地
```

- [ ] **Step 4: Run the check to verify it passes**

Run: `D:/adk/.venv/Scripts/python.exe -c "from pathlib import Path; txt=Path('docs/linhe_dataset_integration.md').read_text(encoding='utf-8'); assert '已标注复核模式' in txt; assert '_labels.parquet' in txt; assert '当前标签' in txt; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add docs/linhe_dataset_integration.md
git commit -m "docs: describe labeling review workflow"
```

---

### Task 5: Run regression checks for review mode

**Files:**
- Test: `tests/test_labels_api.py`
- Manual: `ae_frontend/index.html` review flow

- [ ] **Step 1: Run backend label API tests**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_labels_api.py -v`
Expected: PASS

- [ ] **Step 2: Start the app for manual verification**

Run: `PYTHONPATH="D:/adk/AlphaEarth-System/.claude/worktrees/linhe-peft/ae_backend" NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost" D:/adk/.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8087`
Expected: app starts successfully.

- [ ] **Step 3: Manually verify review mode flow**

Run in browser:
1. Open `http://127.0.0.1:8087/`
2. Click `Linhe 标注`
3. Switch to `已标注复核模式`
4. Filter `其他`
5. Verify `当前标签` is shown
6. Change one sample to another label
7. Verify stats update and the sample no longer appears in the same filtered queue

Expected: full review/relabel flow works.

- [ ] **Step 4: Commit final verified state**

```bash
git add ae_backend/app/api/labels.py ae_frontend/index.html tests/test_labels_api.py docs/linhe_dataset_integration.md
git commit -m "feat: add linhe label review mode"
```

---

## Self-Review

### Spec coverage
- Mode switch: Task 2
- Label filter: Task 2
- Show current label: Task 2
- Overwrite label flow: Task 3
- Backend support for labeled mode and label filtering: Task 1
- Manual focused review workflow: Tasks 3 and 5
- Documentation: Task 4

### Placeholder scan
- No `TBD`, `TODO`, or vague "implement later" steps remain.
- Commands, code snippets, and expected outcomes are concrete.

### Type consistency
- Review mode values are consistently `unlabeled` and `labeled`
- Filter query param is consistently `label`
- Existing label field is consistently `current_label`
- Labels continue to persist through `_labels.parquet`

Plan complete and saved to `docs/superpowers/plans/2026-04-18-linhe-label-review-mode.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
