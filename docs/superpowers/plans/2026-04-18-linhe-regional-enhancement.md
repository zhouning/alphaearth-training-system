# Linhe Regional AlphaEarth Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, testable engineering loop that turns Linhe private imagery into a selectable regional dataset, trains a Prithvi-based PEFT model on it, and documents the workflow inside AlphaEarth-System.

**Architecture:** Keep the scope narrow: reuse the existing `scripts/` workflow for data preparation, make the backend training path recognize `data/linhe_patches`, and expose the dataset through the existing `/api/ae/pipeline/datasets` path so the frontend picks it up automatically. Do not add new model methods or Sentinel-2 pairing in this plan; the plan only closes the RGB-only regional engineering loop.

**Tech Stack:** Python, FastAPI, PyTorch, rasterio, geopandas, pandas, pytest, existing `geoadapter/` Prithvi PEFT stack

---

## File Structure

- Modify: `ae_backend/app/api/pipeline.py`
  - Extend the dataset listing endpoint to include `data/linhe_patches/_index.parquet` as a first-class training dataset.
- Modify: `ae_backend/app/services/trainer.py`
  - Teach `RealPatchDataset` to recursively load `.npz` patch files and normalize RGB patches to tensors.
  - Route `dataset_id == "linhe_patches"` to `data/linhe_patches` instead of `data/weights/raw_data/dataset_*`.
- Create: `tests/test_linhe_backend_integration.py`
  - Add focused regression tests for dataset discovery and `.npz` loading.
- Modify: `scripts/linhe_build_patches.py`
  - Keep the offline RGB patch builder stable and testable.
- Create: `tests/test_linhe_scripts.py`
  - Add focused unit tests for patch tiling and pseudo-label generation.
- Modify: `scripts/linhe_finetune.py`
  - Keep the minimal Houlsby + GeoAdapter PEFT scaffold runnable with Linhe patches.
- Modify: `docs/linhe_dataset_integration.md`
  - Update the doc so it reflects implemented behavior instead of future intent.

---

### Task 1: Make backend dataset discovery expose `linhe_patches`

**Files:**
- Modify: `ae_backend/app/api/pipeline.py:131-165`
- Test: `tests/test_linhe_backend_integration.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app


def test_pipeline_datasets_includes_linhe_patches(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    linhe_dir = data_dir / "linhe_patches"
    linhe_dir.mkdir(parents=True)
    (linhe_dir / "_index.parquet").write_bytes(b"placeholder")

    from app.core.config import settings
    monkeypatch.setattr(settings, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(data_dir / "weights" / "raw_data"))

    client = TestClient(app)
    response = client.get("/api/ae/pipeline/datasets")

    assert response.status_code == 200
    payload = response.json()
    ids = [item["id"] for item in payload["data"]]
    assert "linhe_patches" in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_linhe_backend_integration.py::test_pipeline_datasets_includes_linhe_patches -v`
Expected: FAIL because `linhe_patches` is not returned yet.

- [ ] **Step 3: Write minimal implementation**

```python
@router.get("/datasets")
async def list_available_datasets():
    from app.core.config import settings
    work_dir = settings.RAW_DATA_DIR
    datasets = []

    if os.path.exists(work_dir):
        for item in os.listdir(work_dir):
            item_path = os.path.join(work_dir, item)
            if os.path.isdir(item_path) and item.startswith("dataset_"):
                pt_files = [f for f in os.listdir(item_path) if f.endswith('.pt')]
                if pt_files:
                    datasets.append({
                        "id": item,
                        "name": f"数据集 {item[-8:]} ({len(pt_files)} 切片) [磁盘]",
                        "mtime": os.path.getmtime(item_path),
                    })

    linhe_index = os.path.join(settings.DATA_DIR, "linhe_patches", "_index.parquet")
    if os.path.exists(linhe_index):
        datasets.append({
            "id": "linhe_patches",
            "name": "临河样例数据 [RGB→Prithvi PEFT]",
            "mtime": os.path.getmtime(linhe_index),
        })

    for mem_id, tensors in IN_MEMORY_DATASETS.items():
        datasets.append({
            "id": f"memory_{mem_id}",
            "name": f"数据集 {mem_id[-8:]} ({len(tensors)} 切片) [内存直通 🚀]",
            "mtime": time.time(),
        })

    datasets.sort(key=lambda x: x["mtime"], reverse=True)
    return {"status": "success", "data": datasets}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_linhe_backend_integration.py::test_pipeline_datasets_includes_linhe_patches -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ae_backend/app/api/pipeline.py tests/test_linhe_backend_integration.py
git commit -m "feat: expose linhe patches dataset"
```

---

### Task 2: Make `RealPatchDataset` load Linhe `.npz` patches

**Files:**
- Modify: `ae_backend/app/services/trainer.py:30-82`
- Test: `tests/test_linhe_backend_integration.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from app.services.trainer import RealPatchDataset


def test_real_patch_dataset_reads_npz_rgb_patch(tmp_path):
    sample_dir = tmp_path / "linhe_patches" / "scene_a"
    sample_dir.mkdir(parents=True)
    np.savez_compressed(sample_dir / "p_00000_00000.npz", rgb=np.ones((3, 128, 128), dtype=np.uint8) * 255)

    ds = RealPatchDataset(str(tmp_path / "linhe_patches"), patch_size=128)
    item = ds[0]

    assert item.shape == (5, 128, 128)
    assert item[:3].max().item() <= 1.0
    assert item[3:].sum().item() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_linhe_backend_integration.py::test_real_patch_dataset_reads_npz_rgb_patch -v`
Expected: FAIL because current loader only scans `.pt` files.

- [ ] **Step 3: Write minimal implementation**

```python
import glob
import numpy as np


class RealPatchDataset(Dataset):
    def __init__(self, data_dir_or_id, patch_size=128):
        self.data_dir_or_id = data_dir_or_id
        self.patch_size = patch_size
        self.files = []
        self.in_memory_tensors = None

        if "memory_" in data_dir_or_id:
            ...
        else:
            if os.path.exists(data_dir_or_id):
                self.files = sorted(
                    glob.glob(os.path.join(data_dir_or_id, "**", "*.pt"), recursive=True)
                    + glob.glob(os.path.join(data_dir_or_id, "**", "*.npz"), recursive=True)
                )

    def __getitem__(self, idx):
        if self.in_memory_tensors is not None:
            ...
        else:
            if not self.files:
                return torch.zeros(5, self.patch_size, self.patch_size)

            file_path = self.files[idx % len(self.files)]
            try:
                if file_path.endswith('.npz'):
                    arr = np.load(file_path)["rgb"]
                    tensor = torch.from_numpy(arr).float()
                    if tensor.max() > 1:
                        tensor = tensor / 255.0
                else:
                    tensor = torch.load(file_path, weights_only=True)
            except Exception:
                return torch.zeros(5, self.patch_size, self.patch_size)

        if tensor.shape != (5, self.patch_size, self.patch_size):
            c, h, w = tensor.shape
            new_tensor = torch.zeros(5, self.patch_size, self.patch_size)
            new_tensor[:min(c, 5), :min(h, self.patch_size), :min(w, self.patch_size)] = tensor[:min(c, 5), :min(h, self.patch_size), :min(w, self.patch_size)]
            tensor = new_tensor
        return tensor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_linhe_backend_integration.py::test_real_patch_dataset_reads_npz_rgb_patch -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ae_backend/app/services/trainer.py tests/test_linhe_backend_integration.py
git commit -m "feat: support linhe npz patches in trainer"
```

---

### Task 3: Route `linhe_patches` through the backend trainer

**Files:**
- Modify: `ae_backend/app/services/trainer.py:154-173`
- Test: `tests/test_linhe_backend_integration.py`

- [ ] **Step 1: Write the failing test**

```python
from app.services.trainer import AlphaEarthTrainer


def test_alphaearth_trainer_uses_linhe_dataset_path(monkeypatch, tmp_path):
    captured = {}

    def fake_dataset(path, patch_size=128):
        captured["path"] = path
        captured["patch_size"] = patch_size
        return [0]

    from app.core.config import settings
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("app.services.trainer.RealPatchDataset", fake_dataset)
    monkeypatch.setattr("app.services.trainer.PrithviAlphaEarthEncoder", lambda **kwargs: type("M", (), {"to": lambda self, d: self, "parameters": lambda self: []})())
    monkeypatch.setattr("app.services.trainer.optim.Adam", lambda params, lr: object())
    monkeypatch.setattr("app.services.trainer.DataLoader", lambda dataset, batch_size, shuffle: dataset)

    AlphaEarthTrainer(job_id="job1", dataset_id="linhe_patches", peft_method="houlsby")
    assert captured["path"].endswith("data/linhe_patches")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_linhe_backend_integration.py::test_alphaearth_trainer_uses_linhe_dataset_path -v`
Expected: FAIL because trainer currently builds `dataset_<id>` paths only.

- [ ] **Step 3: Write minimal implementation**

```python
if dataset_id == 'linhe_patches':
    dataset_path = os.path.join(settings.DATA_DIR, 'linhe_patches')
else:
    dataset_path = (
        os.path.join(settings.RAW_DATA_DIR, 'dataset_' + dataset_id)
        if not dataset_id.startswith('memory_')
        else dataset_id
    )
self.dataset = RealPatchDataset(dataset_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_linhe_backend_integration.py::test_alphaearth_trainer_uses_linhe_dataset_path -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ae_backend/app/services/trainer.py tests/test_linhe_backend_integration.py
git commit -m "feat: route linhe dataset through trainer"
```

---

### Task 4: Add focused tests for the Linhe patch builder and fine-tune scaffold

**Files:**
- Create: `tests/test_linhe_scripts.py`
- Test: `tests/test_linhe_scripts.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from scripts.linhe_build_patches import tile_array
from scripts.linhe_finetune import build_pseudo_labels


def test_tile_array_skips_mostly_zero_tiles():
    arr = np.zeros((3, 256, 256), dtype=np.uint8)
    arr[:, 0:128, 0:128] = 255
    tiles = tile_array(arr, patch=128, stride=128)
    assert len(tiles) == 1
    assert tiles[0][0] == 0
    assert tiles[0][1] == 0


def test_build_pseudo_labels_returns_one_label_per_patch(tmp_path):
    paths = []
    for i in range(4):
        path = tmp_path / f"patch_{i}.npz"
        rgb = np.full((3, 128, 128), fill_value=i * 40, dtype=np.uint8)
        np.savez_compressed(path, rgb=rgb)
        paths.append(path)

    labels = build_pseudo_labels(paths, n_clusters=2)
    assert len(labels) == 4
    assert set(labels.tolist()) <= {0, 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_linhe_scripts.py -v`
Expected: FAIL until the new test file exists and imports succeed.

- [ ] **Step 3: Write minimal implementation support**

```python
# No production code change is needed if imports already work.
# If imports fail, add an empty file:
# scripts/__init__.py
```

```python
# scripts/__init__.py
"""Project scripts package for test imports."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_linhe_scripts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_linhe_scripts.py scripts/__init__.py
git commit -m "test: cover linhe patch and peft scripts"
```

---

### Task 5: Update the integration document to match implemented behavior

**Files:**
- Modify: `docs/linhe_dataset_integration.md`
- Test: manual read-through

- [ ] **Step 1: Write the failing documentation check**

```text
Read docs/linhe_dataset_integration.md and verify these statements are true:
- linhe_patches is discoverable through /api/ae/pipeline/datasets
- AlphaEarthTrainer can consume linhe_patches
- RealPatchDataset supports .npz RGB patches
```

- [ ] **Step 2: Run the check to verify it fails or is incomplete**

Run: `D:/adk/.venv/Scripts/python.exe -c "from pathlib import Path; print(Path('docs/linhe_dataset_integration.md').read_text(encoding='utf-8'))"`
Expected: The document is missing one or more implementation details above.

- [ ] **Step 3: Write the minimal documentation update**

```markdown
### 4. 平台可见数据集
后端 `GET /api/ae/pipeline/datasets` 现在会自动识别：
- `data/linhe_patches/_index.parquet`

并将其暴露为：
- `linhe_patches`

### 5. 训练链路兼容
- `AlphaEarthTrainer` 已支持 `dataset_id == "linhe_patches"`
- `RealPatchDataset` 已支持递归读取 `.npz` RGB patch
- `.npz` patch 会按 `[3, H, W] -> [5, H, W]` 零填充到现有训练输入形状
```

- [ ] **Step 4: Run the check to verify it passes**

Run: `D:/adk/.venv/Scripts/python.exe -c "from pathlib import Path; txt=Path('docs/linhe_dataset_integration.md').read_text(encoding='utf-8'); assert 'linhe_patches' in txt; assert '.npz' in txt; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add docs/linhe_dataset_integration.md
git commit -m "docs: document linhe training integration"
```

---

### Task 6: Run regression checks for the minimal engineering loop

**Files:**
- Test: `tests/test_linhe_backend_integration.py`
- Test: `tests/test_linhe_scripts.py`
- Test: existing `tests/test_trainer.py`
- Test: existing `tests/test_datasets.py`

- [ ] **Step 1: Run backend integration tests**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_linhe_backend_integration.py -v`
Expected: PASS

- [ ] **Step 2: Run script tests**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_linhe_scripts.py -v`
Expected: PASS

- [ ] **Step 3: Run existing trainer and dataset tests**

Run: `D:/adk/.venv/Scripts/python.exe -m pytest tests/test_trainer.py tests/test_datasets.py -v`
Expected: PASS

- [ ] **Step 4: Run a smoke build of the offline patch script**

Run: `D:/adk/.venv/Scripts/python.exe scripts/linhe_build_patches.py --target-gsd 10 --patch 128 --max-scenes 1`
Expected: prints `[done]` and writes `data/linhe_patches/_index.parquet`

- [ ] **Step 5: Run a smoke start of the PEFT scaffold**

Run: `D:/adk/.venv/Scripts/python.exe scripts/linhe_finetune.py --epochs 1 --batch-size 4 --max-patches 32 --prithvi-ckpt D:/adk/AlphaEarth-System/data/weights/prithvi/Prithvi_100M.pt`
Expected: prints `epoch 1/1` and writes `results/linhe/linhe_houlsby_rgb.pt`

- [ ] **Step 6: Commit final verified state**

```bash
git add ae_backend/app/api/pipeline.py ae_backend/app/services/trainer.py tests/test_linhe_backend_integration.py tests/test_linhe_scripts.py docs/linhe_dataset_integration.md scripts/__init__.py
git commit -m "feat: close linhe regional training loop"
```

---

## Self-Review

### Spec coverage
- Regional dataset discovery: covered by Task 1
- Regional training path: covered by Tasks 2 and 3
- Minimal runnable RGB-only PEFT loop: covered by Tasks 4 and 6
- Documentation aligned with engineering positioning: covered by Task 5

### Placeholder scan
- No `TBD`, `TODO`, or “implement later” placeholders remain.
- Commands, code snippets, and file paths are concrete.

### Type consistency
- Dataset id is consistently `linhe_patches`
- Backend path is consistently `data/linhe_patches`
- `.npz` payload key is consistently `rgb`
- Trainer input shape is consistently padded to `(5, 128, 128)`

Plan complete and saved to `docs/superpowers/plans/2026-04-18-linhe-regional-enhancement.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
