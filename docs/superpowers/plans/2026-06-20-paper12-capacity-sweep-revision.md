# Paper 12 Capacity-Sweep Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Colab-ready PEFT capacity-sweep experiment track and revise Paper 12 so its innovation claim centers on architecture-aware PEFT failure diagnosis and capacity-boundary evidence.

**Architecture:** Keep the existing `PrithviBackbone` and `run_benchmark.py` experiment runner. Add one config and one generated notebook for the EuroSAT capacity sweep, then update manuscript/status documents with bounded language that treats this experiment as pending until Colab results are returned.

**Tech Stack:** Python, PyTorch, YAML configs, Jupyter notebook JSON, pytest, LaTeX manuscript sections.

---

### Task 1: Add Capacity-Sweep Config

**Files:**
- Create: `geoadapter/bench/configs/eurosat_peft_capacity_sweep.yaml`
- Test: `tests/test_paper12_colab_notebooks.py`

- [ ] **Step 1: Create config**

Create a EuroSAT config with:

```yaml
experiment:
  name: eurosat_peft_capacity_sweep
  dataset: eurosat
  dataset_root: ./data/eurosat
  epochs: 50
  batch_size: 64
  seeds: [42, 123, 456]

modalities:
  - preset: s2_full

methods:
  - name: linear_probe
    adapter: zero_pad
    peft: null
  - name: lora_split_qkv_r4
    adapter: zero_pad
    peft: lora_split_qkv
    rank: 4
  - name: lora_split_qkv_r8
    adapter: zero_pad
    peft: lora_split_qkv
    rank: 8
  - name: lora_split_qkv_r16
    adapter: zero_pad
    peft: lora_split_qkv
    rank: 16
  - name: lora_split_qkv_r32
    adapter: zero_pad
    peft: lora_split_qkv
    rank: 32
  - name: lora_split_qkv_r64
    adapter: zero_pad
    peft: lora_split_qkv
    rank: 64
  - name: houlsby_d8
    adapter: zero_pad
    peft: houlsby
    bottleneck_dim: 8
  - name: houlsby_d16
    adapter: zero_pad
    peft: houlsby
    bottleneck_dim: 16
  - name: houlsby_d32
    adapter: zero_pad
    peft: houlsby
    bottleneck_dim: 32
  - name: houlsby_d64
    adapter: zero_pad
    peft: houlsby
    bottleneck_dim: 64

training:
  lr: 1.0e-3
  lr_peft: 1.0e-4
  scheduler: cosine
  weight_decay: 0.01

prithvi:
  pretrained: true
  checkpoint: data/weights/prithvi/Prithvi_100M.pt
```

- [ ] **Step 2: Add config contract test**

Add a pytest assertion that loads this config and checks:

```python
assert cfg["experiment"]["dataset"] == "eurosat"
assert cfg["experiment"]["seeds"] == [42, 123, 456]
assert cfg["prithvi"]["checkpoint"] == "data/weights/prithvi/Prithvi_100M.pt"
assert len(cfg["methods"]) == 10
```

- [ ] **Step 3: Run focused config test**

Run:

```powershell
python -m pytest tests/test_paper12_colab_notebooks.py -q
```

Expected: PASS.

### Task 2: Generate Capacity-Sweep Colab Notebook

**Files:**
- Modify: `scripts/make_paper12_colab_notebooks.py`
- Create: `colab/paper12_peft_capacity_sweep_colab.ipynb`
- Test: `tests/test_paper12_colab_notebooks.py`

- [ ] **Step 1: Extend notebook generator**

Add a `CAPACITY_OUT` path and a `capacity_sweep_notebook()` function. The notebook must include:

```python
expected_rows = 30
```

and write:

```python
/content/drive/MyDrive/paper12_results/peft_capacity_sweep.json
/content/drive/MyDrive/paper12_results/peft_capacity_sweep_summary.json
```

- [ ] **Step 2: Include notebook in generator outputs**

Add `CAPACITY_OUT: capacity_sweep_notebook()` to the generator's `outputs` dictionary.

- [ ] **Step 3: Run generator**

Run:

```powershell
python scripts\make_paper12_colab_notebooks.py
```

Expected: the new notebook is written under `colab/`.

- [ ] **Step 4: Add notebook contract test**

Assert the notebook contains:

```python
"paper12_peft_capacity_sweep_colab.ipynb"
"--branch paper12-results-colab-20260619"
"eurosat_peft_capacity_sweep.yaml"
"peft_capacity_sweep_summary.json"
"expected_rows = 30"
```

### Task 3: Revise Manuscript Framing

**Files:**
- Modify: `submission/paper12_isprs_jprs_20260606/02_latex_source/main_isprs_jprs.tex`
- Modify: `submission/paper12_isprs_jprs_20260606/02_latex_source/sections/abstract.tex`
- Modify: `submission/paper12_isprs_jprs_20260606/02_latex_source/sections/introduction.tex`
- Modify: `submission/paper12_isprs_jprs_20260606/02_latex_source/sections/discussion.tex`
- Modify: `submission/paper12_isprs_jprs_20260606/02_latex_source/sections/conclusion.tex`
- Mirror equivalent edits under `paper12/`.

- [ ] **Step 1: Retitle around diagnosis**

Use a title along these lines:

```latex
Architecture-Aware Diagnosis of Parameter-Efficient Adaptation in Prithvi-100M: Fused-QKV Failure, Capacity Boundaries, and Cross-Domain Remote-Sensing Validation
```

- [ ] **Step 2: Rebuild abstract**

Keep completed results, but state that capacity matching is the decisive remaining experiment until results exist. Avoid presenting pending evidence as complete.

- [ ] **Step 3: Rebuild contribution list**

Make the first two contributions diagnostic:

```latex
\item We separate implementation failure from post-fix capacity limits for LoRA on a fused-QKV Prithvi-100M backbone.
\item We introduce a parameter-capacity audit protocol that compares split-QKV LoRA ranks against Houlsby bottleneck widths under matched data, seeds, and metrics.
```

- [ ] **Step 4: Tighten LoveDA capacity language**

Use "suggests" and "capacity-boundary hypothesis" until the parameter-matched sweep result exists.

- [ ] **Step 5: Preserve Linhe boundary**

Keep Linhe as production-style weak supervision, not independent ground truth.

### Task 4: Update Submission Status Documents

**Files:**
- Modify: `submission/paper12_isprs_jprs_20260606/REQUIRED_EXPERIMENTS_ISPRS.md`
- Modify: `submission/paper12_isprs_jprs_20260606/00_ACTION_REQUIRED.md`
- Modify: `paper12/README.md`
- Modify: `submission/paper12_isprs_jprs_20260606/06_supplementary_material/README_supplementary.md`

- [ ] **Step 1: Add capacity-sweep status**

Record the capacity sweep as "prepared; awaiting Colab run".

- [ ] **Step 2: Add exact Colab instructions**

Reference:

```text
colab/paper12_peft_capacity_sweep_colab.ipynb
```

and the two Drive output files.

- [ ] **Step 3: Keep EuroSAT channel bridge marked complete**

Do not regress the completed channel-bridge status.

### Task 5: Verify and Commit

**Files:**
- All modified files.

- [ ] **Step 1: Run tests**

Run:

```powershell
python -m pytest tests/test_paper12_public_dataset_results.py tests/test_paper12_colab_notebooks.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Inspect diff**

Run:

```powershell
git -C D:\tmp\alphaearth-paper12-results-20260619 diff --stat
git -C D:\tmp\alphaearth-paper12-results-20260619 diff -- geoadapter/bench/configs/eurosat_peft_capacity_sweep.yaml tests/test_paper12_colab_notebooks.py
```

- [ ] **Step 3: Commit**

Run:

```powershell
git -C D:\tmp\alphaearth-paper12-results-20260619 add docs/superpowers/specs/2026-06-20-paper12-capacity-sweep-revision-design.md docs/superpowers/plans/2026-06-20-paper12-capacity-sweep-revision.md geoadapter/bench/configs/eurosat_peft_capacity_sweep.yaml scripts/make_paper12_colab_notebooks.py colab/paper12_peft_capacity_sweep_colab.ipynb tests/test_paper12_colab_notebooks.py paper12 submission/paper12_isprs_jprs_20260606
git -C D:\tmp\alphaearth-paper12-results-20260619 commit -m "feat: prepare paper12 peft capacity sweep revision"
```

- [ ] **Step 4: Push**

Run:

```powershell
git -C D:\tmp\alphaearth-paper12-results-20260619 push
```

Expected: branch `paper12-results-colab-20260619` updates on GitHub.
