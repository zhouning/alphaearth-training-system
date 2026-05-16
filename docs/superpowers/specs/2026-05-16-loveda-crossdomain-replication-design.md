# LoveDA Cross-Domain PEFT Replication for paper12 Section 9

**Date**: 2026-05-16
**Author**: zhouning (with Claude Code)
**Status**: Approved (brainstorming complete, ready for implementation plan)
**Target**: paper12 RSE submission (Remote Sensing of Environment)

## 1. Goal and Scope

**Goal**: Run a 5-method × 3-seed × 2-direction (urban→rural, rural→urban) cross-domain PEFT
benchmark on the LoveDA dataset and integrate the results into paper12 as a new subsection
`Section 9.4 Public-Data Replication: LoveDA Cross-Domain LULC`. The purpose is to provide
RSE reviewers a fully reproducible public-data track that confirms the central PEFT
ranking (Houlsby ≫ Linear, LoRA collapse) reported on Linhe County without requiring
access to the non-redistributable Chinese-domestic high-resolution imagery used in the
main Section 9.

**Why now**: paper12 has been routed to RSE (rolling submission, no hard deadline) per
[[paper12_next_steps]]. The Linhe imagery cannot be openly redistributed under its source
licenses; without a public-data replication, the data-availability statement is the most
likely cause of a desk-reject or major-revision request. The LoveDA replication
neutralizes that risk while strengthening the cross-dataset evidence chain from 4 datasets
(EuroSAT / BigEarthNet-S2 / LandCover.ai / Linhe) to 5.

**Non-goals** (explicitly excluded to prevent scope creep):

- Do not target the LoveDA leaderboard; standard urban+rural mixed split is not run.
- Do not introduce new PEFT methods; reuse the five methods already in paper12.
- Do not modify the first 8 sections of paper12 or the existing Section 9 main body.
- Do not replace Prithvi-100M; the single-backbone limitation is left as future work.
- Do not re-run the Linhe experiments.
- No multi-modal Prithvi-EO 2.0, no SatMAE / Scale-MAE / SpectralGPT additions.

**Success criteria**:

- 30 runs complete with no missing entries (5 methods × 2 directions × 3 seeds).
- Output JSON `results/loveda/loveda_lulc_seg.json` follows the schema of
  `linhe_results/linhe_lulc_seg.json` with an additional `direction` field.
- The PEFT ranking Houlsby ≫ Linear and LoRA ≈ Linear reproduces on at least one
  direction. If both directions invert the ranking, see Section 5 reversal protocol.
- Section 9.4 written, compiled into `main.pdf`, `cover_letter_rse.tex` updated with
  a new contribution and a reproducibility statement, references.bib gains the
  LoveDA citation. Page count goes from 22 to ~24.

## 2. Data and Protocol

**Dataset**: LoveDA (Wang et al., NeurIPS 2021 Datasets and Benchmarks Track)

- Source: Zenodo `https://zenodo.org/records/5706578`
- 5,987 high-resolution (0.3 m GSD) RGB images covering Nanjing, Changzhou, Wuhan
- 7 classes: background (ignore), building, road, water, barren, forest, agriculture
- Native urban (3 city tiles) vs rural (3 city tiles) domain split provided by the dataset

**Patch generation**:

- Original 1024 × 1024 images sliced into 128 × 128 patches with stride 128 (no overlap)
- Aligned with Linhe protocol (128 × 128 RGB)
- Estimated patches: ~5987 × 64 = ~383k; after dataset-natural train/val split, ~250k
  training patches per domain
- Patches are stored as an index file `(image_path, x, y)` rather than pre-sliced PNGs to
  control disk usage; runtime crop on dataloader.

**Cross-domain split** (Q2 decision):

| Direction | Train set | Val set | Purpose |
|---|---|---|---|
| **U→R** | LoveDA Train Urban (1156 imgs) | LoveDA Val Rural (677 imgs) | Urban-trained → rural generalization |
| **R→U** | LoveDA Train Rural (1366 imgs) | LoveDA Val Urban (593 imgs) | Rural-trained → urban generalization |

LoveDA's official test set is unlabeled; the official val set serves as the holdout for
both directions. Train/val pairs are mutually disjoint to keep cross-domain evaluation
clean.

**Label handling**:

- Use original 7 classes without remapping. Linhe's 6-class Esri remap is independent
  and the dataset names disambiguate to the reader.
- Apply ignore_index = 0 (LoveDA convention).

**Preprocessing aligned with Prithvi**:

- Reuse the existing `zero_pad` input adapter: 3-band RGB → 6-band HLS template
  (consistent with Linhe / EuroSAT runs).
- ImageNet stats for normalization (consistent with paper12 elsewhere).

**Correspondence to Linhe** (presented in the paper for reviewer clarity):

| Dimension | Linhe (Section 9 main) | LoveDA (Section 9.4) |
|---|---|---|
| Imagery | 22 satellite families, 0.42–4.29 m, **private** | Single source, 0.3 m, **public** |
| Patches | 128 × 128 RGB | 128 × 128 RGB |
| Labels | Esri 10 m weak supervision | LoveDA human annotation (strong supervision) |
| Split | Scene-level 80/20 | **Cross-domain (urban↔rural)** |
| Classes | 6 (Esri remap) | 7 (LoveDA native) |

## 3. Experimental Matrix and Training Hyperparameters

**Five methods** (identical to Linhe Section 9). All five methods live in a single
config file `loveda_lulc.yaml` following the existing `linhe_lulc.yaml` pattern
(single experiment block with a `methods:` list — not five separate files):

| Method | Trainable params (Linhe, 6 cls) | Adapter / PEFT |
|---|---|---|
| Linear Probe | 4,614 | `zero_pad` / null |
| BitFit | 107,526 | `zero_pad` / `bitfit` |
| LoRA (r=8, split-QKV) | 152,070 | `zero_pad` / `lora` |
| Houlsby (bottleneck=64) | 1,194,246 | `zero_pad` / `houlsby` |
| GeoAdapter | 4,756 | `geo_adapter` / null |

Parameter counts above are the Linhe (6-class head) numbers; the LoveDA 7-class head
adds ~770 parameters to the head only, leaving adapter/PEFT counts unchanged.

**Training hyperparameters** (locked to Linhe values per Q3 full-budget decision):

| Hyperparameter | Value |
|---|---|
| Backbone | Prithvi-100M (frozen) |
| Epochs | 30 |
| Batch size | 16 |
| Optimizer | AdamW, lr=1e-3 (head) / 1e-4 (adapter) |
| Scheduler | Cosine with 5% warmup |
| Seeds | {42, 123, 456} |
| Loss | Cross-entropy (matches Linhe `loss: ce`; no Dice term) |
| Segmentation head | Linear upsampler |
| GPU | Colab Pro L4 |

**Run matrix** = 30 runs (5 methods × 2 directions × 3 seeds).

**Compute estimate** (extrapolated from Linhe Section 9 wall-clock):

- Single run: ~50 min on L4 (250k train patches × 30 epochs / bs=16)
- Total: 30 × 50 min ≈ 25 h L4 GPU
- Colab units: ~250 (May budget remaining: 600, safe)
- Calendar: 2–3 Colab sessions (each ≤ 12 h to avoid idle disconnect)

**Output metrics per run**:

- Primary: mIoU averaged over 7 classes (excluding ignore=0)
- Auxiliary: per-class IoU, OA, mF1
- File: `results/loveda/loveda_lulc_seg.json`, schema mirrors
  `linhe_results/linhe_lulc_seg.json` with an added `direction` field.

## 4. Code and File Layout

**New files** (deliberately small to maximize reuse):

```
geoadapter/
  bench/
    configs/
      loveda_lulc.yaml              # new, single config, 5 methods inside (mirrors linhe_lulc.yaml)
    run_loveda_crossdomain.py       # new, 30-run scheduler (5 methods × 2 directions × 3 seeds)
  data/
    loveda.py                       # new, dataset loader

paper12/
  sections/
    linhe_validation.tex            # modify, append Section 9.4
  figures/
    loveda_crossdomain.pdf          # new
  scripts/
    make_loveda_figure.py           # new

results/
  loveda/
    loveda_lulc_seg.json            # new, 30-run results

colab/
  train_loveda_crossdomain.ipynb    # new, Colab driver notebook
```

**Reuse, do not modify**:

- `geoadapter/adapters/*.py` — five PEFT implementations
- `geoadapter/models/prithvi_encoder.py` — backbone wrapper
- `geoadapter/engine/trainer.py` — training loop
- `geoadapter/data/_base.py` — dataset base
- `geoadapter/bench/run_benchmark.py` — single-run entry point

**`geoadapter/data/loveda.py`** (~150 lines, modeled on `landcoverai.py`):

```python
class LoveDADataset(BaseSegDataset):
    """LoveDA cross-domain LULC.

    Args:
        root: data root containing Train/ and Val/ subdirectories
        domain: one of {"urban", "rural"}
        split: one of {"train", "val"}
        patch_size: 128
        stride: 128
    """
    NUM_CLASSES = 7  # ignore_index=0
    URBAN_CITIES = {1, 2, 3}
    RURAL_CITIES = {4, 5, 6}

    def __init__(self, root, domain, split, patch_size=128, stride=128):
        ...
        self._patch_index = self._build_patch_index()

    def _build_patch_index(self):
        # walk LoveDA/{Train|Val}/{Urban|Rural}/images_png/*.png
        # generate (image_path, x, y) tuples for 128x128 crops
        ...
```

**`geoadapter/bench/run_loveda_crossdomain.py`** (~80 lines):

```python
DIRECTIONS = [
    ("urban", "rural"),  # U→R
    ("rural", "urban"),  # R→U
]

base_cfg = load_yaml("loveda_lulc.yaml")  # contains all 5 methods inline

for train_dom, val_dom in DIRECTIONS:
    for method_cfg in base_cfg["methods"]:  # 5 methods
        for seed in base_cfg["experiment"]["seeds"]:  # [42, 123, 456]
            run_cfg = deepcopy(base_cfg)
            run_cfg["experiment"]["train_domain"] = train_dom
            run_cfg["experiment"]["val_domain"] = val_dom
            run_cfg["experiment"]["seeds"] = [seed]
            run_cfg["methods"] = [method_cfg]  # single method per run
            result = run_single(run_cfg)
            append_to_json("results/loveda/loveda_lulc_seg.json", {
                "direction": f"{train_dom[0].upper()}->{val_dom[0].upper()}",
                "method": method_cfg["name"],
                "seed": seed,
                "miou": result["miou"],
                "per_class_iou": result["per_class_iou"],
                "oa": result["oa"],
                "trainable_params": result["trainable_params"],
            })
```

The scheduler script supports resume-from-checkpoint by skipping any
`(direction, method, seed)` triple already present in the JSON.

**Single YAML config** `loveda_lulc.yaml` (~50 lines): clone `linhe_lulc.yaml`, change
`experiment.dataset: loveda`, `experiment.num_classes: 7`, `experiment.split_mode:
cross_domain`, and add `experiment.train_domain` / `experiment.val_domain` placeholders
that the scheduler overrides per run. The `methods:` list, modality preset, and training
block carry over unchanged from Linhe.

**Colab notebook structure** (~10 cells):

1. Mount Drive, clone the repo branch
2. Install dependencies, copy Prithvi weights from Drive
3. Download LoveDA from Zenodo (~3 GB, cached on Drive)
4. Build patch index (one-time, cached on Drive)
5. Display patch distribution sanity check
6. Run U→R direction (15 runs, resume-aware)
7. Run R→U direction (15 runs, resume-aware)
8. Sync results to Drive and to git
9. Run figure script to produce `loveda_crossdomain.pdf`
10. Emit a Markdown summary table for LaTeX

**Caching strategy** (defends against Colab session disconnect):

- LoveDA raw zips: `MyDrive/loveda/Train.zip`, `Val.zip` (download once manually)
- Patch index: `MyDrive/loveda/patches_128/index.parquet`
- Checkpoints: `MyDrive/loveda/runs/{direction}_{method}_seed{seed}/`
- Result JSON flushed to Drive after each run
- Final results pushed to git for double backup (Drive + git)

## 5. Risks and Failure Handling

**Primary risk register**:

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Houlsby ≫ Linear ranking does not reproduce | Low | **Critical** — paper reversal | Reversal protocol below |
| LoRA no longer collapses (≥ +0.05 mIoU on LoveDA) | Low | High — weakens fused-QKV claim | Reversal protocol below |
| Cross-domain underfit (mIoU < 0.10) | Medium | Medium — numbers ugly but ranking may hold | Tolerated; reported as cross-domain difficulty signal |
| Colab L4 OOM at bs=16 | Medium | Low | Reduce to bs=8 with grad accumulation, matching LandCover.ai protocol |
| Session idle disconnect loses runs | High | Low | Resume-aware scheduler (JSON append + Drive checkpoints) |
| Zenodo download throttle | Medium | Low | One-time download cached permanently on Drive |
| Patch slicing memory blowup (~383k patches) | Medium | Medium | Index-based loader, runtime crop |
| Single-run wall-clock exceeds 50 min | Medium | Medium | +50% buffer planned, total 25 h → 37 h still under 600-unit budget |
| Prithvi weights missing on Drive | Low | High | Confirmed present on Drive (used by paper58 ablation) |

**Reversal protocol** (decided in advance to avoid post-hoc rationalization):

| Reversal scenario | Response |
|---|---|
| Houlsby > Linear with gap < +0.05 mIoU | Report as is. Paper text discusses cross-domain narrowing of in-domain advantage. |
| Houlsby ≈ Linear or Houlsby < Linear | **Pause submission.** Re-examine Linhe results for unisolated confounders. Re-evaluate the paper's central claim. |
| LoRA ≥ +0.05 mIoU over Linear on LoveDA | **Pause submission.** Re-evaluate fused-QKV claim. Possibly add a mechanistic analysis of why Linhe collapses but LoveDA does not. |
| GeoAdapter positive on LoveDA (consistent with Linhe) | Reinforce existing Section 7.2 nuance; no rewrite. |
| GeoAdapter negative on LoveDA (consistent with EuroSAT, not Linhe) | Revise Sections 7.2 and 9.4: claim becomes "GeoAdapter is unstable under RGB-only modality shift, performance is context-dependent." |

**Bottom line**: Reversals are not hidden. If the paper's core claim does not survive
LoveDA, the paper is revised or the submission target re-evaluated. This is committed to
the spec to remove ambiguity later.

**Compute / time budget**:

- 25 h estimate + 50% buffer = 37 h ceiling
- Colab Pro session ≤ 12 h → at least 4 sessions
- Calendar window: post-2026-06-09 demo through 2026-06-16 RSE submission target = 6 days
- After every session, manually push results to git for local + Drive double backup

**Deferred until after 2026-06-09 demo** (per [[strategy_pivot_2026_05_16]]):

- LoveDA download and patch slicing (consumes Drive space and attention)
- Any GPU training
- Any modification to paper12 .tex files

**Permitted before 2026-06-09 demo** (does not conflict with the strategy pivot):

- Write the `loveda_lulc.yaml` config
- Write `geoadapter/data/loveda.py` data loader
- Write `run_loveda_crossdomain.py` scheduler
- Write Colab notebook skeleton
- Smoke-test pipeline locally on a 100-patch dummy run
- Write spec, write plan, commit (no paper12 changes)

## 6. Paper Integration: Section 9.4 Draft

**Location**: append to `paper12/sections/linhe_validation.tex` after the existing 9.3
discussion. Renumber the existing 9.4 Discussion to 9.5.

**Subsection title**: `9.4 Public-Data Replication: LoveDA Cross-Domain LULC`

**Structure** (~1.5 pages, ~600 words + 1 table + 1 figure):

```
Setup
─────
- One paragraph: what LoveDA is, why selected, cross-domain protocol
- Emphasize that the protocol mirrors Linhe: 128×128 RGB / Prithvi frozen / 30 epoch / 3 seed
- Plain-English statement to reviewers: "this section exists because the
  Linhe imagery is not redistributable under its source licenses; readers
  can reproduce the methodological conclusion of Section 9 entirely from
  publicly available data"

Results
───────
- Table: 5 methods × 2 directions, mIoU mean ± std, Δ vs Linear
  (10 rows × 4 columns, compact two-column layout)
- Figure: side-by-side bar chart (U→R left / R→U right), 5 methods × 3 seeds scatter
- Three discussion paragraphs:
  1) Houlsby ranking reproduction + cross-domain numerics
  2) LoRA collapse on a fifth dataset — strengthens fused-QKV claim
  3) Domain-shift effect (U→R vs R→U asymmetry, why)

Cross-dataset summary
────────────────────
- One paragraph + one compact table: ranking consistency across 5 datasets
  (EuroSAT / BigEarthNet-S2 / LandCover.ai / Linhe / LoveDA)
- This summary table is the heaviest single piece of evidence in the paper;
  one glance gives the reviewer the full story
```

**Cover letter revisions** (`cover_letter_rse.tex`):

- Append contribution #5 to "Principal contributions":

  > A fully reproducible public-data replication on LoveDA under cross-domain
  > (urban↔rural) split protocol, confirming Houlsby ≫ Linear Probe and LoRA
  > collapse on a fifth dataset for which all imagery and labels are freely
  > available.

- Add to "Reproducibility" section:

  > The LoveDA replication (Section 9.4) uses only publicly available Zenodo
  > data and is provided as a self-contained reproducibility track requiring
  > no access to private imagery.

- "Confirmation" section unchanged (EarthVision history note remains).

**Expected LaTeX diff size**:

- `linhe_validation.tex`: +120 lines
- `abstract.tex`: +1 sentence
- `introduction.tex`: contribution list +1 entry
- `cover_letter_rse.tex`: +2 paragraphs
- `references.bib`: +1 entry (LoveDA citation, Wang et al. NeurIPS 2021 D&B Track)
- `figures/loveda_crossdomain.pdf`: new
- Page count: 22 → 24

**RSE page-limit compliance**: RSE soft ceiling is ~30 pages two-column; 24 pages leaves
headroom for revision rounds.

**Plain-English summary inserted as the first paragraph of Section 9.4**:

> We provide this replication so that any reviewer or reader without access to
> Chinese-domestic high-resolution imagery can independently verify the
> methodological conclusions of Section 9 using only publicly available data.
> The LoveDA setup deliberately differs from Linhe on sensor resolution, label
> provenance, and class taxonomy, testing whether the PEFT ranking generalizes
> beyond a single deployment context. The cross-domain split (train on urban
> scenes, evaluate on rural, and vice versa) further stresses the ranking
> under geographic distribution shift, which the scene-level Linhe split tests
> on a smaller scale.

## 7. References to Other Memory and Plans

- [[paper12_next_steps]] — current submission strategy (RSE primary, ISPRS backup, TGRS fallback)
- [[paper12_innovation_assessment]] — honest assessment of paper's contribution tier
- [[paper58_x_paper12_ablation]] — sibling Colab pipeline that this work mirrors operationally
- [[strategy_pivot_2026_05_16]] — 2026-06-09 demo deadline that gates GPU work
- [[colab_training_state]] — Colab Drive layout, Prithvi weights location

## 8. Open Items Deferred to Implementation Plan

- Exact patch-stride choice if LoveDA images are not divisible by 128 (likely 1024 ÷ 128 = 8 even, no edge handling needed; verify on first download)
- Whether to add a sub-experiment varying r in LoRA on LoveDA (deferred; not in current spec)
- Final figure aesthetic: bar-chart palette consistency with `linhe_lulc_seg.pdf`
- Citation format for LoveDA bibtex entry: confirm against existing references.bib style
