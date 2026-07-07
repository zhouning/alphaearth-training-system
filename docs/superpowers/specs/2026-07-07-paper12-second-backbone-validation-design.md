# Paper12 Second-Backbone Validation Design

Date: 2026-07-07
Workspace: `D:\adk\AlphaEarth-System`
Branch: `master`

## Goal

Add a minimal second-backbone validation track for Paper12 so the manuscript can
answer the reviewer risk that all PEFT conclusions are currently bounded to
Prithvi-100M.

The first implementation target is a compact EuroSAT experiment:

- backbone: one official or official-compatible non-Prithvi GeoFM, selected by
  fastest reliable Colab execution from SatMAE, Scale-MAE, or SpectralGPT;
- modalities: `s2_full` and `rgb`;
- methods: linear probe, low-rank attention adaptation, and bottleneck adapter;
- seeds: 42, 123, and 456;
- metrics: overall accuracy, macro F1, and trainable parameters.

The full matrix is 18 rows. Results are written to
`paper12_results/second_backbone_eurosat.json` and
`paper12_results/second_backbone_eurosat_summary.json`.

## Current Context

Paper12 already has completed Prithvi-100M evidence for EuroSAT, BigEarthNet,
LandCover.ai, Linhe, LoveDA, channel-bridge ablation, and PEFT capacity sweep.
The remaining high-risk reviewer question is whether the method ranking is a
Prithvi-specific artifact.

The codebase currently hard-codes `PrithviBackbone` inside
`geoadapter.bench.run_benchmark`. The PEFT insertion functions already operate
on PyTorch transformer blocks, so a narrow backbone factory can add a second
backbone without rewriting the benchmark runner.

Local weights exist for Prithvi only. The second-backbone real experiment must
therefore be explicit Colab or prepared-weight work, while local unit tests must
remain offline and deterministic.

## Scope

### In Scope

- Add a backbone factory that selects `prithvi` or a second-backbone adapter from
  benchmark config.
- Preserve the existing Prithvi path and result files unchanged.
- Add a second-backbone EuroSAT config with the 18-row matrix.
- Add a generated Colab notebook that stages the selected second-backbone
  weights, downloads EuroSAT to local SSD, runs the matrix, and writes result
  and summary JSON files to Drive.
- Add tests for config shape, dry-run matrix size, backbone factory behavior,
  and summary schema.
- Add status-document updates that mark second-backbone validation as prepared
  until result JSON files are returned.
- Add manuscript edits only after real result JSON files are available.

### Out of Scope

- No large weights, datasets, or checkpoints committed to git.
- No automatic network downloads in unit tests.
- No broad runner rewrite beyond the factory and dimension plumbing required for
  the second backbone.
- No claim that the ranking generalizes to all GeoFMs unless the second-backbone
  result supports it.
- No rerun of existing Prithvi capacity-sweep, channel-bridge, Linhe, or LoveDA
  experiments.

## Approach

Use a two-layer implementation.

The local code layer adds only reusable infrastructure and contracts. It makes
the runner configurable by backbone, exposes feature dimension and target input
channel count through metadata, and keeps tests offline by using synthetic
forward checks and dry-run matrix checks.

The real experiment layer lives in a Colab notebook. It explicitly downloads or
copies the selected backbone weights, prepares EuroSAT, runs the 18-row matrix,
aggregates OA and macro F1 by method and modality, and writes both raw and
summary JSON files to Drive.

This keeps the git change reviewable while still producing manuscript-grade
evidence when the notebook is run.

## Architecture

### Backbone Factory

Create `geoadapter.models.backbone_factory` with:

- `BackboneSpec`: a small dataclass-like object with `name`, `feature_dim`,
  `input_channels`, `blocks`, and `model`;
- `build_backbone(config)`: returns a frozen backbone and its metadata;
- existing Prithvi support implemented by delegating to `PrithviBackbone`;
- second-backbone support implemented behind a clear model id such as
  `satmae` once the implementation target is selected.

The runner uses `feature_dim` when constructing classification heads and adapter
bottlenecks. Existing Prithvi configs continue to work because their default
metadata remains `feature_dim=768` and `input_channels=6`.

### PEFT Injection

Keep the current method names for Prithvi. For the second backbone:

- `linear_probe` trains only the task head;
- low-rank attention adaptation uses the nearest architecture-compatible LoRA
  insertion path;
- bottleneck adapter uses the existing Houlsby adapter where block shape allows
  it, otherwise an equivalent residual bottleneck wrapper with the same reported
  parameter accounting.

If a selected backbone exposes attention in a way that cannot support one of
these methods safely, the config must fail early with an explicit error rather
than silently running a different method.

### Config

Add `geoadapter/bench/configs/eurosat_second_backbone.yaml`:

- dataset: EuroSAT;
- task: classification;
- modalities: `s2_full`, `rgb`;
- methods: `linear_probe`, second-backbone low-rank method, second-backbone
  bottleneck adapter;
- seeds: 42, 123, 456;
- epochs and batch size aligned with existing Paper12 EuroSAT experiments unless
  the selected backbone requires a documented reduction.

`allow_synthetic_fallback` should be false for manuscript runs so missing
datasets fail loudly.

### Notebook

Extend `scripts/make_paper12_colab_notebooks.py` to generate
`colab/paper12_second_backbone_eurosat_colab.ipynb`.

The notebook must:

- clone the Paper12 results branch into Colab local SSD;
- install local package dependencies and any documented optional second-backbone
  dependency;
- stage second-backbone weights under `data/weights/<backbone>/`;
- download EuroSAT under `data/eurosat`;
- dry-run the matrix before training;
- run `python -m geoadapter.bench.run_benchmark` with checkpoint resume enabled;
- assert `expected_rows = 18`;
- aggregate OA, macro F1, trainable parameters, seeds, and per-modality method
  rankings into the summary JSON.

### Result Schema

Raw rows must include at least:

- `backbone`;
- `method`;
- `modality`;
- `seed`;
- `trainable_params`;
- `overall_accuracy`;
- `macro_f1`.

Summary records must be grouped by `(backbone, method, modality)` with:

- mean/std OA;
- mean/std macro F1;
- trainable parameter count;
- seeds;
- rank within each modality.

## Data Flow

1. User runs the generated Colab notebook.
2. Notebook writes raw and summary JSON files to Drive.
3. User mirrors those files into `paper12_results/` and the submission
   supplementary results directory.
4. `python -m geoadapter.bench.paper12_audit` is rerun after result files are
   present.
5. Figures or tables are regenerated only if the summary is incorporated into
   the manuscript.
6. Manuscript claims are updated based on the decision rule below.

## Decision Rule

If the second backbone shows the same qualitative ranking, the manuscript may
say that the Prithvi finding is supported by a compact second-backbone
validation, while still avoiding universal GeoFM claims.

If the ranking changes, the second-backbone result becomes a boundary result:
Paper12 should keep the current architecture-aware diagnosis framing and state
that PEFT ranking is backbone-dependent.

If the second-backbone run fails due to dependency or weight incompatibility, no
manuscript claim changes are made. The status documents record the blocker and
the work remains infrastructure-only.

## Testing

Local tests must not require network access, real second-backbone weights, or
EuroSAT downloads.

Required tests:

- backbone factory returns existing Prithvi metadata unchanged;
- second-backbone config has 2 modalities, 3 methods, 3 seeds, and 18 dry-run
  combinations;
- notebook generator includes the second-backbone output notebook, selected
  config, expected output JSON paths, and `expected_rows = 18`;
- runner can construct a classification head from backbone metadata;
- missing required real dataset raises when synthetic fallback is disabled;
- summary builder validates raw rows and computes per-modality method ranking.

Manual verification:

- run the generated notebook dry-run cell;
- run a one-epoch smoke matrix if the selected second-backbone weights are
  available;
- run the full 18-row Colab matrix before manuscript edits.

## Manuscript Integration

Do not edit the abstract, results, or discussion until the second-backbone
summary JSON exists locally.

After results are available:

- add one compact table or supplement table for second-backbone EuroSAT;
- update the limitations paragraph from "single-backbone Prithvi-100M" to either
  "compact two-backbone validation" or "backbone-dependent boundary result";
- update the abstract only if the result changes the central claim;
- reconcile experiment counts across abstract, introduction, conclusion, and
  supplementary material.

## Acceptance Criteria

The design is complete when:

- the local code path can be reviewed without large artifacts;
- the generated Colab notebook can produce the full 18-row result file;
- local focused tests pass;
- `git diff --check` is clean;
- no existing Prithvi result file is changed by infrastructure work;
- manuscript changes, if any, are backed by committed second-backbone result
  JSON files.

## Risks

- Official second-backbone checkpoints may have incompatible input normalization
  or band order. Mitigation: document the bridge and fail if the mapping cannot
  be stated clearly.
- Some backbones may not expose PyTorch transformer blocks compatible with the
  current PEFT injection helpers. Mitigation: implement a narrow adapter wrapper
  for the selected backbone and report unsupported methods explicitly.
- Colab dependency drift may break runtime setup. Mitigation: pin optional
  dependency versions in the notebook cell once the selected backbone is known.
- A changed ranking may weaken the current story. Mitigation: treat it as a
  useful boundary result instead of overclaiming.
