# Paper12 GeoVLM Seed-42 Recovery Design

Date: 2026-07-24
Workspace: `D:\adk\.worktrees\paper12-geovlm-seed42-recovery`
Branch: `fix/paper12-geovlm-seed42-recovery`

## Goal

Recover the Paper12 GeoVLM seed-42 smoke experiment without changing the model
architecture, optimizer hyperparameters, loss coefficients, evaluation
thresholds, or acceptance gates. The recovery must fix two failures observed in
the first real LandCoverAI run:

1. the Colab subprocess could not resolve the explicitly downloaded SigLIP
   snapshot from its custom Google Drive cache; and
2. the prompt model collapsed to a constant all-background prediction after the
   training pipeline exposed it to substantially more empty targets than the
   configured 25% cap implied.

The recovery remains an experiment-infrastructure change. It does not make the
GeoVLM MVP complete, authorize the full three-seed matrix, or change Paper12
manuscript claims.

## Evidence From The Failed Run

The external result intake used these Colab Drive artifacts without committing
them:

- `/content/drive/MyDrive/paper12_results/geovlm_prompt_segmentation.json`
- `/content/drive/MyDrive/paper12_results/geovlm_prompt_segmentation_summary.json`

The raw file contained only the prompt method for seed 42, with three class
rows rather than the required 18-row matrix. The independently rebuilt summary
matched the Colab summary and reported `mvp_status: incomplete`.

The seed-42 smoke result failed:

- `finite_decreasing_loss`;
- `nonconstant_predictions`; and
- `prompt_dependent_probability_maps`.

Checkpoint reproduction passed. All trainable checkpoint tensors were finite,
and no row indicated synthetic fallback or missing-weight substitution.

The final previews and row statistics showed the same constant probability,
approximately 0.43137, for building, water, and road prompts. Every thresholded
prediction was empty. Loss reached its minimum of 1.42846 at epoch 5, jumped to
approximately 2.09 at epoch 7, and stayed near that value through epoch 50.

The validation rows also exposed the sampling issue:

- 47.50% of validation tiles contained none of building, water, or road;
- building was present in 17.04% of tiles;
- water was present in 19.85% of tiles; and
- road was present in 37.20% of tiles.

The current sampler caps only avoidable empty prompts. A tile containing none
of the three supported classes always produces an empty target and bypasses the
cap. With the observed no-target share, the expected total empty-target rate is
approximately 60.6%, not 25%.

## Scope

### In Scope

- Pass the configured Hugging Face cache directory explicitly to both SigLIP
  tokenizer and text-model loading.
- Build a deterministic target-present training pool from the LandCoverAI
  training split.
- Reserve a small deterministic probe subset from the training split so final
  validation remains untouched by checkpoint selection.
- Enforce the total per-batch empty-target cap on the target-present training
  pool.
- Record training-pool, class-sampling, and observed empty-target statistics.
- Save resumable last-epoch state separately from the best model state.
- Run a bounded per-epoch probe for nonconstant predictions, prompt-dependent
  probability maps, and probe loss.
- Evaluate and reproduce the best checkpoint rather than the last epoch.
- Reject incompatible failed-run artifacts through a versioned training
  contract instead of silently resuming them.
- Update the generated Colab workflow and recovery documentation.

### Out Of Scope

- Learning-rate changes, gradient clipping, mixed precision, loss-weight
  changes, or positive-weight-clip changes.
- Changes to Prithvi, SigLIP, FiLM, dense similarity, Houlsby dimensions, or the
  decoder architecture.
- Changes to the probability threshold or any MVP acceptance gate.
- Running seeds 123 or 456, or the no-text baseline, before the recovered
  seed-42 smoke passes.
- Deleting or overwriting the failed JSON, checkpoint, or preview artifacts.
- Incorporating GeoVLM results into Paper12 manuscript text.

## Considered Approaches

### A. Controlled Data And Checkpoint Recovery (Selected)

Make cache resolution explicit, remove no-target tiles from the prompt-training
pool, retain bounded negative supervision through absent-class prompts, and
select a best checkpoint using a training-split probe.

This approach changes only the two demonstrated failure mechanisms. It
preserves model and optimizer comparability with the failed run and gives the
next seed-42 result a clear causal interpretation.

### B. Combined Sampling And Optimizer Stabilization (Deferred)

Apply the selected data changes together with a lower learning rate, gradient
clipping, or altered BCE/Dice coefficients.

This may recover more quickly but would make it impossible to determine whether
the collapse came from empty-target exposure or optimization. It is allowed
only as a separately versioned follow-up if approach A still fails.

### C. Architecture Redesign (Rejected For This Recovery)

Replace FiLM/dense similarity with cross-attention, unfreeze more parameters,
or replace the binary decoder.

This has the highest implementation and scientific risk and is not supported
by the current evidence. Architecture work requires a new design after the
controlled recovery is evaluated.

## Design

### Explicit SigLIP Cache Resolution

`SiglipTextEncoder` gains an optional `cache_dir` argument. The value is passed
unchanged to both `AutoTokenizer.from_pretrained` and
`SiglipTextModel.from_pretrained` alongside the existing model identifier,
revision, and `local_files_only` flag.

The checked-in config uses `cache_dir: null`. The generated Colab config sets
the field to the absolute Google Drive Hugging Face cache directory. The model
identifier and resolved revision remain stable in checkpoint metadata; an
environment-specific snapshot path must not replace the model identifier.

Offline callers that omit `cache_dir` preserve current behavior.

### Target-Present Training Pool

Before training, scan the training masks once and build an in-memory index of
tiles containing at least one supported target class. Tiles containing only
background and/or woodland are excluded from the prompt-training pool but
remain untouched in the source dataset and final validation set.

The scan also records the supported classes present in each retained tile. The
pool must raise an actionable error if no supported target is present.

Negative supervision is retained without no-target tiles. For at most
`floor(batch_size * empty_target_cap)` positions in each batch, the prompt class
may be a supported class absent from that image. Every other position must use
a class present in its image. Therefore the observed total empty-target share
cannot exceed the configured cap.

Prompt classes for non-empty positions are balanced across building, water,
and road using a seeded deterministic schedule. If a scheduled class has no
available image, the runner fails rather than silently changing the taxonomy.

The final validation dataset remains the complete official validation split,
including no-target tiles. Empty validation cases continue to test false
positive behavior. Reports must keep empty-mask rate adjacent to class IoU so
empty-mask agreement cannot be described as foreground segmentation skill.

### Training-Split Probe

Reserve a deterministic bounded probe set from the target-present training
pool before constructing the training loader. The default is two positive
examples per target class, selected from the seed-specific shuffled index. A
single image may satisfy more than one class, but probe indices are removed
from the training pool exactly once.

At the end of every epoch, evaluate the current model on this fixed probe using
training prompts only. Record:

- finite mean probe loss;
- per-class prediction range and nonconstant status;
- mean absolute change between correct- and wrong-prompt probability maps; and
- the epoch training loss.

The probe is a checkpoint-selection diagnostic, not a reported model result.
Held-out prompts and the official validation split are used only in final
evaluation.

### Best And Resume Checkpoints

Each method/seed pair has two atomic checkpoint files:

- `*.last.pt` stores every completed epoch, optimizer/scheduler state, full
  loss history, and probe history for safe resume;
- `*.best.pt` stores the best trainable model state and its selected loss/probe
  prefix for final evaluation and offline inference.

Best-state ranking is deterministic and lexicographic:

1. all probe values are finite;
2. number of classes with nonconstant predictions;
3. number of classes with positive prompt-map change;
4. larger mean prompt-map change; and
5. lower mean probe loss.

The selected epoch and ranking components are stored in checkpoint metadata.
Training still runs the configured 50 epochs; this recovery does not add early
stopping. At the end, the runner reloads `*.best.pt`, verifies its logits, and
uses that model for official seed-42 evaluation.

The raw row stores both `full_loss_history` and the selected-prefix
`loss_history`. The existing `finite_decreasing_loss` smoke check applies to the
selected model's loss prefix. This is not a threshold relaxation: the evaluated
checkpoint, its selected epoch, and the complete post-selection training
history all remain auditable.

### Training Contract And Failed-Run Isolation

Add a versioned training-contract identifier to config, checkpoint metadata,
and raw results. The recovery contract covers:

- target-present pool policy;
- maximum total empty-target share;
- probe selection policy;
- best-checkpoint ranking policy; and
- SigLIP model identifier/revision, without environment-specific cache paths.

Existing artifacts lacking this contract are incompatible. The runner must
stop with an instruction to archive them; it must not overwrite, append to, or
resume the failed run.

The Colab documentation instructs the author to move the failed JSON,
checkpoint, and previews into a dated failed-run archive before starting the
recovery. Archive operations remain explicit and user-controlled.

### Diagnostics And Result Contract

Raw rows and checkpoint state add:

- source training size;
- target-present pool size and excluded no-target count/share;
- probe indices or their stable hash;
- per-class prompt counts;
- observed empty-target count/share;
- best epoch and probe ranking values;
- full and selected loss histories; and
- training-contract identifier.

The seed-42 smoke gate remains unchanged:

- finite decreasing selected loss;
- nonconstant predictions for all target classes;
- prompt-dependent probability maps for all target classes; and
- checkpoint reproduction.

If any item fails, the runner persists diagnostics, reports the exact failed
checks, and blocks the full matrix.

## Testing

Tests are written before implementation and must cover:

1. `SiglipTextEncoder` forwards `cache_dir` to tokenizer and model loading while
   preserving the model identifier and revision.
2. Target-present indexing excludes masks with none of the supported classes.
3. Deterministic batching balances non-empty prompt classes and never exceeds
   the configured total empty-target cap.
4. A missing supported class or empty target-present pool fails explicitly.
5. Probe indices are deterministic, bounded, and excluded from training.
6. A tiny injected model whose final epoch degrades retains the earlier best
   checkpoint while `*.last.pt` remains resumable.
7. Final evaluation and checkpoint reproduction use `*.best.pt`.
8. Training-contract mismatches reject old raw results and checkpoints without
   modifying them.
9. The generated Colab config includes the explicit cache directory and failed
   artifact archive guidance.
10. Existing focused GeoVLM tests and the complete maintained suite pass.

## Verification And Decision Boundary

Offline completion requires:

- focused GeoVLM tests pass;
- complete maintained pytest suite passes;
- deterministic Colab generation produces no diff;
- `compileall` and `git diff --check` pass; and
- no result, checkpoint, cache, preview, or failed-run artifact is committed.

Real completion of this recovery requires a fresh A100 seed-42 run under the
new training contract. The full matrix remains disabled until all four existing
smoke checks pass. A second failure does not authorize threshold changes; it
triggers a separate optimizer-stabilization design using the preserved
diagnostics.
