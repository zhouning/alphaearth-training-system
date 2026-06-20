# Paper 12 Capacity-Sweep Revision Design

## Objective

Revise Paper 12 from a ranking-style PEFT benchmark into a stronger ISPRS JPRS submission centered on architecture-aware PEFT diagnosis and a capacity-boundary hypothesis for cross-domain remote-sensing adaptation.

## Reviewer Risks Addressed

The ISPRS-style review identified four major risks:

- The Houlsby advantage may be caused by trainable-parameter budget rather than adapter architecture.
- The conclusions are currently limited to a single Prithvi-100M backbone.
- Linhe uses Esri-derived supervisory labels rather than independent manual ground truth.
- LoveDA cross-domain mIoU values are low, so the manuscript must show whether methods recover real multi-class signal or sit near a majority/background floor.

This revision prioritizes the first and fourth risks because they can be addressed inside the existing Prithvi-centered codebase with Colab-scale experiments. The second-backbone risk remains an important follow-up because the current implementation is built around `PrithviBackbone`. The Linhe label-quality risk remains framed as a manual audit requirement.

## Scientific Reframing

The revised one-sentence argument is:

> In Prithvi-100M adaptation for heterogeneous remote-sensing inputs, we show that PEFT failure has two separable causes, implementation-level insertion failure and capacity/placement limits under domain shift, using architecture-aware LoRA controls, channel-bridge ablations, public benchmark sweeps, and production/cross-domain segmentation diagnostics, with claims bounded to the tested Prithvi setting.

The manuscript should no longer read primarily as "Houlsby wins most tables." It should read as a diagnostic study that explains when a common PEFT method fails, why a bottleneck adapter succeeds, and how much evidence is still needed before generalizing beyond Prithvi-100M.

## New Experiment Track

### PEFT Capacity Sweep

Add a new Colab notebook and benchmark config for a parameter-matched capacity sweep.

Primary public benchmark:

- Dataset: EuroSAT.
- Modality: `s2_full`.
- Seeds: `42, 123, 456`.
- Metrics: OA, macro F1, trainable parameters.
- Methods:
  - Linear probe.
  - Split-QKV LoRA ranks `4, 8, 16, 32, 64`.
  - Houlsby bottlenecks `8, 16, 32, 64`.
  - Optional full fine-tuning reference remains sourced from the existing result file rather than rerun.

Decision rule:

- If high-rank LoRA remains below narrower Houlsby variants near or below the same parameter budget, strengthen the architecture/placement claim.
- If high-rank LoRA catches up, revise the manuscript to say the original Houlsby advantage was mostly a capacity effect.
- If results are mixed, present the curve as a boundary: low-rank attention-only updates saturate early, while bottleneck MLP adapters trade capacity more effectively in the tested Prithvi setting.

Optional cross-domain extension:

- Dataset: LoveDA U->R.
- Seeds: seed `42` initially.
- Metrics: mIoU, per-class IoU, predicted-pixel histogram.
- Methods: `lora_split_qkv` ranks `16, 32, 64` and Houlsby bottlenecks `16, 32, 64`.
- Use this only if Colab time permits. It is not required before the EuroSAT capacity curve is available.

## Notebook Contract

Create `colab/paper12_peft_capacity_sweep_colab.ipynb`.

The notebook must:

- Clone branch `paper12-results-colab-20260619`.
- Stage Prithvi weights at `data/weights/prithvi/Prithvi_100M.pt`.
- Download EuroSAT to `/content/AlphaEarth-System/data/eurosat`.
- Run `python -m geoadapter.bench.run_benchmark --config geoadapter/bench/configs/eurosat_peft_capacity_sweep.yaml`.
- Write Drive outputs:
  - `/content/drive/MyDrive/paper12_results/peft_capacity_sweep.json`
  - `/content/drive/MyDrive/paper12_results/peft_capacity_sweep_summary.json`
- Verify the expected row count before summary generation.

Expected primary matrix size:

- 10 methods x 1 modality x 3 seeds = 30 rows.

## Manuscript Revision Boundary

Before Colab results are available, the manuscript can be improved in these ways:

- Reframe title, abstract, introduction, and discussion toward architecture-aware diagnosis and capacity-boundary testing.
- Add the capacity-sweep protocol as a planned or pending reviewer-strengthening experiment in the action-required and required-experiments files.
- Avoid presenting the new capacity sweep as completed evidence.

After Colab results are available:

- Add a new result subsection and table/figure for the capacity curve.
- Update abstract and conclusion with the actual outcome.
- Mirror JSON files into the supplementary package.
- Add tests that compute summary statistics from raw rows and assert manuscript values match.

## Tests

Add or extend tests to verify:

- The new config includes exactly the intended methods, seeds, checkpoint path, and dataset.
- The new notebook is pinned to `paper12-results-colab-20260619`, uses the staged Prithvi checkpoint path, writes to Drive, and expects 30 rows.
- Required-experiments status marks capacity sweep as pending until the result files exist.
- Existing completed EuroSAT channel-bridge and LoveDA full-finetune tests continue to pass.

## Non-Goals

- Do not implement a second backbone in this revision pass unless explicitly requested later.
- Do not claim Linhe independent accuracy without a manual label audit.
- Do not regenerate final manuscript figures until the capacity-sweep JSON exists.
- Do not mix unrelated main-branch files into the Paper 12 result branch.
