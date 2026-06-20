# Required and Recommended Experiments Before ISPRS JPRS Submission

This document records evidence gaps identified during the ISPRS JPRS-style review. The manuscript has been revised to avoid overclaiming where evidence is not yet available, but the experiments below would materially improve the chance of review success.

## Priority A: Strongly Recommended Before Initial Submission

### 1. Second-backbone validation

Purpose: address the main reviewer concern that the current conclusions are based on Prithvi-100M only.

Minimum design:

- Backbone: SatMAE, Scale-MAE, or SpectralGPT, whichever is fastest to run in the current codebase.
- Dataset: EuroSAT.
- Modalities: `s2_full` and `rgb`.
- Methods: Linear Probe, LoRA or equivalent low-rank method, Houlsby/adapter if available.
- Seeds: 3.
- Metrics: OA, macro F1, trainable parameters.

Decision rule:

- If Houlsby-like adapters still dominate, restore a stronger cross-GeoFM claim.
- If ranking changes, keep the current bounded Prithvi-100M framing and present the second backbone as a boundary result.

### 2. Parameter-matched PEFT capacity sweep

Purpose: separate method architecture from trainable-parameter budget.

Current status:

- Prepared and awaiting Colab execution.
- Config: `geoadapter/bench/configs/eurosat_peft_capacity_sweep.yaml`.
- Notebook: `colab/paper12_peft_capacity_sweep_colab.ipynb`.
- Expected output files:
  - `/content/drive/MyDrive/paper12_results/peft_capacity_sweep.json`
  - `/content/drive/MyDrive/paper12_results/peft_capacity_sweep_summary.json`
- Expected matrix: 10 methods x 1 EuroSAT `s2_full` modality x 3 seeds = 30 rows.

Minimum design:

- Dataset: EuroSAT `s2_full`, plus one harder setting such as LoveDA R->U or Linhe.
- Methods: LoRA ranks selected to approach Houlsby parameter count; Houlsby bottleneck sizes such as 8, 16, 32, 64.
- Seeds: 3 for EuroSAT; at least 1-3 for the expensive segmentation setting.
- Metrics: OA/mIoU, trainable parameters, convergence curves.

Decision rule:

- If high-rank LoRA remains below smaller Houlsby, the architecture claim is stronger.
- If high-rank LoRA catches up, revise the claim to capacity rather than method family.
- Until these JSON files are returned and verified, manuscript language should treat the LoveDA capacity boundary as a supported hypothesis, not a general adapter-capacity rule.

### 3. Linhe label-quality validation

Purpose: address the concern that Esri 2022 labels are pseudo-labels for 2025 imagery.

Minimum design:

- Randomly sample 300-500 Linhe patches, stratified by predicted/Esri class where possible.
- Manually inspect or annotate class labels using the highest-resolution available imagery.
- Report agreement between Esri-derived labels and manual labels.
- Report per-class reliability, especially for `water`, `built`, and rare/near-absent classes.

Decision rule:

- If Esri agreement is high, the production-style validation becomes much stronger.
- If agreement is weak, frame Linhe strictly as weak-supervision robustness rather than LULC accuracy.

### 4. LoveDA strong baseline and per-class reporting

Purpose: address the concern that all cross-domain mIoU values are close to majority/background baselines.

Current status:

- LoveDA full fine-tuning U->R: completed with three seeds. Full fine-tuning reaches 0.1145 $\pm$ 0.0028 mIoU from seed values 0.1119, 0.1142, and 0.1175.
- LoveDA full fine-tuning R->U: completed with three seeds. Full fine-tuning reaches 0.1391 $\pm$ 0.0085 mIoU from seed values 0.1481, 0.1381, and 0.1311.
- The completed two-direction result is recorded in `paper12_results/loveda_full_finetune_u2r.json`, `paper12_results/loveda_full_finetune_r2u.json`, and summarized in `paper12_results/loveda_full_finetune_summary.json`.
- The completed full fine-tuning baseline does not remove the LoveDA concern: it improves over the small-PEFT/all-background cluster but remains below Houlsby in both directions.

Minimum design:

- Add full fine-tuning or a stronger unfrozen segmentation baseline on LoveDA U->R and R->U.
- Move per-class IoU and prediction-pixel histograms into the main manuscript or a main-table supplement.
- Include the all-background baseline explicitly in the table.

Decision rule:

- If Houlsby remains the only method recovering multiple classes, keep the capacity-threshold hypothesis.
- If full fine-tuning is still poor, discuss backbone/domain mismatch rather than PEFT capacity alone.

### 5. Channel-bridge ablation

Purpose: clarify how 10-band Sentinel-2 inputs are reduced to Prithvi's six-channel template.

Current status:

- The EuroSAT channel-bridge rerun has completed and produced a 12-row JSON plus summary.
- The EuroSAT config now loads `data/weights/prithvi/Prithvi_100M.pt`.
- EuroSAT channel-bridge ablation: completed and manuscript-ready.

Minimum design:

- Compare deterministic truncate/pad bridge against a learned 10->6 projection or a reinitialized 10-channel patch embedding.
- Dataset: EuroSAT `s2_full`.
- Methods: Linear Probe and Houlsby at minimum.
- Seeds: 3.

Decision rule:

- If learned bridge changes the ranking, the current channel-bridge design must become a main limitation.
- If ranking is stable, the current deterministic bridge is defensible.
- Treat the current channel-bridge JSON as the final rerun evidence unless a later rerun changes the numbers.

## Priority B: Recommended If Time Allows

### 6. Macro F1 tables for EuroSAT

The method section currently states that OA and macro F1 are reported. Add macro F1 to the main table or a supplementary table.

### 7. Full or official BigEarthNet replication

The current 10K/5K BigEarthNet subset is acceptable as a compute-bounded validation, but reviewers may ask whether the long tail changes the ranking. At minimum, document the sampling strategy and class distribution.

### 8. Regression or temporal task smoke test

This is not required for the current claim, but it would reduce the limitation that regression and temporal forecasting remain untested.

## Manuscript Text Already Revised

- The title now narrows the contribution to architecture-aware Prithvi-100M adaptation.
- The abstract no longer claims a universal GeoFM PEFT ranking.
- The introduction removes the unqualified "first systematic benchmark" wording.
- The methods define standard LoRA versus split-QKV LoRA.
- The methods define the six-channel bridge boundary.
- The EuroSAT channel-bridge notebook and config are aligned with the staged Prithvi checkpoint, and the rerun evidence is complete.
- The EuroSAT PEFT capacity-sweep notebook and config are prepared, but the result JSON files are not yet manuscript evidence.
- The Linhe section now treats Esri-derived labels as supervisory labels, not independent ground truth.
- The LoveDA threshold is now framed as a supported hypothesis, not a universal rule.
- The conclusion now includes Linhe/LoveDA and names the remaining evidence gaps.

