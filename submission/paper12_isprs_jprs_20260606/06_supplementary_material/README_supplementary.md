# Supplementary Material Package

This directory contains reproducibility artifacts for the Paper 12 ISPRS JPRS submission.

## Contents

- `paper12_results/` - BigEarthNet, LandCover.ai, LoRA ablation, full-finetuning, rank-sensitivity, LoveDA two-direction full-finetuning, and EuroSAT channel-bridge result files.
- `results/` - EuroSAT and GeoAdapter result JSON files copied from the AlphaEarth-System working tree.
- `scripts/` - figure-generation and diagnostic scripts copied from the manuscript source tree.

Note: `paper12_results/eurosat_channel_bridge*.json` is the rerun output after the checkpoint-path fix and can be cited as manuscript evidence once the package is uploaded.

Pending: the prepared PEFT capacity sweep writes `paper12_results/peft_capacity_sweep.json` and `paper12_results/peft_capacity_sweep_summary.json` from `colab/paper12_peft_capacity_sweep_colab.ipynb`. Add and mirror those files only after the Colab run completes and the 30-row count is verified.

## Recommended Supplementary Archive

Zip this directory after confirming that all files are allowed to be shared:

```powershell
Compress-Archive -Path .\06_supplementary_material\* -DestinationPath .\paper12_isprs_jprs_supplementary.zip -Force
```

## Checks Before Upload

- Confirm that all result files cited in the manuscript are present.
- If the PEFT capacity sweep is cited, confirm that `peft_capacity_sweep.json` and `peft_capacity_sweep_summary.json` are present and synchronized with the manuscript tables.
- Confirm that `loveda_full_finetune_u2r.json`, `loveda_full_finetune_r2u.json`, and `loveda_full_finetune_summary.json` remain synchronized before citing the two-direction LoveDA full fine-tuning baseline.
- Confirm that the EuroSAT channel-bridge rerun JSON and summary are mirrored into the supplementary package before upload.
- Confirm that any Linhe data-sharing restrictions are documented in the data availability statement.
- Confirm that logs do not expose local machine paths, private tokens, or restricted dataset locations.

