# Supplementary Material Package

This directory contains reproducibility artifacts for the Paper 12 ISPRS JPRS submission.

## Contents

- `paper12_results/` - BigEarthNet, LandCover.ai, LoRA ablation, full-finetuning, rank-sensitivity, LoveDA full-finetuning U->R, and EuroSAT channel-bridge result files.
- `results/` - EuroSAT and GeoAdapter result JSON files copied from the AlphaEarth-System working tree.
- `scripts/` - figure-generation and diagnostic scripts copied from the manuscript source tree.

Note: `paper12_results/eurosat_channel_bridge*.json` is retained as archive output from the first Colab run. The config now loads `data/weights/prithvi/Prithvi_100M.pt`, so the channel-bridge ablation should be rerun before these values are cited as manuscript evidence.

## Recommended Supplementary Archive

Zip this directory after confirming that all files are allowed to be shared:

```powershell
Compress-Archive -Path .\06_supplementary_material\* -DestinationPath .\paper12_isprs_jprs_supplementary.zip -Force
```

## Checks Before Upload

- Confirm that all result files cited in the manuscript are present.
- Confirm that `loveda_full_finetune_r2u.json` has been generated before claiming a two-direction LoveDA full fine-tuning baseline.
- Rerun the EuroSAT channel-bridge ablation with the checkpoint-loaded config before moving it into manuscript tables.
- Confirm that any Linhe data-sharing restrictions are documented in the data availability statement.
- Confirm that logs do not expose local machine paths, private tokens, or restricted dataset locations.

