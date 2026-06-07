# Supplementary Material Package

This directory contains reproducibility artifacts for the Paper 12 ISPRS JPRS submission.

## Contents

- `paper12_results/` - BigEarthNet, LandCover.ai, LoRA ablation, full-finetuning, and rank-sensitivity result files and logs.
- `results/` - EuroSAT and GeoAdapter result JSON files copied from the AlphaEarth-System working tree.
- `scripts/` - figure-generation and diagnostic scripts copied from the manuscript source tree.

## Recommended Supplementary Archive

Zip this directory after confirming that all files are allowed to be shared:

```powershell
Compress-Archive -Path .\06_supplementary_material\* -DestinationPath .\paper12_isprs_jprs_supplementary.zip -Force
```

## Checks Before Upload

- Confirm that all result files cited in the manuscript are present.
- Confirm that any Linhe data-sharing restrictions are documented in the data availability statement.
- Confirm that logs do not expose local machine paths, private tokens, or restricted dataset locations.

