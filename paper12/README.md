# Paper 12 LaTeX Draft

**Title**: *Architecture-Aware Diagnosis of Parameter-Efficient Adaptation in Prithvi-100M: Fused-QKV Failure, Capacity Boundaries, and Cross-Domain Remote-Sensing Validation*

## Structure

```
paper12/
├── main.tex                  # generic article (Overleaf preview / arXiv)
├── main_cvpr.tex             # CVPR author-kit two-column
├── references.bib            # 19 canonical citations
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── method.tex
│   ├── results.tex           # 5-modality main tables + 2 figure refs
│   ├── discussion.tex        # mechanism, methodology, limitations
│   ├── conclusion.tex
│   └── appendix.tex          # per-seed tables, hyperparams, fused-QKV trap
├── figures/
│   ├── acc_vs_params.pdf     # generated
│   └── per_modality_oa.pdf   # generated
├── scripts/
│   └── make_figures.py       # regenerates figures/ from results JSON
└── SUBMISSION_CHECKLIST.md   # pre-submission checklist + venue notes
```

## Build

### Generic preview
```bash
cd paper12
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### CVPR / EarthVision Workshop
Drop `cvpr.sty` and `ieeenat_fullname.bst` from the official author kit next to `main_cvpr.tex`, then:
```bash
pdflatex main_cvpr
bibtex main_cvpr
pdflatex main_cvpr
pdflatex main_cvpr
```

### Regenerate figures from latest experiment JSON
```bash
python paper12/scripts/make_figures.py
```

## Data sources

All numerical values in tables and figures trace back to:
- `results/eurosat_results.json` — EuroSAT per-seed records (linear probe, BitFit, LoRA, Houlsby, GeoAdapter v2 across 5 modalities)
- `paper12_results/summary.csv` — BigEarthNet-S2 per-seed mAP + LoRA Split-QKV ablation + full fine-tuning ceiling
- `paper12_results/full_finetune_20260421_1202.json` — full fine-tuning baseline (single seed)
- `paper12_results/lora_ablation_20260421_1240.json` — split-QKV diagnostic ablation

- `paper12_results/loveda_full_finetune_u2r.json`, `paper12_results/loveda_full_finetune_r2u.json`, and `paper12_results/loveda_full_finetune_summary.json` - LoveDA two-direction full fine-tuning baseline (3 seeds per direction)
- `paper12_results/eurosat_channel_bridge.json` and `paper12_results/eurosat_channel_bridge_summary.json` - EuroSAT channel-bridge rerun output after the checkpoint-path fix. These values are now manuscript-ready evidence.
- `paper12_results/peft_capacity_sweep.json` and `paper12_results/peft_capacity_sweep_summary.json` - completed EuroSAT parameter-capacity sweep outputs, mirrored from `colab/paper12_peft_capacity_sweep_colab.ipynb` and ready to cite.
- `paper12_results/second_backbone_eurosat.json` and
  `paper12_results/second_backbone_eurosat_summary.json` - SatMAE-compatible
  second-backbone EuroSAT validation outputs, mirrored from
  `colab/paper12_second_backbone_eurosat_colab.ipynb` and ready to cite as
  bounded two-backbone evidence.

## GeoVLM prompt segmentation MVP

Implementation status: the offline model, runner, checkpoint contract, Colab
workflow, and inference CLI are available. Evidence status: incomplete. No real
three-seed LandCoverAI result has been accepted or added to the manuscript.

The validated target scope is limited to English prompts for `building`,
`road`, and `water`. The source LandCoverAI taxonomy remains the official five
classes: `0 background`, `1 building`, `2 woodland`, `3 water`, and `4 road`.
The model path is Prithvi-100M with checkpoint positional embeddings, frozen
base weights, Houlsby adapters, a frozen SigLIP text tower, FiLM conditioning,
dense image-text similarity, and a binary decoder. The no-text comparison uses
three binary heads with the same visual adaptation path.

Focused local verification:

```bash
python -m pytest tests/test_prompt_segmentation_data.py tests/test_prithvi_position_embeddings.py tests/test_prompt_segmentation_model.py tests/test_prompt_segmentation_engine.py tests/test_geovlm_prompt_summary.py tests/test_geovlm_prompt_runner.py tests/test_geovlm_prompt_inference.py tests/test_paper12_colab_notebooks.py -v
```

Real execution is defined by
`colab/paper12_geovlm_prompt_segmentation_colab.ipynb`. It persists checkpoints
under `/content/drive/MyDrive/paper12_checkpoints/geovlm_prompt_segmentation`,
previews under `/content/drive/MyDrive/paper12_previews/geovlm_prompt_segmentation`,
and result JSON files under `/content/drive/MyDrive/paper12_results`.

Seed-42 smoke run:

```bash
python -m geoadapter.bench.run_geovlm_prompt_segmentation --config geoadapter/bench/configs/geovlm_prompt_segmentation.yaml --output paper12_results/geovlm_prompt_segmentation.json --summary-output paper12_results/geovlm_prompt_segmentation_summary.json --checkpoint-dir /path/to/checkpoints --preview-dir /path/to/previews --stage seed42
```

Full prompt/baseline matrix after the smoke checks pass:

```bash
python -m geoadapter.bench.run_geovlm_prompt_segmentation --config geoadapter/bench/configs/geovlm_prompt_segmentation.yaml --output paper12_results/geovlm_prompt_segmentation.json --summary-output paper12_results/geovlm_prompt_segmentation_summary.json --checkpoint-dir /path/to/checkpoints --preview-dir /path/to/previews --stage full
```

Rebuild the acceptance summary:

```bash
python -m geoadapter.bench.geovlm_prompt_summary --input paper12_results/geovlm_prompt_segmentation.json --output paper12_results/geovlm_prompt_segmentation_summary.json --bootstrap-iterations 1000
```

Offline checkpoint inference:

```bash
python scripts/run_geovlm_prompt_segmentation.py --image sample.tif --prompt "segment all water bodies" --checkpoint /path/to/checkpoint.pt --output-dir results/geovlm_prompt_inference --threshold 0.5 --local-files-only
```

The MVP must remain `incomplete` until the real prompt and baseline matrix
contains all six method/seed pairs and the three-seed prompt rows pass every
IoU, held-out retention, counterfactual sensitivity, probability-change, and
checkpoint-reproduction gate. See `docs/geovlm_prompt_segmentation_mvp.md` for
the complete evidence contract.

## Status

Current state: ISPRS JPRS revision framing is updated around architecture-aware PEFT diagnosis. EuroSAT channel-bridge, LoveDA full fine-tuning, the completed PEFT capacity sweep, and the SatMAE-compatible second-backbone evidence are mirrored; broader multi-backbone generalization remains bounded rather than universal. See `SUBMISSION_CHECKLIST.md` and `submission/paper12_isprs_jprs_20260606/REQUIRED_EXPERIMENTS_ISPRS.md` for remaining tasks before submission.

Second-backbone validation is complete and manuscript-ready after the 18-row Colab run, local mirroring, supplementary mirroring, and review-audit regeneration.
