# Paper 12 LaTeX Draft

**Title**: *How Well Do PEFT Methods Adapt Geospatial Foundation Models to Heterogeneous Inputs? A Systematic Evaluation on Prithvi-100M*

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

- `paper12_results/loveda_full_finetune_u2r.json` and `paper12_results/loveda_full_finetune_summary.json` - LoveDA U->R full fine-tuning baseline (3 seeds; R->U still open)
- `paper12_results/eurosat_channel_bridge.json` and `paper12_results/eurosat_channel_bridge_summary.json` - EuroSAT channel-bridge archive output. The config has since been fixed to load `data/weights/prithvi/Prithvi_100M.pt`; rerun before using these values as manuscript evidence.

## Status

Current state: complete second-pass academic polish with formal citations, mechanism-level Discussion, full per-seed Appendix, and CVPR template variant. See `SUBMISSION_CHECKLIST.md` for remaining tasks before submission.
