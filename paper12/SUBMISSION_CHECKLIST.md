# Submission Checklist

This checklist covers the last-mile tasks between the current draft and a camera-ready submission to a CV / remote sensing venue.

## Build targets

Two entry points share the same `sections/` directory:

- `main.tex` — generic `article` class, 11pt single-column. Use for Overleaf preview, internal review, and arXiv upload.
- `main_cvpr.tex` — CVPR author-kit template, 10pt two-column. Requires `cvpr.sty` + `ieeenat_fullname.bst` from the official CVPR author kit placed next to `main_cvpr.tex` before compiling.

If the target venue is **IEEE JSTARS** or **IEEE TGRS**, copy `main_cvpr.tex` to `main_ieee.tex`, replace the `cvpr` style with `IEEEtran` (`\documentclass[journal]{IEEEtran}`), and drop the workshop-specific preamble.

If the target is **Springer LNCS** (ECCV workshops, ACCV), replace with `\documentclass{llncs}` and the `splncs04` bibliography style.

## Pre-submission checklist

### Content
- [ ] Abstract fits within venue word limit (CVPR: 200–250 words; current: 255 — trim if needed)
- [ ] Paper length fits venue limit (CVPR main: 8 pages + refs; EarthVision workshop: typically 6 pages + refs; IEEE JSTARS: no hard limit)
- [ ] All claims in abstract are supported by a specific table or figure number
- [ ] Every `\cite{}` resolves (no `[?]` in compiled PDF)
- [ ] Every `\ref{}` / `\label{}` resolves (no `??` in compiled PDF)
- [ ] Figures `acc_vs_params.pdf` and `per_modality_oa.pdf` are regenerated from current JSON via `scripts/make_figures.py`
- [ ] Tables in results section match the numbers in `results/eurosat_results.json` and `paper12_results/summary.csv`
- [ ] Appendix is included only if venue permits (CVPR: yes, as supplementary; IEEE JSTARS: merged into main paper)

### Anonymization (for double-blind submission)
- [ ] Author block replaced with anonymous placeholder (done in `main_cvpr.tex`)
- [ ] No self-citation pattern "as we showed in [X]"
- [ ] No GitHub repository URL, institution name, or acknowledgements in main text
- [ ] `.pdf` metadata stripped (`pdftk input.pdf dump_data | grep -i author`)
- [ ] Filenames inside the tarball do not reveal identity

### Reproducibility
- [ ] `references.bib` is canonical (venue + year + DOI where available)
- [ ] Supplementary contains `summary.csv`, per-method JSON, and `make_figures.py`
- [ ] `README.md` documents exact command to reproduce each figure and table
- [ ] Random seeds (42, 123, 456) are documented in Appendix~\ref{app:hparams}
- [ ] LoRA fused-QKV trap (Appendix~\ref{app:lora-trap}) includes enough detail for a third party to reproduce both the failure and the fix

### Ethics / broader impact
- [ ] Confirm EuroSAT and BigEarthNet licenses permit the reported use
- [ ] Add a short "broader impact" paragraph if required (CVPR 2026: optional)
- [ ] Confirm no human-subjects data, no sensitive location leakage

## Known gaps worth addressing before submission

1. **LoRA rank sensitivity**: completed for split-QKV LoRA on EuroSAT s2_full with r∈{4,8,16,32}. The curve is essentially flat (0.706--0.709 OA), which strengthens the claim that the post-fix LoRA ceiling is not explained by an underpowered rank.
2. **Segmentation task**: all current results are classification. A single downstream segmentation benchmark (e.g., Sen1Floods11 or MADOS) would broaden the claim beyond image-level tasks.
3. **Second backbone**: we cite SatMAE, Scale-MAE, SpectralGPT but only run Prithvi. Repeating one modality sweep on SatMAE would convert the "single-backbone" limitation into a "two-backbone, consistent ranking" strength.
4. **Figure 3 (training curves)** is now included in the appendix. If reviewers ask for denser optimization-dynamics evidence, the raw logs in `paper12_results/*.log` can still be expanded into per-epoch plots beyond the current every-10-epoch snapshots.

## Submission-day commands

```bash
# 1. regenerate figures from latest JSON
python paper12/scripts/make_figures.py

# 2. compile the venue-specific main (example: CVPR)
cd paper12
pdflatex main_cvpr
bibtex main_cvpr
pdflatex main_cvpr
pdflatex main_cvpr

# 3. strip metadata
exiftool -all= main_cvpr.pdf

# 4. bundle supplementary
tar czf supplementary.tar.gz \
    sections/appendix.tex \
    figures/*.pdf \
    scripts/make_figures.py \
    ../results/eurosat_results.json \
    ../paper12_results/summary.csv
```

## Recommended venue order

1. **CVPR EarthVision Workshop** — best fit. Audience expects RS + ML negative results; 6-page limit forces sharp writing; no novelty hammer.
2. **IEEE JSTARS** — if EarthVision rejects, expand appendix into main body and submit as full journal article. Editors are receptive to systematic benchmarks.
3. **GIScience & Remote Sensing** — interdisciplinary, slower turnaround, good fit for the "practitioner guidance" framing.

Avoid CVPR main conference: the paper's contribution is primarily empirical/benchmark, which is historically undersold there.
