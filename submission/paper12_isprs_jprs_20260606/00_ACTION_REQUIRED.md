# Action Required Before ISPRS JPRS Submission

These items require author confirmation before the package is uploaded.

## Author and Affiliation

- Confirm the formal author list and order.
- Confirm institutional affiliations.
- Confirm corresponding author name and email.
- Confirm whether the current working author block `Zhouning / AlphaEarth-System Project` should be replaced by formal names.

## Journal Formatting

- Compile `02_latex_source/main_isprs_jprs.tex` and review the output.
- Decide whether the appendix remains inside the manuscript or becomes supplementary material.
- Confirm whether ISPRS JPRS requests graphical abstract, highlights, or both in the active Editorial Manager workflow.

## Data and Reproducibility

- Confirm what part of the Linhe patch corpus can be shared under source-imagery licensing.
- Confirm repository URL or anonymous archive URL for code, configs, and logs.
- Confirm whether all public datasets are cited with the required dataset papers or official URLs.

## Declarations

- Confirm funding statement.
- Confirm competing-interest statement.
- Confirm generative-AI disclosure wording.
- Confirm whether an ethics statement is required for the Linhe operational dataset.

## Technical Checks

- Verify the completed PEFT capacity-sweep artifacts from `colab/paper12_peft_capacity_sweep_colab.ipynb`: `paper12_results/peft_capacity_sweep.json` and `paper12_results/peft_capacity_sweep_summary.json`.
- Confirm both capacity-sweep files are mirrored into `06_supplementary_material/paper12_results/` before final upload.
- Regenerate all figures from the latest result files.
- Reconcile experiment counts in the abstract, introduction, and conclusion against the latest expanded experiment set.
- Check every table value against `paper12_results/summary.csv`, `results/eurosat_results.json`, and the Linhe/LoveDA result files.
- Recheck that the LoveDA table values for the completed two-direction full fine-tuning baseline match `loveda_full_finetune_summary.json` and the two raw direction JSON files.
- Regenerate `paper12_results/review_audit_summary.json` with `python -m geoadapter.bench.paper12_audit` after any result-file change, and mirror it into `06_supplementary_material/paper12_results/`. Confirm schema version 2 includes model-scope, label-source, and decoder-capacity checks.
- Verify the EuroSAT channel-bridge rerun JSON and summary against `paper12_results/eurosat_channel_bridge.json` and `paper12_results/eurosat_channel_bridge_summary.json` before final upload.
- Remove or explain any claims using "first" unless the literature search is up to date.
- Review `REQUIRED_EXPERIMENTS_ISPRS.md` and decide which Priority A experiments will be completed before initial submission.
