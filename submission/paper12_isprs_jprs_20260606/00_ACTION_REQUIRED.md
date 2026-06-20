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

- Run the prepared PEFT capacity-sweep notebook if this experiment will be included before submission: `colab/paper12_peft_capacity_sweep_colab.ipynb`.
- After the capacity sweep completes, copy back and verify `/content/drive/MyDrive/paper12_results/peft_capacity_sweep.json` and `/content/drive/MyDrive/paper12_results/peft_capacity_sweep_summary.json`; do not cite the capacity curve as completed evidence until those files are present and checked.
- Regenerate all figures from the latest result files.
- Reconcile experiment counts in the abstract, introduction, and conclusion against the latest expanded experiment set.
- Check every table value against `paper12_results/summary.csv`, `results/eurosat_results.json`, and the Linhe/LoveDA result files.
- Recheck that the LoveDA table values for the completed two-direction full fine-tuning baseline match `loveda_full_finetune_summary.json` and the two raw direction JSON files.
- Verify the EuroSAT channel-bridge rerun JSON and summary against `paper12_results/eurosat_channel_bridge.json` and `paper12_results/eurosat_channel_bridge_summary.json` before final upload.
- Remove or explain any claims using "first" unless the literature search is up to date.
- Review `REQUIRED_EXPERIMENTS_ISPRS.md` and decide which Priority A experiments will be completed before initial submission.
