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

- Regenerate all figures from the latest result files.
- Reconcile the conclusion count: the abstract mentions public benchmarks plus production and LoveDA; the conclusion currently says "109 total experiments" and should be checked against the latest expanded experiment count.
- Check every table value against `paper12_results/summary.csv`, `results/eurosat_results.json`, and the Linhe/LoveDA result files.
- Remove or explain any claims using "first" unless the literature search is up to date.
- Review `REQUIRED_EXPERIMENTS_ISPRS.md` and decide which Priority A experiments will be completed before initial submission.
