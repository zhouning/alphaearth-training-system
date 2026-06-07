# Paper 12 ISPRS JPRS Submission Package

Target journal: ISPRS Journal of Photogrammetry and Remote Sensing.

This package reorganizes the current Paper 12 manuscript for journal submission without modifying the working draft in `AlphaEarth-System/paper12`.

## Directory Map

- `01_manuscript_pdf/` - current compiled manuscript PDF copied from the working paper.
- `02_latex_source/` - ISPRS/Elsevier-oriented LaTeX source package.
- `03_cover_letter/` - cover letter draft for ISPRS JPRS.
- `04_declarations/` - declaration, data availability, funding, ethics, author contribution, and AI-use statement drafts.
- `05_highlights_abstract_keywords/` - Elsevier-style highlights, plain-text abstract, and keywords.
- `06_supplementary_material/` - appendix source, result files, logs, and figure-generation scripts.
- `00_ACTION_REQUIRED.md` - items that must be confirmed before submission.
- `SUBMISSION_CHECKLIST_ISPRS_JPRS.md` - practical pre-submission checklist.

## Recommended Submission File Order

1. Main manuscript PDF: `01_manuscript_pdf/paper12_isprs_jprs_current_compiled.pdf`
2. LaTeX source archive: zip the contents of `02_latex_source/`
3. Cover letter: `03_cover_letter/cover_letter_isprs_jprs.md`
4. Highlights: `05_highlights_abstract_keywords/highlights.md`
5. Abstract and keywords: `05_highlights_abstract_keywords/abstract_plain_text.md`, `keywords.md`
6. Declarations: files under `04_declarations/`
7. Supplementary material: zip the contents of `06_supplementary_material/`

## Notes

The ISPRS JPRS LaTeX entry point is `02_latex_source/main_isprs_jprs.tex`. It uses the Elsevier `elsarticle` class and the existing section files. The original article-class entry point is retained as `02_latex_source/main_original_article.tex` for traceability.

