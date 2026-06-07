# Build Instructions

This directory contains an ISPRS/Elsevier-oriented LaTeX source package.

## Build

Run from this directory:

```powershell
pdflatex -interaction=nonstopmode main_isprs_jprs.tex
bibtex main_isprs_jprs
pdflatex -interaction=nonstopmode main_isprs_jprs.tex
pdflatex -interaction=nonstopmode main_isprs_jprs.tex
```

Expected output:

- `main_isprs_jprs.pdf`

## Source Layout

- `main_isprs_jprs.tex` - Elsevier `elsarticle` entry point for ISPRS JPRS.
- `main_original_article.tex` - original generic article entry point copied from the working draft.
- `sections/` - manuscript body.
- `figures/` - PDF figures used by the manuscript.
- `references.bib` - bibliography database.

## Before Upload

- Confirm author details in `main_isprs_jprs.tex`.
- Confirm appendix placement.
- Confirm that all citations and labels resolve.
- Check overfull boxes in the compiled log.

