# Report

LaTeX source for the dimensionality reduction lab report.

## Full project download

Source code, this report folder, and build scripts are bundled on Google Drive:

**[ml-lab-2-code — Google Drive](https://drive.google.com/drive/folders/1o9qE7YEfH3MTnR-EyIGOmkCo16rRXGG9)**

## Build

From the repository root:

```powershell
.\scripts\compile_latex.ps1
```

Or manually from `report/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Output: `report/main.pdf`.

## Layout

```
report/
├── main.tex
├── main.pdf              # after build (or from Drive bundle)
├── content/
│   ├── title.tex
│   ├── preamble.tex
│   ├── introduction.tex
│   ├── fundamental.tex
│   ├── method.tex
│   ├── research_directions.tex
│   ├── experiment_analysis.tex
│   └── conclusion.tex
├── figures/
├── appendix/
└── ref/
    ├── ref.tex
    └── ref.bib
```

Figures for the experiment section can be regenerated with
`code/experiments/report_pca_kpca_isomap.py` (writes to `report/figures/`).
