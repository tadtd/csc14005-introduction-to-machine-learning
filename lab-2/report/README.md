# Báo cáo

Tệp chính: `main.tex`. Tài liệu tham khảo BibTeX: `ref/ref.bib` (được nạp qua `ref/ref.tex`).

Biên dịch từ thư mục `report/`:

```powershell
pdflatex -jobname=report -interaction=nonstopmode -halt-on-error main.tex
bibtex report
pdflatex -jobname=report -interaction=nonstopmode -halt-on-error main.tex
pdflatex -jobname=report -interaction=nonstopmode -halt-on-error main.tex
```

Kết quả nộp là `report.pdf`.

```text
report/
|-- main.tex
|-- report.pdf
|-- content/
|   |-- title.tex
|   |-- preamble.tex
|   |-- introduction.tex
|   |-- fundamental.tex
|   |-- method.tex
|   |-- experiment_analysis.tex
|   |-- related_research.tex
|   `-- conclusion.tex
|-- figures/
|-- appendix/
`-- ref/
    |-- ref.tex
    `-- ref.bib
```
