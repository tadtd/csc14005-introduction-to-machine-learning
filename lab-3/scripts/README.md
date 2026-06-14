# Scripts

This directory contains PowerShell helper scripts for building the LaTeX artifacts and cleaning generated files in this lab.

## Requirements

- Windows PowerShell or PowerShell 7+
- `latexmk` available in `PATH` for the LaTeX build scripts

Run scripts from the project root unless noted otherwise.

## Build the Report

```powershell
scripts\compile_report.ps1
```

By default, this compiles `report/main.tex` into `report/build/main.pdf`.

Optional parameters:

```powershell
scripts\compile_report.ps1 `
  -MainFile main.tex `
  -WorkingDir report `
  -OutputDir build
```

## Build the Slides

```powershell
scripts\compile_slide.ps1
```

By default, this compiles `slide/main.tex` into `slide/build/main.pdf`.

Use `-Clean` to run `latexmk -C` before compiling:

```powershell
scripts\compile_slide.ps1 -Clean
```

## Clear Generated Artifacts

Preview the cleanup without deleting files:

```powershell
scripts\clear_artifacts.ps1 -WhatIf
```

Run the default cleanup:

```powershell
scripts\clear_artifacts.ps1
```

The default cleanup removes common generated artifacts only:

- LaTeX build directories and auxiliary files under `report/` and `slide/`
- Python cache directories such as `__pycache__`
- Python bytecode files such as `.pyc` and `.pyo`
- Tool caches such as `.uv-cache`, `.pytest_cache`, `.mypy_cache`, and `.ruff_cache`

Additional cleanup options are opt-in:

```powershell
# Also remove virtual environments such as .venv
scripts\clear_artifacts.ps1 -IncludeVirtualEnvs

# Also remove local data directories
scripts\clear_artifacts.ps1 -IncludeData

# Also clear result/output directories while preserving .gitkeep files
scripts\clear_artifacts.ps1 -IncludeResults
```

All cleanup paths are checked before deletion and must stay inside the project root.
