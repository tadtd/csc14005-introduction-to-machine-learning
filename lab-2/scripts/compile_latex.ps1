# Compile report/main.tex to report/main.pdf
# Prefer latexmk if available, otherwise fallback to pdflatex + bibtex.

$ErrorActionPreference = "Stop"

$ReportDir = (Resolve-Path (Join-Path $PSScriptRoot (Join-Path ".." "report")))
$MainTex = "main.tex"
$PdfPath = Join-Path $ReportDir "main.pdf"

function Invoke-Latex {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args
    )

    & $Exe @Args

    if ($LASTEXITCODE -ne 0) {
        throw "$Exe failed with exit code $LASTEXITCODE"
    }
}

Push-Location $ReportDir

try {
    Write-Host "Compiling $MainTex in $ReportDir ..."

    # Use latexmk if available
    if (Get-Command latexmk -ErrorAction SilentlyContinue) {

        Write-Host "Using latexmk ..."

        Invoke-Latex `
            -Exe latexmk `
            -pdf `
            -interaction=nonstopmode `
            -halt-on-error `
            $MainTex

    }
    else {

        Write-Warning "latexmk not found. Falling back to pdflatex + bibtex."

        if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
            throw "pdflatex not found. Install MiKTeX or TeX Live and add it to PATH."
        }

        Write-Host "pdflatex (pass 1) ..."
        Invoke-Latex `
            -Exe pdflatex `
            -interaction=nonstopmode `
            -halt-on-error `
            $MainTex

        if (Get-Command bibtex -ErrorAction SilentlyContinue) {

            Write-Host "bibtex ..."
            Invoke-Latex -Exe bibtex main

        }
        else {
            Write-Warning "bibtex not found; skipping bibliography pass."
        }

        foreach ($i in 2..3) {

            Write-Host "pdflatex (pass $i) ..."

            Invoke-Latex `
                -Exe pdflatex `
                -interaction=nonstopmode `
                -halt-on-error `
                $MainTex
        }
    }

    Write-Host "Done: $PdfPath"

}
finally {
    Pop-Location
}