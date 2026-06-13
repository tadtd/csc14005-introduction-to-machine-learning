# scripts/compile-latex.ps1

param (
    [string]$MainFile = "main.tex",
    [string]$WorkingDir = "report",
    [string]$OutputDir = "build"
)

$ErrorActionPreference = "Stop"

Write-Host "Compiling LaTeX project..."

Push-Location $WorkingDir

try {

    if (!(Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir | Out-Null
    }

    # Clean previous failed state
    latexmk -C -outdir="$OutputDir"

    # Compile
    latexmk `
        -pdf `
        -interaction=nonstopmode `
        -synctex=1 `
        -outdir="$OutputDir" `
        "$MainFile"

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Build successful!"
        Write-Host "PDF: $WorkingDir/$OutputDir/main.pdf"
    }
    else {
        Write-Host ""
        Write-Host "Build failed!"
        exit 1
    }
}
finally {
    Pop-Location
}