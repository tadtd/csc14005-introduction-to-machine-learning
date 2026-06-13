# scripts/compile_slide.ps1

param (
    [string]$MainFile = "main.tex",
    [string]$WorkingDir = "slide",
    [string]$OutputDir = "build",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $WorkingDir)) {
    throw "Slide directory not found: $WorkingDir"
}

Write-Host "Compiling Beamer slides..."
Write-Host "WorkingDir: $WorkingDir"
Write-Host "MainFile:   $MainFile"

Push-Location $WorkingDir

try {
    if (!(Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir | Out-Null
    }

    if ($Clean) {
        latexmk -C -outdir="$OutputDir"
    }

    latexmk `
        -pdf `
        -interaction=nonstopmode `
        -synctex=1 `
        -outdir="$OutputDir" `
        "$MainFile"

    if ($LASTEXITCODE -ne 0) {
        throw "Slide build failed."
    }

    $pdfPath = Join-Path $WorkingDir (Join-Path $OutputDir ([System.IO.Path]::ChangeExtension($MainFile, ".pdf")))
    Write-Host ""
    Write-Host "Build successful!"
    Write-Host "PDF: $pdfPath"
}
finally {
    Pop-Location
}
