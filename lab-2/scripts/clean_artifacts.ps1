param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

# Common generated directories to remove
$DirPatterns = @(
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    ".cache"
)

# Common generated files to remove (Python + LaTeX)
$FilePatterns = @(
    "*.pyc",
    "*.pyo",
    "*.aux",
    "*.bbl",
    "*.bcf",
    "*.blg",
    "*.fdb_latexmk",
    "*.fls",
    "*.lof",
    "*.log",
    "*.lot",
    "*.nav",
    "*.out",
    "*.run.xml",
    "*.snm",
    "*.synctex.gz",
    "*.toc",
    "*.xdv"
)

Write-Host "Cleaning generated artifacts under: $Root"

$removedDirs = 0
$removedFiles = 0

foreach ($pattern in $DirPatterns) {
    Get-ChildItem -Path $Root -Directory -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq $pattern } |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop
            $removedDirs++
            Write-Host "Removed directory: $($_.FullName)"
        }
}

foreach ($pattern in $FilePatterns) {
    Get-ChildItem -Path $Root -File -Recurse -Force -Filter $pattern -ErrorAction SilentlyContinue |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Stop
            $removedFiles++
            Write-Host "Removed file: $($_.FullName)"
        }
}

Write-Host ""
Write-Host "Done."
Write-Host "Directories removed: $removedDirs"
Write-Host "Files removed: $removedFiles"
