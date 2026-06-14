[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [switch]$IncludeVirtualEnvs,
    [switch]$IncludeData,
    [switch]$IncludeResults
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$RootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
)

$Removed = New-Object System.Collections.Generic.List[string]
$Skipped = New-Object System.Collections.Generic.List[string]

function Resolve-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return [System.IO.Path]::GetFullPath($Path)
}

function Test-IsInsideRoot {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = (Resolve-FullPath $Path).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )

    if ($fullPath -eq $RootFull) {
        return $true
    }

    $rootPrefix = $RootFull + [System.IO.Path]::DirectorySeparatorChar
    return $fullPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-RelativePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = Resolve-FullPath $Path
    $rootUri = [System.Uri]::new($RootFull + [System.IO.Path]::DirectorySeparatorChar)
    $pathUri = [System.Uri]::new($fullPath)
    $relativeUri = $rootUri.MakeRelativeUri($pathUri)
    return [System.Uri]::UnescapeDataString($relativeUri.ToString()).Replace(
        '/',
        [System.IO.Path]::DirectorySeparatorChar
    )
}

function Test-IsInExcludedTree {
    param([Parameter(Mandatory = $true)][string]$Path)

    $relative = Get-RelativePath $Path
    $parts = $relative -split '[\\/]'
    $excludedNames = @('.git', '.venv', 'venv', 'env', 'ENV', '.uv-cache')

    foreach ($part in $parts) {
        if ($excludedNames -contains $part) {
            return $true
        }
    }

    return $false
}

function Remove-ArtifactPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $fullPath = Resolve-FullPath $Path
    if (-not (Test-IsInsideRoot $fullPath)) {
        $Skipped.Add($fullPath)
        Write-Warning "Skipped path outside project root: $fullPath"
        return
    }

    $relative = Get-RelativePath $fullPath
    if ($PSCmdlet.ShouldProcess($relative, 'Remove artifact')) {
        Remove-Item -LiteralPath $fullPath -Recurse -Force -ErrorAction Stop
        $Removed.Add($relative)
    }
}

function Remove-ChildrenExceptGitKeep {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return
    }

    Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne '.gitkeep' } |
        ForEach-Object { Remove-ArtifactPath $_.FullName }
}

Write-Host "Cleaning artifacts under: $RootFull"

$fixedArtifactDirs = @(
    'report/build',
    'slide/build',
    '.uv-cache',
    'probabilistic-circuit/.uv-cache',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache'
)

foreach ($relativePath in $fixedArtifactDirs) {
    Remove-ArtifactPath (Join-Path $RootFull $relativePath)
}

$artifactDirectoryNames = @(
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    '.ipynb_checkpoints'
)

Get-ChildItem -LiteralPath $RootFull -Directory -Recurse -Force -ErrorAction SilentlyContinue |
    Where-Object {
        ($artifactDirectoryNames -contains $_.Name) -and
        (-not (Test-IsInExcludedTree $_.FullName))
    } |
    ForEach-Object { Remove-ArtifactPath $_.FullName }

Get-ChildItem -LiteralPath $RootFull -File -Recurse -Force -ErrorAction SilentlyContinue |
    Where-Object {
        ($_.Extension -in @('.pyc', '.pyo')) -and
        (-not (Test-IsInExcludedTree $_.FullName))
    } |
    ForEach-Object { Remove-ArtifactPath $_.FullName }

$latexBases = @('report', 'slide')
$latexPatterns = @(
    '*.aux',
    '*.log',
    '*.out',
    '*.toc',
    '*.lof',
    '*.lot',
    '*.bbl',
    '*.blg',
    '*.brf',
    '*.fls',
    '*.fdb_latexmk',
    '*.synctex.gz',
    '*.synctex(busy)'
)

foreach ($base in $latexBases) {
    $basePath = Join-Path $RootFull $base
    if (-not (Test-Path -LiteralPath $basePath -PathType Container)) {
        continue
    }

    foreach ($pattern in $latexPatterns) {
        Get-ChildItem -LiteralPath $basePath -Filter $pattern -File -Recurse -Force -ErrorAction SilentlyContinue |
            ForEach-Object { Remove-ArtifactPath $_.FullName }
    }
}

if ($IncludeVirtualEnvs) {
    $virtualEnvDirs = @(
        '.venv',
        'venv',
        'env',
        'ENV',
        'deep-learning/.venv',
        'data-mining/.venv',
        'probabilistic-circuit/.venv'
    )

    foreach ($relativePath in $virtualEnvDirs) {
        Remove-ArtifactPath (Join-Path $RootFull $relativePath)
    }
}

if ($IncludeData) {
    $dataDirs = @(
        'data',
        'data-mining/data',
        'deep-learning/data',
        'probabilistic-circuit/data'
    )

    foreach ($relativePath in $dataDirs) {
        Remove-ArtifactPath (Join-Path $RootFull $relativePath)
    }
}

if ($IncludeResults) {
    $resultDirsToClear = @(
        'deep-learning/results',
        'deep-learning/results/runs',
        'deep-learning/results/plots',
        'data-mining/results',
        'probabilistic-circuit/results',
        'probabilistic-circuit/outputs'
    )

    foreach ($relativePath in $resultDirsToClear) {
        Remove-ChildrenExceptGitKeep (Join-Path $RootFull $relativePath)
    }
}

Write-Host ''
Write-Host "Removed artifacts: $($Removed.Count)"
foreach ($path in $Removed) {
    Write-Host "  - $path"
}

if ($Skipped.Count -gt 0) {
    Write-Host ''
    Write-Host "Skipped unsafe paths: $($Skipped.Count)"
    foreach ($path in $Skipped) {
        Write-Host "  - $path"
    }
}
