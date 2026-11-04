<#
.SYNOPSIS
    Fetches the ProteoWizard commit pinned in pwiz.lock into .\pwiz.
.DESCRIPTION
    Shallow (that commit only), blobless and sparse (scripts\pwiz-sparse-paths.txt), so only the
    library sources Centrix builds from are downloaded. Rerun after changing pwiz.lock; an
    existing .\pwiz is moved to the new commit.
.PARAMETER Target
    Where to put the clone; .\pwiz by default.
#>
param(
    [string]$Target = (Join-Path (Split-Path $PSScriptRoot -Parent) 'pwiz')
)
$ErrorActionPreference = 'Stop'

$root = Split-Path $PSScriptRoot -Parent
$lock = @{}
foreach ($line in Get-Content (Join-Path $root 'pwiz.lock')) {
    if ($line -match '^\s*([a-z]+)=(.+)$') { $lock[$Matches[1]] = $Matches[2].Trim() }
}
if (-not $lock.repository -or -not $lock.commit) { throw 'pwiz.lock must name a repository and a commit' }

function Invoke-Git { & git -C $Target @args; if ($LASTEXITCODE -ne 0) { throw "git $args failed" } }

if (-not (Test-Path (Join-Path $Target '.git'))) {
    New-Item -ItemType Directory -Force $Target | Out-Null
    Invoke-Git init --quiet
    Invoke-Git remote add origin $lock.repository
}
Invoke-Git config core.longpaths true
Invoke-Git sparse-checkout init --no-cone
$patterns = Get-Content (Join-Path $PSScriptRoot 'pwiz-sparse-paths.txt') | Where-Object { $_ -and $_ -notmatch '^\s*#' }
[System.IO.File]::WriteAllLines((Join-Path $Target '.git/info/sparse-checkout'), [string[]]$patterns)
$head = & git -C $Target rev-parse --verify --quiet HEAD 2>$null
if ($head -eq $lock.commit) {
    # The paths may have changed.
    Invoke-Git sparse-checkout reapply
}
else {
    Write-Host "Fetching pwiz $($lock.commit)"
    Invoke-Git fetch --quiet --depth 1 --filter=blob:none origin $lock.commit
    Invoke-Git -c advice.detachedHead=false checkout --quiet --force $lock.commit
}
Write-Host "pwiz is at $($lock.commit)"
