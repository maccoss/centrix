<#
.SYNOPSIS
    Runs a built or installed centrix on ProteoWizard's Thermo .raw test file, so a build or a package is shown to read
    a vendor file end to end: calibrate, convert, and check the mzML it writes.
.PARAMETER Centrix
    The centrix executable to run.
.PARAMETER PwizRoot
    The ProteoWizard tree holding pwiz/data/vendor_readers/Thermo/Reader_Thermo_Test.data.tar.bz2; .\pwiz by default.
#>
#Requires -Version 7.0
param(
    [Parameter(Mandatory)] [string]$Centrix,
    [string]$PwizRoot = (Join-Path (Split-Path $PSScriptRoot -Parent) 'pwiz')
)
$ErrorActionPreference = 'Stop'

$archive = Join-Path $PwizRoot 'pwiz/data/vendor_readers/Thermo/Reader_Thermo_Test.data.tar.bz2'
if (-not (Test-Path $archive)) { throw "No Thermo test archive at $archive" }
$work = Join-Path ([System.IO.Path]::GetTempPath()) ('centrix-smoke-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $work | Out-Null
try {
    # Windows' own tar (bsdtar), not one a Git or MSYS PATH puts first, which reads C:\ as a remote host.
    $tar = $IsWindows ? (Join-Path $env:SystemRoot 'System32/tar.exe') : 'tar'
    & $tar -xjf $archive -C $work 'Reader_Thermo_Test.data/090701-LTQVelos-unittest-01.raw'
    if ($LASTEXITCODE -ne 0) { throw 'Could not extract the Thermo test file' }
    $raw = Join-Path $work 'Reader_Thermo_Test.data/090701-LTQVelos-unittest-01.raw'

    & $Centrix --version
    if ($LASTEXITCODE -ne 0) { throw 'centrix --version failed' }
    & $Centrix calibrate $raw
    if ($LASTEXITCODE -ne 0) { throw 'centrix calibrate failed' }
    & $Centrix $raw -o $work
    if ($LASTEXITCODE -ne 0) { throw 'centrix failed to convert the .raw' }

    $output = Join-Path $work '090701-LTQVelos-unittest-01.centrix.mzML'
    $text = Get-Content -Raw $output
    $spectra = ([regex]::Matches($text, '<spectrum ')).Count
    $profile = ([regex]::Matches($text, 'MS:1000128')).Count
    if ($spectra -ne 85) { throw "Expected 85 spectra in $output, found $spectra" }
    if ($profile -ne 0) { throw "$profile spectra of $output are still profile" }
    if ($text -notmatch 'Centrix centroiding') { throw "No Centrix processing record in $output" }
    if (-not $text.TrimEnd().EndsWith('</indexedmzML>')) { throw "$output is not a complete indexed mzML" }

    # --centroid vendor writes Thermo's own centroids, under a filter too: the vendor's peak picker must wrap the
    # reader's list, not Centrix's wrappers, or it falls back to pwiz's wavelet picker.
    $vendorDir = Join-Path $work 'vendor'
    & $Centrix $raw -o $vendorDir --centroid vendor --filter 'msLevel 1-'
    if ($LASTEXITCODE -ne 0) { throw 'centrix --centroid vendor failed to convert the .raw' }
    $vendorText = Get-Content -Raw (Join-Path $vendorDir '090701-LTQVelos-unittest-01.centrix.mzML')
    if ($vendorText -notmatch 'Thermo/Xcalibur peak picking') { throw "--centroid vendor did not write Thermo's centroids" }
    if (([regex]::Matches($vendorText, 'MS:1000128')).Count -ne 0) { throw '--centroid vendor left profile spectra' }
    Write-Host "Smoke test passed: $spectra spectra centroided, and Thermo's centroids with --centroid vendor"
}
finally {
    Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
}
