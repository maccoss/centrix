<#
.SYNOPSIS
    Builds Centrix, and with -RunTests tests it: fetches the pinned ProteoWizard tree (pwiz.lock)
    unless PWIZ_ROOT names one, then builds Centrix.sln.
.PARAMETER Configuration
    Debug or Release (default).
.PARAMETER IAgreeToVendorLicenses
    Build the vendor readers (Thermo .raw and the rest), agreeing to the vendors' licenses
    (pwiz/pwiz/data/vendor_readers/*/EULA.*.txt). Without it centrix reads open formats only.
.PARAMETER RunTests
    Run the tests after building.
#>
#Requires -Version 7.0
param(
    [ValidateSet('Debug', 'Release')] [string]$Configuration = 'Release',
    [switch]$IAgreeToVendorLicenses,
    [switch]$RunTests
)
$ErrorActionPreference = 'Stop'
if (-not $env:PWIZ_ROOT) { & (Join-Path $PSScriptRoot 'scripts/get-pwiz.ps1') }
$arguments = @('-c', $Configuration, '-p:Platform=x64', '--nologo')
if ($IAgreeToVendorLicenses) { $arguments += '-p:IAgreeToVendorLicenses=true' }
$solution = Join-Path $PSScriptRoot 'Centrix.sln'
& dotnet build $solution @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($RunTests) {
    & dotnet test $solution @arguments --no-build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$exe = if ($IsWindows) { 'centrix.exe' } else { 'centrix' }
Write-Host "Built $(Join-Path $PSScriptRoot "src/Centrix/bin/x64/$Configuration/net10.0/$exe")"
