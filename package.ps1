<#
.SYNOPSIS
    Packages Centrix for Windows and Linux: a self-contained folder per runtime, zipped (Windows) or
    tarred (Linux), and on Windows an installer.

.DESCRIPTION
    For each runtime identifier, `dotnet publish --self-contained` of src/Centrix into a folder named
    centrix-<version>-<rid>, which needs no .NET on the machine it runs on, then:
      * win-x64:   centrix-<version>-win-x64.zip, and with -Setup Centrix-Setup-<version>.exe (Inno
                   Setup 6: per-user or per-machine, optional PATH entry; installer\Setup.iss).
      * linux-x64: centrix-<version>-linux-x64.tar.gz, written with the executable bit on the
                   centrix launcher, from either operating system, with install.sh, which installs it
                   under ~/.local (or --prefix) and links bin/centrix to it.

    The vendor SDKs are compiled in (-IAgreeToVendorLicenses) but left out of the package, as the
    ProteoWizard installers leave them out: their licenses do not allow redistributing them. The
    first time a package reads a vendor file, ProteoWizard's VendorSdkLoader downloads that SDK,
    pinned to the ProteoWizard commit in pwiz.lock and hash-checked, into a per-user cache. With
    -BundleVendorSdks the SDKs stay in the package, for machines without internet access (an HPC
    node); such a package is for your own use, not for redistribution. Their EULAs ship either way.

    Runs on PowerShell 7 on Windows or Linux. Fetches the pinned ProteoWizard tree first.

.PARAMETER Rid
    Runtime identifiers to package. Default: win-x64 and linux-x64 (each cross-builds from either OS).

.PARAMETER IAgreeToVendorLicenses
    Build with the vendor readers (Thermo .raw, Bruker .d, SCIEX .wiff, ...). Without it the
    package reads open formats only (mzML, mzXML, mz5, mzMLb, MGF).

.PARAMETER BundleVendorSdks
    Keep the vendor SDKs in the package rather than fetching them on first use.

.PARAMETER Setup
    Also build the Windows installer; needs Inno Setup 6 (ISCC.exe on PATH or in its default folder).

.PARAMETER OutputDir
    Where the packages go; dist\ by default.
#>
#Requires -Version 7.0
param(
    [string[]]$Rid = @('win-x64', 'linux-x64'),
    [switch]$IAgreeToVendorLicenses,
    [switch]$BundleVendorSdks,
    [switch]$Setup,
    [string]$OutputDir
)
$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
if (-not $OutputDir) { $OutputDir = Join-Path $root 'dist' }
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$staging = Join-Path $OutputDir '_staging'
$project = Join-Path $root 'src/Centrix/Centrix.csproj'
$pwiz = if ($env:PWIZ_ROOT) { $env:PWIZ_ROOT } else { Join-Path $root 'pwiz' }

if (-not $env:PWIZ_ROOT) { & (Join-Path $root 'scripts/get-pwiz.ps1') }

# The version: Directory.Build.props's, the same the executable reports.
$version = ([xml](Get-Content (Join-Path $root 'Directory.Build.props'))).Project.PropertyGroup.Version |
    Where-Object { $_ } | Select-Object -First 1
if (-not $version) { throw 'No <Version> in Directory.Build.props' }

# Vendor SDK files, by name prefix: ProteoWizard's own table, which its installers strip by.
$vendorPrefixes = (Get-Content -Raw (Join-Path $pwiz 'build/vendor-sdk-pins.json') | ConvertFrom-Json).vendors |
    ForEach-Object { $_.prefixes } | Sort-Object -Unique
$satellites = 'cs', 'de', 'es', 'fr', 'it', 'ja', 'ko', 'pl', 'pt-BR', 'ru', 'tr', 'zh-Hans', 'zh-Hant'

function Test-VendorFile([string]$leaf) {
    if ($leaf -notmatch '\.(dll|exe|manifest|so|dylib)(\.\d+)*$') { return $false }
    $base = if ($leaf.StartsWith('lib')) { $leaf.Substring(3) } else { $leaf }
    foreach ($p in $vendorPrefixes) {
        if ($leaf -like "$p*" -or $base -like "$p*") { return $true }
    }
    return $false
}

# What a package leaves out: debug symbols and XML docs, localization satellites, the other OS's
# natives and launchers, and (unless bundled) the vendor SDKs.
function Test-Skip([string]$relative, [bool]$windows) {
    $leaf = Split-Path -Leaf $relative
    if ($relative -match '\.(pdb|xml)$') { return $true }
    if ($relative -match ('^(' + ($satellites -join '|') + ')[\\/]')) { return $true }
    if ($windows) {
        if ($relative -match '\.(so|dylib|a)(\.\d+)*$') { return $true }
        if ($leaf -in @('7zz', 'centrix')) { return $true }
        if ($relative -match '^runtimes[\\/](?!win-x64[\\/]|win[\\/])') { return $true }
    }
    else {
        if ($leaf -in @('7za.exe', 'centrix.exe')) { return $true }
        if ($relative -match '^runtimes[\\/](?!linux-x64[\\/]|linux[\\/]|unix[\\/])') { return $true }
    }
    if ($relative -match '\.(installed|cxttmp)$') { return $true }
    if (-not $BundleVendorSdks -and (Test-VendorFile $leaf)) { return $true }
    return $false
}

function New-Stage([string]$rid) {
    $windows = $rid -like 'win-*'
    $name = "centrix-$version-$rid"
    $publish = Join-Path $staging "$name-publish"
    $stage = Join-Path $staging $name
    foreach ($dir in $publish, $stage) {
        if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
    }
    Write-Host "==> Publishing $name" -ForegroundColor Cyan
    # PwizTargetIsWindows is a global property, so the vendor projects stage the target's natives
    # whatever the host (a RuntimeIdentifier does not reach them).
    $arguments = @('publish', $project, '-c', 'Release', '-r', $rid, '--self-contained', 'true', '-p:Platform=x64',
        "-p:PwizTargetIsWindows=$($windows.ToString().ToLowerInvariant())", '-o', $publish, '-v', 'minimal', '--nologo')
    if ($IAgreeToVendorLicenses) { $arguments += '-p:IAgreeToVendorLicenses=true' }
    & dotnet @arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed for $rid" }
    $launcher = if ($windows) { 'centrix.exe' } else { 'centrix' }
    if (-not (Test-Path (Join-Path $publish $launcher))) { throw "No $launcher in the $rid publish" }
    if ($IAgreeToVendorLicenses -and -not (Test-Path (Join-Path $publish 'ThermoFisher.CommonCore.RawFileReader.dll'))) {
        throw "Published $rid without the Thermo SDK although -IAgreeToVendorLicenses was given"
    }

    $kept = 0; $skipped = 0
    Get-ChildItem $publish -Recurse -File | ForEach-Object {
        $relative = [System.IO.Path]::GetRelativePath($publish, $_.FullName)
        if (Test-Skip $relative $windows) { $skipped++; return }
        $target = Join-Path $stage $relative
        New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
        Copy-Item $_.FullName $target
        $kept++
    }
    Remove-Item $publish -Recurse -Force
    if (-not $BundleVendorSdks) {
        $leaked = Get-ChildItem $stage -Recurse -File | Where-Object { Test-VendorFile $_.Name }
        if ($leaked) { throw "Vendor SDK files in the $rid package: $($leaked.Name -join ', ')" }
    }
    Write-Host "    $kept files kept, $skipped left out"

    Copy-Item (Join-Path $root 'README.md'), (Join-Path $root 'LICENSE'), (Join-Path $root 'NOTICE') $stage
    if (-not $windows) {
        # Unix line endings, whatever the checkout's: bash reads a CR as part of each line.
        $install = (Get-Content -Raw (Join-Path $root 'installer/install.sh')).Replace("`r`n", "`n")
        [System.IO.File]::WriteAllText((Join-Path $stage 'install.sh'), $install)
    }
    if ($IAgreeToVendorLicenses) {
        Get-ChildItem (Join-Path $pwiz 'pwiz/data/vendor_readers') -Filter 'EULA.*.txt' -Recurse -File |
            ForEach-Object { Copy-Item $_.FullName $stage }
    }
    return $stage
}

function New-Zip([string]$stage) {
    $zip = "$stage.zip" -replace [regex]::Escape($staging), $OutputDir
    if (Test-Path $zip) { Remove-Item $zip -Force }
    [System.IO.Compression.ZipFile]::CreateFromDirectory($stage, $zip, [System.IO.Compression.CompressionLevel]::Optimal, $true)
    return $zip
}

# A tar.gz written by .NET's tar writer, so it is the same from Windows as from Linux: one top-level
# folder, files 0644 and the launcher 0755.
function New-TarGz([string]$stage) {
    $tarball = "$stage.tar.gz" -replace [regex]::Escape($staging), $OutputDir
    if (Test-Path $tarball) { Remove-Item $tarball -Force }
    $top = Split-Path -Leaf $stage
    $file = [System.IO.File]::Create($tarball)
    try {
        $gzip = [System.IO.Compression.GZipStream]::new($file, [System.IO.Compression.CompressionLevel]::Optimal)
        $writer = [System.Formats.Tar.TarWriter]::new($gzip, [System.Formats.Tar.TarEntryFormat]::Pax, $false)
        $directories = @($stage) + (Get-ChildItem $stage -Recurse -Directory | ForEach-Object FullName)
        foreach ($dir in $directories) {
            $relative = [System.IO.Path]::GetRelativePath($stage, $dir).Replace('\', '/')
            $entryName = if ($relative -eq '.') { "$top/" } else { "$top/$relative/" }
            $entry = [System.Formats.Tar.PaxTarEntry]::new([System.Formats.Tar.TarEntryType]::Directory, $entryName)
            $entry.Mode = [System.IO.UnixFileMode]'UserRead, UserWrite, UserExecute, GroupRead, GroupExecute, OtherRead, OtherExecute'
            $writer.WriteEntry($entry)
        }
        foreach ($item in Get-ChildItem $stage -Recurse -File) {
            $relative = [System.IO.Path]::GetRelativePath($stage, $item.FullName).Replace('\', '/')
            $entry = [System.Formats.Tar.PaxTarEntry]::new([System.Formats.Tar.TarEntryType]::RegularFile, "$top/$relative")
            $executable = $relative -in @('centrix', '7zz', 'install.sh')
            $entry.Mode = if ($executable) {
                [System.IO.UnixFileMode]'UserRead, UserWrite, UserExecute, GroupRead, GroupExecute, OtherRead, OtherExecute'
            } else {
                [System.IO.UnixFileMode]'UserRead, UserWrite, GroupRead, OtherRead'
            }
            $entry.ModificationTime = $item.LastWriteTimeUtc
            $data = [System.IO.File]::OpenRead($item.FullName)
            try {
                $entry.DataStream = $data
                $writer.WriteEntry($entry)
            }
            finally { $data.Dispose() }
        }
        $writer.Dispose()
        $gzip.Dispose()
    }
    finally { $file.Dispose() }
    return $tarball
}

function Find-Iscc {
    $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    foreach ($dir in "${env:ProgramFiles(x86)}\Inno Setup 6", "$env:ProgramFiles\Inno Setup 6", "$env:LOCALAPPDATA\Programs\Inno Setup 6") {
        $candidate = Join-Path $dir 'ISCC.exe'
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function New-Setup([string]$stage) {
    $iscc = Find-Iscc
    if (-not $iscc) { throw 'Inno Setup 6 (ISCC.exe) was not found; install it from https://jrsoftware.org/isinfo.php' }
    $name = "Centrix-Setup-$version"
    & $iscc '/Q' "/DStagingDir=$stage" "/DOutputDir=$OutputDir" "/DOutputBaseFilename=$name" "/DMyAppVersion=$version" `
        (Join-Path $root 'installer/Setup.iss') | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "ISCC failed (exit $LASTEXITCODE)" }
    return Join-Path $OutputDir "$name.exe"
}

New-Item -ItemType Directory -Force $OutputDir, $staging | Out-Null
Write-Host "Centrix ${version}: $($Rid -join ', ') into $OutputDir" -ForegroundColor Green
if (-not $IAgreeToVendorLicenses) { Write-Warning 'Without -IAgreeToVendorLicenses the packages read open formats only.' }
$artifacts = @()
foreach ($r in $Rid) {
    $stage = New-Stage $r
    if ($r -like 'win-*') {
        $artifacts += New-Zip $stage
        if ($Setup) { $artifacts += New-Setup $stage }
    }
    else {
        $artifacts += New-TarGz $stage
    }
}
Remove-Item $staging -Recurse -Force
foreach ($artifact in $artifacts) {
    $item = Get-Item $artifact
    $hash = (Get-FileHash $artifact -Algorithm SHA256).Hash
    Write-Host ("{0}  {1:N1} MB  SHA-256 {2}" -f $item.Name, ($item.Length / 1MB), $hash)
}
