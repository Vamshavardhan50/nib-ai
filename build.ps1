# Build script for cactus-rs
# Uses the GNU toolchain + MinGW (no Visual Studio required).
#
# Usage:  .\build.ps1          (debug build)
#         .\build.ps1 -Release (release build, smallest binary)

param(
    [switch]$Release,
    [switch]$Test
)

$ErrorActionPreference = "Stop"

# Locate MinGW (WinLibs installed via winget).
$mingw = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
    -Recurse -Filter dlltool.exe -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty DirectoryName

if (-not $mingw) {
    Write-Error "MinGW not found. Install it with: winget install BrechtSanders.WinLibs.POSIX.UCRT"
}

$env:PATH = "$mingw;$env:PATH"
$cargo = "$env:USERPROFILE\.cargo\bin\cargo.exe"
$toolchain = "+stable-x86_64-pc-windows-gnu"

if ($Test) {
    & $cargo $toolchain test
    exit $LASTEXITCODE
}

if ($Release) {
    & $cargo $toolchain build --release
    if ($LASTEXITCODE -eq 0) {
        $bin = "target\release\cactus.exe"
        $size = (Get-Item $bin).Length
        Write-Host ""
        Write-Host "Built $bin  ($([math]::Round($size / 1MB, 2)) MB)" -ForegroundColor Green
    }
} else {
    & $cargo $toolchain build
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Built target\debug\cactus.exe" -ForegroundColor Green
    }
}
exit $LASTEXITCODE
