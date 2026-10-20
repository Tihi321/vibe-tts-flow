<#
.SYNOPSIS
  Builds the portable VibeTTS app (PyInstaller one-folder) and three zips:
    dist\VibeTTS-chrome-extension-<version>.zip   the Chrome extension (load unpacked)
    dist\VibeTTS-<version>-win64.zip             app only; the model is downloaded from Settings
    dist\VibeTTS-<version>-win64-with-model.zip  app + int8 model + voices, works offline
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\build.ps1
  powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -SkipTests -NoModel
.NOTES
  Never copies config.toml or logs. The model files are downloaded once into build\models
  (cached). After the build, dist\VibeTTS holds the model too, so VibeTTS.exe runs as is.
#>
param(
    [switch]$SkipTests,
    [switch]$NoModel   # skip the with-model zip (no model download)
)

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "Missing $python. Create the venv first: python -m venv .venv"
}

function Invoke-Native {
    param([string]$What, [scriptblock]$Cmd)
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

function New-Zip {
    param([string]$Source, [string]$Zip)
    if (Test-Path $Zip) { Remove-Item -Force $Zip }
    Write-Host "== Creating $Zip" -ForegroundColor Cyan
    Compress-Archive -Path $Source -DestinationPath $Zip -CompressionLevel Optimal
}

# --- 1. dependencies ---------------------------------------------------------
Write-Host "== Installing build dependencies" -ForegroundColor Cyan
Invoke-Native "pip install" { & $python -m pip install -r (Join-Path $root 'requirements-build.txt') }

# --- 2. tests ----------------------------------------------------------------
if (-not $SkipTests) {
    Write-Host "== Running tests" -ForegroundColor Cyan
    Invoke-Native "pytest" { & $python -m pytest -q tests }
}

# --- 3. icon -----------------------------------------------------------------
Write-Host "== Generating icon" -ForegroundColor Cyan
$buildDir = Join-Path $root 'build'
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
$icoPath = Join-Path $buildDir 'vibetts.ico'
$iconScript = @"
import sys
from vibetts.tray import make_icon
img = make_icon('#e8e8e8')
img.save(sys.argv[1], format='ICO', sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64)])
"@
Invoke-Native "icon generation" { & $python -c $iconScript $icoPath }

# --- 4. PyInstaller ----------------------------------------------------------
Write-Host "== Running PyInstaller" -ForegroundColor Cyan
$dist = Join-Path $root 'dist'
$appDir = Join-Path $dist 'VibeTTS'
if (Test-Path $appDir) { Remove-Item -Recurse -Force $appDir }
Invoke-Native "PyInstaller" {
    & $python -m PyInstaller (Join-Path $root 'packaging\VibeTTS.spec') --noconfirm `
        --distpath $dist --workpath (Join-Path $buildDir 'pyinstaller')
}

# --- 5. extras ---------------------------------------------------------------
Copy-Item -Path (Join-Path $root 'README.md') -Destination $appDir -Force
# the Chrome extension, to load unpacked from chrome://extensions (Settings > Reader opens it)
Copy-Item -Path (Join-Path $root 'extension') -Destination (Join-Path $appDir 'extension') -Recurse -Force
$modelsDir = Join-Path $appDir 'models'
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

$versionLine = Select-String -Path (Join-Path $root 'vibetts\__init__.py') -Pattern '__version__\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $versionLine) { throw "Could not read __version__ from vibetts/__init__.py" }
$version = $versionLine.Matches[0].Groups[1].Value

# --- 5b. Chrome extension zip (the contents of extension\ at the zip root) ---------------
$extZip = Join-Path $dist "VibeTTS-chrome-extension-$version.zip"
if (Test-Path $extZip) { Remove-Item -Force $extZip }
Write-Host "== Creating $extZip" -ForegroundColor Cyan
Compress-Archive -Path (Join-Path $root 'extension\*') -DestinationPath $extZip -CompressionLevel Optimal

# --- 6. app-only zip ---------------------------------------------------------
$zips = @($extZip, (Join-Path $dist "VibeTTS-$version-win64.zip"))
New-Zip $appDir $zips[1]

# --- 7. model (cached) + with-model zip --------------------------------------
if (-not $NoModel) {
    $cache = Join-Path $buildDir 'models'
    Write-Host "== Fetching model files into $cache (cached)" -ForegroundColor Cyan
    Invoke-Native "model download" { & $python -m vibetts.model_setup --model int8 --dest $cache }
    Copy-Item -Path (Join-Path $cache 'kokoro-v1.0.int8.onnx') -Destination $modelsDir -Force
    Copy-Item -Path (Join-Path $cache 'voices-v1.0.bin') -Destination $modelsDir -Force
    $zips += Join-Path $dist "VibeTTS-$version-win64-with-model.zip"
    New-Zip $appDir $zips[2]
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "  Folder: $appDir"
Write-Host "  Exe:    $(Join-Path $appDir 'VibeTTS.exe')"
foreach ($z in $zips) { Write-Host "  Zip:    $z" }
