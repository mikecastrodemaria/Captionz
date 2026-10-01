# Captionz installation (PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "=== Captionz installation ===" -ForegroundColor Cyan

# Find a complete Python installation (venv + pip + tkinter). Prefer the "py" launcher:
# "python" on PATH may be MSYS2, Anaconda, or the Microsoft Store stub.
$candidates = @(@("py", "-3"), @("python"), @("python3"))
$py = $null
foreach ($c in $candidates) {
    $exe = $c[0]; $pre = @($c | Select-Object -Skip 1)
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    & $exe @pre -c "import venv, ensurepip, tkinter" 2>$null
    if ($LASTEXITCODE -eq 0) { $py = $c; break }
}
if (-not $py) {
    Write-Host "[ERROR] No complete Python 3.10+ installation found (venv + pip + tkinter)." -ForegroundColor Red
    Write-Host "Install Python from https://www.python.org/downloads/ and select 'Add to PATH' and 'tcl/tk and IDLE'."
    exit 1
}
Write-Host "Using Python: $($py -join ' ')"

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "Creating virtual environment .venv..."
    if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
    & $py[0] @($py | Select-Object -Skip 1) -m venv .venv
}
if (-not (Test-Path $venvPy)) {
    Write-Host "[ERROR] Could not create the virtual environment." -ForegroundColor Red
    exit 1
}

& $venvPy -m pip install --upgrade pip | Out-Null
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Dependency installation failed." -ForegroundColor Red; exit 1 }
& $venvPy -c "import tkinter, PIL; print('OK: tkinter + Pillow', PIL.__version__)"
Write-Host ""
Write-Host "Installation complete. Run .\start.ps1" -ForegroundColor Green
