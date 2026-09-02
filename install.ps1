# Captionz : installation (PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "=== Captionz : installation ===" -ForegroundColor Cyan

# Trouver un Python complet (venv + pip + tkinter). Le lanceur "py" est prefere :
# "python" du PATH peut etre MSYS2 / Anaconda / le stub du Microsoft Store.
$candidates = @(@("py", "-3"), @("python"), @("python3"))
$py = $null
foreach ($c in $candidates) {
    $exe = $c[0]; $pre = @($c | Select-Object -Skip 1)
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    & $exe @pre -c "import venv, ensurepip, tkinter" 2>$null
    if ($LASTEXITCODE -eq 0) { $py = $c; break }
}
if (-not $py) {
    Write-Host "[ERREUR] Aucun Python 3.10+ complet trouve (venv + pip + tkinter)." -ForegroundColor Red
    Write-Host "Installe Python depuis https://www.python.org/downloads/ en cochant 'Add to PATH' et 'tcl/tk and IDLE'."
    exit 1
}
Write-Host "Python utilise : $($py -join ' ')"

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "Creation de l'environnement virtuel .venv ..."
    if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
    & $py[0] @($py | Select-Object -Skip 1) -m venv .venv
}
if (-not (Test-Path $venvPy)) {
    Write-Host "[ERREUR] creation du venv impossible." -ForegroundColor Red
    exit 1
}

& $venvPy -m pip install --upgrade pip | Out-Null
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "[ERREUR] installation des dependances." -ForegroundColor Red; exit 1 }
& $venvPy -c "import tkinter, PIL; print('OK : tkinter + Pillow', PIL.__version__)"
Write-Host ""
Write-Host "Installation terminee. Lance .\start.ps1" -ForegroundColor Green
