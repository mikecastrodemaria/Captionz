# Captionz : interface web NiceGUI (PowerShell)
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Environnement absent : lance d'abord .\install.ps1" -ForegroundColor Yellow
    exit 1
}
& ".venv\Scripts\python.exe" app.py --ui web @args
