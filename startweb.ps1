# Captionz : interface web NiceGUI (PowerShell). Ctrl+C pour arreter.
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Environnement absent : lance d'abord .\install.ps1" -ForegroundColor Yellow
    exit 1
}
Write-Host "Captionz - interface web NiceGUI (Ctrl+C pour arreter)" -ForegroundColor Cyan
& ".venv\Scripts\python.exe" app.py --ui web @args
