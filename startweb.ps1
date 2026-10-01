# Captionz NiceGUI web interface (PowerShell). Press Ctrl+C to stop.
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Environment not found. Run .\install.ps1 first." -ForegroundColor Yellow
    exit 1
}
Write-Host "Captionz - NiceGUI web interface (press Ctrl+C to stop)" -ForegroundColor Cyan
& ".venv\Scripts\python.exe" app.py --ui web @args
