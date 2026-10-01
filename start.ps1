# Captionz startup (PowerShell)
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\pythonw.exe")) {
    Write-Host "Environment not found. Run .\install.ps1 first." -ForegroundColor Yellow
    exit 1
}
Start-Process -FilePath ".venv\Scripts\pythonw.exe" -ArgumentList (@("app.py") + $args) -WorkingDirectory $PSScriptRoot
