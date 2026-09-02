# Captionz : demarrage (PowerShell)

Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\pythonw.exe")) {

    Write-Host "Environnement absent : lance d'abord .\install.ps1" -ForegroundColor Yellow

    exit 1

}

Start-Process -FilePath ".venv\Scripts\pythonw.exe" -ArgumentList (@("app.py") + $args) -WorkingDirectory $PSScriptRoot

