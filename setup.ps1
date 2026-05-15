# Windows CPU setup entry point — mirrors the Linux Makefile targets.
#
# Usage:
#   .\setup.ps1 venv          # Create virtualEnv/ and install requirements-windows-cpu.txt
#   .\setup.ps1 set-models    # Pull Ollama models for CPU profile (qwen2.5:7b + gemma3:12b)
#   .\setup.ps1 ollama        # Start two Ollama instances on 11434/11435
#   .\setup.ps1 ollama-stop   # Stop all local Ollama instances
#   .\setup.ps1 backend       # Run the FastAPI server (python backend.py)
#   .\setup.ps1 frontend      # Run the Vite dev server (python frontend.py)
#   .\setup.ps1 all           # venv + set-models + ollama (one-shot first-run)
#
# Prereqs (install separately, see Windows_README.md):
#   * Python 3.11
#   * Node.js 22
#   * MongoDB
#   * Tesseract (UB-Mannheim build)
#   * Ollama for Windows

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("venv", "set-models", "ollama", "ollama-stop", "backend", "frontend", "all")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
$scriptRoot = $PSScriptRoot

function Invoke-Venv {
    if (-not (Test-Path "$scriptRoot\virtualEnv")) {
        Write-Host "Creating virtualEnv..."
        python -m venv "$scriptRoot\virtualEnv"
    }
    $py = "$scriptRoot\virtualEnv\Scripts\python.exe"
    Write-Host "Upgrading pip..."
    & $py -m pip install --upgrade pip
    Write-Host "Installing requirements-windows-cpu.txt..."
    & $py -m pip install -r "$scriptRoot\requirements-windows-cpu.txt"
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    Write-Host "Done. Activate with: .\virtualEnv\Scripts\Activate.ps1"
}

function Invoke-SetModels {
    & "$scriptRoot\scripts\setmodel.ps1"
}

function Invoke-Ollama {
    & "$scriptRoot\scripts\start_ollama.ps1"
}

function Invoke-OllamaStop {
    Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "Stopped all ollama processes."
}

function Invoke-Backend {
    $py = if (Test-Path "$scriptRoot\virtualEnv\Scripts\python.exe") {
        "$scriptRoot\virtualEnv\Scripts\python.exe"
    } else {
        "python"
    }
    & $py "$scriptRoot\backend.py"
}

function Invoke-Frontend {
    $py = if (Test-Path "$scriptRoot\virtualEnv\Scripts\python.exe") {
        "$scriptRoot\virtualEnv\Scripts\python.exe"
    } else {
        "python"
    }
    & $py "$scriptRoot\frontend.py"
}

switch ($Target) {
    "venv"        { Invoke-Venv }
    "set-models"  { Invoke-SetModels }
    "ollama"      { Invoke-Ollama }
    "ollama-stop" { Invoke-OllamaStop }
    "backend"     { Invoke-Backend }
    "frontend"    { Invoke-Frontend }
    "all"         {
        Invoke-Venv
        Invoke-SetModels
        Invoke-Ollama
        Write-Host ""
        Write-Host "Setup complete. Next:"
        Write-Host "  .\setup.ps1 backend    # in one terminal"
        Write-Host "  .\setup.ps1 frontend   # in another terminal"
    }
}
