# Windows CPU setup entry point - mirrors the Linux Makefile targets.
#
# Usage:
#   .\setup.ps1 venv          # Create virtualEnv/ with Python 3.11 and install requirements-windows-cpu.txt
#   .\setup.ps1 set-models    # Pull Ollama models for CPU profile (qwen2.5:7b + gemma3:12b)
#   .\setup.ps1 ollama        # Start two Ollama instances on 11434/11435
#   .\setup.ps1 ollama-stop   # Stop all local Ollama instances
#   .\setup.ps1 doctor        # Pre-download embedding + cross-encoder models from HuggingFace
#   .\setup.ps1 backend       # Run the FastAPI server (python backend.py)
#   .\setup.ps1 frontend      # Run the Vite dev server (python frontend.py)
#   .\setup.ps1 all           # venv + set-models + ollama + doctor (one-shot first-run)
#
# Prereqs (install separately, see Windows_README.md):
#   * Python 3.11 (NOT 3.12+; some ML deps lack newer wheels)
#   * Node.js 22
#   * MongoDB
#   * Tesseract (UB-Mannheim build)
#   * Ollama for Windows

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("venv", "set-models", "ollama", "ollama-stop", "doctor", "backend", "frontend", "all")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
$scriptRoot = $PSScriptRoot

function Resolve-Python311 {
    # Prefer the Windows Python launcher 'py -3.11' - it knows about all
    # installed Python versions and is the canonical way to pick one. Fall back
    # to python3.11 on PATH, then to whatever 'python' is, with a version warning.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $null = & py -3.11 --version 2>&1
            if ($LASTEXITCODE -eq 0) { return @("py", "-3.11") }
        } catch {}
    }
    if (Get-Command python3.11 -ErrorAction SilentlyContinue) { return @("python3.11") }
    $pyVer = & python --version 2>&1
    if ($pyVer -notmatch "3\.11\.") {
        Write-Warning "Python 3.11 not found. Falling back to: $pyVer"
        Write-Warning "Some ML deps may fail to install. Install 3.11 from python.org and re-run."
    }
    return @("python")
}

function Invoke-Venv {
    if (-not (Test-Path "$scriptRoot\virtualEnv")) {
        $pyCmd = Resolve-Python311
        Write-Host "Creating virtualEnv with: $($pyCmd -join ' ')"
        if ($pyCmd.Length -gt 1) {
            & $pyCmd[0] $pyCmd[1..($pyCmd.Length-1)] -m venv "$scriptRoot\virtualEnv"
        } else {
            & $pyCmd[0] -m venv "$scriptRoot\virtualEnv"
        }
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    } else {
        # Sanity-check existing venv version
        $existing = & "$scriptRoot\virtualEnv\Scripts\python.exe" --version 2>&1
        if ($existing -notmatch "3\.11\.") {
            Write-Warning "Existing virtualEnv uses $existing - project requires Python 3.11."
            Write-Warning "Delete virtualEnv\ and re-run '.\setup.ps1 venv' to recreate."
        }
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

function Invoke-Doctor {
    # Pre-warm the HuggingFace cache so backend.py doesn't try to download
    # models during module import (which crashes the server on first boot if
    # the network blip / SSL handshake fails). pip-system-certs (installed by
    # requirements-windows-cpu.txt) lets HTTPS use the Windows cert store, so
    # corporate proxy CAs work transparently.
    $py = "$scriptRoot\virtualEnv\Scripts\python.exe"
    if (-not (Test-Path $py)) { throw "virtualEnv missing - run '.\setup.ps1 venv' first" }

    Write-Host "Pre-downloading nomic-embed-text-v1.5 (embedding model, ~150 MB)..."
    & $py -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"
    if ($LASTEXITCODE -ne 0) { throw "Embedding model download failed. See Windows_README.md SSL section." }

    Write-Host "Pre-downloading ms-marco-MiniLM-L-6-v2 (cross-encoder, ~90 MB)..."
    & $py -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)"
    if ($LASTEXITCODE -ne 0) { throw "Cross-encoder download failed. See Windows_README.md SSL section." }

    Write-Host ""
    Write-Host "Models cached. backend.py will load them from disk on next boot."
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
    "doctor"      { Invoke-Doctor }
    "backend"     { Invoke-Backend }
    "frontend"    { Invoke-Frontend }
    "all"         {
        Invoke-Venv
        Invoke-SetModels
        Invoke-Ollama
        Invoke-Doctor
        Write-Host ""
        Write-Host "Setup complete. Next:"
        Write-Host "  .\setup.ps1 backend    # in one terminal"
        Write-Host "  .\setup.ps1 frontend   # in another terminal"
    }
}
