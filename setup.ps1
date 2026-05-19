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

function Test-Python311Available {
    # Returns $true if py -3.11 OR python3.11 resolves to a working Python 3.11.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $null = & py -3.11 --version 2>&1
            if ($LASTEXITCODE -eq 0) { return $true }
        } catch {}
    }
    if (Get-Command python3.11 -ErrorAction SilentlyContinue) {
        try {
            $v = & python3.11 --version 2>&1
            if ($LASTEXITCODE -eq 0 -and $v -match "3\.11\.") { return $true }
        } catch {}
    }
    return $false
}

function Install-Python311 {
    # Idempotent: only acts when 3.11 is not already installed. Per-user
    # install so no admin prompt, no PATH override (your existing 'python'
    # stays put), but the py launcher picks up 3.11 via the registry so
    # 'py -3.11' starts working immediately.
    if (Test-Python311Available) { return }

    $version = "3.11.9"
    $url = "https://www.python.org/ftp/python/$version/python-$version-amd64.exe"
    $installer = Join-Path $env:TEMP "python-$version-amd64.exe"

    Write-Host "Python 3.11 not found. Downloading installer..."
    Write-Host "  URL:    $url"
    Write-Host "  Target: $installer"
    try {
        # Invoke-WebRequest uses .NET HttpWebRequest which already trusts the
        # Windows certificate store - works behind corporate SSL inspection
        # without any extra setup. Suppressing the inline progress bar makes
        # the download ~10x faster on PS 5.1.
        $oldPref = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
        $ProgressPreference = $oldPref
    } catch {
        throw "Failed to download Python 3.11 installer: $_`nIf you are behind a proxy that blocks python.org, download manually from $url and run with: $installer /passive InstallAllUsers=0 PrependPath=0 Include_launcher=1"
    }

    Write-Host "Installing Python $version per-user (no admin, no PATH override)..."
    $installArgs = @(
        "/passive",
        "InstallAllUsers=0",
        "PrependPath=0",
        "Include_launcher=1",
        "AssociateFiles=0",
        "Shortcuts=0",
        "Include_test=0",
        "Include_doc=0",
        "Include_dev=0",
        "Include_debug=0",
        "Include_symbols=0",
        "SimpleInstall=1"
    )
    $proc = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        throw "Python 3.11 installer exited with code $($proc.ExitCode). Try running '$installer' manually."
    }
    Remove-Item $installer -Force -ErrorAction SilentlyContinue

    if (-not (Test-Python311Available)) {
        throw "Python 3.11 installer reported success but 'py -3.11' still does not resolve. Close and reopen PowerShell, then re-run '.\setup.ps1 venv'."
    }
    Write-Host "Python 3.11 installed successfully."
}

function Resolve-Python311 {
    # Prefer the Windows Python launcher 'py -3.11' - it knows about all
    # installed Python versions and is the canonical way to pick one. Fall
    # back to python3.11 on PATH if the launcher is missing.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $null = & py -3.11 --version 2>&1
            if ($LASTEXITCODE -eq 0) { return @("py", "-3.11") }
        } catch {}
    }
    if (Get-Command python3.11 -ErrorAction SilentlyContinue) { return @("python3.11") }
    throw "Python 3.11 not found even after install. Restart your shell and re-run '.\setup.ps1 venv'."
}

function Invoke-Venv {
    Install-Python311  # idempotent; no-op when 3.11 already present
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
        # Hard-stop on a wrong-version venv: pip install would otherwise build
        # wheels for the wrong Python and the resulting venv would be broken.
        $existing = & "$scriptRoot\virtualEnv\Scripts\python.exe" --version 2>&1
        if ($existing -notmatch "3\.11\.") {
            throw "Existing virtualEnv uses $existing but the project requires Python 3.11. Delete the folder and re-run:`n  Remove-Item -Recurse -Force virtualEnv`n  .\setup.ps1 venv"
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
