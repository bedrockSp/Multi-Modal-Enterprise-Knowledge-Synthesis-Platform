# Start two Ollama instances on 11434 (queries) and 11435 (VLM / doc parsing).
# Equivalent to Makefile `ollama-1` + `ollama-2`. Logs go to logs/ollama-*.log.
# Use Stop-Ollama (see Stop-Ollama.ps1) or `Get-Process ollama | Stop-Process`
# to shut them down.

$ErrorActionPreference = "Stop"

$logsDir = Join-Path $PSScriptRoot "..\logs"
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }

function Start-OllamaInstance {
    param([int]$Port)

    $log = Join-Path $logsDir "ollama-$Port.log"
    Write-Host "Starting Ollama on :$Port (log: $log)"

    # OLLAMA_KEEP_ALIVE=-1 keeps models resident; matches Linux Makefile behavior.
    Start-Process -FilePath "ollama" -ArgumentList "serve" `
        -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -Environment @{
            "OLLAMA_HOST" = "0.0.0.0:$Port"
            "OLLAMA_KEEP_ALIVE" = "-1"
        }
}

# Note: Start-Process -Environment requires PowerShell 7+. On Windows PowerShell
# 5.1, fall back to per-process env via cmd /c.
if ($PSVersionTable.PSVersion.Major -lt 7) {
    function Start-OllamaInstance {
        param([int]$Port)
        $log = Join-Path $logsDir "ollama-$Port.log"
        Write-Host "Starting Ollama on :$Port (log: $log)"
        $cmd = "set OLLAMA_HOST=0.0.0.0:$Port && set OLLAMA_KEEP_ALIVE=-1 && ollama serve > `"$log`" 2>&1"
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmd -WindowStyle Hidden
    }
}

Start-OllamaInstance -Port 11434
Start-OllamaInstance -Port 11435

Write-Host ""
Write-Host "Ollama running on 11434 and 11435. To stop: Get-Process ollama | Stop-Process"
