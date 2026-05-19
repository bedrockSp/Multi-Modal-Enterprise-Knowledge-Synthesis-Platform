# PowerShell mirror of scripts/setmodel.sh - pulls Ollama models for the
# Windows CPU profile. The default 20B model is replaced with a 7B-class model
# that is actually usable on a CPU laptop. Pass -Model to override.
param(
    [string]$Model = "qwen2.5:7b"
)

$ErrorActionPreference = "Stop"

Write-Host "Pulling main model: $Model"
ollama pull $Model
if ($LASTEXITCODE -ne 0) { throw "ollama pull $Model failed" }

Write-Host "Pulling image-parser fallback model: gemma3:12b"
ollama pull gemma3:12b
if ($LASTEXITCODE -ne 0) { throw "ollama pull gemma3:12b failed" }

Write-Host ""
Write-Host "Models pulled. Verify with: ollama list"
Write-Host "Make sure MAIN_MODEL=$Model in your .env."
