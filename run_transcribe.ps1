param([int]$MaxRuns = 12)

$Root = $PSScriptRoot
$Python = Join-Path $Root "venv\Scripts\python.exe"
$Manifest = Join-Path $Root "data\manifest.json"
$Output = Join-Path $Root "data\sravaani_out.jsonl"
$Log = Join-Path $Root "data\transcribe.log"
$env:PYTHONIOENCODING = "utf-8"

Set-Location $Root
$total = (Get-Content $Manifest -Raw | ConvertFrom-Json).Count
for ($i = 1; $i -le $MaxRuns; $i++) {
    "=== run $i" | Out-File -Append -Encoding utf8 $Log
    & $Python -m sravaani_eval.transcribe *>> $Log
    "exit code $LASTEXITCODE" | Out-File -Append -Encoding utf8 $Log
    $done = if (Test-Path $Output) { (Get-Content $Output).Count } else { 0 }
    Write-Host "run $i finished: $done of $total calls transcribed"
    if ($done -ge $total) { break }
}
