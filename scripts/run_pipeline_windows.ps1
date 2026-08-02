param(
    [string]$Dataset = "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing",
    [string]$Output = "outputs\exploratory_full",
    [string]$InitiatedBy = "Ahmed",
    [switch]$RunFinalTest
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Python virtual environment not found: $Python"
}

if (-not (Test-Path $Dataset)) {
    throw "Dataset not found: $Dataset"
}

function Invoke-Python {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Python @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "Python failed with exit code ${LASTEXITCODE}: python $($Arguments -join ' ')"
    }
}

New-Item -ItemType Directory -Force -Path $Output | Out-Null

Write-Host "`n=== Environment verification ==="
Invoke-Python --version
Invoke-Python -m compileall src main.py streamlit_app.py
Invoke-Python -m ruff check src tests main.py streamlit_app.py
Invoke-Python -m pytest -q

Write-Host "`n=== GPU verification ==="
Invoke-Python main.py gpu-smoke `
    --device auto `
    --batch-size 4 `
    --output "$Output\gpu"

$GpuFile = Get-ChildItem "$Output\gpu" -Filter "*.json" -File |
    Select-Object -First 1

if (-not $GpuFile) {
    throw "GPU smoke JSON report was not created."
}

$GpuReport = Get-Content $GpuFile.FullName -Raw | ConvertFrom-Json

if (-not $GpuReport.cuda_used) {
    throw "CUDA was not used. Stop before long training."
}

Write-Host "CUDA verified."

$Override = @(
    "--allow-leakage-override",
    "--override-justification",
    "Known dataset duplicates accepted for exploratory run; not final thesis evidence."
)

Write-Host "`n=== Manifest and objective features ==="
Invoke-Python main.py build-manifest `
    --dataset $Dataset `
    --output "$Output\manifest_full.csv"

Import-Csv "$Output\manifest_full.csv" |
    Where-Object { $_.split -in @("train", "valid") } |
    Export-Csv "$Output\manifest_selection.csv" -NoTypeInformation

Invoke-Python main.py extract-features `
    --manifest "$Output\manifest_selection.csv" `
    --output "$Output\features" `
    @Override

Write-Host "`n=== Objective baseline models ==="
Invoke-Python main.py train-feature-model `
    --features "$Output\features\features.csv" `
    --output "$Output\feature_models" `
    --seeds "42,123,2026"

Write-Host "`n=== Deep-model comparison ==="
Invoke-Python main.py compare-deep-models `
    --dataset $Dataset `
    --output "$Output\deep" `
    --models "small_cnn,mobilenet_v3_small,resnet18,efficientnet_b0" `
    --seeds "42,123,2026" `
    --epochs 30 `
    --image-size 224 `
    --batch-size 4 `
    --grad-accum-steps 4 `
    --device auto `
    @Override

$Comparison = "$Output\deep\deep_comparison.json"

if (-not (Test-Path $Comparison)) {
    throw "Deep-comparison result was not created: $Comparison"
}

$DeepResult = Get-Content $Comparison -Raw | ConvertFrom-Json
$WinnerRun = $DeepResult.winning_run

if (-not $WinnerRun) {
    throw "winning_run is missing from deep_comparison.json"
}

if (-not (Test-Path $WinnerRun.checkpoint)) {
    throw "Winning checkpoint does not exist: $($WinnerRun.checkpoint)"
}

Write-Host "Validation winner:"
Write-Host "Model: $($WinnerRun.model)"
Write-Host "Seed: $($WinnerRun.seed)"
Write-Host "Checkpoint: $($WinnerRun.checkpoint)"

Write-Host "`n=== Generic embeddings ==="
Invoke-Python main.py extract-embeddings `
    --manifest "$Output\manifest_selection.csv" `
    --backbone resnet18 `
    --batch-size 4 `
    --device auto `
    --output "$Output\emb_generic" `
    @Override

Write-Host "`n=== Fine-tuned winner embeddings ==="
Invoke-Python main.py extract-embeddings `
    --manifest "$Output\manifest_selection.csv" `
    --deep-comparison $Comparison `
    --batch-size 4 `
    --device auto `
    --output "$Output\emb_finetuned" `
    @Override

Write-Host "`n=== Embedding comparison ==="
Invoke-Python main.py compare-embeddings `
    --features "$Output\features\features.csv" `
    --generic "$Output\emb_generic\embeddings.npz" `
    --finetuned "$Output\emb_finetuned\embeddings.npz" `
    --output "$Output\emb_compare"

Write-Host "`n=== Fusion ==="
Invoke-Python main.py train-fusion-model `
    --features "$Output\features\features.csv" `
    --embeddings "$Output\emb_finetuned\embeddings.npz" `
    --output "$Output\fusion" `
    --seeds "42,123,2026"

$FusionResult = Get-Content "$Output\fusion\fusion_leaderboard.json" -Raw |
    ConvertFrom-Json

$FusionCheckpoint = $FusionResult.winner.checkpoint

if (-not $FusionCheckpoint) {
    throw "Fusion winner checkpoint was not found."
}

Write-Host "`n=== Fusion calibration ==="
Invoke-Python main.py calibrate-fusion `
    --bundle $FusionCheckpoint `
    --features "$Output\features\features.csv" `
    --embeddings "$Output\emb_finetuned\embeddings.npz" `
    --output "$Output\calibration"

Write-Host "`n=== Ablation ==="
Invoke-Python main.py run-ablation `
    --features "$Output\features\features.csv" `
    --output "$Output\ablation" `
    --seeds "42,123,2026"

$Example = Get-ChildItem (Join-Path $Dataset "valid") -File -Recurse |
    Where-Object {
        $_.Extension -match "^\.(jpg|jpeg|png|bmp|webp|tif|tiff)$"
    } |
    Get-Random

if ($Example) {
    Write-Host "`n=== Example prediction and report ==="

    Invoke-Python main.py explain-gradcam `
        --image $Example.FullName `
        --deep-comparison $Comparison `
        --output "$Output\gradcam" `
        --device auto

    Invoke-Python main.py predict-image `
        --image $Example.FullName `
        --deep-comparison $Comparison `
        --device auto

    Invoke-Python main.py analyze-image `
        --image $Example.FullName `
        --deep-comparison $Comparison `
        --output "$Output\case_001"
}

if ($RunFinalTest) {
    Write-Host "`n=== Locked final test ==="

    Invoke-Python main.py evaluate `
        --manifest "$Output\manifest_full.csv" `
        --deep-comparison $Comparison `
        --split test `
        --unlock-test `
        --confirm-final-evaluation `
        --initiated-by $InitiatedBy `
        --output "$Output\final_test"
}
else {
    Write-Host "`nFinal test remains LOCKED."
    Write-Host "Run again with -RunFinalTest only after accepting the validation winner."
}

Write-Host "`n=== Thesis outputs ==="
Invoke-Python main.py generate-thesis-outputs `
    --output $Output

Write-Host "`nPipeline completed."
Write-Host "Output root: $Output"
Write-Host "Launch UI with:"
Write-Host "& '$Python' -m streamlit run streamlit_app.py"

