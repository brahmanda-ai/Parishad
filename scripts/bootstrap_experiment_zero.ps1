param(
    [string]$WorkspacePath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$PythonExe = "python",
    [string]$VenvDir = "venv",
    [string]$HfRepo = "bartowski/Qwen2.5-3B-Instruct-GGUF",
    [string]$HfFile = "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    [string]$ModelOutputDir = "models",
    [ValidateSet("both", "gsm8k", "humaneval")]
    [string]$Benchmark = "both",
    [ValidateSet("all", "1", "2", "3", "4")]
    [string]$Part = "all",
    [int]$Seed = 42,
    [int]$TotalProblems = 100,
    [int]$ChunkSize = 25,
    [string]$OutputDir = "results/experiment_zero",
    [switch]$SkipModelPull,
    [switch]$SkipDatasetDownload,
    [switch]$SkipBenchmarkRun
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "==== $Message ====" -ForegroundColor Cyan
}

function Assert-Command {
    param(
        [string]$CommandName,
        [string]$InstallHint
    )
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Required command '$CommandName' not found. $InstallHint"
    }
}

function Update-ParishadConfig {
    param([string]$TargetModelPath)

    $configDir = Join-Path $HOME ".parishad"
    $configPath = Join-Path $configDir "config.json"

    New-Item -ItemType Directory -Force -Path $configDir | Out-Null

    if (Test-Path $configPath) {
        $cfg = Get-Content -Raw $configPath | ConvertFrom-Json
    }
    else {
        $cfg = [pscustomobject]@{}
    }

    if (-not $cfg.PSObject.Properties.Name.Contains("session")) {
        $cfg | Add-Member -NotePropertyName session -NotePropertyValue ([pscustomobject]@{})
    }
    if (-not $cfg.PSObject.Properties.Name.Contains("model_config")) {
        $cfg | Add-Member -NotePropertyName model_config -NotePropertyValue ([pscustomobject]@{})
    }
    if (-not $cfg.PSObject.Properties.Name.Contains("models")) {
        $cfg | Add-Member -NotePropertyName models -NotePropertyValue ([pscustomobject]@{})
    }

    $cfg.setup_complete = $true
    $cfg.session.backend = "llama_cpp"
    $cfg.session.model = $TargetModelPath

    if (-not $cfg.session.PSObject.Properties.Name.Contains("sabha") -or -not $cfg.session.sabha) {
        $cfg.session | Add-Member -NotePropertyName sabha -NotePropertyValue "madhyam" -Force
    }

    $cfg.session.model_map = [pscustomobject]@{
        small = $TargetModelPath
        mid   = $TargetModelPath
        big   = $TargetModelPath
    }

    $modelKey = "local:qwen2.5-3b"
    $cfg.models.$modelKey = [pscustomobject]@{
        source = "local"
        format = "gguf"
        path = $TargetModelPath
        size_bytes = (Get-Item $TargetModelPath).Length
    }

    $cfg.model_config.n_gpu_layers = -1
    $cfg.model_config.n_ctx = 8192

    $cfg | ConvertTo-Json -Depth 100 | Set-Content -Path $configPath -Encoding UTF8

    Write-Host "Configured Parishad model backend: llama_cpp" -ForegroundColor Green
    Write-Host "Configured Parishad model file: $TargetModelPath" -ForegroundColor Green
}

function Get-HfGgufModel {
    param(
        [string]$Repo,
        [string]$File,
        [string]$OutputDir
    )

    $targetDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
        $OutputDir
    }
    else {
        Join-Path $WorkspacePath $OutputDir
    }

    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    $downloadScript = @"
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id=r"$Repo",
    filename=r"$File",
    local_dir=r"$targetDir",
    local_dir_use_symlinks=False,
)
print(path)
"@

    $tempScript = Join-Path $env:TEMP "parishad_hf_download.py"
    Set-Content -Path $tempScript -Encoding UTF8 -Value $downloadScript

    $output = & python $tempScript 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download model from Hugging Face. Output:`n$($output -join "`n")"
    }

    $modelPath = ($output | Select-Object -Last 1).ToString().Trim()
    if (-not (Test-Path $modelPath)) {
        throw "Model download reported path '$modelPath' but file was not found."
    }

    Write-Host "Downloaded model: $modelPath" -ForegroundColor Green
    return $modelPath
}

function Initialize-BenchmarkDatasets {
    param([string]$Root)

    $rawDir = Join-Path $Root "dataset/raw"
    $processedDir = Join-Path $Root "dataset/processed"

    New-Item -ItemType Directory -Force -Path $rawDir | Out-Null
    New-Item -ItemType Directory -Force -Path $processedDir | Out-Null

    $gsmRawPath = Join-Path $rawDir "gsm8k_rows.json"
    $humRawPath = Join-Path $rawDir "humaneval_rows.json"

    $gsmUrl = "https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k&config=main&split=train&offset=0&length=100"
    $humUrl = "https://datasets-server.huggingface.co/rows?dataset=openai%2Fopenai_humaneval&config=openai_humaneval&split=test&offset=0&length=100"

    Write-Host "Downloading GSM8K rows..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $gsmUrl -OutFile $gsmRawPath

    Write-Host "Downloading HumanEval rows..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $humUrl -OutFile $humRawPath

    $gsmOutPath = Join-Path $processedDir "gsm8k_100.jsonl"
    $humOutPath = Join-Path $processedDir "humaneval_100.jsonl"

    if (Test-Path $gsmOutPath) { Remove-Item $gsmOutPath -Force }
    if (Test-Path $humOutPath) { Remove-Item $humOutPath -Force }

    $gsmRaw = Get-Content -Raw $gsmRawPath | ConvertFrom-Json
    foreach ($item in $gsmRaw.rows) {
        $row = $item.row
        $obj = [ordered]@{
            question = [string]$row.question
            answer   = [string]$row.answer
        }
        ($obj | ConvertTo-Json -Compress) | Add-Content -Path $gsmOutPath -Encoding UTF8
    }

    $humRaw = Get-Content -Raw $humRawPath | ConvertFrom-Json
    foreach ($item in $humRaw.rows) {
        $row = $item.row
        $obj = [ordered]@{
            task_id            = [string]$row.task_id
            prompt             = [string]$row.prompt
            entry_point        = [string]$row.entry_point
            test               = [string]$row.test
            canonical_solution = [string]$row.canonical_solution
        }
        ($obj | ConvertTo-Json -Compress) | Add-Content -Path $humOutPath -Encoding UTF8
    }

    $gsmCount = (Get-Content $gsmOutPath).Count
    $humCount = (Get-Content $humOutPath).Count

    Write-Host "Prepared GSM8K records: $gsmCount" -ForegroundColor Green
    Write-Host "Prepared HumanEval records: $humCount" -ForegroundColor Green
}

function Test-GpuCapacity {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $memMiB = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | Select-Object -First 1).Trim()
        if ($memMiB) {
            $memGiB = [math]::Round(([double]$memMiB / 1024), 2)
            Write-Host "Detected GPU VRAM: $memGiB GiB" -ForegroundColor Green
            if ([double]$memMiB -lt 22000) {
                Write-Host "Warning: VRAM is below 22 GiB. The script will continue." -ForegroundColor Yellow
            }
        }
    }
    else {
        Write-Host "nvidia-smi not found. Skipping GPU capacity check." -ForegroundColor Yellow
    }
}

Write-Section "Bootstrap Parishad Experiment Zero"
Assert-Command -CommandName $PythonExe -InstallHint "Install Python 3.10+ and make sure it is in PATH."

Set-Location $WorkspacePath
Test-GpuCapacity

Write-Section "Create and Activate Virtual Environment"
$venvPath = Join-Path $WorkspacePath $VenvDir
if (-not (Test-Path $venvPath)) {
    & $PythonExe -m venv $venvPath
}

$activatePath = Join-Path $venvPath "Scripts/Activate.ps1"
if (-not (Test-Path $activatePath)) {
    throw "Cannot find activation script at $activatePath"
}
. $activatePath

Write-Section "Install Dependencies"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[benchmark]"
python -m pip install huggingface_hub

Write-Section "Download HF GGUF and Configure Parishad"
if (-not [string]::IsNullOrWhiteSpace($env:HF_TOKEN)) {
    Write-Host "HF_TOKEN detected in environment (will be used automatically by huggingface_hub)." -ForegroundColor Green
}

$resolvedModelPath = Join-Path (if ([System.IO.Path]::IsPathRooted($ModelOutputDir)) { $ModelOutputDir } else { Join-Path $WorkspacePath $ModelOutputDir }) $HfFile

if (-not $SkipModelPull) {
    $resolvedModelPath = Get-HfGgufModel -Repo $HfRepo -File $HfFile -OutputDir $ModelOutputDir
}
elseif (-not (Test-Path $resolvedModelPath)) {
    throw "-SkipModelPull was set, but model file does not exist at '$resolvedModelPath'."
}

Update-ParishadConfig -TargetModelPath $resolvedModelPath

if (-not $SkipDatasetDownload) {
    Write-Section "Download and Prepare Datasets"
    Initialize-BenchmarkDatasets -Root $WorkspacePath
}

if (-not $SkipBenchmarkRun) {
    Write-Section "Run Benchmark Scripts"
    python scripts/experiment_zero.py `
        --benchmark $Benchmark `
        --part $Part `
        --seed $Seed `
        --total-problems $TotalProblems `
        --chunk-size $ChunkSize `
        --output-dir $OutputDir `
        --gsm8k-dataset-path dataset/processed/gsm8k_100.jsonl `
        --humaneval-dataset-path dataset/processed/humaneval_100.jsonl
}

Write-Section "Done"
Write-Host "Results directory: $OutputDir" -ForegroundColor Green
Write-Host "Model source: https://huggingface.co/$HfRepo/resolve/main/$HfFile" -ForegroundColor Green
Write-Host "To re-run only benchmarks, use: .\scripts\bootstrap_experiment_zero.ps1 -SkipDatasetDownload -SkipModelPull" -ForegroundColor Green
