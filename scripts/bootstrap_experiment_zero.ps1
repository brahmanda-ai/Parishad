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

    function Write-Utf8NoBomFile {
        param(
            [string]$Path,
            [string]$Content
        )

        $encoding = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($Path, $Content, $encoding)
    }

    function Add-Utf8NoBomLine {
        param(
            [string]$Path,
            [string]$Line
        )

        $encoding = New-Object System.Text.UTF8Encoding($false)
        $stream = New-Object System.IO.StreamWriter($Path, $true, $encoding)
        try {
            $stream.WriteLine($Line)
        }
        finally {
            $stream.Dispose()
        }
    }

    function Update-ParishadConfig {
        param([string]$TargetModelPath)

        function Set-OrAddProperty {
            param(
                [object]$Object,
                [string]$Name,
                [object]$Value
            )

            if ($Object.PSObject.Properties[$Name]) {
                $Object.$Name = $Value
            }
            else {
                $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value -Force
            }
        }

        $configDir = Join-Path $HOME ".parishad"
        $configPath = Join-Path $configDir "config.json"

        New-Item -ItemType Directory -Force -Path $configDir | Out-Null

        $cfg = $null
        if (Test-Path $configPath) {
            try {
                $raw = Get-Content -Raw $configPath
                if (-not [string]::IsNullOrWhiteSpace($raw)) {
                    $cfg = $raw | ConvertFrom-Json
                }
            }
            catch {
                Write-Host "Warning: Existing config.json is invalid JSON; recreating minimal config." -ForegroundColor Yellow
                $cfg = $null
            }
        }

        if ($null -eq $cfg -or $cfg -is [string] -or $cfg -is [array]) {
            $cfg = [pscustomobject]@{}
        }

        $rootPropNames = @($cfg.PSObject.Properties | ForEach-Object { $_.Name })
        if ($rootPropNames -notcontains "session") {
            $cfg | Add-Member -NotePropertyName session -NotePropertyValue ([pscustomobject]@{})
        }
        elseif ($null -eq $cfg.session -or $cfg.session -is [string] -or $cfg.session -is [array]) {
            $cfg.session = [pscustomobject]@{}
        }

        if ($rootPropNames -notcontains "model_config") {
            $cfg | Add-Member -NotePropertyName model_config -NotePropertyValue ([pscustomobject]@{})
        }
        elseif ($null -eq $cfg.model_config -or $cfg.model_config -is [string] -or $cfg.model_config -is [array]) {
            $cfg.model_config = [pscustomobject]@{}
        }

        if ($rootPropNames -notcontains "models") {
            $cfg | Add-Member -NotePropertyName models -NotePropertyValue ([pscustomobject]@{})
        }
        elseif ($null -eq $cfg.models -or $cfg.models -is [string] -or $cfg.models -is [array]) {
            $cfg.models = [pscustomobject]@{}
        }

        $modelKey = "local:qwen2.5-3b"

        Set-OrAddProperty -Object $cfg -Name "setup_complete" -Value $true
        Set-OrAddProperty -Object $cfg.session -Name "backend" -Value "llama_cpp"
        Set-OrAddProperty -Object $cfg.session -Name "model" -Value $modelKey

        $sessionPropNames = @($cfg.session.PSObject.Properties | ForEach-Object { $_.Name })
        if ($sessionPropNames -notcontains "sabha" -or -not $cfg.session.sabha) {
            Set-OrAddProperty -Object $cfg.session -Name "sabha" -Value "madhyam"
        }

        Set-OrAddProperty -Object $cfg.session -Name "model_map" -Value ([pscustomobject]@{
            small = $modelKey
            mid   = $modelKey
            big   = $modelKey
        })

        $modelEntry = [pscustomobject]@{
            source = "local"
            format = "gguf"
            path = $TargetModelPath
            size_bytes = (Get-Item $TargetModelPath).Length
        }
        $cfg.models | Add-Member -NotePropertyName $modelKey -NotePropertyValue $modelEntry -Force

        Set-OrAddProperty -Object $cfg.model_config -Name "n_gpu_layers" -Value -1
        Set-OrAddProperty -Object $cfg.model_config -Name "n_ctx" -Value 8192

        Write-Utf8NoBomFile -Path $configPath -Content ($cfg | ConvertTo-Json -Depth 100)

        Write-Host "Configured Parishad model backend: llama_cpp" -ForegroundColor Green
        Write-Host "Configured Parishad model file: $TargetModelPath" -ForegroundColor Green
    }

    function Get-HfGgufModel {
        param(
            [string]$Repo,
            [string]$File,
            [string]$OutputDir,
            [string]$PythonCmd
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
)
print(path)
"@

        $tempScript = Join-Path $env:TEMP "parishad_hf_download.py"
        Set-Content -Path $tempScript -Encoding UTF8 -Value $downloadScript

        $oldNativePref = $null
        if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
            $oldNativePref = $Global:PSNativeCommandUseErrorActionPreference
            $Global:PSNativeCommandUseErrorActionPreference = $false
        }

        $output = & $PythonCmd $tempScript 2>&1
        $exitCode = $LASTEXITCODE

        if ($null -ne $oldNativePref) {
            $Global:PSNativeCommandUseErrorActionPreference = $oldNativePref
        }

        if ($exitCode -ne 0) {
            throw "Failed to download model from Hugging Face (exit $exitCode). Output:`n$($output -join "`n")"
        }

        $modelPath = ""
        $outputLines = @($output)
        for ($i = $outputLines.Count - 1; $i -ge 0; $i--) {
            $candidate = $outputLines[$i].ToString().Trim()
            if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path $candidate)) {
                $modelPath = $candidate
                break
            }
        }

        if ([string]::IsNullOrWhiteSpace($modelPath)) {
            $modelPath = ($output | Select-Object -Last 1).ToString().Trim()
        }
        if (-not (Test-Path $modelPath)) {
            throw "Model download reported path '$modelPath' but file was not found."
        }

        Write-Host "Downloaded model: $modelPath" -ForegroundColor Green
        return $modelPath
    }

    function Assert-NonGatedHfModel {
        param(
            [string]$Repo,
            [string]$PythonCmd
        )

        $checkScript = @"
import json
from huggingface_hub import HfApi

repo = r"$Repo"
info = HfApi().model_info(repo_id=repo)
gated = info.gated

if isinstance(gated, str):
    gated = gated.strip().lower() not in {"", "false", "none", "no"}

print(json.dumps({"repo": repo, "gated": bool(gated)}))
"@

        $tempScript = Join-Path $env:TEMP "parishad_hf_gated_check.py"
        Set-Content -Path $tempScript -Encoding UTF8 -Value $checkScript

        $oldNativePref = $null
        if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
            $oldNativePref = $Global:PSNativeCommandUseErrorActionPreference
            $Global:PSNativeCommandUseErrorActionPreference = $false
        }

        $output = & $PythonCmd $tempScript 2>&1
        $exitCode = $LASTEXITCODE

        if ($null -ne $oldNativePref) {
            $Global:PSNativeCommandUseErrorActionPreference = $oldNativePref
        }

        if ($exitCode -ne 0) {
            Write-Host "Warning: Could not verify gating status for '$Repo'. Continuing with download attempt." -ForegroundColor Yellow
            return
        }

        $jsonLine = ""
        $outputLines = @($output)
        for ($i = $outputLines.Count - 1; $i -ge 0; $i--) {
            $candidate = $outputLines[$i].ToString().Trim()
            if ($candidate.StartsWith("{")) {
                $jsonLine = $candidate
                break
            }
        }

        if ([string]::IsNullOrWhiteSpace($jsonLine)) {
            Write-Host "Warning: Could not parse gating metadata for '$Repo'. Continuing with download attempt." -ForegroundColor Yellow
            return
        }

        $meta = $null
        try {
            $meta = $jsonLine | ConvertFrom-Json
        }
        catch {
            Write-Host "Warning: Could not parse gating status for '$Repo'. Continuing with download attempt." -ForegroundColor Yellow
            return
        }

        if ($meta.gated -eq $true) {
            throw "Selected model repository '$Repo' is gated. Choose a non-gated model for benchmarks. Suggested non-gated GGUF repos: bartowski/Qwen2.5-3B-Instruct-GGUF, bartowski/Qwen2.5-1.5B-Instruct-GGUF, bartowski/SmolLM2-1.7B-Instruct-GGUF."
        }

        Write-Host "Verified non-gated Hugging Face model repo: $Repo" -ForegroundColor Green
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
            Add-Utf8NoBomLine -Path $gsmOutPath -Line ($obj | ConvertTo-Json -Compress)
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
            Add-Utf8NoBomLine -Path $humOutPath -Line ($obj | ConvertTo-Json -Compress)
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

    $resolvedModelDir = if ([System.IO.Path]::IsPathRooted($ModelOutputDir)) {
        $ModelOutputDir
    }
    else {
        Join-Path $WorkspacePath $ModelOutputDir
    }
    $resolvedModelPath = Join-Path $resolvedModelDir $HfFile

    if (-not $SkipModelPull) {
        Assert-NonGatedHfModel -Repo $HfRepo -PythonCmd $PythonExe
        $resolvedModelPath = Get-HfGgufModel -Repo $HfRepo -File $HfFile -OutputDir $ModelOutputDir -PythonCmd $PythonExe
    }
    elseif (-not (Test-Path $resolvedModelPath)) {
        if (Test-Path $resolvedModelDir) {
            $existingModel = Get-ChildItem -Path $resolvedModelDir -Filter "*.gguf" -File -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
            if ($existingModel) {
                $resolvedModelPath = $existingModel.FullName
                Write-Host "SkipModelPull: requested file not found; using existing GGUF '$resolvedModelPath'" -ForegroundColor Yellow
            }
            else {
                throw "-SkipModelPull was set, but no .gguf file exists in '$resolvedModelDir'."
            }
        }
        else {
            throw "-SkipModelPull was set, but model directory does not exist at '$resolvedModelDir'."
        }
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
