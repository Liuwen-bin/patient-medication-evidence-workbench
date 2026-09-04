[CmdletBinding()]
param(
    [string]$HealthRoot = "C:\Users\Administrator\Desktop\mcp\health-record-mcp\Agent",
    [string]$DrugRoot = "C:\Users\Administrator\Downloads\dm_spl_release_homeopathic\homeopathic\dailymed_lightrag",
    [string]$ModelEnvPath = "C:\Users\Administrator\Downloads\dm_spl_release_homeopathic\homeopathic\LightRAG\.env",
    [switch]$CommitSynthetic
)

$ErrorActionPreference = "Stop"
$reviewRoot = Split-Path $PSScriptRoot -Parent
$python = (Get-Command python -ErrorAction Stop).Source
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$runDirectory = Join-Path $reviewRoot "artifacts/live-runs/$timestamp-$([guid]::NewGuid().ToString('N').Substring(0, 8))"
$canonicalReport = Join-Path $reviewRoot "artifacts/evaluation/online-integration-report.json"
$runReport = Join-Path $runDirectory "online-integration-report.json"
$sourceHealthDb = Join-Path $HealthRoot "data/chinese-demo-record.sqlite"
$evaluationHealthDb = Join-Path $runDirectory "health-eval.sqlite"
$startedProcesses = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()
$priorMilvusUri = $env:MILVUS_URI

function Write-FailureReport {
    param([string]$Code, [string]$Stage, [string]$ExceptionType)

    $caseIds = @(
        "online-single-complete",
        "online-ambiguous-variant",
        "online-allergy-ingredient",
        "online-partial-unmapped",
        "online-evidence-degraded"
    )
    $report = [ordered]@{
        schemaVersion = "1.0"
        generatedAt = (Get-Date).ToUniversalTime().ToString("o")
        execution = [ordered]@{
            mode = "online_integration"
            network = $true
            realModel = $false
            realDatabases = $false
            completed = $false
        }
        thresholds = @{}
        metrics = [ordered]@{ taskCompletionRate = 0; metricsCoverage = 0 }
        acceptancePassed = $false
        failure = [ordered]@{
            code = $Code
            stage = $Stage
            exceptionType = $ExceptionType
        }
        cases = @($caseIds | ForEach-Object {
            [ordered]@{
                caseId = $_
                passed = $false
                durationMs = 0
                missingMetrics = @("run")
                failureCode = $Code
            }
        })
    }
    New-Item -ItemType Directory -Force (Split-Path $canonicalReport -Parent) | Out-Null
    $json = $report | ConvertTo-Json -Depth 10
    Set-Content -LiteralPath $runReport -Value $json -Encoding UTF8
    Set-Content -LiteralPath $canonicalReport -Value $json -Encoding UTF8
}

function Test-TcpPort {
    param([int]$Port, [int]$TimeoutMilliseconds = 500)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $async = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne($TimeoutMilliseconds)) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Wait-TcpPort {
    param([int]$Port, [string]$ServiceName, [int]$TimeoutSeconds = 60)

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-TcpPort -Port $Port) {
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw [System.TimeoutException]::new("$ServiceName did not become ready on its expected port.")
}

function Start-OwnedProcess {
    param(
        [string]$Name,
        [string]$WorkingDirectory,
        [string[]]$Arguments
    )

    $stdout = Join-Path $runDirectory "$Name.stdout.log"
    $stderr = Join-Path $runDirectory "$Name.stderr.log"
    $process = Start-Process -FilePath $python -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $startedProcesses.Add($process)
    return $process
}

New-Item -ItemType Directory -Force $runDirectory | Out-Null
$stage = "validate_inputs"
try {
    foreach ($requiredPath in @($HealthRoot, $DrugRoot, $ModelEnvPath, $sourceHealthDb)) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw [System.IO.FileNotFoundException]::new("A required live-evaluation path is unavailable.")
        }
    }
    $sourceHashBefore = (Get-FileHash -LiteralPath $sourceHealthDb -Algorithm SHA256).Hash
    $sourceTimestampBefore = (Get-Item -LiteralPath $sourceHealthDb).LastWriteTimeUtc

    $stage = "seed_health_database"
    & $python (Join-Path $PSScriptRoot "seed-live-evaluation.py") `
        --source-db $sourceHealthDb `
        --output-db $evaluationHealthDb `
        --resources (Join-Path $reviewRoot "evaluation/live-health-resources.jsonl")
    if ($LASTEXITCODE -ne 0) {
        throw [System.InvalidOperationException]::new("Health evaluation database seeding failed.")
    }

    # Pass the model configuration by path only. The launcher never reads or prints the file.
    $env:AGENT_MODEL_ENV_PATH = (Resolve-Path -LiteralPath $ModelEnvPath).Path
    $env:AGENT_LLM_ENABLED = "true"
    $env:EHR_DB_PATH = (Resolve-Path -LiteralPath $evaluationHealthDb).Path
    $env:HEALTH_MCP_URL = "http://127.0.0.1:8000/mcp"
    $env:DRUG_MCP_URL = "http://127.0.0.1:8010/mcp"
    $env:REVIEW_API_HOST = "127.0.0.1"
    $env:REVIEW_API_PORT = "8020"
    $env:REVIEW_API_KEY = "local-synthetic-evaluation"
    $env:REVIEW_API_REVIEWER_ID = "pharmacist-eval"
    $env:REVIEW_DB_PATH = Join-Path $runDirectory "reviews.sqlite"
    $env:REVIEW_CHECKPOINT_DB = Join-Path $runDirectory "review-checkpoints.sqlite"
    $env:EVAL_RUN_DIR = $runDirectory
    $env:EVAL_HEALTH_DB_PATH = $evaluationHealthDb
    $env:EVAL_SOURCE_HEALTH_DB_PATH = $sourceHealthDb
    $env:PYTHONPATH = "$reviewRoot\src;$DrugRoot\src;$($env:PYTHONPATH)"

    $stage = "start_health_mcp"
    if (-not (Test-TcpPort -Port 8000)) {
        Start-OwnedProcess -Name "health-mcp" -WorkingDirectory $HealthRoot `
            -Arguments @("mcp/mcp_server.py", "--transport", "http", "--host", "127.0.0.1", "--port", "8000") | Out-Null
        Wait-TcpPort -Port 8000 -ServiceName "Health MCP"
    }
    elseif ($CommitSynthetic) {
        throw [System.InvalidOperationException]::new("Port 8000 is already owned; isolated commit cannot be attested.")
    }

    $stage = "start_drug_mcp"
    if (-not (Test-TcpPort -Port 8010)) {
        Start-OwnedProcess -Name "drug-mcp" -WorkingDirectory $DrugRoot `
            -Arguments @("-m", "dailymed_lightrag.drug_mcp_server", "--transport", "http", "--host", "127.0.0.1", "--port", "8010") | Out-Null
        Wait-TcpPort -Port 8010 -ServiceName "Drug MCP"
    }

    $stage = "start_review_api"
    if (-not (Test-TcpPort -Port 8020)) {
        Start-OwnedProcess -Name "review-api" -WorkingDirectory $reviewRoot `
            -Arguments @("-c", '"from medication_review_agent.api import main; main()"') | Out-Null
        Wait-TcpPort -Port 8020 -ServiceName "Review API"
    }
    elseif ($CommitSynthetic) {
        throw [System.InvalidOperationException]::new("Port 8020 is already owned; isolated commit cannot be attested.")
    }
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8020/api/health" -TimeoutSec 5
    if ($health.status -ne "healthy") {
        throw [System.InvalidOperationException]::new("Review API health contract failed.")
    }

    $stage = "start_degraded_profile"
    if ((Test-TcpPort -Port 8011) -or (Test-TcpPort -Port 8021)) {
        throw [System.InvalidOperationException]::new("Degraded profile ports are already owned.")
    }
    $env:MILVUS_URI = "http://127.0.0.1:65534"
    Start-OwnedProcess -Name "drug-mcp-rag-unavailable" -WorkingDirectory $DrugRoot `
        -Arguments @("-m", "dailymed_lightrag.drug_mcp_server", "--transport", "http", "--host", "127.0.0.1", "--port", "8011") | Out-Null
    Wait-TcpPort -Port 8011 -ServiceName "Degraded Drug MCP"
    $env:DRUG_MCP_URL = "http://127.0.0.1:8011/mcp"
    $env:REVIEW_API_PORT = "8021"
    $env:REVIEW_DB_PATH = Join-Path $runDirectory "reviews-rag-unavailable.sqlite"
    $env:REVIEW_CHECKPOINT_DB = Join-Path $runDirectory "review-checkpoints-rag-unavailable.sqlite"
    Start-OwnedProcess -Name "review-api-rag-unavailable" -WorkingDirectory $reviewRoot `
        -Arguments @("-c", '"from medication_review_agent.api import main; main()"') | Out-Null
    Wait-TcpPort -Port 8021 -ServiceName "Degraded Review API"
    $degradedHealth = Invoke-RestMethod -Uri "http://127.0.0.1:8021/api/health" -TimeoutSec 5
    if ($degradedHealth.status -ne "healthy") {
        throw [System.InvalidOperationException]::new("Degraded Review API health contract failed.")
    }
    if ([string]::IsNullOrEmpty($priorMilvusUri)) {
        Remove-Item Env:MILVUS_URI -ErrorAction SilentlyContinue
    } else {
        $env:MILVUS_URI = $priorMilvusUri
    }

    $stage = "run_online_cases"
    $arguments = @(
        "-m", "medication_review_agent.online_evaluation",
        "--cases", (Join-Path $reviewRoot "evaluation/online-cases.jsonl"),
        "--output", $runReport,
        "--base-url", "http://127.0.0.1:8020",
        "--rag-unavailable-base-url", "http://127.0.0.1:8021",
        "--api-key", $env:REVIEW_API_KEY,
        "--reviewer-id", $env:REVIEW_API_REVIEWER_ID
    )
    if ($CommitSynthetic) {
        $arguments += "--commit-synthetic"
    }
    & $python @arguments
    $runnerExitCode = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $runReport)) {
        throw [System.InvalidOperationException]::new("Online runner did not produce its report.")
    }
    New-Item -ItemType Directory -Force (Split-Path $canonicalReport -Parent) | Out-Null
    Copy-Item -LiteralPath $runReport -Destination $canonicalReport -Force
    $stage = "verify_source_database"
    $sourceHashAfter = (Get-FileHash -LiteralPath $sourceHealthDb -Algorithm SHA256).Hash
    $sourceTimestampAfter = (Get-Item -LiteralPath $sourceHealthDb).LastWriteTimeUtc
    if ($sourceHashAfter -ne $sourceHashBefore -or $sourceTimestampAfter -ne $sourceTimestampBefore) {
        Write-FailureReport -Code "SOURCE_DATABASE_CHANGED" -Stage $stage `
            -ExceptionType "SourceDatabaseIntegrityError"
        exit 1
    }
    exit $runnerExitCode
}
catch {
    $failureCode = if ($stage -eq "verify_source_database") {
        "SOURCE_DATABASE_CHANGED"
    } else {
        "DEPENDENCY_START_FAILED"
    }
    Write-FailureReport -Code $failureCode -Stage $stage `
        -ExceptionType $_.Exception.GetType().Name
    Write-Error "Online evaluation failed during '$stage'. See the sanitized report."
    exit 1
}
finally {
    if ([string]::IsNullOrEmpty($priorMilvusUri)) {
        Remove-Item Env:MILVUS_URI -ErrorAction SilentlyContinue
    } else {
        $env:MILVUS_URI = $priorMilvusUri
    }
    foreach ($process in $startedProcesses) {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
