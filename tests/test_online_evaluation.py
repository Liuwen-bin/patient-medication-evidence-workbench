from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from medication_review_agent.online_evaluation import (
    OnlineCase,
    OnlineCaseResult,
    OnlineEvaluationRunner,
    _run_all,
    build_online_report,
    load_online_cases,
    write_sanitized_online_report,
)


def _snapshot(status: str, version: int) -> dict:
    base = {
        "reviewId": "online-review",
        "version": version,
        "status": status,
        "patientRef": "FHIR:Patient/p1",
        "medicationMappings": [],
        "findings": [],
        "auditEvents": [],
        "metrics": {
            "toolLatencyMs": 10,
            "retries": 0,
            "inputTokens": 12,
            "outputTokens": 4,
            "estimatedCost": 0.001,
        },
        "modelCalls": [{
            "modelId": "configured-model",
            "promptVersion": "intent-v1",
            "inputTokens": 12,
            "outputTokens": 4,
            "estimatedCost": 0.001,
            "latencyMs": 5,
            "fallback": False,
        }],
        "writebackStatus": "NOT_REQUESTED",
        "writebackJob": None,
    }
    if status == "AWAITING_PATIENT_CONFIRMATION":
        base["candidates"] = [{"id": "p1", "patientNumber": "DEMO-LIVE-001"}]
    if status == "AWAITING_MAPPING_CONFIRMATION":
        base["medicationMappings"] = [{
            "medicationId": "med-1",
            "mappingConfirmationRequired": True,
            "candidates": [{
                "productId": "DRUG_PRODUCT::10191-1246",
                "productCode": "10191-1246",
            }],
        }]
    if status in {"AWAITING_FINDING_REVIEW", "READY_FOR_SIGN_OFF", "SIGNED_OFF"}:
        base["medicationMappings"] = [{
            "medicationId": "med-1",
            "matchClass": "EXACT_IDENTIFIER",
            "selectedProductId": "DRUG_PRODUCT::10191-1246",
        }]
        base["findings"] = [{
            "findingId": "finding-1",
            "reviewType": "LABEL_EVIDENCE_REVIEW",
            "status": "PENDING" if status == "AWAITING_FINDING_REVIEW" else "ACCEPTED",
            "patientEvidenceRefs": ["FHIR:MedicationRequest/med-1"],
            "labelEvidenceRefs": ["SPL:doc-1#warnings"],
        }]
    if status == "SIGNED_OFF":
        base["writebackStatus"] = "NOT_REQUESTED"
        base["retrievalAttempts"] = {"DRUG_PRODUCT::10191-1246:warnings": 2}
    return base


@pytest.mark.asyncio
async def test_online_runner_drives_each_interrupt_and_previews_writeback() -> None:
    mutations: list[httpx.Request] = []
    decision_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal decision_count
        if request.method == "POST":
            mutations.append(request)
        path = request.url.path
        if path == "/api/reviews" and request.method == "POST":
            return httpx.Response(201, json=_snapshot("CREATED", 0))
        if path.endswith("/run"):
            return httpx.Response(200, json=_snapshot("AWAITING_PATIENT_CONFIRMATION", 1))
        if path.endswith("/decisions"):
            decision_count += 1
            statuses = [
                "AWAITING_MAPPING_CONFIRMATION",
                "AWAITING_FINDING_REVIEW",
                "READY_FOR_SIGN_OFF",
            ]
            return httpx.Response(200, json=_snapshot(statuses[decision_count - 1], decision_count + 1))
        if path.endswith("/complete"):
            return httpx.Response(200, json=_snapshot("SIGNED_OFF", 5))
        if path.endswith("/writeback/prepare"):
            prepared = _snapshot("SIGNED_OFF", 6)
            prepared["writebackStatus"] = "PREPARED"
            prepared["writebackJob"] = {
                "jobId": "writeback-online-review-5",
                "reviewVersion": 5,
                "expectedVersion": 5,
                "bundleHash": "a" * 64,
                "resources": [{"resourceType": "DetectedIssue", "id": "mr-di-1"}],
                "warnings": [],
                "blockedFindings": [],
            }
            return httpx.Response(200, json=prepared)
        if path.endswith("/audit"):
            return httpx.Response(200, json=[{
                "node": "retrieve_label_evidence",
                "tool": "search_label_evidence",
                "resultStatus": "OK",
                "latencyMs": 7,
                "retryCount": 1,
            }])
        return httpx.Response(404, json={"detail": "not found"})

    runner = OnlineEvaluationRunner(
        base_url="http://review.test",
        api_key="test-key",
        reviewer_id="pharmacist-eval",
        transport=httpx.MockTransport(handler),
    )
    case = OnlineCase.model_validate({
        "caseId": "online-single-complete",
        "patientId": "DEMO-LIVE-001",
        "question": "核查标签警告",
        "mappingSelection": {"strategy": "productCode", "value": "10191-1246"},
        "findingDecisions": {"LABEL_EVIDENCE_REVIEW": "ACCEPT_FINDING"},
        "expected": {"requiresWritebackPreview": True},
    })

    result = await runner.run_case(case)

    assert result.finalReviewStatus == "SIGNED_OFF"
    assert result.writebackStatus == "PREPARED"
    assert result.interrupts == [
        "PATIENT_CONFIRMATION", "MAPPING_CONFIRMATION", "FINDING_REVIEW"
    ]
    assert result.operational["nodeTrace"] == ["retrieve_label_evidence"]
    assert result.operational["toolStatuses"] == {
        "search_label_evidence": {"OK": 1}
    }
    assert result.operational["narrativeAttempts"] == {
        "maximumPerScope": 2,
        "total": 2,
    }
    assert result.operational["mappingInterrupts"] == 1
    assert result.operational["writebackPreviewCount"] == 1
    assert all(request.headers["x-reviewer-id"] == "pharmacist-eval" for request in mutations)


def test_online_case_manifest_has_five_stable_cases() -> None:
    cases = load_online_cases("evaluation/online-cases.jsonl")
    assert [case.caseId for case in cases] == [
        "online-single-complete",
        "online-ambiguous-variant",
        "online-allergy-ingredient",
        "online-partial-unmapped",
        "online-evidence-degraded",
    ]


def test_seed_script_never_modifies_source_database(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    output = tmp_path / "run" / "health-eval.sqlite"
    with sqlite3.connect(source) as connection:
        connection.execute(
            "CREATE TABLE fhir_resources (resource_type TEXT NOT NULL, "
            "resource_id TEXT NOT NULL, json TEXT NOT NULL, "
            "PRIMARY KEY(resource_type, resource_id))"
        )
        connection.execute(
            "INSERT INTO fhir_resources VALUES ('Patient', 'original', '{}')"
        )
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/seed-live-evaluation.py",
            "--source-db", str(source),
            "--output-db", str(output),
            "--resources", "evaluation/live-health-resources.jsonl",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with sqlite3.connect(output) as connection:
        assert connection.execute(
            "SELECT count(*) FROM fhir_resources WHERE resource_type='Patient'"
        ).fetchone()[0] == 6


def test_online_report_contains_no_credentials_or_patient_names(tmp_path: Path) -> None:
    report_path = write_sanitized_online_report(
        tmp_path / "online.json",
        {
            "execution": {"mode": "online_integration"},
            "cases": [{"caseId": "case-1", "passed": False, "failureCode": "MODEL_ERROR"}],
        },
    )
    text = report_path.read_text(encoding="utf-8")
    assert "AGENT_LLM_API_KEY" not in text
    assert "张三" not in text
    assert "李四" not in text
    assert "chain_of_thought" not in text.casefold()


def test_live_launcher_enforces_isolation_and_secret_path_boundary() -> None:
    script = Path("scripts/run-live-evaluation.ps1")
    assert script.is_file()
    text = script.read_text(encoding="utf-8")

    assert "AGENT_MODEL_ENV_PATH" in text
    assert "Get-Content $ModelEnvPath" not in text
    assert "artifacts/live-runs" in text.replace("\\", "/")
    assert "seed-live-evaluation.py" in text
    assert "Start-Process" in text
    assert "-WindowStyle Hidden" in text
    assert "from medication_review_agent.api import main; main()" in text
    assert "startedProcesses" in text
    assert "Stop-Process" in text
    assert "online-integration-report.json" in text
    assert "DEPENDENCY_START_FAILED" in text
    assert "Get-FileHash" in text
    assert "SOURCE_DATABASE_CHANGED" in text


@pytest.mark.asyncio
async def test_online_runner_labels_dependency_timeouts() -> None:
    class TimedOutRunner:
        async def run_case(self, case: OnlineCase):
            raise httpx.ReadTimeout("Milvus-backed Drug MCP did not respond")

    case = OnlineCase(
        caseId="timeout-case",
        patientId="DEMO-LIVE-001",
        question="核查标签警告",
    )

    results = await _run_all([case], TimedOutRunner())

    assert results[0].failureCode == "DEPENDENCY_TIMEOUT"


def test_online_report_marks_only_observed_real_components() -> None:
    report = build_online_report([
        OnlineCaseResult(
            caseId="timeout-case",
            passed=False,
            durationMs=120_000,
            failureCode="DEPENDENCY_TIMEOUT",
            missingMetrics=["run"],
        )
    ])

    assert report["generatedAt"]
    assert report["gitSha"]
    assert report["execution"]["realModel"] is False
    assert report["execution"]["realDatabases"] is False
    assert report["execution"]["completed"] is True
    assert report["acceptancePassed"] is False
