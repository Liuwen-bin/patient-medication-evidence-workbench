from datetime import UTC, datetime

import pytest

from medication_review_agent.models import (
    AuditEvent, EvidenceItem, Finding, FindingStatus, HumanDecision, MedicationMapping, MedicationRecord,
    ReviewSnapshot, ReviewStatus, RunMetrics,
)
from medication_review_agent.report import (
    ReportNotSigned, build_signed_report, render_report_html, render_report_json,
)


def signed_review_snapshot() -> ReviewSnapshot:
    now = datetime.now(UTC)
    return ReviewSnapshot(
        reviewId="r1", patientRef="FHIR:Patient/p1", asOf="2026-08-31",
        status=ReviewStatus.SIGNED_OFF, createdAt=now, updatedAt=now,
        medications=[MedicationRecord(medicationId="m1", name="METFORMIN <script>", patientEvidenceRefs=["FHIR:MedicationRequest/m1"])],
        medicationMappings=[MedicationMapping(medicationId="m1", sourceName="METFORMIN <script>", matchClass="UNMAPPED")],
        findings=[Finding(
            findingId="f1", reviewType="EVIDENCE_GAP", summary="No mapped product",
            attentionLevel="HIGH", confidence=1.0, status=FindingStatus.ACCEPTED,
        )],
        evidenceIndex=[EvidenceItem(
            evidenceId="e1", source="SPL", evidenceRef="SPL:doc-1#warnings",
            graphProvenance={
                "graphBackend": "neo4j", "graphWorkspace": "dailymed-workspace",
                "graphDatabase": "neo4j-db", "fallbackUsed": False,
                "consistency": {"status": "CONSISTENT"},
            },
        )],
        humanDecisions=[HumanDecision(
            action="REJECT_FINDING", reviewerId="pharmacist-demo",
            findingId="f2", note="Not clinically applicable",
        )],
        auditEvents=[AuditEvent(
            node="verify_findings", tool="validate_evidence", requestId="request-123",
            resultStatus="OK", modelId="model-1", promptVersion="planner-v1",
            argumentSummary={"claimCount": 1}, evidenceRefs=["SPL:doc-1#warnings"],
            latencyMs=7, retryCount=2, inputTokens=11, outputTokens=13, estimatedCost=0.25,
        )],
        metrics=RunMetrics(toolLatencyMs=12, retries=1, estimatedCost=0.0),
    )


def test_unsigned_review_cannot_render_signed_report() -> None:
    snapshot = signed_review_snapshot().model_copy(update={"status": ReviewStatus.READY_FOR_SIGN_OFF})
    with pytest.raises(ReportNotSigned):
        build_signed_report(snapshot, reviewer_id=None)


def test_report_contains_unmapped_medication_metrics_and_scope() -> None:
    report = build_signed_report(signed_review_snapshot(), reviewer_id="pharmacist-demo")
    assert report.unmappedMedications
    assert report.metrics.toolLatencyMs == 12
    assert "not an autonomous prescribing decision" in report.scopeStatement


def test_report_json_and_html_share_escaped_model_content() -> None:
    report = build_signed_report(signed_review_snapshot(), reviewer_id="pharmacist-demo")
    assert "METFORMIN <script>" in render_report_json(report)
    html = render_report_html(report)
    assert "METFORMIN &lt;script&gt;" in html
    assert "METFORMIN <script>" not in html


def test_report_discloses_evidence_graph_audit_and_reviewer_details() -> None:
    report = build_signed_report(signed_review_snapshot(), reviewer_id="pharmacist-demo")
    assert report.auditTrail[0].requestId == "request-123"
    assert report.evidenceProvenance[0]["graphWorkspace"] == "dailymed-workspace"
    assert report.reviewerDecisions[0].note == "Not clinically applicable"
    html = render_report_html(report)
    for expected in ("dailymed-workspace", "neo4j-db", "request-123", "model-1", "planner-v1", "Not clinically applicable", "claimCount", "SPL:doc-1#warnings", "latency=7", "retries=2", "input_tokens=11", "output_tokens=13", "cost=0.25"):
        assert expected in html
