from __future__ import annotations

import json

import pytest

from medication_review_agent.gateways import TimedToolResult
from medication_review_agent.models import (
    Finding,
    FindingStatus,
    ReviewSnapshot,
    ReviewStatus,
    ToolEnvelope,
    WritebackJob,
)
from medication_review_agent.writeback import (
    WritebackCoordinator,
    WritebackError,
    WritebackStateError,
    build_writeback_payload,
)


def finding(
    finding_id: str,
    *,
    status: FindingStatus = FindingStatus.ACCEPTED,
    review_type: str = "ROUTE_MISMATCH",
) -> Finding:
    return Finding(
        findingId=finding_id,
        reviewType=review_type,
        ruleId="route-v1",
        summary="Verified route mismatch.",
        attentionLevel="MEDIUM",
        confidence=0.9,
        medicationIds=["med-1"],
        selectedProductIds=["DRUG_PRODUCT::1"],
        patientEvidenceRefs=["FHIR:MedicationRequest/med-1"],
        labelEvidenceRefs=["SPL:doc-1#route"],
        labelEvidenceIds=[
            "evidence-60b43af3bfea24d82e6d1dc1f6c83fb174925401b3639345abe0a0cefa2e0138"
        ],
        status=status,
    )


def snapshot(
    *,
    status: ReviewStatus = ReviewStatus.SIGNED_OFF,
    findings: list[Finding] | None = None,
) -> ReviewSnapshot:
    return ReviewSnapshot(
        reviewId="review-1",
        status=status,
        question="核查给药途径",
        patientRef="FHIR:Patient/p1",
        contextSnapshot={
            "patient": {"name": "不应发送", "patientNumber": "secret"}
        },
        evidenceIndex=[{
            "evidenceId": "evidence-60b43af3bfea24d82e6d1dc1f6c83fb174925401b3639345abe0a0cefa2e0138",
            "source": "SPL",
            "evidenceRef": "SPL:doc-1#route",
            "medicationIds": ["med-1"],
            "productIds": ["DRUG_PRODUCT::1"],
            "topic": "route",
            "summary": "Verified route evidence.",
            "documentId": "doc-1",
            "documentVersion": "3",
            "sectionId": "route",
            "sourcePath": "labels/doc-1.xml",
            "contentHash": "a" * 64,
        }],
        findings=findings if findings is not None else [finding("accepted-1")],
        unresolvedItems=[{
            "kind": "PATIENT_FIELD_MISSING",
            "summary": "Pregnancy status is not recorded.",
            "medicationIds": ["med-1"],
            "evidenceRefs": ["FHIR:MedicationRequest/med-1"],
        }],
        humanDecisions=[{
            "action": "SIGN_OFF",
            "reviewerId": "pharmacist-1",
            "occurredAt": "2026-09-04T10:11:12+08:00",
        }],
        version=7,
    )


def timed_envelope(status: str, data: dict) -> TimedToolResult:
    return TimedToolResult(
        envelope=ToolEnvelope.model_validate({
            "schemaVersion": "1.0",
            "status": status,
            "data": data,
            "evidenceRefs": [],
            "warnings": [],
            "errors": [] if status == "OK" else ["writeback failed"],
            "provenance": {},
            "requestId": "request-1",
        }),
        latency_ms=1,
    )


class FakeHealthWritebackGateway:
    def __init__(self, preview: TimedToolResult, commit: TimedToolResult | None = None):
        self.preview = preview
        self.commit_result = commit
        self.preview_payload = None
        self.commit_args = None

    async def validate_writeback(self, payload):
        self.preview_payload = payload
        return self.preview

    async def commit_writeback(
        self, job_id, bundle_hash, expected_version, confirmed, reviewer_id
    ):
        self.commit_args = (
            job_id, bundle_hash, expected_version, confirmed, reviewer_id
        )
        assert self.commit_result is not None
        return self.commit_result


def preview_envelope() -> TimedToolResult:
    return timed_envelope("OK", {
        "jobId": "writeback-review-1-7",
        "reviewId": "review-1",
        "reviewVersion": 7,
        "expectedVersion": 7,
        "patientRef": "Patient/p1",
        "bundleHash": "a" * 64,
        "resources": [{"resourceType": "DetectedIssue", "id": "mr-di-1"}],
        "warnings": [],
        "blockedFindings": [],
    })


def test_payload_contains_only_completed_verified_review_data() -> None:
    review = snapshot(findings=[
        finding("accepted-1"),
        finding("rejected-1", status=FindingStatus.REJECTED),
        finding("pending-1", status=FindingStatus.PENDING),
    ])

    payload = build_writeback_payload(review, "pharmacist-1")
    encoded = json.dumps(payload, ensure_ascii=False)

    assert [item["findingId"] for item in payload["findings"]] == ["accepted-1"]
    assert payload["patientRef"] == "Patient/p1"
    assert payload["signedAt"] == "2026-09-04T02:11:12Z"
    assert "不应发送" not in encoded
    assert "secret" not in encoded
    assert "contextSnapshot" not in encoded


def test_accepted_evidence_gap_becomes_an_unresolved_task() -> None:
    gap = finding("gap-1", review_type="EVIDENCE_GAP")
    gap.labelEvidenceRefs = []
    review = snapshot(findings=[gap])

    payload = build_writeback_payload(review, "pharmacist-1")

    assert payload["findings"] == []
    assert any(
        item["kind"] == "LABEL_EVIDENCE_MISSING"
        for item in payload["unresolvedItems"]
    )


def test_accepted_missing_patient_field_becomes_patient_followup_task() -> None:
    gap = Finding(
        findingId="gap-patient-1",
        reviewType="EVIDENCE_GAP",
        ruleId="missing-patient-field-v1",
        summary="Patient allergy information is not recorded.",
        attentionLevel="HIGH",
        confidence=1.0,
        medicationIds=["med-1"],
        patientEvidenceRefs=["FHIR:MedicationRequest/med-1"],
        status=FindingStatus.ACCEPTED,
        missingField="allergies",
    )
    review = snapshot(findings=[gap])
    review.unresolvedItems = []

    payload = build_writeback_payload(review, "pharmacist-1")

    assert [item["kind"] for item in payload["unresolvedItems"]] == [
        "PATIENT_FIELD_MISSING"
    ]


def test_writeback_rejects_accepted_finding_with_unresolvable_evidence_id() -> None:
    review = snapshot()
    review.evidenceIndex = []

    with pytest.raises(WritebackStateError, match="valid SPL evidence"):
        build_writeback_payload(review, "pharmacist-1")


@pytest.mark.parametrize(
    "review_type", ["PRODUCT_UNMAPPED", "LABEL_EVIDENCE_MISSING"]
)
def test_dedicated_unresolved_finding_becomes_matching_task(
    review_type: str,
) -> None:
    unresolved = Finding(
        findingId="unresolved-1",
        reviewType=review_type,
        ruleId="unresolved-v1",
        summary="Pharmacist follow-up is required.",
        attentionLevel="HIGH",
        confidence=1.0,
        medicationIds=["med-1"],
        patientEvidenceRefs=["FHIR:MedicationRequest/med-1"],
        labelEvidenceRefs=[],
        status=FindingStatus.ACCEPTED,
    )

    payload = build_writeback_payload(
        snapshot(findings=[unresolved]), "pharmacist-1"
    )

    assert payload["findings"] == []
    assert [item["kind"] for item in payload["unresolvedItems"]] == [
        "PATIENT_FIELD_MISSING",
        review_type,
    ]


def test_writeback_payload_requires_a_persisted_signoff_decision() -> None:
    review = snapshot()
    review.humanDecisions = []

    with pytest.raises(WritebackStateError, match="sign-off decision"):
        build_writeback_payload(review, "pharmacist-1")


def test_writeback_payload_rejects_requester_different_from_persisted_signer() -> None:
    with pytest.raises(WritebackStateError, match="signed-off reviewer"):
        build_writeback_payload(snapshot(), "pharmacist-other")


@pytest.mark.asyncio
async def test_prepare_requires_signed_review_and_returns_typed_job() -> None:
    gateway = FakeHealthWritebackGateway(preview_envelope())
    coordinator = WritebackCoordinator(gateway)

    with pytest.raises(WritebackStateError):
        await coordinator.prepare(
            snapshot(status=ReviewStatus.RUNNING), "pharmacist-1"
        )
    job = await coordinator.prepare(snapshot(), "pharmacist-1")

    assert isinstance(job, WritebackJob)
    assert job.jobId == "writeback-review-1-7"
    assert job.reviewVersion == 7


@pytest.mark.asyncio
async def test_commit_sends_stored_job_values_only() -> None:
    committed = timed_envelope("OK", {
        "jobId": "writeback-review-1-7",
        "committed": True,
        "idempotentReplay": False,
        "bundleHash": "a" * 64,
        "created": ["DetectedIssue/mr-di-1"],
    })
    gateway = FakeHealthWritebackGateway(preview_envelope(), committed)
    job = WritebackJob.model_validate(preview_envelope().envelope.data)

    result = await WritebackCoordinator(gateway).commit(
        snapshot(), job, "pharmacist-1", confirmed=True
    )

    assert result["committed"] is True
    assert gateway.commit_args == (
        "writeback-review-1-7", "a" * 64, 7, True, "pharmacist-1"
    )


@pytest.mark.asyncio
async def test_commit_rejects_requester_different_from_persisted_signer() -> None:
    committed = timed_envelope("OK", {
        "jobId": "writeback-review-1-7",
        "committed": True,
        "idempotentReplay": False,
        "bundleHash": "a" * 64,
        "created": ["DetectedIssue/mr-di-1"],
    })
    gateway = FakeHealthWritebackGateway(preview_envelope(), committed)
    job = WritebackJob.model_validate(preview_envelope().envelope.data)

    with pytest.raises(WritebackStateError, match="signed-off reviewer"):
        await WritebackCoordinator(gateway).commit(
            snapshot(), job, "pharmacist-other", confirmed=True
        )

    assert gateway.commit_args is None


@pytest.mark.asyncio
async def test_non_ok_envelope_maps_machine_readable_error() -> None:
    failed = timed_envelope("ERROR", {
        "error": {
            "code": "WRITEBACK_VERSION_CONFLICT",
            "message": "stale version",
            "retryable": False,
        }
    })
    gateway = FakeHealthWritebackGateway(failed)

    with pytest.raises(WritebackError) as captured:
        await WritebackCoordinator(gateway).prepare(snapshot(), "pharmacist-1")

    assert captured.value.code == "WRITEBACK_VERSION_CONFLICT"
    assert captured.value.retryable is False
