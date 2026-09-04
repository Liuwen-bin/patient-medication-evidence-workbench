from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from medication_review_agent.models import (
    EvidenceItem,
    Finding,
    FindingStatus,
    MedicationMapping,
    ReviewSnapshot,
    ReviewStatus,
    ToolEnvelope,
    WritebackStatus,
)


def test_tool_envelope_rejects_wrong_schema() -> None:
    with pytest.raises(ValidationError):
        ToolEnvelope.model_validate({
            "schemaVersion": "2.0",
            "status": "OK",
            "data": {},
            "evidenceRefs": [],
            "warnings": [],
            "errors": [],
            "provenance": {},
            "requestId": "r1",
        })


def test_accepted_finding_requires_paired_evidence() -> None:
    with pytest.raises(ValidationError):
        Finding(
            findingId="f1",
            reviewType="LABEL_WARNING",
            ruleId="label-warning-v1",
            summary="Candidate warning",
            attentionLevel="HIGH",
            confidence=0.9,
            patientEvidenceRefs=["FHIR:MedicationRequest/m1"],
            labelEvidenceRefs=[],
            status=FindingStatus.ACCEPTED,
        )


def test_evidence_gap_can_be_accepted_without_paired_evidence() -> None:
    finding = Finding(
        findingId="f1",
        reviewType="EVIDENCE_GAP",
        ruleId="evidence-gap-v1",
        summary="No mapped DailyMed product",
        attentionLevel="HIGH",
        confidence=1.0,
        status=FindingStatus.ACCEPTED,
    )
    assert finding.status == FindingStatus.ACCEPTED


def test_graph_provenance_survives_mapping_evidence_and_snapshot() -> None:
    provenance = {
        "graphBackend": "neo4j",
        "graphWorkspace": "dailymed",
        "graphDatabase": "neo4j",
        "fallbackUsed": False,
        "consistency": {"status": "CONSISTENT", "checkedAt": "2026-08-31"},
        "futureField": "preserved",
    }
    mapping = MedicationMapping(
        medicationId="m1",
        sourceName="ARNICA",
        matchClass="EXACT_IDENTIFIER",
        selectedProductId="DRUG_PRODUCT::1",
        graphProvenance=provenance,
    )
    evidence = EvidenceItem(
        evidenceId="e1", source="SPL", evidenceRef="SPL:d#s",
        graphProvenance=provenance,
    )
    assert mapping.graphProvenance.graphBackend == "neo4j"
    assert evidence.graphProvenance.model_extra == {"futureField": "preserved"}


def test_snapshot_round_trip_preserves_graph_provenance() -> None:
    now = datetime.now(UTC)
    snapshot = ReviewSnapshot(
        reviewId="r1",
        question="默认用药证据核查",
        patientRef="FHIR:Patient/p1",
        status=ReviewStatus.CREATED,
        createdAt=now,
        updatedAt=now,
        medicationMappings=[MedicationMapping(
            medicationId="m1",
            sourceName="ARNICA",
            matchClass="UNMAPPED",
            graphProvenance={
                "graphBackend": "snapshot",
                "fallbackUsed": True,
                "consistency": {"status": "UNAVAILABLE"},
            },
        )],
    )
    restored = ReviewSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored.medicationMappings[0].graphProvenance.fallbackUsed is True


def test_signed_off_is_a_distinct_terminal_status() -> None:
    assert ReviewStatus.SIGNED_OFF.value == "SIGNED_OFF"


def test_new_snapshot_uses_review_schema_1_1() -> None:
    snapshot = ReviewSnapshot(
        reviewId="review-1",
        status=ReviewStatus.CREATED,
        question="核查成分和标签警告",
    )
    assert snapshot.schemaVersion == "1.1"
    assert snapshot.writebackStatus is WritebackStatus.NOT_REQUESTED
    assert snapshot.intent is None
    assert snapshot.writebackJob is None
    assert snapshot.writebackError is None
