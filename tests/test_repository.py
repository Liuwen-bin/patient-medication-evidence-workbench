import json
import sqlite3
from pathlib import Path

import pytest

from medication_review_agent.models import AuditEvent, ReviewStatus
from medication_review_agent.repository import (
    AuditRedactionError,
    ReviewRepository,
    ReviewVersionConflict,
    UnsupportedReviewSchema,
)


def test_repository_migrates_1_0_snapshot_in_memory(tmp_path: Path) -> None:
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    old = repository.create(
        patient_ref="demo-1",
        question="默认用药证据核查",
    )
    payload = old.model_dump(mode="json")
    for field in (
        "question",
        "intent",
        "writebackStatus",
        "writebackJob",
        "writebackError",
        "modelCalls",
        "retrievalAttempts",
        "reinvestigationCounts",
    ):
        payload.pop(field)
    payload["schemaVersion"] = "1.0"
    with sqlite3.connect(repository.path) as connection:
        connection.execute(
            "UPDATE reviews SET snapshot_json = ? WHERE review_id = ?",
            (json.dumps(payload), old.reviewId),
        )

    migrated = repository.get(old.reviewId)

    assert migrated.schemaVersion == "1.1"
    assert migrated.question == "默认用药证据核查"
    assert migrated.writebackStatus.value == "NOT_REQUESTED"
    with sqlite3.connect(repository.path) as connection:
        stored = json.loads(connection.execute(
            "SELECT snapshot_json FROM reviews WHERE review_id = ?",
            (old.reviewId,),
        ).fetchone()[0])
    assert stored["schemaVersion"] == "1.0"


def test_repository_rejects_unknown_review_schema(tmp_path: Path) -> None:
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    snapshot = repository.create(patient_ref=None, question="核查用药")
    raw = snapshot.model_dump(mode="json")
    raw["schemaVersion"] = "2.0"
    with sqlite3.connect(repository.path) as connection:
        connection.execute(
            "UPDATE reviews SET snapshot_json = ? WHERE review_id = ?",
            (json.dumps(raw), snapshot.reviewId),
        )

    with pytest.raises(UnsupportedReviewSchema):
        repository.get(snapshot.reviewId)


def test_repository_backfills_provenance_for_early_1_1_findings(tmp_path: Path) -> None:
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    snapshot = repository.create(patient_ref="p1", question="核查标签")
    payload = snapshot.model_dump(mode="json")
    payload["findings"] = [{
        "findingId": "legacy-finding",
        "reviewType": "LABEL_WARNING",
        "summary": "Review label warning.",
        "attentionLevel": "HIGH",
        "confidence": 0.8,
        "patientEvidenceRefs": ["FHIR:MedicationRequest/m1"],
        "labelEvidenceRefs": ["SPL:doc-1#warnings"],
    }]
    with sqlite3.connect(repository.path) as connection:
        connection.execute(
            "UPDATE reviews SET snapshot_json = ? WHERE review_id = ?",
            (json.dumps(payload), snapshot.reviewId),
        )

    restored = repository.get(snapshot.reviewId)

    assert restored.findings[0].ruleId == "legacy-label-warning-v1"
    assert restored.findings[0].normalizationVersion is None
    assert restored.findings[0].comparisonInputs == {}


def test_stale_review_update_is_rejected(tmp_path: Path) -> None:
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    created = repository.create(patient_ref="FHIR:Patient/p1", question="默认用药证据核查")
    first = repository.get(created.reviewId)
    second = repository.get(created.reviewId)
    first.status = ReviewStatus.RUNNING
    saved = repository.save(first, expected_version=0)
    assert saved.version == 1
    second.status = ReviewStatus.CANCELLED
    with pytest.raises(ReviewVersionConflict):
        repository.save(second, expected_version=0)


def test_review_survives_repository_reopen(tmp_path: Path) -> None:
    path = tmp_path / "reviews.sqlite"
    created = ReviewRepository(path).create(
        patient_ref="FHIR:Patient/p1",
        question="默认用药证据核查",
    )
    restored = ReviewRepository(path).get(created.reviewId)
    assert restored.patientRef == "FHIR:Patient/p1"
    assert restored.status == ReviewStatus.CREATED


def test_repository_releases_sqlite_file_after_each_operation(tmp_path: Path) -> None:
    path = tmp_path / "reviews.sqlite"
    repository = ReviewRepository(path)
    created = repository.create(patient_ref="FHIR:Patient/p1", question="默认用药证据核查")
    repository.get(created.reviewId)
    path.unlink()
    assert not path.exists()


def test_audit_event_does_not_store_raw_patient_payload(tmp_path: Path) -> None:
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    review = repository.create(patient_ref="FHIR:Patient/p1", question="默认用药证据核查")
    repository.append_audit(
        review.reviewId,
        node="collect_review_context",
        tool="get_medication_review_context",
        request_id="r1",
        result_status="OK",
        argument_summary={"patientIdHash": "a" * 64},
        evidence_refs=["FHIR:MedicationRequest/m1"],
        latency_ms=10,
    )
    serialized = json.dumps(repository.list_audit(review.reviewId))
    assert "patientName" not in serialized
    assert "FHIR:MedicationRequest/m1" in serialized


def test_model_audit_persists_fallback_without_prompt_or_patient_features(
    tmp_path: Path,
) -> None:
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    review = repository.create(
        patient_ref="FHIR:Patient/p1",
        question="核查储存条件",
    )

    event = repository.append_audit(
        review.reviewId,
        node="parse_review_goal",
        tool=None,
        request_id=None,
        result_status="MODEL_FALLBACK",
        argument_summary={"topicCount": 1, "medicationCount": 2},
        evidence_refs=[],
        latency_ms=12,
        model_id="test-model",
        prompt_version="intent-v1",
        input_tokens=17,
        output_tokens=5,
        estimated_cost=0.0,
        model_fallback=True,
    )

    assert event.modelFallback is True
    serialized = event.model_dump_json()
    assert "核查储存条件" not in serialized
    assert "patientFeatures" not in serialized


@pytest.mark.parametrize("unsafe_key", ["name", "birthDate", "content", "attachment", "dosage", "rawPayload"])
def test_audit_rejects_sensitive_argument_keys(tmp_path: Path, unsafe_key: str) -> None:
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    review = repository.create(patient_ref="FHIR:Patient/p1", question="默认用药证据核查")
    with pytest.raises(AuditRedactionError):
        repository.append_audit(
            review.reviewId, node="n", tool="t", request_id="r",
            result_status="OK", argument_summary={unsafe_key: "secret"},
            evidence_refs=[], latency_ms=1,
        )


def test_audit_rejects_patient_payload_hidden_under_innocuous_key(tmp_path: Path) -> None:
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    review = repository.create(patient_ref="FHIR:Patient/p1", question="默认用药证据核查")
    with pytest.raises(AuditRedactionError):
        repository.append_audit(
            review.reviewId, node="n", tool="t", request_id="r",
            result_status="OK",
            argument_summary={"details": {"resourceType": "Patient", "id": "p1"}},
            evidence_refs=[], latency_ms=1,
        )


def test_audit_event_with_same_mutation_id_is_idempotent(tmp_path: Path) -> None:
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    review = repository.create(patient_ref="FHIR:Patient/p1", question="默认用药证据核查")
    kwargs = dict(
        node="collect_review_context", tool="get_medication_review_context",
        request_id="r1", result_status="OK",
        argument_summary={"patientIdHash": "a" * 64},
        evidence_refs=["FHIR:MedicationRequest/m1"], latency_ms=10,
        mutation_id="mutation-1",
    )
    first = repository.append_audit(review.reviewId, **kwargs)
    second = repository.append_audit(review.reviewId, **kwargs)
    events = repository.list_audit(review.reviewId)
    assert first.mutationId == second.mutationId == "mutation-1"
    assert len(events) == 1


def test_checkpoint_advanced_mutation_rolls_forward_after_repository_reopen(tmp_path: Path) -> None:
    path = tmp_path / "reviews.sqlite"
    repository = ReviewRepository(path)
    review = repository.create(patient_ref="FHIR:Patient/p1", question="默认用药证据核查")
    repository.prepare_mutation(
        mutation_id="mutation-crash", review_id=review.reviewId,
        expected_version=0, action_fingerprint="f" * 64,
        checkpoint_backup=b"durable-backup",
    )
    repository.append_audit(
        review.reviewId, node="collect_review_context", tool="health",
        request_id="r1", result_status="OK",
        argument_summary={"patientIdHash": "a" * 64}, evidence_refs=[],
        latency_ms=1, mutation_id="mutation-crash",
    )
    projected = review.model_copy(deep=True)
    projected.status = ReviewStatus.AWAITING_FINDING_REVIEW
    repository.mark_checkpoint_advanced("mutation-crash", projected)

    restarted = ReviewRepository(path)
    recovered = restarted.commit_mutation("mutation-crash")
    assert recovered.version == 1
    assert recovered.status == ReviewStatus.AWAITING_FINDING_REVIEW
    events = restarted.list_audit(review.reviewId)
    assert [AuditEvent.model_validate(item).mutationId for item in events] == ["mutation-crash"]
    assert restarted.commit_mutation("mutation-crash").version == 1
    assert restarted.get_mutation("mutation-crash")["state"] == "COMMITTED"


def test_checkpoint_advanced_1_0_projection_migrates_before_commit(tmp_path: Path) -> None:
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    review = repository.create(
        patient_ref="FHIR:Patient/p1",
        question="原始核查问题",
    )
    repository.prepare_mutation(
        mutation_id="legacy-projection",
        review_id=review.reviewId,
        expected_version=0,
        action_fingerprint="f" * 64,
        checkpoint_backup=b"durable-backup",
    )
    projected = review.model_copy(update={"status": ReviewStatus.RUNNING})
    repository.mark_checkpoint_advanced("legacy-projection", projected)
    legacy_payload = projected.model_dump(mode="json")
    for field in (
        "question",
        "intent",
        "writebackStatus",
        "writebackJob",
        "writebackError",
        "modelCalls",
        "retrievalAttempts",
        "reinvestigationCounts",
    ):
        legacy_payload.pop(field)
    legacy_payload["schemaVersion"] = "1.0"
    with sqlite3.connect(repository.path) as connection:
        connection.execute(
            "UPDATE mutation_journal SET projected_snapshot_json = ? WHERE mutation_id = ?",
            (json.dumps(legacy_payload), "legacy-projection"),
        )

    recovered = repository.commit_mutation("legacy-projection")

    assert recovered.schemaVersion == "1.1"
    assert recovered.question == "默认用药证据核查"
    assert recovered.version == 1
    assert repository.get(review.reviewId).schemaVersion == "1.1"


def test_identical_audit_results_in_distinct_call_slots_are_both_persisted(tmp_path: Path) -> None:
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    review = repository.create(patient_ref="FHIR:Patient/p1", question="默认用药证据核查")
    repository.prepare_mutation(
        mutation_id="mutation-1", review_id=review.reviewId, expected_version=0,
        action_fingerprint="f" * 64, checkpoint_backup=b"backup",
    )
    kwargs = dict(
        node="resolve_medications", tool="resolve_medication", request_id="same",
        result_status="OK", argument_summary={"medicationIdHash": "a" * 64},
        evidence_refs=[], latency_ms=3, mutation_id="mutation-1",
    )
    repository.append_audit(review.reviewId, audit_slot="resolve:0", **kwargs)
    repository.append_audit(review.reviewId, audit_slot="resolve:1", **kwargs)
    repository.append_audit(
        review.reviewId, audit_slot="resolve:0",
        **{**kwargs, "request_id": "retry-request", "latency_ms": 99},
    )
    projected = review.model_copy(deep=True)
    projected.status = ReviewStatus.RUNNING
    repository.mark_checkpoint_advanced("mutation-1", projected)
    repository.commit_mutation("mutation-1")
    events = repository.list_audit(review.reviewId)
    assert [item["auditSlot"] for item in events] == ["resolve:0", "resolve:1"]
    assert events[0]["requestId"] == "same"
    assert events[0]["latencyMs"] == 3
