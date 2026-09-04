import pytest

from medication_review_agent.models import Finding, FindingStatus
from medication_review_agent.verifier import apply_verification, verify_local_policy


def pending_finding(**updates) -> Finding:
    values = {
        "findingId": "f1", "reviewType": "LABEL_WARNING",
        "ruleId": "label-warning-v1",
        "summary": "Review label warning.", "attentionLevel": "HIGH",
        "confidence": 0.8, "patientEvidenceRefs": ["FHIR:MedicationRequest/m1"],
        "labelEvidenceRefs": ["SPL:doc-1#section-1"],
    }
    values.update(updates)
    return Finding(**values)


@pytest.mark.parametrize("unsafe", [
    "Stop the medication immediately.",
    "Double the dose.",
    "Replace the prescription with ARNICA.",
    "The patient has influenza.",
    "立即停药。",
    "建议增加剂量。",
])
def test_verifier_rejects_autonomous_clinical_actions(unsafe: str) -> None:
    result = verify_local_policy(pending_finding(summary=unsafe))
    assert "unsafe_clinical_action" in result.errors


def test_non_gap_finding_without_both_sources_is_downgraded() -> None:
    finding = pending_finding(labelEvidenceRefs=[])
    verified = apply_verification(finding, remote_errors=[])
    assert verified.status == FindingStatus.NEEDS_MORE_EVIDENCE
    assert "missing_label_evidence" in verified.verificationErrors


def test_invalid_reference_syntax_is_downgraded() -> None:
    verified = apply_verification(pending_finding(labelEvidenceRefs=["doc-1"]), remote_errors=[])
    assert "invalid_label_evidence_reference" in verified.verificationErrors


def test_snapshot_fallback_requires_explicit_human_review() -> None:
    verified = apply_verification(pending_finding(graphProvenance={
        "graphBackend": "snapshot", "fallbackUsed": True,
        "consistency": {"status": "UNAVAILABLE"},
    }), remote_errors=[])
    assert verified.requiresHumanReview is True
    assert "graph_fallback_used" in verified.verificationWarnings


def test_graph_drift_is_reported_as_an_evidence_gap() -> None:
    verified = apply_verification(pending_finding(graphProvenance={
        "graphBackend": "neo4j", "fallbackUsed": False,
        "consistency": {"status": "DRIFT"},
    }), remote_errors=[])
    assert "graph_snapshot_drift" in verified.verificationWarnings
    assert verified.requiresHumanReview is True


def test_remote_validation_errors_are_machine_readable() -> None:
    verified = apply_verification(pending_finding(), remote_errors=["out_of_scope_label_evidence"])
    assert verified.status == FindingStatus.NEEDS_MORE_EVIDENCE
    assert verified.verificationErrors == ["out_of_scope_label_evidence"]
