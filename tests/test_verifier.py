import pytest

from medication_review_agent.models import Finding, FindingStatus
from medication_review_agent.verifier import (
    apply_verification,
    verify_label_evidence_bindings,
    verify_local_policy,
)


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


@pytest.mark.parametrize(
    "review_type", ["PRODUCT_UNMAPPED", "LABEL_EVIDENCE_MISSING"]
)
def test_unresolved_finding_types_do_not_require_label_evidence(
    review_type: str,
) -> None:
    finding = pending_finding(
        reviewType=review_type,
        labelEvidenceRefs=[],
    )

    verified = apply_verification(finding, remote_errors=[])

    assert verified.status == FindingStatus.PENDING
    assert verified.verificationErrors == []


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


def test_ingredient_finding_rejects_valid_but_unrelated_warning_evidence() -> None:
    finding = pending_finding(
        reviewType="DUPLICATE_ACTIVE_INGREDIENT",
        medicationIds=["m1"],
        selectedProductIds=["DRUG_PRODUCT::1"],
        labelEvidenceRefs=["SPL:doc-1#warnings"],
        labelEvidenceIds=[
            "evidence-9fd8e7adea5dd7293edf014924aaf2ab3297b13920c24692d486d4be98c7431f"
        ],
    )
    evidence = [{
        "evidenceId": "evidence-9fd8e7adea5dd7293edf014924aaf2ab3297b13920c24692d486d4be98c7431f",
        "source": "SPL",
        "evidenceRef": "SPL:doc-1#warnings",
        "medicationIds": ["m1"],
        "productIds": ["DRUG_PRODUCT::1"],
        "topic": "warnings",
        "summary": "Warning text.",
        "documentId": "doc-1",
        "documentVersion": "3",
        "sectionId": "warnings",
        "sourcePath": "labels/doc-1.xml",
        "contentHash": "a" * 64,
    }]

    assert verify_label_evidence_bindings(finding, evidence) == [
        "invalid_label_evidence_binding"
    ]
