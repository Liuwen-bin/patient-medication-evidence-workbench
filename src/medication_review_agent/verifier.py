from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .models import Finding, FindingStatus


FHIR_REFERENCE = re.compile(r"^FHIR:[A-Za-z][A-Za-z0-9]*/[^#\s]+$")
SPL_REFERENCE = re.compile(r"^SPL:[^#\s]+#[^\s]+$")
UNSAFE_ACTION = re.compile(
    r"\b(?:stop|discontinue|double|increase|decrease|replace|prescribe|recommend|diagnos(?:e|is)|patient\s+has)\b|"
    r"停药|停止用药|加倍|增加剂量|减少剂量|调整剂量|替换|处方|建议|诊断|患有",
    re.IGNORECASE,
)


class VerificationResult(BaseModel):
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def verify_local_policy(finding: Finding) -> VerificationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if UNSAFE_ACTION.search(finding.summary):
        errors.append("unsafe_clinical_action")
    if finding.reviewType != "EVIDENCE_GAP":
        if not finding.patientEvidenceRefs:
            errors.append("missing_patient_evidence")
        if not finding.labelEvidenceRefs:
            errors.append("missing_label_evidence")
    if any(not FHIR_REFERENCE.fullmatch(ref) for ref in finding.patientEvidenceRefs):
        errors.append("invalid_patient_evidence_reference")
    if any(not SPL_REFERENCE.fullmatch(ref) for ref in finding.labelEvidenceRefs):
        errors.append("invalid_label_evidence_reference")
    provenance = finding.graphProvenance
    if provenance is not None:
        if provenance.fallbackUsed:
            warnings.append("graph_fallback_used")
        consistency = provenance.consistency.status.upper() if provenance.consistency else "UNKNOWN"
        if consistency == "DRIFT":
            warnings.append("graph_snapshot_drift")
        elif consistency == "UNAVAILABLE":
            warnings.append("graph_consistency_unavailable")
    return VerificationResult(errors=list(dict.fromkeys(errors)), warnings=list(dict.fromkeys(warnings)))


def apply_verification(finding: Finding, remote_errors: list[str]) -> Finding:
    local = verify_local_policy(finding)
    errors = list(dict.fromkeys([*finding.verificationErrors, *remote_errors, *local.errors]))
    warnings = list(dict.fromkeys([*finding.verificationWarnings, *local.warnings]))
    update = {
        "verificationErrors": errors,
        "verificationWarnings": warnings,
        "requiresHumanReview": finding.requiresHumanReview or bool(warnings),
    }
    if errors:
        update["status"] = FindingStatus.NEEDS_MORE_EVIDENCE
    return finding.model_copy(update=update)


def verify_findings(findings: list[Finding], remote_results: dict[str, list[str]]) -> list[Finding]:
    return [apply_verification(item, remote_results.get(item.findingId, [])) for item in findings]
