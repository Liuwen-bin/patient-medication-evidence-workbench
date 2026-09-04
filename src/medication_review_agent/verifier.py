from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .models import EvidenceItem, Finding, FindingStatus, UNRESOLVED_FINDING_TYPES
from .retrieval import stable_evidence_id


FHIR_REFERENCE = re.compile(r"^FHIR:[A-Za-z][A-Za-z0-9]*/[^#\s]+$")
SPL_REFERENCE = re.compile(r"^SPL:[^#\s]+#[^\s]+$")
UNSAFE_ACTION = re.compile(
    r"\b(?:stop|discontinue|double|increase|decrease|replace|prescribe|recommend|diagnos(?:e|is)|patient\s+has)\b|"
    r"停药|停止用药|加倍|增加剂量|减少剂量|调整剂量|替换|处方|建议|诊断|患有",
    re.IGNORECASE,
)
CONTENT_HASH = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_LABEL_TOPICS = {
    "ROUTE_MISMATCH": frozenset({"route"}),
    "DOSAGE_FORM_MISMATCH": frozenset({"dosage_form"}),
    "INGREDIENT_ALLERGY_NAME_MATCH": frozenset({"ingredients"}),
    "DUPLICATE_ACTIVE_INGREDIENT": frozenset({"ingredients"}),
    "SHARED_ACTIVE_INGREDIENT": frozenset({"ingredients"}),
    "LABEL_WARNING": frozenset({"warnings"}),
}


class VerificationResult(BaseModel):
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def verify_label_evidence_bindings(
    finding: Finding,
    evidence_index: list[EvidenceItem | dict],
) -> list[str]:
    if finding.reviewType in UNRESOLVED_FINDING_TYPES:
        return []
    if not finding.labelEvidenceIds:
        return ["invalid_label_evidence_binding"]

    indexed: dict[str, EvidenceItem | dict] = {}
    for item in evidence_index:
        evidence_id = (
            item.evidenceId
            if isinstance(item, EvidenceItem)
            else str(item.get("evidenceId") or "")
        )
        if evidence_id:
            indexed[evidence_id] = item

    cited: list[EvidenceItem] = []
    try:
        for evidence_id in finding.labelEvidenceIds:
            cited.append(EvidenceItem.model_validate(indexed[evidence_id]))
    except (KeyError, ValueError):
        return ["invalid_label_evidence_binding"]

    if {item.evidenceRef for item in cited} != set(finding.labelEvidenceRefs):
        return ["invalid_label_evidence_binding"]
    required_topics = REQUIRED_LABEL_TOPICS.get(finding.reviewType)
    if required_topics and not any(item.topic in required_topics for item in cited):
        return ["invalid_label_evidence_binding"]

    finding_medications = set(finding.medicationIds)
    finding_products = set(finding.selectedProductIds)
    cited_medications: set[str] = set()
    cited_products: set[str] = set()
    for item in cited:
        medication_ids = set(item.medicationIds)
        product_ids = set(item.productIds)
        cited_medications.update(medication_ids)
        cited_products.update(product_ids)
        document_id = item.documentId or ""
        document_version = item.documentVersion or ""
        content_hash = item.contentHash or ""
        reference_document = (
            item.evidenceRef.removeprefix("SPL:").split("#", 1)[0]
            if item.evidenceRef.startswith("SPL:") and "#" in item.evidenceRef
            else ""
        )
        if (
            item.source != "SPL"
            or not medication_ids.intersection(finding_medications)
            or not product_ids
            or not product_ids <= finding_products
            or not document_id
            or reference_document != document_id
            or not document_version.strip()
            or not item.sectionId
            or not item.sourcePath
            or not item.topic
            or not (item.summary or "").strip()
            or CONTENT_HASH.fullmatch(content_hash) is None
            or item.evidenceId
            != stable_evidence_id(
                item.source,
                item.evidenceRef,
                document_version,
                content_hash,
            )
        ):
            return ["invalid_label_evidence_binding"]
    if (
        not finding_medications <= cited_medications
        or not finding_products <= cited_products
    ):
        return ["invalid_label_evidence_binding"]
    return []


def verify_local_policy(finding: Finding) -> VerificationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if UNSAFE_ACTION.search(finding.summary):
        errors.append("unsafe_clinical_action")
    if finding.reviewType not in UNRESOLVED_FINDING_TYPES:
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
