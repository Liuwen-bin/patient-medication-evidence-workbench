from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ToolStatus(str, Enum):
    OK = "OK"
    AMBIGUOUS = "AMBIGUOUS"
    UNMAPPED = "UNMAPPED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ERROR = "ERROR"


class ReviewStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    AWAITING_PATIENT_CONFIRMATION = "AWAITING_PATIENT_CONFIRMATION"
    AWAITING_MAPPING_CONFIRMATION = "AWAITING_MAPPING_CONFIRMATION"
    AWAITING_FINDING_REVIEW = "AWAITING_FINDING_REVIEW"
    NEEDS_MORE_EVIDENCE = "NEEDS_MORE_EVIDENCE"
    BLOCKED_TOOL_ERROR = "BLOCKED_TOOL_ERROR"
    READY_FOR_SIGN_OFF = "READY_FOR_SIGN_OFF"
    SIGNED_OFF = "SIGNED_OFF"
    CANCELLED = "CANCELLED"


class ReviewTopic(str, Enum):
    IDENTITY = "identity"
    INGREDIENTS = "ingredients"
    ROUTE = "route"
    DOSAGE_FORM = "dosage_form"
    WARNINGS = "warnings"
    DOSAGE = "dosage"
    STORAGE = "storage"
    INDICATIONS = "indications"
    PREGNANCY = "pregnancy"
    STOP_USE = "stop_use"
    IMAGES = "images"


class ReviewIntent(ContractModel):
    type: Literal["MEDICATION_EVIDENCE_REVIEW"] = "MEDICATION_EVIDENCE_REVIEW"
    topics: list[ReviewTopic]
    requiresNarrativeEvidence: bool
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    modelId: str | None = None
    promptVersion: str


class WritebackStatus(str, Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"
    COMMITTING = "COMMITTING"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"


class ModelCallRecord(ContractModel):
    modelId: str
    promptVersion: str
    inputTokens: int = Field(ge=0)
    outputTokens: int = Field(ge=0)
    estimatedCost: float = Field(ge=0.0)
    latencyMs: int = Field(ge=0)
    usageAvailable: bool = False
    costAvailable: bool = False
    fallback: bool = False
    failureCode: str | None = None


class WritebackFailure(ContractModel):
    code: str
    message: str
    retryable: bool


class WritebackJob(ContractModel):
    jobId: str
    reviewVersion: int = Field(ge=0)
    bundleHash: str
    expectedVersion: int = Field(ge=0)
    resources: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blockedFindings: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None


class FindingStatus(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    NEEDS_MORE_EVIDENCE = "NEEDS_MORE_EVIDENCE"


class GraphConsistency(ContractModel):
    status: str = "UNKNOWN"
    checkedAt: str | None = None
    differences: list[dict[str, Any]] = Field(default_factory=list)


class GraphEvidenceProvenance(ContractModel):
    graphBackend: str | None = None
    graphWorkspace: str | None = None
    graphDatabase: str | None = None
    fallbackUsed: bool = False
    consistency: GraphConsistency | None = None


class ToolEnvelope(ContractModel):
    schemaVersion: Literal["1.0"]
    status: ToolStatus
    data: dict[str, Any] = Field(default_factory=dict)
    evidenceRefs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    requestId: str

    @property
    def graph_provenance(self) -> GraphEvidenceProvenance | None:
        graph_keys = {"graphBackend", "graphWorkspace", "graphDatabase", "fallbackUsed", "consistency"}
        if not graph_keys.intersection(self.provenance):
            return None
        return GraphEvidenceProvenance.model_validate(self.provenance)


class MedicationRecord(ContractModel):
    medicationId: str
    name: str
    identifiers: list[dict[str, str]] = Field(default_factory=list)
    strength: str | None = None
    dosageForm: str | None = None
    route: str | None = None
    dosage: str | None = None
    patientEvidenceRefs: list[str] = Field(default_factory=list)


class MedicationMapping(ContractModel):
    medicationId: str
    sourceName: str
    matchClass: str
    selectedProductId: str | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    unmatchedFields: list[str] = Field(default_factory=list)
    graphProvenance: GraphEvidenceProvenance | None = None
    requiresHumanReview: bool = False


class ReviewPlanItem(ContractModel):
    planItemId: str
    reviewType: str
    medicationIds: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    rationale: str
    requiresHumanReview: bool = False


class EvidenceItem(ContractModel):
    evidenceId: str
    source: str
    evidenceRef: str
    medicationIds: list[str] = Field(default_factory=list)
    productIds: list[str] = Field(default_factory=list)
    topic: str | None = None
    summary: str | None = None
    graphProvenance: GraphEvidenceProvenance | None = None


class Finding(ContractModel):
    findingId: str
    reviewType: str
    ruleId: str
    normalizationVersion: str | None = None
    comparisonInputs: dict[str, Any] = Field(default_factory=dict)
    summary: str
    attentionLevel: str
    confidence: float = Field(ge=0.0, le=1.0)
    medicationIds: list[str] = Field(default_factory=list)
    selectedProductIds: list[str] = Field(default_factory=list)
    patientEvidenceRefs: list[str] = Field(default_factory=list)
    labelEvidenceRefs: list[str] = Field(default_factory=list)
    status: FindingStatus = FindingStatus.PENDING
    requiresHumanReview: bool = False
    verificationErrors: list[str] = Field(default_factory=list)
    verificationWarnings: list[str] = Field(default_factory=list)
    graphProvenance: GraphEvidenceProvenance | None = None

    @model_validator(mode="after")
    def accepted_findings_need_evidence(self) -> "Finding":
        if self.status != FindingStatus.ACCEPTED or self.reviewType == "EVIDENCE_GAP":
            return self
        if not self.patientEvidenceRefs or not self.labelEvidenceRefs:
            raise ValueError("accepted findings require patient and label evidence")
        return self


def migrate_finding_payload(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    if not migrated.get("ruleId"):
        review_type = str(migrated.get("reviewType") or "finding")
        slug = re.sub(r"[^a-z0-9]+", "-", review_type.casefold()).strip("-")
        migrated["ruleId"] = f"legacy-{slug or 'finding'}-v1"
    migrated.setdefault("normalizationVersion", None)
    migrated.setdefault("comparisonInputs", {})
    return migrated


class HumanDecision(ContractModel):
    action: str
    reviewerId: str
    occurredAt: datetime = Field(default_factory=lambda: datetime.now(UTC))
    medicationId: str | None = None
    productId: str | None = None
    findingId: str | None = None
    note: str | None = None


class AuditEvent(ContractModel):
    mutationId: str | None = None
    auditSlot: str | None = None
    node: str
    tool: str | None = None
    requestId: str | None = None
    resultStatus: str
    argumentSummary: dict[str, Any] = Field(default_factory=dict)
    evidenceRefs: list[str] = Field(default_factory=list)
    occurredAt: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latencyMs: int = Field(default=0, ge=0)
    retryCount: int = Field(default=0, ge=0)
    modelId: str | None = None
    promptVersion: str | None = None
    inputTokens: int = Field(default=0, ge=0)
    outputTokens: int = Field(default=0, ge=0)
    estimatedCost: float = Field(default=0.0, ge=0.0)
    modelFallback: bool = False


class RunMetrics(ContractModel):
    toolLatencyMs: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    inputTokens: int = Field(default=0, ge=0)
    outputTokens: int = Field(default=0, ge=0)
    estimatedCost: float = Field(default=0.0, ge=0.0)


class ReviewSnapshot(ContractModel):
    reviewId: str
    schemaVersion: Literal["1.1"] = "1.1"
    status: ReviewStatus
    question: str
    patientRef: str | None = None
    asOf: str | None = None
    intent: ReviewIntent | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    contextSnapshot: dict[str, Any] = Field(default_factory=dict)
    contextMissingFields: list[str] = Field(default_factory=list)
    medications: list[MedicationRecord] = Field(default_factory=list)
    medicationMappings: list[MedicationMapping] = Field(default_factory=list)
    reviewPlan: list[ReviewPlanItem] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    evidenceIndex: list[EvidenceItem] = Field(default_factory=list)
    unresolvedItems: list[dict[str, Any]] = Field(default_factory=list)
    humanDecisions: list[HumanDecision] = Field(default_factory=list)
    auditEvents: list[AuditEvent] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    writebackStatus: WritebackStatus = WritebackStatus.NOT_REQUESTED
    writebackJob: WritebackJob | None = None
    writebackError: WritebackFailure | None = None
    modelCalls: list[ModelCallRecord] = Field(default_factory=list)
    retrievalAttempts: dict[str, int] = Field(default_factory=dict)
    reinvestigationCounts: dict[str, int] = Field(default_factory=dict)
    version: int = Field(default=0, ge=0)
    createdAt: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(UTC))
