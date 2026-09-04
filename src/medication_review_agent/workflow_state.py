from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class ReviewState(TypedDict, total=False):
    mutationId: str
    reviewId: str
    schemaVersion: str
    status: str
    question: str
    questionSafety: dict[str, Any]
    patientRef: str | None
    asOf: str | None
    contextSnapshot: dict[str, Any]
    contextMissingFields: list[str]
    medications: list[dict[str, Any]]
    medicationMappings: list[dict[str, Any]]
    intent: dict[str, Any]
    modelCalls: list[dict[str, Any]]
    retrievalAttempts: dict[str, int]
    reinvestigationCounts: dict[str, int]
    reviewPlan: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    evidenceIndex: list[dict[str, Any]]
    unresolvedItems: list[dict[str, Any]]
    humanDecisions: list[dict[str, Any]]
    auditEvents: list[dict[str, Any]]
    metrics: dict[str, Any]
    candidates: list[dict[str, Any]]
    reinvestigateFindingIds: list[str]
    contextMedicationsChanged: bool
    contextChangedMedicationIds: list[str]
    contextRebuildPending: bool


class PatientConfirmation(BaseModel):
    action: Literal["CONFIRM_PATIENT"]
    patientId: str
    reviewerId: str


class MappingConfirmation(BaseModel):
    action: Literal["CONFIRM_MAPPING"]
    medicationId: str
    productId: str
    reviewerId: str


class FindingDecision(BaseModel):
    action: Literal["ACCEPT_FINDING", "REJECT_FINDING", "REQUEST_MORE_EVIDENCE"]
    findingId: str
    note: str | None = None


class CompleteFindingReview(BaseModel):
    action: Literal["COMPLETE_FINDING_REVIEW"]
    reviewerId: str
    decisions: list[FindingDecision] = Field(default_factory=list)


class FinalSignOff(BaseModel):
    action: Literal["SIGN_OFF"]
    reviewerId: str
