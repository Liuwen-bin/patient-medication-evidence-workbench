from __future__ import annotations

import hashlib
from datetime import UTC
from typing import Any, Protocol

from .gateways import TimedToolResult
from .models import (
    FindingStatus,
    ReviewSnapshot,
    ReviewStatus,
    UNRESOLVED_FINDING_TYPES,
    WritebackJob,
)
from .verifier import verify_label_evidence_bindings


class HealthWritebackGateway(Protocol):
    async def validate_writeback(self, payload: dict[str, Any]) -> TimedToolResult: ...

    async def commit_writeback(
        self,
        job_id: str,
        bundle_hash: str,
        expected_version: int,
        confirmed: bool,
        reviewer_id: str,
    ) -> TimedToolResult: ...


class WritebackStateError(RuntimeError):
    pass


class WritebackError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def unresolved_item_id(review_id: str, item: dict[str, Any]) -> str:
    source = "|".join(
        [
            review_id,
            str(item.get("kind") or "UNKNOWN"),
            ",".join(sorted(item.get("medicationIds") or [])),
            str(item.get("summary") or ""),
        ]
    )
    return "unresolved-" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]


def _normalize_patient_reference(reference: str | None) -> str:
    value = str(reference or "").removeprefix("FHIR:")
    if not value:
        raise WritebackStateError("A signed review must have a patient reference.")
    return value if value.startswith("Patient/") else f"Patient/{value}"


def _summary(item: dict[str, Any]) -> str:
    if item.get("summary"):
        return str(item["summary"])
    if item.get("missingField"):
        return f"Required review field is not recorded: {item['missingField']}."
    errors = item.get("errors") or []
    if errors:
        return "; ".join(str(value) for value in errors)
    error = item.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "Unresolved review item.")
    if error:
        return str(error)
    return f"Unresolved medication review item: {item.get('kind') or 'UNKNOWN'}."


def _project_unresolved(review_id: str, item: dict[str, Any]) -> dict[str, Any]:
    projected = {
        "kind": str(item.get("kind") or "UNKNOWN"),
        "summary": _summary(item),
        "medicationIds": sorted(set(item.get("medicationIds") or [])),
        "evidenceRefs": sorted(
            set(
                item.get("evidenceRefs")
                or [
                    *(item.get("patientEvidenceRefs") or []),
                    *(item.get("labelEvidenceRefs") or []),
                ]
            )
        ),
    }
    projected["unresolvedItemId"] = str(
        item.get("unresolvedItemId") or unresolved_item_id(review_id, projected)
    )
    return projected


def _sign_off_decision(snapshot: ReviewSnapshot, reviewer_id: str):
    sign_off = next(
        (
            decision
            for decision in reversed(snapshot.humanDecisions)
            if decision.action == "SIGN_OFF"
        ),
        None,
    )
    if sign_off is None:
        raise WritebackStateError("A signed review must include its sign-off decision.")
    if sign_off.reviewerId != reviewer_id:
        raise WritebackStateError(
            "The writeback requester must match the signed-off reviewer."
        )
    if sign_off.occurredAt.utcoffset() is None:
        raise WritebackStateError("The sign-off timestamp must include a timezone offset.")
    return sign_off


def build_writeback_payload(
    snapshot: ReviewSnapshot, reviewer_id: str
) -> dict[str, Any]:
    sign_off = _sign_off_decision(snapshot, reviewer_id)
    signed_at = sign_off.occurredAt.astimezone(UTC).isoformat().replace("+00:00", "Z")
    findings = []
    unresolved = [
        _project_unresolved(snapshot.reviewId, dict(item))
        for item in snapshot.unresolvedItems
    ]
    for finding in snapshot.findings:
        if finding.status != FindingStatus.ACCEPTED or finding.verificationErrors:
            continue
        if finding.reviewType in UNRESOLVED_FINDING_TYPES:
            missing_field = getattr(finding, "missingField", None)
            source_tool = getattr(finding, "sourceTool", None)
            gap = {
                "kind": (
                    finding.reviewType
                    if finding.reviewType != "EVIDENCE_GAP"
                    else (
                        "PATIENT_FIELD_MISSING"
                        if missing_field
                        else (
                            "LABEL_EVIDENCE_MISSING"
                            if source_tool == "search_label_evidence"
                            or not finding.labelEvidenceRefs
                            else "EVIDENCE_GAP"
                        )
                    )
                ),
                "summary": finding.summary,
                "medicationIds": finding.medicationIds,
                "evidenceRefs": [
                    *finding.patientEvidenceRefs,
                    *finding.labelEvidenceRefs,
                ],
            }
            unresolved.append(_project_unresolved(snapshot.reviewId, gap))
            continue
        if verify_label_evidence_bindings(finding, snapshot.evidenceIndex):
            raise WritebackStateError(
                "Accepted findings must resolve to valid SPL evidence."
            )
        if not finding.patientEvidenceRefs or not finding.labelEvidenceRefs:
            continue
        findings.append(
            {
                "findingId": finding.findingId,
                "reviewType": finding.reviewType,
                "summary": finding.summary,
                "medicationIds": sorted(set(finding.medicationIds)),
                "selectedProductIds": sorted(set(finding.selectedProductIds)),
                "patientEvidenceRefs": sorted(set(finding.patientEvidenceRefs)),
                "labelEvidenceRefs": sorted(set(finding.labelEvidenceRefs)),
                "status": "ACCEPTED",
                "verificationErrors": [],
            }
        )
    unique_unresolved = {
        item["unresolvedItemId"]: item for item in unresolved
    }
    model_ids = sorted(
        {
            call.modelId
            for call in snapshot.modelCalls
            if call.modelId and not call.fallback
        }
    )
    return {
        "schemaVersion": "1.0",
        "reviewSchemaVersion": snapshot.schemaVersion,
        "reviewId": snapshot.reviewId,
        "reviewVersion": snapshot.version,
        "patientRef": _normalize_patient_reference(snapshot.patientRef),
        "reviewerId": sign_off.reviewerId,
        "signedAt": signed_at,
        "findings": sorted(findings, key=lambda item: item["findingId"]),
        "unresolvedItems": sorted(
            unique_unresolved.values(), key=lambda item: item["unresolvedItemId"]
        ),
        "agent": {
            "name": "medication-review-agent",
            "version": "0.1.0",
            "modelIds": model_ids,
        },
        "syntheticData": True,
    }


def _raise_envelope_error(result: TimedToolResult) -> None:
    if result.envelope.status.value == "OK":
        return
    error = result.envelope.data.get("error") or {}
    code = str(error.get("code") or "WRITEBACK_TOOL_ERROR")
    message = str(
        error.get("message")
        or (result.envelope.errors[0] if result.envelope.errors else "Writeback failed.")
    )
    raise WritebackError(code, message, bool(error.get("retryable", False)))


class WritebackCoordinator:
    def __init__(self, gateway: HealthWritebackGateway) -> None:
        self.gateway = gateway

    async def prepare(
        self, snapshot: ReviewSnapshot, reviewer_id: str
    ) -> WritebackJob:
        if snapshot.status != ReviewStatus.SIGNED_OFF:
            raise WritebackStateError("Writeback requires a signed-off review.")
        payload = build_writeback_payload(snapshot, reviewer_id)
        if not payload["findings"] and not payload["unresolvedItems"]:
            raise WritebackStateError("The signed review has no eligible writeback content.")
        result = await self.gateway.validate_writeback(payload)
        _raise_envelope_error(result)
        try:
            job = WritebackJob.model_validate(result.envelope.data)
        except ValueError as exc:
            raise WritebackError(
                "INVALID_WRITEBACK_PREVIEW", str(exc), False
            ) from exc
        if job.reviewVersion != snapshot.version:
            raise WritebackError(
                "WRITEBACK_VERSION_MISMATCH",
                "The preview version does not match the signed review.",
                False,
            )
        return job

    async def commit(
        self,
        snapshot: ReviewSnapshot,
        job: WritebackJob,
        reviewer_id: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        _sign_off_decision(snapshot, reviewer_id)
        result = await self.gateway.commit_writeback(
            job.jobId,
            job.bundleHash,
            job.expectedVersion,
            confirmed,
            reviewer_id,
        )
        _raise_envelope_error(result)
        data = result.envelope.data
        if data.get("jobId") != job.jobId or data.get("bundleHash") != job.bundleHash:
            raise WritebackError(
                "INVALID_WRITEBACK_COMMIT",
                "The commit response does not match the prepared job.",
                False,
            )
        if data.get("committed") is not True:
            raise WritebackError(
                "INVALID_WRITEBACK_COMMIT",
                "The commit response did not confirm persistence.",
                False,
            )
        return dict(data)
