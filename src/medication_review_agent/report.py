from __future__ import annotations

from html import escape
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import AuditEvent, Finding, FindingStatus, HumanDecision, MedicationMapping, ReviewSnapshot, ReviewStatus, RunMetrics


SCOPE_STATEMENT = (
    "This report is a pharmacist-reviewed label-evidence investigation. "
    "It is not an autonomous prescribing decision and does not modify the clinical record."
)


class ReportNotSigned(RuntimeError):
    pass


class ReportDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    schemaVersion: str = "1.0"
    reviewId: str
    status: str
    patientRef: str | None
    sourceAsOf: str | None
    reviewerId: str
    createdAt: str
    updatedAt: str
    mappingDecisions: list[MedicationMapping] = Field(default_factory=list)
    acceptedFindings: list[Finding] = Field(default_factory=list)
    rejectedAuditAppendix: list[Finding] = Field(default_factory=list)
    unmappedMedications: list[dict[str, Any]] = Field(default_factory=list)
    ambiguousMedications: list[dict[str, Any]] = Field(default_factory=list)
    evidenceGaps: list[Finding] = Field(default_factory=list)
    reviewerDecisions: list[HumanDecision] = Field(default_factory=list)
    auditTrail: list[AuditEvent] = Field(default_factory=list)
    metrics: RunMetrics
    graphProvenance: list[dict[str, Any]] = Field(default_factory=list)
    evidenceProvenance: list[dict[str, Any]] = Field(default_factory=list)
    scopeStatement: str = SCOPE_STATEMENT


def build_signed_report(snapshot: ReviewSnapshot, reviewer_id: str | None) -> ReportDocument:
    if snapshot.status != ReviewStatus.SIGNED_OFF or not reviewer_id or not reviewer_id.strip():
        raise ReportNotSigned("A signed report requires SIGNED_OFF status and reviewer ID")
    medication_by_id = {item.medicationId: item for item in snapshot.medications}
    unmapped = []
    ambiguous = []
    provenance = []
    for mapping in snapshot.medicationMappings:
        item = medication_by_id.get(mapping.medicationId)
        disclosure = {
            "medicationId": mapping.medicationId,
            "graphBackend": mapping.graphProvenance.graphBackend if mapping.graphProvenance else None,
            "graphWorkspace": mapping.graphProvenance.graphWorkspace if mapping.graphProvenance else None,
            "graphDatabase": mapping.graphProvenance.graphDatabase if mapping.graphProvenance else None,
            "fallbackUsed": mapping.graphProvenance.fallbackUsed if mapping.graphProvenance else False,
            "consistency": mapping.graphProvenance.consistency.model_dump(mode="json") if mapping.graphProvenance and mapping.graphProvenance.consistency else None,
        }
        provenance.append(disclosure)
        rendered = {"medicationId": mapping.medicationId, "name": item.name if item else mapping.sourceName, "matchClass": mapping.matchClass}
        if mapping.matchClass == "UNMAPPED":
            unmapped.append(rendered)
        if mapping.matchClass.startswith("AMBIGUOUS"):
            ambiguous.append(rendered)
    accepted = [item for item in snapshot.findings if item.status == FindingStatus.ACCEPTED]
    evidence_provenance = []
    for item in snapshot.evidenceIndex:
        graph = item.graphProvenance
        evidence_provenance.append({
            "evidenceId": item.evidenceId, "evidenceRef": item.evidenceRef,
            "graphBackend": graph.graphBackend if graph else None,
            "graphWorkspace": graph.graphWorkspace if graph else None,
            "graphDatabase": graph.graphDatabase if graph else None,
            "fallbackUsed": graph.fallbackUsed if graph else False,
            "consistency": graph.consistency.model_dump(mode="json") if graph and graph.consistency else None,
        })
    return ReportDocument(
        reviewId=snapshot.reviewId, status=snapshot.status.value,
        patientRef=snapshot.patientRef, sourceAsOf=snapshot.asOf,
        reviewerId=reviewer_id.strip(), mappingDecisions=snapshot.medicationMappings,
        createdAt=snapshot.createdAt.isoformat(), updatedAt=snapshot.updatedAt.isoformat(),
        acceptedFindings=accepted,
        rejectedAuditAppendix=[item for item in snapshot.findings if item.status == FindingStatus.REJECTED],
        unmappedMedications=unmapped, ambiguousMedications=ambiguous,
        evidenceGaps=[item for item in snapshot.findings if item.reviewType == "EVIDENCE_GAP"],
        reviewerDecisions=snapshot.humanDecisions, auditTrail=snapshot.auditEvents,
        metrics=snapshot.metrics, graphProvenance=provenance,
        evidenceProvenance=evidence_provenance,
    )


def render_report_json(report: ReportDocument) -> str:
    return report.model_dump_json(indent=2)


def _text(value: Any) -> str:
    return escape(str(value), quote=True)


def render_report_html(report: ReportDocument) -> str:
    mappings = "".join(
        f"<li>{_text(item.sourceName)}: {_text(item.matchClass)}</li>"
        for item in report.mappingDecisions
    )
    findings = "".join(
        f"<li><strong>{_text(item.reviewType)}</strong>: {_text(item.summary)}</li>"
        for item in report.acceptedFindings
    )
    gaps = "".join(f"<li>{_text(item.summary)}</li>" for item in report.evidenceGaps)
    graph = "".join(
        f"<li>{_text(item.get('medicationId'))}: backend={_text(item.get('graphBackend'))}, "
        f"workspace={_text(item.get('graphWorkspace'))}, database={_text(item.get('graphDatabase'))}, "
        f"fallback={_text(item.get('fallbackUsed'))}, consistency={_text(item.get('consistency'))}</li>"
        for item in report.graphProvenance
    )
    evidence_graph = "".join(
        f"<li>{_text(item.get('evidenceRef'))}: backend={_text(item.get('graphBackend'))}, "
        f"workspace={_text(item.get('graphWorkspace'))}, database={_text(item.get('graphDatabase'))}, "
        f"fallback={_text(item.get('fallbackUsed'))}, consistency={_text(item.get('consistency'))}</li>"
        for item in report.evidenceProvenance
    )
    decisions = "".join(
        f"<li>{_text(item.reviewerId)}: {_text(item.action)}; finding={_text(item.findingId)}; note={_text(item.note)}</li>"
        for item in report.reviewerDecisions
    )
    audit = "".join(
        f"<li>{_text(item.occurredAt.isoformat())}: node={_text(item.node)}, tool={_text(item.tool)}, "
        f"request={_text(item.requestId)}, status={_text(item.resultStatus)}, model={_text(item.modelId)}, "
        f"prompt={_text(item.promptVersion)}, arguments={_text(item.argumentSummary)}, "
        f"evidence={_text(item.evidenceRefs)}, latency={_text(item.latencyMs)}, "
        f"retries={_text(item.retryCount)}, input_tokens={_text(item.inputTokens)}, "
        f"output_tokens={_text(item.outputTokens)}, cost={_text(item.estimatedCost)}</li>"
        for item in report.auditTrail
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Medication Review</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:960px;margin:32px auto;line-height:1.5}"
        "h1,h2{color:#173b35}section{border-top:1px solid #ccc;padding-top:12px}</style></head><body>"
        f"<h1>Medication Review {_text(report.reviewId)}</h1>"
        f"<p>Patient: {_text(report.patientRef)} | As of: {_text(report.sourceAsOf)} | Reviewer: {_text(report.reviewerId)}</p>"
        f"<p>Created: {_text(report.createdAt)} | Updated: {_text(report.updatedAt)}</p>"
        f"<p>{_text(report.scopeStatement)}</p>"
        f"<section><h2>Mappings</h2><ul>{mappings}</ul></section>"
        f"<section><h2>Accepted findings</h2><ul>{findings}</ul></section>"
        f"<section><h2>Evidence gaps</h2><ul>{gaps}</ul></section>"
        f"<section><h2>Graph provenance</h2><ul>{graph}</ul></section>"
        f"<section><h2>Evidence provenance</h2><ul>{evidence_graph}</ul></section>"
        f"<section><h2>Reviewer decisions</h2><ul>{decisions}</ul></section>"
        f"<section><h2>Audit trail</h2><ul>{audit}</ul></section>"
        f"<section><h2>Run metrics</h2><p>Tool latency: {_text(report.metrics.toolLatencyMs)} ms; "
        f"retries: {_text(report.metrics.retries)}; input tokens: {_text(report.metrics.inputTokens)}; "
        f"output tokens: {_text(report.metrics.outputTokens)}; estimated cost: {_text(report.metrics.estimatedCost)}</p></section>"
        "</body></html>"
    )
