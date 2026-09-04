from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, TypeAdapter

from .gateways import DrugEvidenceGateway, HealthRecordGateway, TimedToolResult, ToolContractError
from .models import (
    EvidenceItem, Finding, FindingStatus, GraphEvidenceProvenance, HumanDecision,
    MedicationMapping, MedicationRecord, ReviewPlanItem, ReviewSnapshot, ReviewStatus,
)
from .planner import ReviewPlanner
from .repository import ReviewRepository
from .safety import evaluate_review_question
from .verifier import apply_verification


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


@dataclass(frozen=True)
class ReviewDependencies:
    health: HealthRecordGateway
    drug: DrugEvidenceGateway
    repository: ReviewRepository
    planner: ReviewPlanner


def open_sqlite_checkpointer(path: str | Path) -> AsyncSqliteSaver:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    connection = aiosqlite.connect(resolved)
    return AsyncSqliteSaver(connection)


def _graph_provenance(result: TimedToolResult) -> dict[str, Any] | None:
    value = result.envelope.graph_provenance
    return value.model_dump(mode="json") if value else None


def _digest(value: str | None) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _graph_warnings(provenance: GraphEvidenceProvenance | None) -> list[str]:
    if provenance is None:
        return []
    warnings = []
    if provenance.fallbackUsed:
        warnings.append("graph_fallback_used")
    status = provenance.consistency.status.upper() if provenance.consistency else "UNKNOWN"
    if status == "DRIFT":
        warnings.append("graph_snapshot_drift")
    if status == "UNAVAILABLE" and "graph_fallback_used" not in warnings:
        warnings.append("graph_consistency_unavailable")
    return warnings


async def _retry(operation):
    retries = 0
    for attempt in range(3):
        try:
            return await operation(), retries
        except ToolContractError:
            raise
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep((0.5, 1.0)[attempt])
            retries += 1
    raise RuntimeError("unreachable")


def build_review_graph(dependencies: ReviewDependencies, checkpointer: AsyncSqliteSaver):
    def safety_gate(state: ReviewState) -> dict[str, Any]:
        question = state.get("question")
        if not isinstance(question, str) or not question.strip():
            try:
                question = dependencies.repository.get(state["reviewId"]).question
            except (KeyError, TypeError):
                question = ""
        decision = evaluate_review_question(question)
        return {
            "question": question,
            "questionSafety": decision.model_dump(mode="json"),
        }

    def explain_scope(state: ReviewState) -> dict[str, Any]:
        decision = state["questionSafety"]
        return {
            "status": ReviewStatus.CANCELLED.value,
            "unresolvedItems": [{
                "kind": "SCOPE_LIMITATION",
                "code": decision["code"],
                "summary": decision["explanation"],
            }],
        }

    async def audited_call(
        state: ReviewState, node: str, tool: str, summary: dict[str, Any], operation,
        *, audit_slot: str | None = None,
    ):
        try:
            result, retries = await _retry(operation)
            dependencies.repository.append_audit(
                state["reviewId"], node=node, tool=tool,
                request_id=result.envelope.requestId, result_status=result.envelope.status.value,
                argument_summary=summary, evidence_refs=result.envelope.evidenceRefs,
                latency_ms=result.latency_ms, retry_count=retries,
                mutation_id=state.get("mutationId"),
                audit_slot=audit_slot or node,
            )
            return result, retries, None
        except ToolContractError as exc:
            dependencies.repository.append_audit(
                state["reviewId"], node=node, tool=tool, request_id=None,
                result_status="CONTRACT_ERROR", argument_summary=summary,
                evidence_refs=[], latency_ms=0,
                mutation_id=state.get("mutationId"),
                audit_slot=audit_slot or node,
            )
            return None, 0, str(exc)
        except Exception as exc:
            dependencies.repository.append_audit(
                state["reviewId"], node=node, tool=tool, request_id=None,
                result_status="ERROR", argument_summary=summary,
                evidence_refs=[], latency_ms=0, retry_count=2,
                mutation_id=state.get("mutationId"),
                audit_slot=audit_slot or node,
            )
            return None, 2, type(exc).__name__

    def add_metrics(state: ReviewState, result: TimedToolResult | None, retries: int) -> dict[str, Any]:
        metrics = dict(state.get("metrics") or {})
        metrics.setdefault("toolLatencyMs", 0)
        metrics.setdefault("retries", 0)
        metrics.setdefault("inputTokens", 0)
        metrics.setdefault("outputTokens", 0)
        metrics.setdefault("estimatedCost", 0.0)
        metrics["toolLatencyMs"] += result.latency_ms if result else 0
        metrics["retries"] += retries
        return metrics

    async def collect_review_context(state: ReviewState) -> dict[str, Any]:
        result, retries, error = await audited_call(
            state, "collect_review_context", "get_medication_review_context",
            {"patientIdHash": _digest(state.get("patientRef"))},
            lambda: dependencies.health.get_review_context(state.get("patientRef"), state.get("asOf")),
        )
        metrics = add_metrics(state, result, retries)
        if error or result is None:
            return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}], "metrics": metrics}
        envelope = result.envelope
        if envelope.status.value == "AMBIGUOUS":
            return {"status": ReviewStatus.AWAITING_PATIENT_CONFIRMATION.value, "candidates": envelope.data.get("candidates", []), "metrics": metrics}
        if envelope.status.value in {"ERROR", "UNMAPPED"}:
            return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "PATIENT_CONTEXT", "errors": envelope.errors}], "metrics": metrics}
        return {"status": ReviewStatus.RUNNING.value, "contextSnapshot": envelope.data, "contextMissingFields": envelope.data.get("missingFields", []), "metrics": metrics}

    def select_patient(state: ReviewState) -> dict[str, Any]:
        payload = interrupt({"kind": "PATIENT_CONFIRMATION", "candidates": state.get("candidates", [])})
        decision = PatientConfirmation.model_validate(payload)
        candidate_ids = {str(item.get("id")) for item in state.get("candidates", [])}
        if decision.patientId not in candidate_ids:
            raise ValueError("Confirmed patient is not a returned candidate")
        human = HumanDecision(action=decision.action, reviewerId=decision.reviewerId, note=f"FHIR:Patient/{decision.patientId}")
        return {"patientRef": decision.patientId, "humanDecisions": [*(state.get("humanDecisions") or []), human.model_dump(mode="json")], "status": ReviewStatus.RUNNING.value}

    def validate_context(state: ReviewState) -> dict[str, Any]:
        context = state.get("contextSnapshot") or {}
        patient = context.get("patient") or {}
        return {"patientRef": f"FHIR:Patient/{patient.get('id')}" if patient.get("id") else state.get("patientRef"), "status": ReviewStatus.RUNNING.value}

    def normalized_medications(context: dict[str, Any]) -> list[dict[str, Any]]:
        records = []
        for item in context.get("activeMedications", []):
            records.append(MedicationRecord(
                medicationId=str(item.get("id")), name=str(item.get("medication") or "Unknown"),
                identifiers=item.get("identifiers") or [], strength=item.get("strength"),
                dosageForm=item.get("dosageForm"), route=item.get("route"), dosage=item.get("dosage"),
                patientEvidenceRefs=[item["evidenceRef"]] if item.get("evidenceRef") else [],
            ).model_dump(mode="json"))
        return records

    def normalize_medications(state: ReviewState) -> dict[str, Any]:
        return {"medications": normalized_medications(state.get("contextSnapshot") or {})}

    def mapping_from_result(medication: MedicationRecord, result: TimedToolResult) -> dict[str, Any]:
        status = result.envelope.status.value
        data = result.envelope.data
        provenance = result.envelope.graph_provenance
        graph_warnings = _graph_warnings(provenance)
        if status == "UNMAPPED":
            return MedicationMapping(
                medicationId=medication.medicationId, sourceName=medication.name,
                matchClass="UNMAPPED", selectedProductId=None, candidates=[],
                unmatchedFields=data.get("unmatchedFields") or [],
                graphProvenance=provenance, requiresHumanReview=bool(graph_warnings),
                mappingConfirmationRequired=False,
            ).model_dump(mode="json")
        match_class = str(data.get("matchClass") or status)
        unmatched_fields = data.get("unmatchedFields") or []
        candidates = list(data.get("candidates") or [])
        selected_product = data.get("selectedProductId")
        scoped_name_is_safe = (
            status == "OK"
            and data.get("autoAcceptable") is True
            and len(candidates) == 1
            and candidates[0].get("productId") == selected_product
            and not unmatched_fields
        )
        mapping_confirmation_required = (
            status == "AMBIGUOUS"
            or match_class not in {"EXACT_IDENTIFIER", "EXACT_SCOPED_NAME"}
            or (match_class == "EXACT_SCOPED_NAME" and not scoped_name_is_safe)
        )
        if mapping_confirmation_required and selected_product and selected_product not in {
            item.get("productId") for item in candidates
        }:
            candidates.append({"productId": selected_product})
        return MedicationMapping(
            medicationId=medication.medicationId, sourceName=medication.name,
            matchClass=match_class, selectedProductId=selected_product,
            candidates=candidates, unmatchedFields=unmatched_fields,
            graphProvenance=provenance,
            requiresHumanReview=mapping_confirmation_required or bool(graph_warnings),
            mappingConfirmationRequired=mapping_confirmation_required,
        ).model_dump(mode="json")

    async def resolve_medications(state: ReviewState) -> dict[str, Any]:
        mappings = []
        total_retries = 0
        metrics = dict(state.get("metrics") or {})
        for raw in state.get("medications", []):
            medication = MedicationRecord.model_validate(raw)
            result, retries, error = await audited_call(
                state, "resolve_medications", "resolve_medication",
                {"medicationIdHash": _digest(medication.medicationId)},
                lambda medication=medication: dependencies.drug.resolve_medication(
                    name=medication.name, identifiers=medication.identifiers,
                    strength=medication.strength, dosage_form=medication.dosageForm,
                    route=medication.route,
                ),
                audit_slot=f"resolve_medications:{_digest(medication.medicationId)}",
            )
            total_retries += retries
            if error or result is None:
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}], "metrics": add_metrics({"metrics": metrics}, result, total_retries)}
            if result.envelope.status.value not in {"OK", "AMBIGUOUS", "UNMAPPED"}:
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{
                        "kind": "MAPPING_ERROR", "status": result.envelope.status.value,
                        "errors": result.envelope.errors,
                    }],
                    "metrics": add_metrics({"metrics": metrics}, result, total_retries),
                }
            mappings.append(mapping_from_result(medication, result))
            metrics = add_metrics({"metrics": metrics}, result, retries)
        status = ReviewStatus.AWAITING_MAPPING_CONFIRMATION.value if any(
            item.get("mappingConfirmationRequired")
            for item in mappings
        ) else ReviewStatus.RUNNING.value
        return {"medicationMappings": mappings, "status": status, "metrics": metrics}

    def confirm_mapping(state: ReviewState) -> dict[str, Any]:
        ambiguous = [
            item for item in state.get("medicationMappings", [])
            if item.get("mappingConfirmationRequired")
        ]
        payload = interrupt({"kind": "MAPPING_CONFIRMATION", "mappings": ambiguous})
        decision = MappingConfirmation.model_validate(payload)
        mappings = [dict(item) for item in state.get("medicationMappings", [])]
        matched = False
        for item in mappings:
            if item["medicationId"] == decision.medicationId:
                candidate_ids = {candidate.get("productId") for candidate in item.get("candidates", [])}
                if decision.productId not in candidate_ids:
                    raise ValueError("Confirmed product is not a returned candidate")
                item["selectedProductId"] = decision.productId
                item["matchClass"] = "HUMAN_CONFIRMED"
                item["requiresHumanReview"] = False
                item["mappingConfirmationRequired"] = False
                matched = True
        if not matched:
            raise ValueError("Medication mapping decision does not match unresolved medication")
        human = HumanDecision(action=decision.action, reviewerId=decision.reviewerId, medicationId=decision.medicationId, productId=decision.productId)
        status = ReviewStatus.AWAITING_MAPPING_CONFIRMATION.value if any(
            item.get("mappingConfirmationRequired") for item in mappings
        ) else ReviewStatus.RUNNING.value
        findings = promote_mapping_findings(state, mappings)
        return {
            "medicationMappings": mappings, "findings": findings,
            "humanDecisions": [*(state.get("humanDecisions") or []), human.model_dump(mode="json")],
            "status": status,
        }

    def promote_mapping_findings(state: ReviewState, mappings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        requested_ids = set(state.get("reinvestigateFindingIds") or [])
        findings = [dict(item) for item in state.get("findings", [])]
        medications = {item["medicationId"]: item for item in state.get("medications", [])}
        mapping_by_medication = {item["medicationId"]: item for item in mappings}
        for finding in findings:
            if finding.get("findingId") not in requested_ids:
                continue
            mapping = next((
                mapping_by_medication.get(medication_id)
                for medication_id in finding.get("medicationIds", [])
                if mapping_by_medication.get(medication_id, {}).get("selectedProductId")
            ), None)
            if mapping is None:
                continue
            medication = medications.get(mapping["medicationId"], {})
            finding.update({
                "reviewType": "LABEL_EVIDENCE_REVIEW",
                "summary": f"Review retrieved label evidence for {mapping['sourceName']}.",
                "selectedProductIds": [mapping["selectedProductId"]],
                "patientEvidenceRefs": medication.get("patientEvidenceRefs", []),
                "labelEvidenceRefs": [], "status": FindingStatus.PENDING.value,
                "verificationErrors": [],
            })
        return findings

    async def plan_review(state: ReviewState) -> dict[str, Any]:
        context = state.get("contextSnapshot") or {}
        features = {"age": (context.get("patient") or {}).get("age"), "allergies": context.get("allergies") or [], "specialPopulations": context.get("specialPopulations") or []}
        planning = await dependencies.planner.plan(
            state.get("question") or "默认用药证据核查",
            features,
            [MedicationMapping.model_validate(item) for item in state.get("medicationMappings", [])],
            state.get("contextMissingFields", []),
        )
        return {"reviewPlan": [item.model_dump(mode="json") for item in planning.items]}

    async def retrieve_evidence(state: ReviewState) -> dict[str, Any]:
        selected = [item["selectedProductId"] for item in state.get("medicationMappings", []) if item.get("selectedProductId")]
        evidence: list[dict[str, Any]] = []
        unresolved_items = [
            dict(item) for item in state.get("unresolvedItems", [])
            if item.get("kind") != "EVIDENCE_GAP"
        ]
        metrics = dict(state.get("metrics") or {})
        for product_id in selected:
            result, retries, error = await audited_call(state, "retrieve_evidence", "get_product_facts", {"productIdHash": _digest(product_id)}, lambda product_id=product_id: dependencies.drug.get_product_facts(product_id), audit_slot=f"retrieve_facts:{_digest(product_id)}")
            metrics = add_metrics({"metrics": metrics}, result, retries)
            if error:
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}], "metrics": metrics}
            if result is not None and result.envelope.status.value == "INSUFFICIENT_EVIDENCE":
                unresolved_items.append({
                    "kind": "EVIDENCE_GAP", "sourceTool": "get_product_facts",
                    "summary": f"Product facts were unavailable for {product_id}.",
                    "productIds": [product_id], "evidenceRefs": result.envelope.evidenceRefs,
                    "graphProvenance": _graph_provenance(result), "errors": result.envelope.errors,
                })
                continue
            if result is None or result.envelope.status.value != "OK":
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "DRUG_EVIDENCE_ERROR", "errors": result.envelope.errors if result else ["missing response"]}], "metrics": metrics}
            evidence.append(EvidenceItem(
                evidenceId=f"facts-{uuid4()}", source="SPL-GRAPH",
                evidenceRef=next(iter(result.envelope.evidenceRefs), f"SPL-GRAPH:{product_id}"),
                productIds=[product_id], topic="product_facts",
                summary="Deterministic product graph facts retrieved.",
                graphProvenance=_graph_provenance(result),
            ).model_dump(mode="json"))
        topics = sorted({topic for item in state.get("reviewPlan", []) for topic in item.get("topics", [])})
        for product_id in selected:
            result, retries, error = await audited_call(
                state, "retrieve_evidence", "search_label_evidence",
                {"productCount": 1, "topicCount": len(topics)},
                lambda product_id=product_id: dependencies.drug.search_label_evidence([product_id], topics, None),
                audit_slot=f"retrieve_search:{_digest(product_id)}",
            )
            metrics = add_metrics({"metrics": metrics}, result, retries)
            if error or result is None:
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}], "metrics": metrics}
            if result.envelope.status.value == "INSUFFICIENT_EVIDENCE":
                unresolved_items.append({
                    "kind": "EVIDENCE_GAP", "sourceTool": "search_label_evidence",
                    "summary": f"Label evidence was insufficient for {product_id}.",
                    "productIds": [product_id], "evidenceRefs": result.envelope.evidenceRefs,
                    "graphProvenance": _graph_provenance(result), "errors": result.envelope.errors,
                })
                continue
            if result.envelope.status.value != "OK":
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "DRUG_EVIDENCE_ERROR", "errors": result.envelope.errors}], "metrics": metrics}
            provenance = _graph_provenance(result)
            for item in result.envelope.data.get("evidence", []):
                evidence.append(EvidenceItem(
                    evidenceId=str(item.get("referenceId") or uuid4()), source="SPL",
                    evidenceRef=item.get("evidenceRef"), productIds=[product_id],
                    summary=item.get("content"), graphProvenance=provenance,
                ).model_dump(mode="json"))
        if len(selected) > 1:
            result, retries, error = await audited_call(state, "retrieve_evidence", "compare_product_ingredients", {"productCount": len(selected)}, lambda: dependencies.drug.compare_product_ingredients(selected))
            metrics = add_metrics({"metrics": metrics}, result, retries)
            if error:
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}], "metrics": metrics}
            if result is not None and result.envelope.status.value == "INSUFFICIENT_EVIDENCE":
                unresolved_items.append({
                    "kind": "EVIDENCE_GAP", "sourceTool": "compare_product_ingredients",
                    "summary": "Ingredient comparison evidence was insufficient for the selected products.",
                    "productIds": selected, "evidenceRefs": result.envelope.evidenceRefs,
                    "graphProvenance": _graph_provenance(result), "errors": result.envelope.errors,
                })
                return {"evidenceIndex": evidence, "unresolvedItems": unresolved_items, "metrics": metrics}
            if result is None or result.envelope.status.value != "OK":
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "DRUG_EVIDENCE_ERROR", "errors": result.envelope.errors if result else ["missing response"]}], "metrics": metrics}
            shared = result.envelope.data.get("sharedActiveIngredients", []) if result else []
            if shared:
                graph_ref = next(
                    (ref for ref in result.envelope.evidenceRefs if ref.startswith("SPL:")),
                    next((item["evidenceRef"] for item in evidence if item.get("evidenceRef", "").startswith("SPL:")), "SPL-GRAPH:ingredient-comparison"),
                )
                evidence.append(EvidenceItem(
                    evidenceId=f"comparison-{uuid4()}", source="SPL-GRAPH",
                    evidenceRef=graph_ref, productIds=selected,
                    topic="shared_active_ingredients", summary=str(shared),
                    graphProvenance=_graph_provenance(result),
                    sharedActiveIngredients=shared,
                ).model_dump(mode="json"))
        return {"evidenceIndex": evidence, "unresolvedItems": unresolved_items, "metrics": metrics}

    def build_findings(state: ReviewState) -> dict[str, Any]:
        medications = {item["medicationId"]: MedicationRecord.model_validate(item) for item in state.get("medications", [])}
        findings = [Finding(
            findingId=str(uuid4()), reviewType="EVIDENCE_GAP",
            summary=f"Patient information is not recorded: {missing_field}.",
            attentionLevel="HIGH", confidence=1.0, requiresHumanReview=True,
            missingField=missing_field,
        ).model_dump(mode="json") for missing_field in state.get("contextMissingFields", [])]
        for gap in [item for item in state.get("unresolvedItems", []) if item.get("kind") == "EVIDENCE_GAP"]:
            findings.append(Finding(
                findingId=str(uuid4()), reviewType="EVIDENCE_GAP",
                summary=str(gap.get("summary") or "Drug evidence is insufficient."),
                attentionLevel="HIGH", confidence=1.0,
                selectedProductIds=gap.get("productIds", []),
                labelEvidenceRefs=[
                    ref for ref in gap.get("evidenceRefs", []) if str(ref).startswith("SPL:")
                ],
                graphProvenance=gap.get("graphProvenance"), requiresHumanReview=True,
                sourceTool=gap.get("sourceTool"), sourceErrors=gap.get("errors", []),
            ).model_dump(mode="json"))
        provenance_gaps: set[tuple[str | None, bool, str]] = set()
        for item in state.get("evidenceIndex", []):
            provenance = GraphEvidenceProvenance.model_validate(item["graphProvenance"]) if item.get("graphProvenance") else None
            warnings = _graph_warnings(provenance)
            if not warnings or provenance is None:
                continue
            consistency = provenance.consistency.status if provenance.consistency else "UNKNOWN"
            identity = (provenance.graphBackend, provenance.fallbackUsed, consistency)
            if identity in provenance_gaps:
                continue
            provenance_gaps.add(identity)
            findings.append(Finding(
                findingId=str(uuid4()), reviewType="EVIDENCE_GAP",
                summary="Drug evidence graph provenance requires pharmacist review.",
                attentionLevel="HIGH", confidence=1.0, requiresHumanReview=True,
                verificationWarnings=warnings, graphProvenance=provenance,
            ).model_dump(mode="json"))
        for raw in state.get("medicationMappings", []):
            mapping = MedicationMapping.model_validate(raw)
            changed_ids = set(state.get("contextChangedMedicationIds") or [])
            if changed_ids and mapping.medicationId not in changed_ids:
                continue
            medication = medications[mapping.medicationId]
            relevant_evidence = [item for item in state.get("evidenceIndex", []) if mapping.selectedProductId in item.get("productIds", [])]
            label_refs = [
                item["evidenceRef"] for item in relevant_evidence
                if item.get("source") == "SPL" and str(item.get("evidenceRef", "")).startswith("SPL:")
            ]
            evidence_provenances = [GraphEvidenceProvenance.model_validate(item["graphProvenance"]) for item in relevant_evidence if item.get("graphProvenance")]
            warnings = list(dict.fromkeys([
                *_graph_warnings(mapping.graphProvenance),
                *(warning for provenance in evidence_provenances for warning in _graph_warnings(provenance)),
            ]))
            finding_provenance = next((item for item in evidence_provenances if _graph_warnings(item)), mapping.graphProvenance)
            if not mapping.selectedProductId:
                findings.append(Finding(
                    findingId=str(uuid4()), reviewType="EVIDENCE_GAP",
                    summary=f"No DailyMed product was mapped for {mapping.sourceName}.",
                    attentionLevel="HIGH", confidence=1.0, medicationIds=[mapping.medicationId],
                    patientEvidenceRefs=medication.patientEvidenceRefs,
                    requiresHumanReview=True,
                ).model_dump(mode="json"))
                continue
            findings.append(Finding(
                findingId=str(uuid4()), reviewType="LABEL_EVIDENCE_REVIEW",
                summary=f"Review retrieved label evidence for {mapping.sourceName}.",
                attentionLevel="HIGH" if warnings else "MEDIUM", confidence=0.8,
                medicationIds=[mapping.medicationId], selectedProductIds=[mapping.selectedProductId],
                patientEvidenceRefs=medication.patientEvidenceRefs, labelEvidenceRefs=label_refs,
                requiresHumanReview=True, verificationWarnings=warnings,
                graphProvenance=finding_provenance,
            ).model_dump(mode="json"))
            if warnings:
                findings.append(Finding(
                    findingId=str(uuid4()), reviewType="EVIDENCE_GAP",
                    summary="Graph provenance requires pharmacist review.", attentionLevel="HIGH",
                    confidence=1.0, medicationIds=[mapping.medicationId],
                    patientEvidenceRefs=medication.patientEvidenceRefs,
                    requiresHumanReview=True, verificationWarnings=warnings,
                    graphProvenance=mapping.graphProvenance,
                ).model_dump(mode="json"))
        for comparison in [item for item in state.get("evidenceIndex", []) if item.get("source") == "SPL-GRAPH" and item.get("topic") == "shared_active_ingredients"]:
            provenance = comparison.get("graphProvenance")
            findings.append(Finding(
                findingId=str(uuid4()), reviewType="DUPLICATE_ACTIVE_INGREDIENT",
                summary=f"Mapped products share active ingredient graph entities: {comparison.get('summary')}.",
                attentionLevel="HIGH", confidence=1.0,
                medicationIds=list(medications),
                selectedProductIds=comparison.get("productIds", []),
                patientEvidenceRefs=[ref for item in medications.values() for ref in item.patientEvidenceRefs],
                labelEvidenceRefs=[comparison["evidenceRef"]] if str(comparison.get("evidenceRef", "")).startswith("SPL:") else [],
                requiresHumanReview=True, graphProvenance=provenance,
                sharedActiveIngredients=comparison.get("sharedActiveIngredients", []),
            ).model_dump(mode="json"))
        changed_ids = set(state.get("contextChangedMedicationIds") or [])
        if changed_ids:
            new_ids = {item.get("findingId") for item in findings}
            findings.extend(
                dict(item) for item in state.get("findings", [])
                if item.get("findingId") not in new_ids
                and item.get("medicationIds")
                and not changed_ids.intersection(item.get("medicationIds", []))
            )
        return {"findings": findings}

    async def reinvestigate_context(state: ReviewState) -> dict[str, Any]:
        patient_ref = state.get("patientRef")
        patient_id = patient_ref.removeprefix("FHIR:Patient/") if patient_ref else None
        result, retries, error = await audited_call(
            state, "reinvestigate_context", "get_medication_review_context",
            {"patientIdHash": _digest(patient_id)},
            lambda: dependencies.health.get_review_context(patient_id, state.get("asOf")),
        )
        metrics = add_metrics(state, result, retries)
        if error or result is None:
            return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}], "metrics": metrics}
        if result.envelope.status.value not in {"OK", "INSUFFICIENT_EVIDENCE"}:
            return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "PATIENT_CONTEXT", "errors": result.envelope.errors}], "metrics": metrics}
        findings = [dict(item) for item in state.get("findings", [])]
        requested_ids = set(state.get("reinvestigateFindingIds") or [])
        missing = set(result.envelope.data.get("missingFields", []))
        for item in findings:
            if item.get("findingId") not in requested_ids:
                continue
            item["verificationErrors"] = []
            still_missing = item.get("missingField") in missing
            item["stillMissing"] = still_missing
            if still_missing:
                item["status"] = FindingStatus.PENDING.value
            else:
                item["status"] = FindingStatus.REJECTED.value
                item["summary"] = f"Resolved: patient information is now recorded: {item.get('missingField')}."
                item["resolution"] = "RESOLVED_BY_CONTEXT_REFRESH"
        context = result.envelope.data
        medications = normalized_medications(context)
        old_medications = {
            item["medicationId"]: MedicationRecord.model_validate(item).model_dump(mode="json")
            for item in state.get("medications", [])
        }
        new_medications = {
            item["medicationId"]: MedicationRecord.model_validate(item).model_dump(mode="json")
            for item in medications
        }
        medications_changed = old_medications != new_medications
        changed_ids = sorted({*old_medications, *new_medications} - {
            medication_id for medication_id in set(old_medications) & set(new_medications)
            if old_medications[medication_id] == new_medications[medication_id]
        })
        features = {
            "age": (context.get("patient") or {}).get("age"),
            "allergies": context.get("allergies") or [],
            "specialPopulations": context.get("specialPopulations") or [],
        }
        planning = await dependencies.planner.plan(
            state.get("question") or "默认用药证据核查",
            features,
            [MedicationMapping.model_validate(item) for item in state.get("medicationMappings", [])],
            list(missing),
        )
        return {
            "contextSnapshot": context,
            "contextMissingFields": list(missing),
            "medications": medications,
            "reviewPlan": [item.model_dump(mode="json") for item in planning.items],
            "findings": findings, "reinvestigateFindingIds": [], "metrics": metrics,
            "contextMedicationsChanged": medications_changed,
            "contextChangedMedicationIds": changed_ids,
            **({"medicationMappings": [], "evidenceIndex": []} if medications_changed else {}),
        }

    async def reinvestigate_mapping(state: ReviewState) -> dict[str, Any]:
        requested_ids = set(state.get("reinvestigateFindingIds") or [])
        target_medication_ids = {
            medication_id
            for item in state.get("findings", [])
            if item.get("findingId") in requested_ids
            for medication_id in item.get("medicationIds", [])
        }
        medications = {
            item["medicationId"]: MedicationRecord.model_validate(item)
            for item in state.get("medications", [])
        }
        mappings = [dict(item) for item in state.get("medicationMappings", [])]
        mapping_positions = {item["medicationId"]: index for index, item in enumerate(mappings)}
        metrics = dict(state.get("metrics") or {})
        for medication_id in target_medication_ids:
            medication = medications[medication_id]
            result, retries, error = await audited_call(
                state, "reinvestigate_mapping", "resolve_medication",
                {"medicationIdHash": _digest(medication_id)},
                lambda medication=medication: dependencies.drug.resolve_medication(
                    name=medication.name, identifiers=medication.identifiers,
                    strength=medication.strength, dosage_form=medication.dosageForm,
                    route=medication.route,
                ),
                audit_slot=f"reinvestigate_mapping:{_digest(medication_id)}",
            )
            metrics = add_metrics({"metrics": metrics}, result, retries)
            if error or result is None:
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}], "metrics": metrics}
            if result.envelope.status.value not in {"OK", "AMBIGUOUS", "UNMAPPED"}:
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "MAPPING_ERROR", "status": result.envelope.status.value, "errors": result.envelope.errors}], "metrics": metrics}
            mapped = mapping_from_result(medication, result)
            if medication_id in mapping_positions:
                mappings[mapping_positions[medication_id]] = mapped
            else:
                mappings.append(mapped)
        confirmation_required = any(
            item.get("mappingConfirmationRequired")
            for item in mappings if item.get("medicationId") in target_medication_ids
        )
        findings = promote_mapping_findings(state, mappings)
        mapped_targets = any(
            item.get("selectedProductId")
            for item in mappings if item.get("medicationId") in target_medication_ids
        )
        return {
            "medicationMappings": mappings, "findings": findings, "metrics": metrics,
            "status": ReviewStatus.AWAITING_MAPPING_CONFIRMATION.value if confirmation_required else ReviewStatus.RUNNING.value,
            "reinvestigateFindingIds": list(requested_ids) if confirmation_required or mapped_targets else [],
        }

    async def reinvestigate_evidence(state: ReviewState) -> dict[str, Any]:
        requested_ids = set(state.get("reinvestigateFindingIds") or [])
        findings = [dict(item) for item in state.get("findings", [])]
        directly_requested = [item for item in findings if item.get("findingId") in requested_ids]
        canonical_ids = {
            item.get("provenanceForFindingId") or item["findingId"]
            for item in directly_requested
        }
        requested = [item for item in findings if item.get("findingId") in canonical_ids]
        if not requested:
            requested = directly_requested
        target_products = {
            product_id
            for item in requested
            for product_id in item.get("selectedProductIds", [])
        }
        target_medication_ids = {
            medication_id
            for item in requested
            for medication_id in item.get("medicationIds", [])
        }
        topics = sorted({
            topic
            for item in state.get("reviewPlan", [])
            if target_medication_ids.intersection(item.get("medicationIds", []))
            for topic in item.get("topics", [])
        })
        retained_evidence = [
            dict(item) for item in state.get("evidenceIndex", [])
            if not target_products.intersection(item.get("productIds", []))
        ]
        refreshed_evidence: list[dict[str, Any]] = []
        refreshed_gaps: list[dict[str, Any]] = []
        comparison_provenance: dict[str, GraphEvidenceProvenance | None] = {}
        comparison_references: dict[str, list[str]] = {}
        stale_gap_ids = {
            gap_id
            for target_id in canonical_ids
            for gap_id in (
                f"provenance-gap-{target_id}",
                f"evidence-gap-get_product_facts-{target_id}",
                f"evidence-gap-search_label_evidence-{target_id}",
                f"evidence-gap-compare_product_ingredients-{target_id}",
            )
        }
        stale_gap_ids.update(item["findingId"] for item in directly_requested if item.get("provenanceForFindingId"))
        findings = [item for item in findings if item.get("findingId") not in stale_gap_ids]
        metrics = dict(state.get("metrics") or {})
        for product_id in sorted(target_products):
            facts, retries, error = await audited_call(
                state, "reinvestigate_evidence", "get_product_facts",
                {"productIdHash": _digest(product_id)},
                lambda product_id=product_id: dependencies.drug.get_product_facts(product_id),
                audit_slot=f"reinvestigate_facts:{_digest(product_id)}",
            )
            metrics = add_metrics({"metrics": metrics}, facts, retries)
            if error or facts is None:
                errors = facts.envelope.errors if facts else [error or "missing response"]
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "DRUG_EVIDENCE_ERROR", "errors": errors}], "metrics": metrics}
            if facts.envelope.status.value == "INSUFFICIENT_EVIDENCE":
                refreshed_gaps.append({
                    "sourceTool": "get_product_facts", "productIds": [product_id],
                    "summary": f"Product facts were unavailable for {product_id}.",
                    "graphProvenance": _graph_provenance(facts),
                    "errors": facts.envelope.errors,
                })
            elif facts.envelope.status.value == "OK":
                refreshed_evidence.append(EvidenceItem(
                    evidenceId=f"facts-{uuid4()}", source="SPL-GRAPH",
                    evidenceRef=next(iter(facts.envelope.evidenceRefs), f"SPL-GRAPH:{product_id}"),
                    productIds=[product_id], topic="product_facts",
                    summary="Deterministic product graph facts retrieved.",
                    graphProvenance=_graph_provenance(facts),
                ).model_dump(mode="json"))
            else:
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "DRUG_EVIDENCE_ERROR", "errors": facts.envelope.errors}], "metrics": metrics}
            search, retries, error = await audited_call(
                state, "reinvestigate_evidence", "search_label_evidence",
                {"productCount": 1, "topicCount": len(topics)},
                lambda product_id=product_id: dependencies.drug.search_label_evidence([product_id], topics, None),
                audit_slot=f"reinvestigate_search:{_digest(product_id)}",
            )
            metrics = add_metrics({"metrics": metrics}, search, retries)
            if error or search is None:
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}], "metrics": metrics}
            if search.envelope.status.value == "INSUFFICIENT_EVIDENCE":
                refreshed_gaps.append({
                    "sourceTool": "search_label_evidence", "productIds": [product_id],
                    "summary": f"Label evidence was insufficient for {product_id}.",
                    "graphProvenance": _graph_provenance(search),
                    "errors": search.envelope.errors,
                })
                continue
            if search.envelope.status.value != "OK":
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "DRUG_EVIDENCE_ERROR", "errors": search.envelope.errors}], "metrics": metrics}
            provenance = _graph_provenance(search)
            for evidence in search.envelope.data.get("evidence", []):
                refreshed_evidence.append(EvidenceItem(
                    evidenceId=str(evidence.get("referenceId") or uuid4()), source="SPL",
                    evidenceRef=evidence.get("evidenceRef"), productIds=[product_id],
                    summary=evidence.get("content"), graphProvenance=provenance,
                ).model_dump(mode="json"))
        resolved_comparison_ids: set[str] = set()
        comparison_targets_by_id = {
            (item.get("provenanceForFindingId") or item["findingId"]): item
            for item in [*requested, *directly_requested]
            if (
                item.get("reviewType") == "DUPLICATE_ACTIVE_INGREDIENT"
                or item.get("sourceTool") == "compare_product_ingredients"
            )
            and len(item.get("selectedProductIds", [])) > 1
        }
        for target in comparison_targets_by_id.values():
            product_ids = list(target.get("selectedProductIds", []))
            comparison, retries, error = await audited_call(
                state, "reinvestigate_evidence", "compare_product_ingredients",
                {"productCount": len(product_ids)},
                lambda product_ids=product_ids: dependencies.drug.compare_product_ingredients(product_ids),
                audit_slot=f"reinvestigate_compare:{_digest('|'.join(sorted(product_ids)))}",
            )
            metrics = add_metrics({"metrics": metrics}, comparison, retries)
            if error or comparison is None:
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}], "metrics": metrics}
            if comparison.envelope.status.value == "INSUFFICIENT_EVIDENCE":
                target_id = target.get("provenanceForFindingId") or target["findingId"]
                parent = next((item for item in findings if item.get("findingId") == target_id), None)
                if parent is not None:
                    parent["status"] = FindingStatus.NEEDS_MORE_EVIDENCE.value
                    parent["verificationErrors"] = list(dict.fromkeys([
                        *parent.get("verificationErrors", []), "comparison_evidence_insufficient",
                    ]))
                refreshed_gaps.append({
                    "sourceTool": "compare_product_ingredients", "productIds": product_ids,
                    "summary": "Ingredient comparison evidence was insufficient for the selected products.",
                    "graphProvenance": _graph_provenance(comparison),
                    "errors": comparison.envelope.errors,
                })
                continue
            if comparison.envelope.status.value != "OK":
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "DRUG_EVIDENCE_ERROR", "errors": comparison.envelope.errors}], "metrics": metrics}
            shared = comparison.envelope.data.get("sharedActiveIngredients", [])
            target_id = target.get("provenanceForFindingId") or target["findingId"]
            current = next((item for item in findings if item.get("findingId") == target_id), None)
            if current is None:
                current = dict(target)
                current["findingId"] = target_id
                findings.append(current)
            current["reviewType"] = "DUPLICATE_ACTIVE_INGREDIENT"
            current["sourceTool"] = "compare_product_ingredients"
            current["medicationIds"] = list(target.get("medicationIds") or [
                mapping["medicationId"] for mapping in state.get("medicationMappings", [])
                if mapping.get("selectedProductId") in product_ids
            ])
            current["patientEvidenceRefs"] = [
                ref for medication in state.get("medications", [])
                if medication.get("medicationId") in current["medicationIds"]
                for ref in medication.get("patientEvidenceRefs", [])
            ]
            current["sharedActiveIngredients"] = shared
            comparison_provenance[target_id] = comparison.envelope.graph_provenance
            comparison_references[target_id] = [
                ref for ref in comparison.envelope.evidenceRefs if ref.startswith("SPL:")
            ]
            if not shared:
                current["status"] = FindingStatus.REJECTED.value
                current["summary"] = "Resolved: refreshed comparison found no shared active ingredient."
                current["resolution"] = "RESOLVED_BY_INGREDIENT_REFRESH"
                resolved_comparison_ids.add(target_id)
            else:
                current["summary"] = f"Mapped products share active ingredient graph entities: {shared}."
                current["status"] = FindingStatus.PENDING.value
                current["verificationErrors"] = []
                reference = next((ref for ref in comparison.envelope.evidenceRefs if ref.startswith("SPL:")), None)
                if reference:
                    current["labelEvidenceRefs"] = [reference]
                current["graphProvenance"] = _graph_provenance(comparison)
        evidence_index = [*retained_evidence, *refreshed_evidence]
        provenance_gaps: list[dict[str, Any]] = []
        for item in findings:
            if item.get("findingId") not in requested_ids:
                continue
            product_ids = set(item.get("selectedProductIds", []))
            item["labelEvidenceRefs"] = [
                evidence["evidenceRef"] for evidence in refreshed_evidence
                if evidence.get("source") == "SPL"
                and product_ids.intersection(evidence.get("productIds", []))
                and str(evidence.get("evidenceRef", "")).startswith("SPL:")
            ]
            item["labelEvidenceRefs"] = list(dict.fromkeys([
                *item["labelEvidenceRefs"], *comparison_references.get(item["findingId"], []),
            ]))
            if item["findingId"] not in resolved_comparison_ids and "comparison_evidence_insufficient" not in item.get("verificationErrors", []):
                item["status"] = FindingStatus.PENDING.value
            if "comparison_evidence_insufficient" not in item.get("verificationErrors", []):
                item["verificationErrors"] = []
            provenances = [
                GraphEvidenceProvenance.model_validate(evidence["graphProvenance"])
                for evidence in refreshed_evidence
                if product_ids.intersection(evidence.get("productIds", []))
                and evidence.get("graphProvenance")
            ]
            warnings = list(dict.fromkeys(
                warning for provenance in provenances for warning in _graph_warnings(provenance)
            ))
            warning_provenance = next((provenance for provenance in provenances if _graph_warnings(provenance)), None)
            refreshed_provenance = warning_provenance or next(iter(provenances), None)
            if refreshed_provenance is not None:
                item["graphProvenance"] = refreshed_provenance.model_dump(mode="json")
            item["verificationWarnings"] = warnings
            if warnings:
                provenance_gaps.append(Finding(
                    findingId=f"provenance-gap-{item['findingId']}",
                    reviewType="EVIDENCE_GAP",
                    summary="Refreshed graph provenance requires pharmacist review.",
                    attentionLevel="HIGH", confidence=1.0,
                    medicationIds=item.get("medicationIds", []),
                    selectedProductIds=item.get("selectedProductIds", []),
                    patientEvidenceRefs=item.get("patientEvidenceRefs", []),
                    requiresHumanReview=True, verificationWarnings=warnings,
                    graphProvenance=warning_provenance,
                    provenanceForFindingId=item["findingId"],
                ).model_dump(mode="json"))
        for finding_id, provenance in comparison_provenance.items():
            if provenance is None:
                continue
            parent = next((item for item in findings if item.get("findingId") == finding_id), None)
            if parent is None:
                continue
            warnings = _graph_warnings(provenance)
            parent["graphProvenance"] = provenance.model_dump(mode="json")
            parent["verificationWarnings"] = list(dict.fromkeys([*parent.get("verificationWarnings", []), *warnings]))
            if warnings:
                gap_id = f"provenance-gap-{finding_id}"
                provenance_gaps = [item for item in provenance_gaps if item.get("findingId") != gap_id]
                provenance_gaps.append(Finding(
                    findingId=gap_id, reviewType="EVIDENCE_GAP",
                    summary="Refreshed comparison graph provenance requires pharmacist review.",
                    attentionLevel="HIGH", confidence=1.0,
                    medicationIds=parent.get("medicationIds", []),
                    selectedProductIds=parent.get("selectedProductIds", []),
                    patientEvidenceRefs=parent.get("patientEvidenceRefs", []),
                    requiresHumanReview=True, verificationWarnings=warnings,
                    graphProvenance=provenance, provenanceForFindingId=finding_id,
                ).model_dump(mode="json"))
        findings.extend(provenance_gaps)
        for gap in refreshed_gaps:
            for target in requested:
                if not set(target.get("selectedProductIds", [])).intersection(gap["productIds"]):
                    continue
                gap_id = f"evidence-gap-{gap['sourceTool']}-{target['findingId']}"
                findings = [item for item in findings if item.get("findingId") != gap_id]
                findings.append(Finding(
                    findingId=gap_id, reviewType="EVIDENCE_GAP", summary=gap["summary"],
                    attentionLevel="HIGH", confidence=1.0,
                    medicationIds=target.get("medicationIds", []),
                    selectedProductIds=gap["productIds"],
                    patientEvidenceRefs=target.get("patientEvidenceRefs", []),
                    graphProvenance=gap.get("graphProvenance"), requiresHumanReview=True,
                    sourceTool=gap["sourceTool"], sourceErrors=gap.get("errors", []),
                    provenanceForFindingId=target["findingId"],
                ).model_dump(mode="json"))
        return {
            "findings": findings, "evidenceIndex": evidence_index,
            "reinvestigateFindingIds": [], "metrics": metrics,
        }

    async def verify_findings_node(state: ReviewState) -> dict[str, Any]:
        claims = [{
            "claimId": item["findingId"], "reviewType": item["reviewType"],
            "patientEvidenceRefs": item.get("patientEvidenceRefs", []),
            "labelEvidenceRefs": item.get("labelEvidenceRefs", []),
            "selectedProductIds": item.get("selectedProductIds", []),
        } for item in state.get("findings", [])]
        result, retries, error = await audited_call(state, "verify_findings", "validate_evidence", {"claimCount": len(claims)}, lambda: dependencies.drug.validate_evidence(claims))
        metrics = add_metrics(state, result, retries)
        if error:
            return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}], "metrics": metrics}
        if result is None or result.envelope.status.value not in {"OK", "INSUFFICIENT_EVIDENCE"}:
            return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "VALIDATOR_ERROR", "errors": result.envelope.errors if result else ["missing response"]}], "metrics": metrics}
        findings = [dict(item) for item in state.get("findings", [])]
        remote = {item.get("claimId"): item for item in result.envelope.data.get("claims", [])} if result else {}
        verified_findings = []
        for item in findings:
            if item.get("status") == FindingStatus.REJECTED.value:
                verified_findings.append(item)
                continue
            verified_findings.append(apply_verification(
                Finding.model_validate(item),
                remote.get(item["findingId"], {}).get("errors", [])
                if item["findingId"] in remote
                else ([] if item["reviewType"] == "EVIDENCE_GAP" else ["remote_validation_incomplete"]),
            ).model_dump(mode="json"))
        findings = verified_findings
        return {"findings": findings, "status": ReviewStatus.AWAITING_FINDING_REVIEW.value, "metrics": metrics}

    def pharmacist_review(state: ReviewState) -> dict[str, Any]:
        payload = interrupt({"kind": "FINDING_REVIEW", "findings": state.get("findings", [])})
        decision = CompleteFindingReview.model_validate(payload)
        findings = [dict(item) for item in state.get("findings", [])]
        by_id = {item.findingId: item for item in decision.decisions}
        reinvestigate_ids: list[str] = []
        decision_events: list[dict[str, Any]] = []
        for item in findings:
            selected = by_id.get(item["findingId"])
            if selected is None:
                continue
            if selected.action == "ACCEPT_FINDING":
                lacks_pair = item["reviewType"] != "EVIDENCE_GAP" and (
                    not item.get("patientEvidenceRefs") or not item.get("labelEvidenceRefs")
                )
                if item.get("verificationErrors") or lacks_pair:
                    item["status"] = FindingStatus.NEEDS_MORE_EVIDENCE.value
                    item["verificationErrors"] = list(dict.fromkeys([
                        *item.get("verificationErrors", []),
                        *(["missing_paired_evidence"] if lacks_pair else []),
                    ]))
                else:
                    item["status"] = FindingStatus.ACCEPTED.value
            elif selected.action == "REJECT_FINDING":
                item["status"] = FindingStatus.REJECTED.value
            else:
                item["status"] = FindingStatus.NEEDS_MORE_EVIDENCE.value
                reinvestigate_ids.append(item["findingId"])
            decision_events.append(HumanDecision(
                action=selected.action, reviewerId=decision.reviewerId,
                findingId=selected.findingId, note=selected.note,
            ).model_dump(mode="json"))
        if reinvestigate_ids:
            status = ReviewStatus.NEEDS_MORE_EVIDENCE.value
        elif any(item["status"] == FindingStatus.NEEDS_MORE_EVIDENCE.value for item in findings):
            status = ReviewStatus.NEEDS_MORE_EVIDENCE.value
        elif any(item["status"] == FindingStatus.PENDING.value for item in findings):
            status = ReviewStatus.AWAITING_FINDING_REVIEW.value
        else:
            status = ReviewStatus.READY_FOR_SIGN_OFF.value
        human = HumanDecision(action=decision.action, reviewerId=decision.reviewerId)
        return {"findings": findings, "humanDecisions": [*(state.get("humanDecisions") or []), *decision_events, human.model_dump(mode="json")], "status": status, "reinvestigateFindingIds": reinvestigate_ids}

    def render_report(state: ReviewState) -> dict[str, Any]:
        if state.get("status") != ReviewStatus.READY_FOR_SIGN_OFF.value:
            return {}
        payload = interrupt({"kind": "FINAL_SIGN_OFF", "reviewId": state["reviewId"]})
        decision = FinalSignOff.model_validate(payload)
        human = HumanDecision(action=decision.action, reviewerId=decision.reviewerId)
        return {"humanDecisions": [*(state.get("humanDecisions") or []), human.model_dump(mode="json")], "status": ReviewStatus.SIGNED_OFF.value}

    graph = StateGraph(ReviewState)
    graph.add_node("safety_gate", safety_gate)
    graph.add_node("explain_scope", explain_scope)
    graph.add_node("collect_review_context", collect_review_context)
    graph.add_node("select_patient", select_patient)
    graph.add_node("validate_context", validate_context)
    graph.add_node("normalize_medications", normalize_medications)
    graph.add_node("resolve_medications", resolve_medications)
    graph.add_node("confirm_mapping", confirm_mapping)
    graph.add_node("plan_review", plan_review)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("build_findings", build_findings)
    graph.add_node("reinvestigate_context", reinvestigate_context)
    graph.add_node("reinvestigate_mapping", reinvestigate_mapping)
    graph.add_node("reinvestigate_evidence", reinvestigate_evidence)
    graph.add_node("verify_findings", verify_findings_node)
    graph.add_node("pharmacist_review", pharmacist_review)
    graph.add_node("render_report", render_report)
    graph.add_edge(START, "safety_gate")
    graph.add_conditional_edges(
        "safety_gate",
        lambda state: "collect" if state["questionSafety"]["allowed"] else "explain",
        {"collect": "collect_review_context", "explain": "explain_scope"},
    )
    graph.add_edge("explain_scope", END)
    graph.add_conditional_edges("collect_review_context", lambda state: "end" if state["status"] == ReviewStatus.BLOCKED_TOOL_ERROR.value else "select" if state["status"] == ReviewStatus.AWAITING_PATIENT_CONFIRMATION.value else "validate", {"end": END, "select": "select_patient", "validate": "validate_context"})
    graph.add_edge("select_patient", "collect_review_context")
    graph.add_edge("validate_context", "normalize_medications")
    graph.add_edge("normalize_medications", "resolve_medications")
    graph.add_conditional_edges("resolve_medications", lambda state: "end" if state["status"] == ReviewStatus.BLOCKED_TOOL_ERROR.value else "confirm" if state["status"] == ReviewStatus.AWAITING_MAPPING_CONFIRMATION.value else "plan", {"end": END, "confirm": "confirm_mapping", "plan": "plan_review"})
    graph.add_conditional_edges(
        "confirm_mapping",
        lambda state: "confirm" if state["status"] == ReviewStatus.AWAITING_MAPPING_CONFIRMATION.value else "reinvestigate" if state.get("reinvestigateFindingIds") else "plan",
        {"confirm": "confirm_mapping", "reinvestigate": "reinvestigate_evidence", "plan": "plan_review"},
    )
    graph.add_edge("plan_review", "retrieve_evidence")
    graph.add_conditional_edges("retrieve_evidence", lambda state: "end" if state.get("status") == ReviewStatus.BLOCKED_TOOL_ERROR.value else "build", {"end": END, "build": "build_findings"})
    graph.add_edge("build_findings", "verify_findings")
    graph.add_conditional_edges("verify_findings", lambda state: "end" if state["status"] == ReviewStatus.BLOCKED_TOOL_ERROR.value else "review", {"end": END, "review": "pharmacist_review"})
    graph.add_conditional_edges(
        "pharmacist_review",
        lambda state: (
            "report" if state["status"] == ReviewStatus.READY_FOR_SIGN_OFF.value
            else "context" if any(
                item.get("findingId") in set(state.get("reinvestigateFindingIds") or []) and item.get("missingField")
                for item in state.get("findings", [])
            )
            else "evidence" if any(
                item.get("findingId") in set(state.get("reinvestigateFindingIds") or []) and item.get("selectedProductIds")
                for item in state.get("findings", [])
            )
            else "mapping" if state.get("reinvestigateFindingIds")
            else "review"
        ),
        {
            "report": "render_report", "context": "reinvestigate_context",
            "mapping": "reinvestigate_mapping", "evidence": "reinvestigate_evidence",
            "review": "pharmacist_review",
        },
    )
    graph.add_conditional_edges(
        "reinvestigate_context",
        lambda state: "end" if state.get("status") == ReviewStatus.BLOCKED_TOOL_ERROR.value else "resolve" if state.get("contextMedicationsChanged") else "verify",
        {"end": END, "resolve": "resolve_medications", "verify": "verify_findings"},
    )
    graph.add_conditional_edges(
        "reinvestigate_mapping",
        lambda state: (
            "end" if state.get("status") == ReviewStatus.BLOCKED_TOOL_ERROR.value
            else "confirm" if state.get("status") == ReviewStatus.AWAITING_MAPPING_CONFIRMATION.value
            else "evidence" if state.get("reinvestigateFindingIds")
            else "verify"
        ),
        {
            "end": END, "confirm": "confirm_mapping",
            "evidence": "reinvestigate_evidence", "verify": "verify_findings",
        },
    )
    graph.add_conditional_edges(
        "reinvestigate_evidence",
        lambda state: "end" if state.get("status") == ReviewStatus.BLOCKED_TOOL_ERROR.value else "verify",
        {"end": END, "verify": "verify_findings"},
    )
    graph.add_edge("render_report", END)
    return graph.compile(checkpointer=checkpointer)


def state_to_snapshot(existing: ReviewSnapshot, state: dict[str, Any]) -> ReviewSnapshot:
    fields = ReviewSnapshot.model_fields
    merged = existing.model_dump(mode="python")
    merged.update({key: value for key, value in state.items() if key in fields})
    merged["updatedAt"] = datetime.now(UTC)
    return ReviewSnapshot.model_validate(merged)
