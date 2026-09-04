from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from .gateways import DrugEvidenceGateway, HealthRecordGateway, TimedToolResult, ToolContractError
from .models import (
    EvidenceItem, Finding, FindingStatus, GraphEvidenceProvenance, HumanDecision,
    MedicationMapping, MedicationRecord, ReviewIntent, ReviewPlanItem, ReviewSnapshot,
    ReviewStatus, ReviewTopic, UNRESOLVED_FINDING_TYPES, migrate_finding_payload,
)
from .planner import DeterministicPlanner, ReviewPlanner
from .repository import ReviewRepository
from .retrieval import (
    BoundedEvidenceRetriever,
    EvidenceGrader,
    ScopedRetrievalRequest,
    retrieval_attempt_key,
    stable_evidence_id,
)
from .safety import evaluate_review_question
from .verifier import apply_verification, verify_label_evidence_bindings
from .workflow_state import (
    CompleteFindingReview,
    FinalSignOff,
    FindingDecision as FindingDecision,
    MappingConfirmation,
    PatientConfirmation,
    ReviewState,
)


@dataclass(frozen=True)
class ReviewDependencies:
    health: HealthRecordGateway
    drug: DrugEvidenceGateway
    repository: ReviewRepository
    planner: ReviewPlanner
    grader: EvidenceGrader | None = None


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


def _document_ids(result: TimedToolResult) -> frozenset[str]:
    product = result.envelope.data.get("product") or {}
    values: set[str] = set()
    if isinstance(product, dict):
        document_id = product.get("documentId")
        if isinstance(document_id, str) and document_id:
            values.add(document_id)
        for document in product.get("documents") or []:
            if isinstance(document, dict) and document.get("documentId"):
                values.add(str(document["documentId"]))
    if not values:
        for reference in result.envelope.evidenceRefs:
            if reference.startswith("SPL:"):
                values.add(reference.removeprefix("SPL:").split("#", 1)[0])
    return frozenset(values)


def _document_versions(result: TimedToolResult) -> dict[str, str]:
    product = result.envelope.data.get("product") or {}
    if not isinstance(product, dict):
        return {}
    documents = [product, *(product.get("documents") or [])]
    versions: dict[str, str] = {}
    for document in documents:
        if not isinstance(document, dict):
            continue
        document_id = document.get("documentId")
        version = document.get("documentVersion")
        if isinstance(document_id, str) and document_id and isinstance(version, str) and version:
            versions[document_id] = version
    return versions


def _document_identities(
    result: TimedToolResult,
) -> frozenset[tuple[str, str, str]]:
    product = result.envelope.data.get("product") or {}
    if not isinstance(product, dict):
        return frozenset()
    documents = [product, *(product.get("documents") or [])]
    return frozenset(
        (
            str(document["documentId"]),
            str(document["documentVersion"]),
            str(document["contentHash"]),
        )
        for document in documents
        if isinstance(document, dict)
        and all(document.get(field) for field in (
            "documentId", "documentVersion", "contentHash",
        ))
    )


def _facts_match_product(result: TimedToolResult, product_id: str) -> bool:
    product = result.envelope.data.get("product")
    return isinstance(product, dict) and product.get("productId") == product_id


def _fact_evidence(
    product_id: str, result: TimedToolResult,
) -> EvidenceItem:
    product = result.envelope.data.get("product") or {}
    product = product if isinstance(product, dict) else {}
    document_ids = sorted(_document_ids(result))
    evidence_ref = next(
        iter(result.envelope.evidenceRefs),
        f"SPL-GRAPH:{product_id}",
    )
    return EvidenceItem(
        evidenceId=stable_evidence_id(
            "SPL-GRAPH",
            evidence_ref,
            product.get("documentVersion"),
            product.get("contentHash"),
        ),
        source="SPL-GRAPH",
        evidenceRef=evidence_ref,
        productIds=[product_id],
        topic="product_facts",
        summary="Deterministic product graph facts retrieved.",
        documentId=next(iter(document_ids), None),
        documentVersion=product.get("documentVersion"),
        effectiveTime=product.get("effectiveTime"),
        sourcePath=product.get("sourcePath"),
        contentHash=product.get("contentHash"),
        graphProvenance=_graph_provenance(result),
        activeIngredients=product.get("activeIngredients") or [],
        inactiveIngredients=product.get("inactiveIngredients") or [],
    )


def _comparison_evidence(
    product_ids: list[str],
    shared_ingredients: list[dict[str, Any]],
    result: TimedToolResult,
    *,
    fallback_ref: str = "SPL-GRAPH:ingredient-comparison",
) -> EvidenceItem:
    evidence_ref = next(
        (ref for ref in result.envelope.evidenceRefs if ref.startswith("SPL:")),
        fallback_ref,
    )
    content_hash = hashlib.sha256(json.dumps(
        shared_ingredients,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return EvidenceItem(
        evidenceId=stable_evidence_id(
            "SPL-GRAPH", evidence_ref, None, content_hash,
        ),
        source="SPL-GRAPH",
        evidenceRef=evidence_ref,
        productIds=product_ids,
        topic="shared_active_ingredients",
        summary=str(shared_ingredients),
        contentHash=content_hash,
        graphProvenance=_graph_provenance(result),
        sharedActiveIngredients=shared_ingredients,
    )


def _upsert_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    positions: dict[tuple[str, ...], int] = {}
    for raw in items:
        item = dict(raw)
        evidence_ref = str(item.get("evidenceRef") or "")
        if evidence_ref:
            identity = (
                "reference",
                str(item.get("source") or ""),
                evidence_ref,
                str(item.get("documentVersion") or ""),
                str(item.get("contentHash") or ""),
            )
        else:
            identity = (
                "id",
                str(item.get("source") or ""),
                str(item.get("evidenceId") or ""),
            )
        position = positions.get(identity)
        if position is None:
            positions[identity] = len(merged)
            merged.append(item)
            continue
        previous = merged[position]
        item["medicationIds"] = sorted({
            *previous.get("medicationIds", []),
            *item.get("medicationIds", []),
        })
        item["productIds"] = sorted({
            *previous.get("productIds", []),
            *item.get("productIds", []),
        })
        merged[position] = {**previous, **item}
    return merged


def _evidence_for_unchanged_findings(
    findings: list[dict[str, Any]],
    evidence_index: list[dict[str, Any]],
    changed_medication_ids: set[str],
) -> list[dict[str, Any]]:
    unchanged_findings = [
        finding for finding in findings
        if finding.get("medicationIds")
        and not changed_medication_ids.intersection(finding["medicationIds"])
    ]
    protected_ids = {
        evidence_id
        for finding in unchanged_findings
        for evidence_id in finding.get("labelEvidenceIds", [])
    }
    legacy_refs = {
        reference
        for finding in unchanged_findings
        if not finding.get("labelEvidenceIds")
        for reference in finding.get("labelEvidenceRefs", [])
    }
    return [
        dict(item) for item in evidence_index
        if (
            item.get("evidenceId") in protected_ids
            or item.get("evidenceRef") in legacy_refs
        )
    ]


def _label_evidence_bindings(
    evidence_index: list[dict[str, Any]],
    product_ids: set[str],
    *,
    sources: frozenset[str] = frozenset({"SPL"}),
    topics: frozenset[str] | None = None,
    document_identities: dict[str, frozenset[tuple[str, str, str]]] | None = None,
) -> tuple[list[str], list[str]]:
    def matches_current_document(item: dict[str, Any]) -> bool:
        if document_identities is None:
            return True
        identity = (
            str(item.get("documentId") or ""),
            str(item.get("documentVersion") or ""),
            str(item.get("contentHash") or ""),
        )
        claimed_products = set(item.get("productIds", []))
        return (
            bool(identity[0] and identity[1] and identity[2])
            and bool(claimed_products)
            and claimed_products <= product_ids
            and all(
            identity in document_identities.get(product_id, frozenset())
                for product_id in claimed_products
            )
        )

    relevant = [
        item for item in evidence_index
        if item.get("source") in sources
        and product_ids.intersection(item.get("productIds", []))
        and str(item.get("evidenceRef", "")).startswith("SPL:")
        and (topics is None or item.get("topic") in topics)
        and matches_current_document(item)
    ]
    return (
        list(dict.fromkeys(item["evidenceRef"] for item in relevant)),
        list(dict.fromkeys(
            item["evidenceId"] for item in relevant if item.get("evidenceId")
        )),
    )


async def _bounded_ordered_map(items, key, operation):
    semaphore = asyncio.Semaphore(4)

    async def run(item):
        async with semaphore:
            return key(item), await operation(item)

    completed = await asyncio.gather(*(
        run(item) for item in sorted(items, key=key)
    ))
    return [result for _, result in sorted(completed, key=lambda pair: pair[0])]


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


def deidentified_patient_features(context: dict[str, Any]) -> dict[str, Any]:
    patient = context.get("patient") or {}
    try:
        age = float(patient["age"])
    except (KeyError, TypeError, ValueError):
        age_band = "unknown"
    else:
        age_band = "child" if age < 18 else "adult" if age < 65 else "older_adult"
    allergy_terms: set[str] = set()
    for item in context.get("allergies") or []:
        if not isinstance(item, dict):
            continue
        value = item.get("substance") or item.get("name")
        if isinstance(value, str) and value.strip():
            allergy_terms.add(value.strip())
    special_population_flags: set[str] = set()
    for item in context.get("specialPopulations") or []:
        if isinstance(item, str) and item.strip():
            special_population_flags.add(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        value = item.get("value")
        name_text = name.strip() if isinstance(name, str) else ""
        value_text = value.strip() if isinstance(value, str) else ""
        if name_text and value_text:
            special_population_flags.add(f"{name_text}: {value_text}")
        elif value_text:
            special_population_flags.add(value_text)
        elif name_text:
            special_population_flags.add(name_text)
    return {
        "ageBand": age_band,
        "allergyTerms": sorted(allergy_terms),
        "specialPopulationFlags": sorted(special_population_flags),
    }


def relevant_missing_fields(
    topics: list[ReviewTopic], missing_fields: list[str],
) -> list[str]:
    topic_values = {topic.value for topic in topics}

    def relevant(field: str) -> bool:
        return (
            (ReviewTopic.WARNINGS.value in topic_values and field == "allergies")
            or (ReviewTopic.ROUTE.value in topic_values and field.endswith(".route"))
            or (
                ReviewTopic.DOSAGE_FORM.value in topic_values
                and field.endswith(".dosageForm")
            )
            or (ReviewTopic.DOSAGE.value in topic_values and field.endswith(".dosage"))
            or (
                ReviewTopic.PREGNANCY.value in topic_values
                and field == "specialPopulations"
            )
        )

    return sorted({field for field in missing_fields if relevant(field)})


def _reinvestigation_kind(finding: dict[str, Any] | None) -> str:
    if finding and finding.get("missingField"):
        return "context"
    if finding and finding.get("selectedProductIds"):
        return "evidence"
    return "mapping"


def _reinvestigation_ids(state: ReviewState, kind: str) -> list[str]:
    findings = {
        item.get("findingId"): item
        for item in state.get("findings", [])
        if item.get("findingId")
    }
    return [
        finding_id
        for finding_id in state.get("reinvestigateFindingIds") or []
        if _reinvestigation_kind(findings.get(finding_id)) == kind
    ]


def _next_reinvestigation_node(state: ReviewState, default: str) -> str:
    for kind in ("context", "evidence", "mapping"):
        if _reinvestigation_ids(state, kind):
            return kind
    return default


def bind_review_plan(
    intent: ReviewIntent,
    mappings: list[MedicationMapping],
    missing_fields: list[str],
) -> list[ReviewPlanItem]:
    medication_ids = [item.medicationId for item in mappings]
    items = [
        ReviewPlanItem(
            planItemId=f"topic-{topic.value}",
            reviewType=topic.value.upper(),
            medicationIds=medication_ids,
            topics=[topic.value],
            rationale=intent.rationale,
        )
        for topic in intent.topics
    ]
    items.extend(
        ReviewPlanItem(
            planItemId=f"missing-{_digest(field)[:16]}",
            reviewType="EVIDENCE_GAP",
            medicationIds=medication_ids,
            topics=[],
            rationale=f"Patient information is not recorded: {field}.",
            requiresHumanReview=True,
            missingField=field,
        )
        for field in missing_fields
    )
    return items


def make_finding(
    *,
    rule_id: str,
    normalization_version: str | None = None,
    comparison_inputs: dict[str, Any] | None = None,
    **values: Any,
) -> Finding:
    return Finding(
        ruleId=rule_id,
        normalizationVersion=normalization_version,
        comparisonInputs=comparison_inputs or {},
        **values,
    )


def _normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).casefold().split())
    return normalized or None


class _ToolInvocationError(RuntimeError):
    def __init__(self, cause: Exception, retries: int) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.retries = retries


def _is_retryable_tool_error(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, ConnectionResetError)):
        return True
    if getattr(error, "retryable", False) is True:
        return True
    response = getattr(error, "response", None)
    status_code = getattr(error, "status_code", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    return status_code == 429


async def _retry(operation, *, max_attempts: int = 3):
    retries = 0
    for attempt in range(max_attempts):
        try:
            return await operation(), retries
        except ToolContractError:
            raise
        except Exception as exc:
            if not _is_retryable_tool_error(exc) or attempt == max_attempts - 1:
                raise _ToolInvocationError(exc, retries) from exc
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
        *, audit_slot: str | None = None, max_attempts: int = 3,
    ):
        try:
            result, retries = await _retry(operation, max_attempts=max_attempts)
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
        except _ToolInvocationError as exc:
            dependencies.repository.append_audit(
                state["reviewId"], node=node, tool=tool, request_id=None,
                result_status="ERROR", argument_summary=summary,
                evidence_refs=[], latency_ms=0, retry_count=exc.retries,
                mutation_id=state.get("mutationId"),
                audit_slot=audit_slot or node,
            )
            return None, exc.retries, type(exc.cause).__name__

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
                strengthSource=item.get("strengthSource"),
                dosageForm=item.get("dosageForm"),
                dosageFormCodings=item.get("dosageFormCodings") or [],
                route=item.get("route"), routeCodings=item.get("routeCodings") or [],
                dosage=item.get("dosage"),
                medicationReference=item.get("medicationReference"),
                medicationEvidenceRefs=item.get("evidenceRefs") or [],
                authoredOn=item.get("authoredOn"),
                effectivePeriod=item.get("effectivePeriod"),
                patientEvidenceRefs=(
                    item.get("evidenceRefs")
                    or ([item["evidenceRef"]] if item.get("evidenceRef") else [])
                ),
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
        metrics = dict(state.get("metrics") or {})

        async def resolve_one(medication: MedicationRecord):
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
            return medication, result, retries, error

        medications = [
            MedicationRecord.model_validate(raw)
            for raw in state.get("medications", [])
        ]
        resolved = await _bounded_ordered_map(
            medications,
            lambda medication: medication.medicationId,
            resolve_one,
        )
        for _, result, retries, _ in resolved:
            metrics = add_metrics({"metrics": metrics}, result, retries)
        for medication, result, _, error in resolved:
            if error or result is None:
                return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}], "metrics": metrics}
            if result.envelope.status.value not in {"OK", "AMBIGUOUS", "UNMAPPED"}:
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{
                        "kind": "MAPPING_ERROR", "status": result.envelope.status.value,
                        "errors": result.envelope.errors,
                    }],
                    "metrics": metrics,
                }
            mappings.append(mapping_from_result(medication, result))
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

    def promote_mapping_findings(
        state: ReviewState,
        mappings: list[dict[str, Any]],
        finding_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        requested_ids = set(
            state.get("reinvestigateFindingIds") or []
            if finding_ids is None
            else finding_ids
        )
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
                "ruleId": "label-evidence-review-v1",
                "normalizationVersion": "normalization-v1",
                "comparisonInputs": {
                    "recordedRoute": _normalized_text(medication.get("route")),
                    "recordedDosageForm": _normalized_text(medication.get("dosageForm")),
                    "recordedDosage": _normalized_text(medication.get("dosage")),
                },
                "summary": f"Review retrieved label evidence for {mapping['sourceName']}.",
                "selectedProductIds": [mapping["selectedProductId"]],
                "patientEvidenceRefs": medication.get("patientEvidenceRefs", []),
                "labelEvidenceRefs": [], "labelEvidenceIds": [],
                "status": FindingStatus.PENDING.value,
                "verificationErrors": [],
            })
        return findings

    async def parse_review_goal(state: ReviewState) -> dict[str, Any]:
        context = state.get("contextSnapshot") or {}
        mappings = [
            MedicationMapping.model_validate(item)
            for item in state.get("medicationMappings", [])
        ]
        model_calls = list(state.get("modelCalls") or [])
        budget_exhausted = len(model_calls) >= 3
        planner = DeterministicPlanner() if budget_exhausted else dependencies.planner
        planning = await planner.plan(
            state.get("question") or "默认用药证据核查",
            deidentified_patient_features(context),
            mappings,
            state.get("contextMissingFields", []),
        )
        model_call = planning.modelCall
        dependencies.repository.append_audit(
            state["reviewId"],
            node="parse_review_goal",
            tool=None,
            request_id=None,
            result_status=(
                "MODEL_CALL_BUDGET_EXHAUSTED"
                if budget_exhausted
                else "MODEL_FALLBACK" if planning.modelFallback else "OK"
            ),
            argument_summary={
                "topicCount": len(planning.intent.topics),
                "medicationCount": len(mappings),
            },
            evidence_refs=[],
            latency_ms=model_call.latencyMs if model_call else 0,
            model_id=model_call.modelId if model_call else planning.intent.modelId,
            prompt_version=(
                model_call.promptVersion if model_call else planning.intent.promptVersion
            ),
            input_tokens=model_call.inputTokens if model_call else 0,
            output_tokens=model_call.outputTokens if model_call else 0,
            estimated_cost=model_call.estimatedCost if model_call else 0.0,
            model_fallback=planning.modelFallback,
            mutation_id=state.get("mutationId"),
            audit_slot="parse_review_goal",
        )
        metrics = dict(state.get("metrics") or {})
        if model_call:
            metrics["inputTokens"] = metrics.get("inputTokens", 0) + model_call.inputTokens
            metrics["outputTokens"] = metrics.get("outputTokens", 0) + model_call.outputTokens
            metrics["estimatedCost"] = (
                metrics.get("estimatedCost", 0.0) + model_call.estimatedCost
            )
        return {
            "intent": planning.intent.model_dump(mode="json"),
            "reviewPlan": [item.model_dump(mode="json") for item in planning.items],
            "modelCalls": [
                *model_calls,
                *([model_call.model_dump(mode="json")] if model_call else []),
            ],
            "metrics": metrics,
            **({
                "unresolvedItems": [
                    *(state.get("unresolvedItems") or []),
                    {
                        "kind": "MODEL_BUDGET",
                        "unresolvedReason": "MODEL_CALL_BUDGET_EXHAUSTED",
                        "summary": "The review continued with deterministic planning.",
                    },
                ],
            } if budget_exhausted else {}),
        }

    def plan_review(state: ReviewState) -> dict[str, Any]:
        intent = ReviewIntent.model_validate(state["intent"])
        mappings = [
            MedicationMapping.model_validate(item)
            for item in state.get("medicationMappings", [])
        ]
        missing_fields = relevant_missing_fields(
            intent.topics,
            state.get("contextMissingFields", []),
        )
        return {
            "reviewPlan": [
                item.model_dump(mode="json")
                for item in bind_review_plan(intent, mappings, missing_fields)
            ],
        }

    async def retrieve_evidence(state: ReviewState) -> dict[str, Any]:
        selected_for_comparison = sorted([
            item["selectedProductId"]
            for item in state.get("medicationMappings", [])
            if item.get("selectedProductId")
        ])
        selected = sorted(set(selected_for_comparison))
        medication_ids_by_product = {
            product_id: sorted({
                item["medicationId"]
                for item in state.get("medicationMappings", [])
                if item.get("selectedProductId") == product_id
            })
            for product_id in selected
        }
        changed_medication_ids = set(
            state.get("contextChangedMedicationIds") or []
        )
        evidence = (
            _evidence_for_unchanged_findings(
                [dict(item) for item in state.get("findings", [])],
                [dict(item) for item in state.get("evidenceIndex", [])],
                changed_medication_ids,
            )
            if state.get("contextMedicationsChanged")
            else []
        )
        unresolved_items = [
            dict(item) for item in state.get("unresolvedItems", [])
            if item.get("kind") != "EVIDENCE_GAP"
        ]
        metrics = dict(state.get("metrics") or {})
        retrieval_attempts = dict(state.get("retrievalAttempts") or {})
        model_calls = list(state.get("modelCalls") or [])
        document_versions_by_product: dict[str, dict[str, str]] = {}

        async def retrieve_facts(product_id: str):
            result, retries, error = await audited_call(
                state,
                "retrieve_evidence",
                "get_product_facts",
                {"productIdHash": _digest(product_id)},
                lambda: dependencies.drug.get_product_facts(product_id),
                audit_slot=f"retrieve_facts:{_digest(product_id)}",
            )
            return product_id, result, retries, error

        facts_results = await _bounded_ordered_map(
            selected,
            lambda product_id: product_id,
            retrieve_facts,
        )
        for _, result, retries, _ in facts_results:
            metrics = add_metrics({"metrics": metrics}, result, retries)
        for product_id, result, _, error in facts_results:
            if error:
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            if result is not None and result.envelope.status.value == "INSUFFICIENT_EVIDENCE":
                unresolved_items.append({
                    "kind": "EVIDENCE_GAP", "sourceTool": "get_product_facts",
                    "summary": f"Product facts were unavailable for {product_id}.",
                    "productIds": [product_id], "evidenceRefs": result.envelope.evidenceRefs,
                    "graphProvenance": _graph_provenance(result), "errors": result.envelope.errors,
                })
                continue
            if result is None or result.envelope.status.value != "OK":
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{
                        "kind": "DRUG_EVIDENCE_ERROR",
                        "errors": result.envelope.errors if result else ["missing response"],
                    }],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            if not _facts_match_product(result, product_id):
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{
                        "kind": "DRUG_EVIDENCE_SCOPE_ERROR",
                        "productIds": [product_id],
                    }],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            document_versions_by_product[product_id] = _document_versions(result)
            evidence.append(_fact_evidence(product_id, result).model_dump(mode="json"))
        allowed_topics = {topic.value for topic in ReviewTopic}
        topics = tuple(
            ReviewTopic(topic)
            for topic in sorted({
                topic
                for item in state.get("reviewPlan", [])
                for topic in item.get("topics", [])
                if topic in allowed_topics
            })
        )
        for product_id in selected:
            if not topics:
                continue
            document_versions = document_versions_by_product.get(product_id, {})
            if not document_versions:
                unresolved_items.append({
                    "kind": "EVIDENCE_GAP", "sourceTool": "search_label_evidence",
                    "summary": f"No confirmed SPL document scope was available for {product_id}.",
                    "productIds": [product_id], "evidenceRefs": [],
                    "unresolvedReason": "MISSING_DOCUMENT_SCOPE",
                    "errors": [],
                })
                continue
            attempt_keys = [retrieval_attempt_key(product_id, topic) for topic in topics]
            prior_attempts = max(
                (retrieval_attempts.get(key, 0) for key in attempt_keys),
                default=0,
            )
            captured_calls: list[tuple[TimedToolResult | None, int]] = []
            captured_error: str | None = None
            search_number = 0

            class AuditedSearchGateway:
                async def search_label_evidence(
                    self,
                    product_ids: list[str],
                    topic_values: list[str],
                    question: str | None,
                ) -> TimedToolResult:
                    nonlocal captured_error, search_number
                    search_number += 1
                    result, retries, error = await audited_call(
                        state,
                        "retrieve_evidence",
                        "search_label_evidence",
                        {"productCount": 1, "topicCount": len(topic_values)},
                        lambda: dependencies.drug.search_label_evidence(
                            product_ids, topic_values, question
                        ),
                        audit_slot=(
                            f"retrieve_search:{_digest(product_id)}:{search_number}"
                        ),
                        max_attempts=1,
                    )
                    captured_calls.append((result, retries))
                    if error or result is None:
                        captured_error = error or "missing response"
                        raise RuntimeError(captured_error)
                    return result

            has_model_budget = len(model_calls) < 3
            retriever = BoundedEvidenceRetriever(
                AuditedSearchGateway(),
                dependencies.grader if has_model_budget else None,
            )
            try:
                outcome = await retriever.retrieve(ScopedRetrievalRequest(
                    productId=product_id,
                    documentIds=frozenset(document_versions),
                    documentVersions=document_versions,
                    topics=topics,
                    question=state.get("question") or "默认用药证据核查",
                    priorAttempts=prior_attempts,
                ))
            except RuntimeError:
                if captured_error is None:
                    raise
                outcome = None
            for tool_result, retries in captured_calls:
                metrics = add_metrics({"metrics": metrics}, tool_result, retries)
            if outcome is None:
                for key in attempt_keys:
                    retrieval_attempts[key] = min(
                        2, retrieval_attempts.get(key, 0) + len(captured_calls)
                    )
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{
                        "kind": "TOOL_ERROR", "error": captured_error,
                    }],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            for key in attempt_keys:
                retrieval_attempts[key] = min(
                    2, retrieval_attempts.get(key, 0) + outcome.attempts
                )
            if outcome.modelCall is not None and len(model_calls) < 3:
                model_call = outcome.modelCall
                model_calls.append(model_call.model_dump(mode="json"))
                metrics["inputTokens"] = metrics.get("inputTokens", 0) + model_call.inputTokens
                metrics["outputTokens"] = metrics.get("outputTokens", 0) + model_call.outputTokens
                metrics["estimatedCost"] = (
                    metrics.get("estimatedCost", 0.0) + model_call.estimatedCost
                )
                dependencies.repository.append_audit(
                    state["reviewId"],
                    node="retrieve_evidence",
                    tool=None,
                    request_id=None,
                    result_status=model_call.failureCode or "OK",
                    argument_summary={
                        "topicCount": len(topics),
                        "evidenceCount": len(outcome.results),
                    },
                    evidence_refs=[],
                    latency_ms=model_call.latencyMs,
                    model_id=model_call.modelId,
                    prompt_version=model_call.promptVersion,
                    input_tokens=model_call.inputTokens,
                    output_tokens=model_call.outputTokens,
                    estimated_cost=model_call.estimatedCost,
                    model_fallback=model_call.failureCode is not None,
                    mutation_id=state.get("mutationId"),
                    audit_slot=f"retrieve_grader:{_digest(product_id)}",
                )
            for item in outcome.results:
                evidence.append(item.model_copy(update={
                    "medicationIds": medication_ids_by_product[product_id],
                }).model_dump(mode="json"))
            unresolved_reason = outcome.unresolvedReason
            if (
                unresolved_reason == "SEMANTIC_GRADING_UNAVAILABLE"
                and dependencies.grader is not None
                and not has_model_budget
            ):
                unresolved_reason = "MODEL_CALL_BUDGET_EXHAUSTED"
            if unresolved_reason in {
                "DRUG_EVIDENCE_ERROR",
                "EVIDENCE_CONTRACT_ERROR",
            }:
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{
                        "kind": unresolved_reason,
                        "errors": [unresolved_reason],
                    }],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            if unresolved_reason:
                unresolved_items.append({
                    "kind": "EVIDENCE_GAP",
                    "sourceTool": "search_label_evidence",
                    "summary": f"Label evidence remains unresolved for {product_id}.",
                    "productIds": [product_id],
                    "evidenceRefs": [item.evidenceRef for item in outcome.results],
                    "unresolvedReason": unresolved_reason,
                    "errors": [],
                })
        if len(selected_for_comparison) > 1:
            result, retries, error = await audited_call(state, "retrieve_evidence", "compare_product_ingredients", {"productCount": len(selected_for_comparison)}, lambda: dependencies.drug.compare_product_ingredients(selected_for_comparison))
            metrics = add_metrics({"metrics": metrics}, result, retries)
            if error:
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            if result is not None and result.envelope.status.value == "INSUFFICIENT_EVIDENCE":
                unresolved_items.append({
                    "kind": "EVIDENCE_GAP", "sourceTool": "compare_product_ingredients",
                    "summary": "Ingredient comparison evidence was insufficient for the selected products.",
                    "productIds": selected_for_comparison, "evidenceRefs": result.envelope.evidenceRefs,
                    "graphProvenance": _graph_provenance(result), "errors": result.envelope.errors,
                })
                return {
                    "evidenceIndex": _upsert_evidence(evidence),
                    "unresolvedItems": unresolved_items,
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            if result is None or result.envelope.status.value != "OK":
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{
                        "kind": "DRUG_EVIDENCE_ERROR",
                        "errors": result.envelope.errors if result else ["missing response"],
                    }],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            shared = result.envelope.data.get("sharedActiveIngredients", []) if result else []
            if shared:
                evidence.append(_comparison_evidence(
                    selected_for_comparison,
                    shared,
                    result,
                    fallback_ref=next((
                        item["evidenceRef"] for item in evidence
                        if item.get("evidenceRef", "").startswith("SPL:")
                    ), "SPL-GRAPH:ingredient-comparison"),
                ).model_dump(mode="json"))
        return {
            "evidenceIndex": _upsert_evidence(evidence),
            "unresolvedItems": unresolved_items,
            "metrics": metrics,
            "retrievalAttempts": retrieval_attempts,
            "modelCalls": model_calls,
        }

    def build_findings(state: ReviewState) -> dict[str, Any]:
        medications = {item["medicationId"]: MedicationRecord.model_validate(item) for item in state.get("medications", [])}
        planned_missing_fields = [
            item["missingField"]
            for item in state.get("reviewPlan", [])
            if item.get("reviewType") == "EVIDENCE_GAP" and item.get("missingField")
        ]
        findings = [make_finding(
            rule_id="missing-patient-field-v1",
            comparison_inputs={"missingField": missing_field},
            findingId=str(uuid4()), reviewType="EVIDENCE_GAP",
            summary=f"Patient information is not recorded: {missing_field}.",
            attentionLevel="HIGH", confidence=1.0, requiresHumanReview=True,
            missingField=missing_field,
        ).model_dump(mode="json") for missing_field in planned_missing_fields]
        for gap in [item for item in state.get("unresolvedItems", []) if item.get("kind") == "EVIDENCE_GAP"]:
            review_type = (
                "LABEL_EVIDENCE_MISSING"
                if gap.get("sourceTool") == "search_label_evidence"
                else "EVIDENCE_GAP"
            )
            findings.append(make_finding(
                rule_id=(
                    "label-evidence-missing-v1"
                    if review_type == "LABEL_EVIDENCE_MISSING"
                    else "drug-evidence-gap-v1"
                ),
                comparison_inputs={
                    "sourceTool": gap.get("sourceTool"),
                    "productIds": sorted(gap.get("productIds", [])),
                },
                findingId=str(uuid4()), reviewType=review_type,
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
            findings.append(make_finding(
                rule_id="graph-provenance-review-v1",
                comparison_inputs={
                    "graphBackend": provenance.graphBackend,
                    "fallbackUsed": provenance.fallbackUsed,
                    "consistencyStatus": consistency,
                },
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
            label_refs, label_ids = _label_evidence_bindings(
                relevant_evidence,
                {mapping.selectedProductId} if mapping.selectedProductId else set(),
            )
            evidence_provenances = [GraphEvidenceProvenance.model_validate(item["graphProvenance"]) for item in relevant_evidence if item.get("graphProvenance")]
            warnings = list(dict.fromkeys([
                *_graph_warnings(mapping.graphProvenance),
                *(warning for provenance in evidence_provenances for warning in _graph_warnings(provenance)),
            ]))
            finding_provenance = next((item for item in evidence_provenances if _graph_warnings(item)), mapping.graphProvenance)
            if not mapping.selectedProductId:
                findings.append(make_finding(
                    rule_id="unmapped-medication-v1",
                    comparison_inputs={"medicationId": mapping.medicationId},
                    findingId=str(uuid4()), reviewType="PRODUCT_UNMAPPED",
                    summary=f"No DailyMed product was mapped for {mapping.sourceName}.",
                    attentionLevel="HIGH", confidence=1.0, medicationIds=[mapping.medicationId],
                    patientEvidenceRefs=medication.patientEvidenceRefs,
                    requiresHumanReview=True,
                ).model_dump(mode="json"))
                continue
            findings.append(make_finding(
                rule_id="label-evidence-review-v1",
                normalization_version="normalization-v1",
                comparison_inputs={
                    "recordedRoute": _normalized_text(medication.route),
                    "recordedDosageForm": _normalized_text(medication.dosageForm),
                    "recordedDosage": _normalized_text(medication.dosage),
                    "allergyTerms": deidentified_patient_features(
                        state.get("contextSnapshot") or {}
                    )["allergyTerms"],
                },
                findingId=str(uuid4()), reviewType="LABEL_EVIDENCE_REVIEW",
                summary=f"Review retrieved label evidence for {mapping.sourceName}.",
                attentionLevel="HIGH" if warnings else "MEDIUM", confidence=0.8,
                medicationIds=[mapping.medicationId], selectedProductIds=[mapping.selectedProductId],
                patientEvidenceRefs=medication.patientEvidenceRefs,
                labelEvidenceRefs=label_refs, labelEvidenceIds=label_ids,
                requiresHumanReview=True, verificationWarnings=warnings,
                graphProvenance=finding_provenance,
            ).model_dump(mode="json"))
            planned_topics = {
                str(topic)
                for item in state.get("reviewPlan", [])
                for topic in item.get("topics", [])
            }
            if "ingredients" in planned_topics:
                ingredient_label_refs, ingredient_label_ids = (
                    _label_evidence_bindings(
                        relevant_evidence,
                        {mapping.selectedProductId},
                        topics=frozenset({"ingredients"}),
                    )
                )
                allergies = [
                    item for item in (state.get("contextSnapshot") or {}).get("allergies", [])
                    if isinstance(item, dict)
                    and _normalized_text(item.get("substance") or item.get("name"))
                ]
                ingredients = [
                    ingredient
                    for evidence_item in relevant_evidence
                    if evidence_item.get("topic") == "product_facts"
                    for ingredient in [
                        *(evidence_item.get("activeIngredients") or []),
                        *(evidence_item.get("inactiveIngredients") or []),
                    ]
                    if isinstance(ingredient, dict)
                    and _normalized_text(ingredient.get("name"))
                ]
                observed_matches: set[tuple[str, str]] = set()
                for allergy in allergies:
                    allergy_name = _normalized_text(
                        allergy.get("substance") or allergy.get("name")
                    )
                    for ingredient in ingredients:
                        ingredient_name = _normalized_text(ingredient.get("name"))
                        ingredient_id = str(ingredient.get("entityId") or "")
                        identity = (str(allergy.get("evidenceRef") or ""), ingredient_id)
                        if (
                            allergy_name != ingredient_name
                            or identity in observed_matches
                        ):
                            continue
                        observed_matches.add(identity)
                        findings.append(make_finding(
                            rule_id="ingredient-allergy-name-match-v1",
                            normalization_version="normalization-v1",
                            comparison_inputs={
                                "allergyName": allergy_name,
                                "ingredientName": ingredient_name,
                                "ingredientId": ingredient_id,
                            },
                            findingId=str(uuid4()),
                            reviewType="INGREDIENT_ALLERGY_NAME_MATCH",
                            summary=(
                                "Allergy name and product ingredient normalize to "
                                f"the same term: {allergy_name}."
                            ),
                            attentionLevel="HIGH",
                            confidence=1.0,
                            medicationIds=[mapping.medicationId],
                            selectedProductIds=[mapping.selectedProductId],
                            patientEvidenceRefs=sorted({
                                *medication.patientEvidenceRefs,
                                str(allergy.get("evidenceRef") or ""),
                            } - {""}),
                            labelEvidenceRefs=ingredient_label_refs,
                            labelEvidenceIds=ingredient_label_ids,
                            requiresHumanReview=True,
                            graphProvenance=finding_provenance,
                        ).model_dump(mode="json"))
            if warnings:
                findings.append(make_finding(
                    rule_id="graph-provenance-review-v1",
                    comparison_inputs={"medicationId": mapping.medicationId},
                    findingId=str(uuid4()), reviewType="EVIDENCE_GAP",
                    summary="Graph provenance requires pharmacist review.", attentionLevel="HIGH",
                    confidence=1.0, medicationIds=[mapping.medicationId],
                    patientEvidenceRefs=medication.patientEvidenceRefs,
                    requiresHumanReview=True, verificationWarnings=warnings,
                    graphProvenance=mapping.graphProvenance,
                ).model_dump(mode="json"))
        for comparison in [item for item in state.get("evidenceIndex", []) if item.get("source") == "SPL-GRAPH" and item.get("topic") == "shared_active_ingredients"]:
            provenance = comparison.get("graphProvenance")
            shared_ingredients = comparison.get("sharedActiveIngredients", [])
            label_refs, label_ids = _label_evidence_bindings(
                state.get("evidenceIndex", []),
                set(comparison.get("productIds", [])),
                topics=frozenset({"ingredients"}),
            )
            findings.append(make_finding(
                rule_id="shared-active-ingredient-v1",
                normalization_version="normalization-v1",
                comparison_inputs={
                    "productIds": sorted(comparison.get("productIds", [])),
                    "sharedActiveIngredientIds": sorted(
                        str(item.get("entityId"))
                        for item in shared_ingredients
                        if isinstance(item, dict) and item.get("entityId")
                    ),
                },
                findingId=str(uuid4()), reviewType="DUPLICATE_ACTIVE_INGREDIENT",
                summary=f"Mapped products share active ingredient graph entities: {comparison.get('summary')}.",
                attentionLevel="HIGH", confidence=1.0,
                medicationIds=list(medications),
                selectedProductIds=comparison.get("productIds", []),
                patientEvidenceRefs=[ref for item in medications.values() for ref in item.patientEvidenceRefs],
                labelEvidenceRefs=label_refs,
                labelEvidenceIds=label_ids,
                requiresHumanReview=True, graphProvenance=provenance,
                sharedActiveIngredients=shared_ingredients,
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
        reinvestigate_ids = list(state.get("reinvestigateFindingIds") or [])
        if state.get("contextRebuildPending"):
            satisfied_ids = set(_reinvestigation_ids(state, "evidence"))
            previous_findings = {
                item.get("findingId"): item for item in state.get("findings", [])
            }
            for finding in findings:
                if finding.get("findingId") not in satisfied_ids:
                    continue
                previous_ids = set(
                    previous_findings.get(finding["findingId"], {}).get(
                        "labelEvidenceIds", []
                    )
                )
                product_ids = set(finding.get("selectedProductIds", []))
                sources = (
                    frozenset({"SPL-GRAPH"})
                    if finding.get("reviewType") == "DUPLICATE_ACTIVE_INGREDIENT"
                    else frozenset({"SPL"})
                )
                all_evidence = list(state.get("evidenceIndex", []))
                refreshed_evidence = [
                    item for item in all_evidence
                    if item.get("evidenceId") not in previous_ids
                    and product_ids.intersection(item.get("productIds", []))
                ]
                label_refs, label_ids = _label_evidence_bindings(
                    refreshed_evidence,
                    product_ids,
                    sources=sources,
                )
                if not label_refs and not label_ids:
                    label_refs, label_ids = _label_evidence_bindings(
                        all_evidence, product_ids, sources=sources,
                    )
                finding["labelEvidenceRefs"] = label_refs
                finding["labelEvidenceIds"] = label_ids
                finding["status"] = FindingStatus.PENDING.value
                finding["verificationErrors"] = []
            reinvestigate_ids = [
                finding_id for finding_id in reinvestigate_ids
                if finding_id not in satisfied_ids
            ]
        return {
            "findings": findings,
            "reinvestigateFindingIds": reinvestigate_ids,
            "contextMedicationsChanged": False,
            "contextChangedMedicationIds": [],
            "contextRebuildPending": False,
        }

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
        all_requested_ids = list(state.get("reinvestigateFindingIds") or [])
        requested_ids = set(_reinvestigation_ids(state, "context"))
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
        intent = ReviewIntent.model_validate(state["intent"])
        mappings = [
            MedicationMapping.model_validate(item)
            for item in state.get("medicationMappings", [])
        ]
        relevant_missing = relevant_missing_fields(intent.topics, list(missing))
        retained_evidence = _evidence_for_unchanged_findings(
            findings,
            [dict(item) for item in state.get("evidenceIndex", [])],
            set(changed_ids),
        )
        return {
            "contextSnapshot": context,
            "contextMissingFields": list(missing),
            "medications": medications,
            "reviewPlan": [
                item.model_dump(mode="json")
                for item in bind_review_plan(intent, mappings, relevant_missing)
            ],
            "findings": findings,
            "reinvestigateFindingIds": [
                finding_id for finding_id in all_requested_ids
                if finding_id not in requested_ids
            ],
            "metrics": metrics,
            "contextMedicationsChanged": medications_changed,
            "contextChangedMedicationIds": changed_ids,
            "contextRebuildPending": medications_changed,
            **({
                "medicationMappings": [],
                "evidenceIndex": retained_evidence,
            } if medications_changed else {}),
        }

    async def reinvestigate_mapping(state: ReviewState) -> dict[str, Any]:
        all_requested_ids = list(state.get("reinvestigateFindingIds") or [])
        requested_ids = set(_reinvestigation_ids(state, "mapping"))
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
        findings = promote_mapping_findings(state, mappings, list(requested_ids))
        findings_by_id = {item.get("findingId"): item for item in findings}
        confirmation_medication_ids = {
            item.get("medicationId") for item in mappings
            if item.get("mappingConfirmationRequired")
        }
        forwarded_ids = {
            finding_id for finding_id in requested_ids
            if findings_by_id.get(finding_id, {}).get("selectedProductIds")
            or confirmation_medication_ids.intersection(
                findings_by_id.get(finding_id, {}).get("medicationIds", [])
            )
        }
        return {
            "medicationMappings": mappings, "findings": findings, "metrics": metrics,
            "status": ReviewStatus.AWAITING_MAPPING_CONFIRMATION.value if confirmation_required else ReviewStatus.RUNNING.value,
            "reinvestigateFindingIds": [
                finding_id for finding_id in all_requested_ids
                if finding_id not in requested_ids or finding_id in forwarded_ids
            ],
        }

    async def reinvestigate_evidence(state: ReviewState) -> dict[str, Any]:
        all_requested_ids = list(state.get("reinvestigateFindingIds") or [])
        requested_ids = set(_reinvestigation_ids(state, "evidence"))
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
        allowed_topics = {topic.value for topic in ReviewTopic}
        topics = tuple(
            ReviewTopic(topic)
            for topic in sorted({
                topic
                for item in state.get("reviewPlan", [])
                if target_medication_ids.intersection(item.get("medicationIds", []))
                for topic in item.get("topics", [])
                if topic in allowed_topics
            })
        )
        refreshed_finding_ids = {
            *requested_ids,
            *canonical_ids,
        }
        protected_findings = [
            item for item in findings
            if item.get("findingId") not in refreshed_finding_ids
        ]
        protected_evidence_ids = {
            evidence_id
            for item in protected_findings
            for evidence_id in item.get("labelEvidenceIds", [])
        }
        protected_legacy_refs = {
            reference
            for item in protected_findings
            if not item.get("labelEvidenceIds")
            for reference in item.get("labelEvidenceRefs", [])
        }
        retained_evidence = [
            dict(item) for item in state.get("evidenceIndex", [])
            if (
                not target_products.intersection(item.get("productIds", []))
                or item.get("evidenceId") in protected_evidence_ids
                or item.get("evidenceRef") in protected_legacy_refs
            )
        ]
        refreshed_evidence: list[dict[str, Any]] = []
        refreshed_gaps: list[dict[str, Any]] = []
        current_document_identities: dict[
            str, frozenset[tuple[str, str, str]]
        ] = {}
        comparison_provenance: dict[str, GraphEvidenceProvenance | None] = {}
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
        retrieval_attempts = dict(state.get("retrievalAttempts") or {})
        model_calls = list(state.get("modelCalls") or [])
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
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{"kind": "DRUG_EVIDENCE_ERROR", "errors": errors}],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            if facts.envelope.status.value == "INSUFFICIENT_EVIDENCE":
                refreshed_gaps.append({
                    "sourceTool": "get_product_facts", "productIds": [product_id],
                    "summary": f"Product facts were unavailable for {product_id}.",
                    "graphProvenance": _graph_provenance(facts),
                    "unresolvedReason": "INSUFFICIENT_EVIDENCE",
                    "errors": facts.envelope.errors,
                })
                continue
            elif facts.envelope.status.value == "OK":
                if not _facts_match_product(facts, product_id):
                    return {
                        "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                        "unresolvedItems": [{
                            "kind": "DRUG_EVIDENCE_SCOPE_ERROR",
                            "productIds": [product_id],
                        }],
                        "metrics": metrics,
                        "retrievalAttempts": retrieval_attempts,
                        "modelCalls": model_calls,
                    }
                refreshed_evidence.append(
                    _fact_evidence(product_id, facts).model_dump(mode="json")
                )
                current_document_identities[product_id] = _document_identities(facts)
            else:
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{
                        "kind": "DRUG_EVIDENCE_ERROR",
                        "errors": facts.envelope.errors,
                    }],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            if not topics:
                continue
            document_versions = _document_versions(facts)
            if not document_versions:
                refreshed_gaps.append({
                    "sourceTool": "search_label_evidence", "productIds": [product_id],
                    "summary": f"No confirmed SPL document scope was available for {product_id}.",
                    "graphProvenance": _graph_provenance(facts),
                    "unresolvedReason": "MISSING_DOCUMENT_SCOPE",
                    "errors": [],
                })
                continue
            attempt_keys = [retrieval_attempt_key(product_id, topic) for topic in topics]
            prior_attempts = max(
                (retrieval_attempts.get(key, 0) for key in attempt_keys),
                default=0,
            )
            captured_calls: list[tuple[TimedToolResult | None, int]] = []
            captured_error: str | None = None
            search_number = 0

            class AuditedReinvestigationGateway:
                async def search_label_evidence(
                    self,
                    product_ids: list[str],
                    topic_values: list[str],
                    question: str | None,
                ) -> TimedToolResult:
                    nonlocal captured_error, search_number
                    search_number += 1
                    result, retries, error = await audited_call(
                        state,
                        "reinvestigate_evidence",
                        "search_label_evidence",
                        {"productCount": 1, "topicCount": len(topic_values)},
                        lambda: dependencies.drug.search_label_evidence(
                            product_ids, topic_values, question
                        ),
                        audit_slot=(
                            f"reinvestigate_search:{_digest(product_id)}:{search_number}"
                        ),
                        max_attempts=1,
                    )
                    captured_calls.append((result, retries))
                    if error or result is None:
                        captured_error = error or "missing response"
                        raise RuntimeError(captured_error)
                    return result

            has_model_budget = len(model_calls) < 3
            retriever = BoundedEvidenceRetriever(
                AuditedReinvestigationGateway(),
                dependencies.grader if has_model_budget else None,
            )
            try:
                outcome = await retriever.retrieve(ScopedRetrievalRequest(
                    productId=product_id,
                    documentIds=frozenset(document_versions),
                    documentVersions=document_versions,
                    topics=topics,
                    question=state.get("question") or "默认用药证据核查",
                    priorAttempts=prior_attempts,
                ))
            except RuntimeError:
                if captured_error is None:
                    raise
                outcome = None
            for search_result, retries in captured_calls:
                metrics = add_metrics({"metrics": metrics}, search_result, retries)
            if outcome is None:
                for key in attempt_keys:
                    retrieval_attempts[key] = min(
                        2, retrieval_attempts.get(key, 0) + len(captured_calls)
                    )
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{
                        "kind": "TOOL_ERROR", "error": captured_error,
                    }],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            for key in attempt_keys:
                retrieval_attempts[key] = min(
                    2, retrieval_attempts.get(key, 0) + outcome.attempts
                )
            if outcome.modelCall is not None and len(model_calls) < 3:
                model_call = outcome.modelCall
                model_calls.append(model_call.model_dump(mode="json"))
                metrics["inputTokens"] = metrics.get("inputTokens", 0) + model_call.inputTokens
                metrics["outputTokens"] = metrics.get("outputTokens", 0) + model_call.outputTokens
                metrics["estimatedCost"] = (
                    metrics.get("estimatedCost", 0.0) + model_call.estimatedCost
                )
                dependencies.repository.append_audit(
                    state["reviewId"],
                    node="reinvestigate_evidence",
                    tool=None,
                    request_id=None,
                    result_status=model_call.failureCode or "OK",
                    argument_summary={
                        "topicCount": len(topics),
                        "evidenceCount": len(outcome.results),
                    },
                    evidence_refs=[],
                    latency_ms=model_call.latencyMs,
                    model_id=model_call.modelId,
                    prompt_version=model_call.promptVersion,
                    input_tokens=model_call.inputTokens,
                    output_tokens=model_call.outputTokens,
                    estimated_cost=model_call.estimatedCost,
                    model_fallback=model_call.failureCode is not None,
                    mutation_id=state.get("mutationId"),
                    audit_slot=f"reinvestigate_grader:{_digest(product_id)}",
                )
            for evidence_item in outcome.results:
                refreshed_evidence.append(evidence_item.model_copy(update={
                    "medicationIds": sorted(target_medication_ids),
                }).model_dump(mode="json"))
            unresolved_reason = outcome.unresolvedReason
            if (
                unresolved_reason == "SEMANTIC_GRADING_UNAVAILABLE"
                and dependencies.grader is not None
                and not has_model_budget
            ):
                unresolved_reason = "MODEL_CALL_BUDGET_EXHAUSTED"
            if unresolved_reason in {
                "DRUG_EVIDENCE_ERROR",
                "EVIDENCE_CONTRACT_ERROR",
            }:
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{
                        "kind": unresolved_reason,
                        "errors": [unresolved_reason],
                    }],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            if unresolved_reason:
                refreshed_gaps.append({
                    "sourceTool": "search_label_evidence",
                    "productIds": [product_id],
                    "summary": f"Label evidence remains unresolved for {product_id}.",
                    "graphProvenance": None,
                    "unresolvedReason": unresolved_reason,
                    "errors": [],
                })
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
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
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
                return {
                    "status": ReviewStatus.BLOCKED_TOOL_ERROR.value,
                    "unresolvedItems": [{
                        "kind": "DRUG_EVIDENCE_ERROR",
                        "errors": comparison.envelope.errors,
                    }],
                    "metrics": metrics,
                    "retrievalAttempts": retrieval_attempts,
                    "modelCalls": model_calls,
                }
            shared = comparison.envelope.data.get("sharedActiveIngredients", [])
            target_id = target.get("provenanceForFindingId") or target["findingId"]
            current = next((item for item in findings if item.get("findingId") == target_id), None)
            if current is None:
                current = dict(target)
                current["findingId"] = target_id
                findings.append(current)
            current["reviewType"] = "DUPLICATE_ACTIVE_INGREDIENT"
            current["sourceTool"] = "compare_product_ingredients"
            current["ruleId"] = "shared-active-ingredient-v1"
            current["normalizationVersion"] = "normalization-v1"
            current["comparisonInputs"] = {
                "productIds": sorted(product_ids),
                "sharedActiveIngredientIds": sorted(
                    str(item.get("entityId"))
                    for item in shared
                    if isinstance(item, dict) and item.get("entityId")
                ),
            }
            current["selectedProductIds"] = sorted(set(product_ids))
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
            if not shared:
                current["status"] = FindingStatus.REJECTED.value
                current["summary"] = "Resolved: refreshed comparison found no shared active ingredient."
                current["resolution"] = "RESOLVED_BY_INGREDIENT_REFRESH"
                resolved_comparison_ids.add(target_id)
            else:
                current["summary"] = f"Mapped products share active ingredient graph entities: {shared}."
                current["status"] = FindingStatus.PENDING.value
                current["verificationErrors"] = []
                comparison_evidence = _comparison_evidence(
                    product_ids,
                    shared,
                    comparison,
                    fallback_ref=next((
                        evidence["evidenceRef"] for evidence in refreshed_evidence
                        if evidence.get("source") == "SPL"
                        and set(product_ids).intersection(evidence.get("productIds", []))
                        and str(evidence.get("evidenceRef", "")).startswith("SPL:")
                    ), "SPL-GRAPH:ingredient-comparison"),
                ).model_dump(mode="json")
                refreshed_evidence.append(comparison_evidence)
                current["graphProvenance"] = _graph_provenance(comparison)
        evidence_index = _upsert_evidence([
            *retained_evidence,
            *refreshed_evidence,
        ])
        provenance_gaps: list[dict[str, Any]] = []
        for item in findings:
            if item.get("findingId") not in requested_ids:
                continue
            product_ids = set(item.get("selectedProductIds", []))
            binding_source = refreshed_evidence
            label_refs, label_ids = _label_evidence_bindings(
                binding_source,
                product_ids,
                topics=(
                    frozenset({"ingredients"})
                    if item.get("reviewType") == "DUPLICATE_ACTIVE_INGREDIENT"
                    else None
                ),
            )
            if (
                item.get("reviewType") == "DUPLICATE_ACTIVE_INGREDIENT"
                and not label_ids
            ):
                label_refs, label_ids = _label_evidence_bindings(
                    evidence_index,
                    product_ids,
                    topics=frozenset({"ingredients"}),
                    document_identities=current_document_identities,
                )
            item["labelEvidenceRefs"] = label_refs
            item["labelEvidenceIds"] = label_ids
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
                provenance_gaps.append(make_finding(
                    rule_id="graph-provenance-review-v1",
                    comparison_inputs={"provenanceForFindingId": item["findingId"]},
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
                provenance_gaps.append(make_finding(
                    rule_id="graph-provenance-review-v1",
                    comparison_inputs={"provenanceForFindingId": finding_id},
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
                findings.append(make_finding(
                    rule_id="drug-evidence-gap-v1",
                    comparison_inputs={
                        "sourceTool": gap["sourceTool"],
                        "productIds": sorted(gap["productIds"]),
                    },
                    findingId=gap_id, reviewType="EVIDENCE_GAP", summary=gap["summary"],
                    attentionLevel="HIGH", confidence=1.0,
                    medicationIds=target.get("medicationIds", []),
                    selectedProductIds=gap["productIds"],
                    patientEvidenceRefs=target.get("patientEvidenceRefs", []),
                    graphProvenance=gap.get("graphProvenance"), requiresHumanReview=True,
                    sourceTool=gap["sourceTool"], sourceErrors=gap.get("errors", []),
                    unresolvedReason=gap.get("unresolvedReason"),
                    provenanceForFindingId=target["findingId"],
                ).model_dump(mode="json"))
        return {
            "findings": findings, "evidenceIndex": evidence_index,
            "reinvestigateFindingIds": [
                finding_id for finding_id in all_requested_ids
                if finding_id not in requested_ids
            ],
            "metrics": metrics,
            "retrievalAttempts": retrieval_attempts,
            "modelCalls": model_calls,
        }

    async def verify_findings_node(state: ReviewState) -> dict[str, Any]:
        normalized_findings = [
            migrate_finding_payload(item)
            for item in state.get("findings", [])
        ]
        claims = [{
            "claimId": item["findingId"],
            "reviewType": (
                "EVIDENCE_GAP"
                if item["reviewType"] in UNRESOLVED_FINDING_TYPES
                else item["reviewType"]
            ),
            "ruleId": item["ruleId"],
            "normalizationVersion": item.get("normalizationVersion"),
            "comparisonInputs": item.get("comparisonInputs", {}),
            "patientEvidenceRefs": item.get("patientEvidenceRefs", []),
            "labelEvidenceRefs": item.get("labelEvidenceRefs", []),
            "selectedProductIds": item.get("selectedProductIds", []),
        } for item in normalized_findings]
        result, retries, error = await audited_call(state, "verify_findings", "validate_evidence", {"claimCount": len(claims)}, lambda: dependencies.drug.validate_evidence(claims))
        metrics = add_metrics(state, result, retries)
        if error:
            return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "TOOL_ERROR", "error": error}], "metrics": metrics}
        if result is None or result.envelope.status.value not in {"OK", "INSUFFICIENT_EVIDENCE"}:
            return {"status": ReviewStatus.BLOCKED_TOOL_ERROR.value, "unresolvedItems": [{"kind": "VALIDATOR_ERROR", "errors": result.envelope.errors if result else ["missing response"]}], "metrics": metrics}
        findings = normalized_findings
        remote = {item.get("claimId"): item for item in result.envelope.data.get("claims", [])} if result else {}
        verified_findings = []
        for item in findings:
            if item.get("status") == FindingStatus.REJECTED.value:
                verified_findings.append(item)
                continue
            finding = Finding.model_validate(item)
            remote_errors = (
                remote.get(item["findingId"], {}).get("errors", [])
                if item["findingId"] in remote
                else (
                    []
                    if item["reviewType"] in UNRESOLVED_FINDING_TYPES
                    else ["remote_validation_incomplete"]
                )
            )
            verified_findings.append(apply_verification(
                finding,
                [
                    *remote_errors,
                    *verify_label_evidence_bindings(
                        finding, state.get("evidenceIndex", [])
                    ),
                ],
            ).model_dump(mode="json"))
        findings = verified_findings
        return {"findings": findings, "status": ReviewStatus.AWAITING_FINDING_REVIEW.value, "metrics": metrics}

    def pharmacist_review(state: ReviewState) -> dict[str, Any]:
        payload = interrupt({"kind": "FINDING_REVIEW", "findings": state.get("findings", [])})
        decision = CompleteFindingReview.model_validate(payload)
        findings = [dict(item) for item in state.get("findings", [])]
        by_id = {item.findingId: item for item in decision.decisions}
        reinvestigate_ids: list[str] = []
        reinvestigation_counts = dict(state.get("reinvestigationCounts") or {})
        decision_events: list[dict[str, Any]] = []
        for item in findings:
            selected = by_id.get(item["findingId"])
            if selected is None:
                continue
            if selected.action == "ACCEPT_FINDING":
                lacks_pair = item["reviewType"] not in UNRESOLVED_FINDING_TYPES and (
                    not item.get("patientEvidenceRefs") or not item.get("labelEvidenceRefs")
                )
                binding_errors = verify_label_evidence_bindings(
                    Finding.model_validate(item), state.get("evidenceIndex", [])
                )
                if item.get("verificationErrors") or lacks_pair or binding_errors:
                    item["status"] = FindingStatus.NEEDS_MORE_EVIDENCE.value
                    item["verificationErrors"] = list(dict.fromkeys([
                        *item.get("verificationErrors", []),
                        *(["missing_paired_evidence"] if lacks_pair else []),
                        *binding_errors,
                    ]))
                else:
                    item["status"] = FindingStatus.ACCEPTED.value
            elif selected.action == "REJECT_FINDING":
                item["status"] = FindingStatus.REJECTED.value
            else:
                item["status"] = FindingStatus.NEEDS_MORE_EVIDENCE.value
                prior_count = reinvestigation_counts.get(item["findingId"], 0)
                if prior_count >= 1:
                    item["verificationErrors"] = list(dict.fromkeys([
                        *item.get("verificationErrors", []),
                        "REINVESTIGATION_BUDGET_EXHAUSTED",
                    ]))
                else:
                    reinvestigation_counts[item["findingId"]] = 1
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
        return {
            "findings": findings,
            "humanDecisions": [
                *(state.get("humanDecisions") or []),
                *decision_events,
                human.model_dump(mode="json"),
            ],
            "status": status,
            "reinvestigateFindingIds": reinvestigate_ids,
            "reinvestigationCounts": reinvestigation_counts,
        }

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
    graph.add_node("parse_review_goal", parse_review_goal)
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
    graph.add_conditional_edges(
        "resolve_medications",
        lambda state: (
            "end" if state["status"] == ReviewStatus.BLOCKED_TOOL_ERROR.value
            else "confirm" if state["status"] == ReviewStatus.AWAITING_MAPPING_CONFIRMATION.value
            else "plan" if state.get("intent")
            else "parse"
        ),
        {
            "end": END,
            "confirm": "confirm_mapping",
            "parse": "parse_review_goal",
            "plan": "plan_review",
        },
    )
    graph.add_conditional_edges(
        "confirm_mapping",
        lambda state: (
            "confirm" if state["status"] == ReviewStatus.AWAITING_MAPPING_CONFIRMATION.value
            else "bind" if state.get("contextRebuildPending")
            else _next_reinvestigation_node(
                state, "bind" if state.get("intent") else "parse"
            )
        ),
        {
            "confirm": "confirm_mapping",
            "context": "reinvestigate_context",
            "evidence": "reinvestigate_evidence",
            "mapping": "reinvestigate_mapping",
            "bind": "plan_review",
            "parse": "parse_review_goal",
        },
    )
    graph.add_edge("parse_review_goal", "plan_review")
    graph.add_edge("plan_review", "retrieve_evidence")
    graph.add_conditional_edges("retrieve_evidence", lambda state: "end" if state.get("status") == ReviewStatus.BLOCKED_TOOL_ERROR.value else "build", {"end": END, "build": "build_findings"})
    graph.add_edge("build_findings", "verify_findings")
    graph.add_conditional_edges(
        "verify_findings",
        lambda state: (
            "end"
            if state["status"] == ReviewStatus.BLOCKED_TOOL_ERROR.value
            else _next_reinvestigation_node(state, "review")
        ),
        {
            "end": END,
            "context": "reinvestigate_context",
            "evidence": "reinvestigate_evidence",
            "mapping": "reinvestigate_mapping",
            "review": "pharmacist_review",
        },
    )
    graph.add_conditional_edges(
        "pharmacist_review",
        lambda state: (
            "report" if state["status"] == ReviewStatus.READY_FOR_SIGN_OFF.value
            else _next_reinvestigation_node(state, "review")
        ),
        {
            "report": "render_report", "context": "reinvestigate_context",
            "mapping": "reinvestigate_mapping", "evidence": "reinvestigate_evidence",
            "review": "pharmacist_review",
        },
    )
    graph.add_conditional_edges(
        "reinvestigate_context",
        lambda state: (
            "end" if state.get("status") == ReviewStatus.BLOCKED_TOOL_ERROR.value
            else "resolve" if state.get("contextMedicationsChanged")
            else _next_reinvestigation_node(state, "verify")
        ),
        {
            "end": END, "resolve": "resolve_medications",
            "context": "reinvestigate_context",
            "evidence": "reinvestigate_evidence",
            "mapping": "reinvestigate_mapping",
            "verify": "verify_findings",
        },
    )
    graph.add_conditional_edges(
        "reinvestigate_mapping",
        lambda state: (
            "end" if state.get("status") == ReviewStatus.BLOCKED_TOOL_ERROR.value
            else "confirm" if state.get("status") == ReviewStatus.AWAITING_MAPPING_CONFIRMATION.value
            else _next_reinvestigation_node(state, "verify")
        ),
        {
            "end": END, "confirm": "confirm_mapping",
            "context": "reinvestigate_context",
            "evidence": "reinvestigate_evidence",
            "mapping": "reinvestigate_mapping",
            "verify": "verify_findings",
        },
    )
    graph.add_conditional_edges(
        "reinvestigate_evidence",
        lambda state: (
            "end" if state.get("status") == ReviewStatus.BLOCKED_TOOL_ERROR.value
            else _next_reinvestigation_node(state, "verify")
        ),
        {
            "end": END,
            "context": "reinvestigate_context",
            "evidence": "reinvestigate_evidence",
            "mapping": "reinvestigate_mapping",
            "verify": "verify_findings",
        },
    )
    graph.add_edge("render_report", END)
    return graph.compile(checkpointer=checkpointer)


def state_to_snapshot(existing: ReviewSnapshot, state: dict[str, Any]) -> ReviewSnapshot:
    fields = ReviewSnapshot.model_fields
    merged = existing.model_dump(mode="python")
    merged.update({key: value for key, value in state.items() if key in fields})
    merged["findings"] = [
        migrate_finding_payload(item)
        for item in merged.get("findings") or []
    ]
    merged["updatedAt"] = datetime.now(UTC)
    return ReviewSnapshot.model_validate(merged)
