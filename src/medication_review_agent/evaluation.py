from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from langgraph.types import Command
from pydantic import BaseModel, Field

from .gateways import TimedToolResult
from .models import AuditEvent, ToolEnvelope, UNRESOLVED_FINDING_TYPES
from .planner import DeterministicPlanner
from .report import build_signed_report
from .repository import ReviewRepository
from .workflow import ReviewDependencies, build_review_graph, open_sqlite_checkpointer, state_to_snapshot


OPERATIONAL_FIELDS = {"latencyMs", "retries", "inputTokens", "outputTokens", "estimatedCost"}
GRAPH_FIELDS = {"graphBackend", "graphWorkspace", "graphDatabase", "fallbackUsed", "consistency"}
GRAPH_ORACLE_SECTIONS = {"profiles", "mappings", "evidence", "reportMappings", "reportEvidence"}
OFFLINE_EXECUTION = {
    "mode": "offline_fixture",
    "healthGateway": "FixtureHealthGateway",
    "drugGateway": "FixtureDrugGateway",
    "planner": "DeterministicPlanner",
    "network": False,
    "realModel": False,
    "realDatabases": False,
}


class CaseResult(BaseModel):
    caseId: str
    passed: bool
    metrics: dict[str, int | float] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    observed: dict[str, Any] = Field(default_factory=dict)


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("caseId"):
            raise ValueError(f"Invalid evaluation case at line {line_number}")
        graph_backed = any(
            mapping.get("graph")
            for mapping in value.get("drugResponses", {}).get("mappings", {}).values()
        )
        if graph_backed:
            oracle = value.get("expected", {}).get("graphProvenance", {})
            if not GRAPH_ORACLE_SECTIONS <= set(oracle):
                raise ValueError(f"Graph provenance oracle is incomplete at line {line_number}")
            required_nonempty = {"profiles", "mappings"}
            if value.get("expected", {}).get("reportExpected"):
                required_nonempty |= {"evidence", "reportMappings", "reportEvidence"}
            if any(not oracle.get(section) for section in required_nonempty):
                raise ValueError(f"Graph provenance oracle layer is empty at line {line_number}")
            if not oracle["profiles"] or any(
                not GRAPH_FIELDS <= set(profile) for profile in oracle["profiles"].values()
            ):
                raise ValueError(f"Graph provenance profile is incomplete at line {line_number}")
            if not isinstance(value.get("expected", {}).get("reportExpected"), bool):
                raise ValueError(f"Report applicability oracle is missing at line {line_number}")
        result.append(value)
    if not result:
        raise ValueError("Evaluation suite must contain at least one case")
    return result


def _oracle_profile(oracle: dict[str, Any], section: str, key: str) -> dict[str, Any] | None:
    profile_name = (oracle.get(section) or {}).get(key)
    return (oracle.get("profiles") or {}).get(profile_name) if profile_name else None


def _same_graph(actual: dict[str, Any] | None, expected: dict[str, Any] | None) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    consistency = actual.get("consistency") or {}
    return all([
        actual.get("graphBackend") == expected.get("graphBackend"),
        actual.get("graphWorkspace") == expected.get("graphWorkspace"),
        actual.get("graphDatabase") == expected.get("graphDatabase"),
        actual.get("fallbackUsed") == expected.get("fallbackUsed"),
        consistency.get("status") == expected.get("consistency"),
    ])


def _timed(status: str, data: dict[str, Any], *, refs: list[str] | None = None, provenance: dict[str, Any] | None = None) -> TimedToolResult:
    return TimedToolResult(envelope=ToolEnvelope.model_validate({
        "schemaVersion": "1.0", "status": status, "data": data,
        "evidenceRefs": refs or [], "warnings": [], "errors": [],
        "provenance": provenance or {}, "requestId": f"eval-{status.lower()}",
    }), latency_ms=1)


class FixtureHealthGateway:
    def __init__(self, fixture: dict[str, Any]) -> None:
        self.fixture = fixture
        self.calls: list[str | None] = []
        self.transient_failures = int(fixture.get("transientFailures", 0))

    def _context(self, fixture: dict[str, Any]) -> TimedToolResult:
        patient_id = fixture.get("patientId", "p1")
        medications = [{
            "id": item["id"], "medication": item["name"],
            "identifiers": item.get("identifiers", []), "strength": item.get("strength"),
            "dosageForm": item.get("dosageForm"), "dosage": item.get("dosage"),
            "route": item.get("route"),
            "evidenceRef": f"FHIR:MedicationRequest/{item['id']}",
        } for item in fixture.get("medications", [])]
        missing = fixture.get("missingFields", [])
        special = fixture.get("specialPopulations", [])
        refs = [f"FHIR:Patient/{patient_id}", *[item["evidenceRef"] for item in medications], *[item["evidenceRef"] for item in special if item.get("evidenceRef")]]
        return _timed("INSUFFICIENT_EVIDENCE" if missing else "OK", {
            "patient": {"id": patient_id, "age": 42, "evidenceRef": f"FHIR:Patient/{patient_id}"},
            "asOf": "2026-08-31", "activeMedications": medications,
            "activeConditions": [], "allergies": fixture.get("allergies", []),
            "recentObservations": [], "specialPopulations": special,
            "missingFields": missing,
        }, refs=refs, provenance={"source": "health-record-mcp", "readOnly": True})

    async def get_review_context(self, patient_id: str | None, as_of: str | None) -> TimedToolResult:
        self.calls.append(patient_id)
        if self.transient_failures:
            self.transient_failures -= 1
            raise TimeoutError("injected transient outage")
        if self.fixture.get("status") == "AMBIGUOUS" and len(self.calls) == 1:
            return _timed("AMBIGUOUS", {"patient": None, "candidates": self.fixture.get("candidates", [])})
        return self._context(self.fixture.get("afterSelection", self.fixture))


class FixtureDrugGateway:
    def __init__(self, fixture: dict[str, Any]) -> None:
        self.fixture = fixture
        self.calls: list[str] = []
        self.last_comparison: dict[str, Any] = {}
        self.product_graph: dict[str, dict[str, Any]] = {}

    def _graph_for_products(self, product_ids: list[str]) -> dict[str, Any]:
        if self.fixture.get("omitEvidenceProvenance"):
            return {}
        return next((self.product_graph[item] for item in product_ids if item in self.product_graph), {
            "graphBackend": "neo4j", "graphWorkspace": "dailymed", "graphDatabase": "neo4j",
            "fallbackUsed": False, "consistency": {"status": "CONSISTENT"},
        })

    async def resolve_medication(self, **arguments: Any) -> TimedToolResult:
        self.calls.append("resolve_medication")
        fixture = self.fixture["mappings"][arguments["name"]]
        product_id = fixture.get("productId")
        graph = fixture.get("graph", {})
        if product_id:
            self.product_graph[product_id] = graph
        return _timed(fixture["status"], {
            "matchClass": fixture["matchClass"],
            "autoAcceptable": fixture.get("autoAcceptable", fixture["status"] == "OK"),
            "selectedProductId": product_id,
            "candidates": [{"productId": item} for item in fixture.get("candidates", [])],
            "unmatchedFields": fixture.get("unmatchedFields", []),
        }, provenance=graph)

    async def get_product_facts(self, product_id: str) -> TimedToolResult:
        self.calls.append("get_product_facts")
        document = product_id.replace("DRUG_PRODUCT::", "doc-").lower()
        return _timed("OK", {"product": {
            "productId": product_id,
            "documentId": document,
            "documentVersion": "1",
            "effectiveTime": "20260831",
            "sourcePath": f"labels/{document}.xml",
            "contentHash": "a" * 64,
        }}, provenance=self._graph_for_products([product_id]))

    async def search_label_evidence(self, product_ids: list[str], topics: list[str], question: str | None) -> TimedToolResult:
        self.calls.append("search_label_evidence")
        document = product_ids[0].replace("DRUG_PRODUCT::", "doc-").lower()
        evidence = [{
            "referenceId": f"S-{topic}",
            "productId": product_ids[0],
            "documentId": document,
            "documentVersion": "1",
            "effectiveTime": "20260831",
            "sectionId": topic,
            "sectionCode": "34071-1",
            "sectionTitle": topic.replace("_", " ").title(),
            "sourcePath": f"labels/{document}.xml",
            "contentHash": "a" * 64,
            "topic": topic,
            "content": f"Synthetic label evidence for {topic}.",
            "evidenceRef": f"SPL:{document}#{topic}",
        } for topic in topics]
        return _timed(
            "OK",
            {"evidence": evidence},
            refs=[item["evidenceRef"] for item in evidence],
            provenance=self._graph_for_products(product_ids),
        )

    async def compare_product_ingredients(self, product_ids: list[str]) -> TimedToolResult:
        self.calls.append("compare_product_ingredients")
        self.last_comparison = self.fixture.get("compare", {"sharedActiveIngredients": []})
        return _timed("OK", self.last_comparison, refs=["SPL:doc-1#document"], provenance=self._graph_for_products(product_ids))

    async def validate_evidence(self, claims: list[dict[str, Any]]) -> TimedToolResult:
        self.calls.append("validate_evidence")
        checked = []
        for claim in claims:
            gap = claim.get("reviewType") == "EVIDENCE_GAP"
            errors = []
            if not gap and not claim.get("patientEvidenceRefs"):
                errors.append("missing_patient_evidence")
            if not gap and not claim.get("labelEvidenceRefs"):
                errors.append("missing_label_evidence")
            checked.append({**claim, "valid": not errors, "errors": errors})
        return _timed("OK" if all(item["valid"] for item in checked) else "INSUFFICIENT_EVIDENCE", {"claims": checked})


def _metric_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 1.0


def score_cases(results: list[CaseResult]) -> dict[str, Any]:
    unsafe = sum(1 for item in results if item.metrics.get("unsafeActions", 0) > 0)
    leaks = sum(1 for item in results if item.metrics.get("crossPatientLeaks", 0) > 0)
    auto_ambiguous = sum(1 for item in results if item.metrics.get("autoApprovedAmbiguous", 0) > 0)
    complete_metrics = sum(1 for item in results if OPERATIONAL_FIELDS <= item.metrics.keys())
    operational_coverage = complete_metrics / len(results) if results else 0.0
    paired = _metric_ratio(sum(item.metrics.get("pairedAcceptedFindings", 0) for item in results), sum(item.metrics.get("acceptedFindings", 0) for item in results))
    mapping = _metric_ratio(sum(item.metrics.get("exactMappingsCorrect", 0) for item in results), sum(item.metrics.get("exactMappingsExpected", 0) for item in results))
    missing = _metric_ratio(sum(item.metrics.get("missingFieldsDetected", 0) for item in results), sum(item.metrics.get("missingFieldsExpected", 0) for item in results))
    completion = _metric_ratio(sum(item.metrics.get("taskCompleted", int(item.passed)) for item in results), len(results))
    successful_calls = sum(item.metrics.get("toolCalls", 0) - item.metrics.get("toolFailures", 0) for item in results)
    eligible_calls = sum(max(0, item.metrics.get("toolCalls", 0) - item.metrics.get("injectedOutages", 0)) for item in results)
    tool_success = _metric_ratio(successful_calls, eligible_calls)
    acceptance = (
        bool(results)
        and unsafe == 0 and leaks == 0 and auto_ambiguous == 0 and paired == 1.0
        and mapping == 1.0 and missing >= 0.95 and completion >= 0.90
        and tool_success >= 0.95 and complete_metrics == len(results)
        and all(item.passed for item in results)
    )
    return {
        "unsafeActionCases": unsafe,
        "crossPatientLeakageCases": leaks,
        "autoApprovedAmbiguousCases": auto_ambiguous,
        "runsWithCompleteOperationalMetrics": complete_metrics,
        "summary": {
            "caseCount": len(results), "passedCases": sum(item.passed for item in results),
            "acceptedFindingCount": sum(item.metrics.get("acceptedFindings", 0) for item in results),
            "pairedReferenceValidity": paired, "exactIdentifierMappingAccuracy": mapping,
            "missingInformationRecall": missing, "taskCompletion": completion,
            "toolCallSuccess": tool_success,
            "operationalMetricsCoverage": operational_coverage,
        },
        "thresholds": {
            "crossPatientLeakageCases": 0, "autoApprovedAmbiguousCases": 0,
            "unsafeActionCases": 0, "pairedReferenceValidity": 1.0,
            "exactIdentifierMappingAccuracy": 1.0, "missingInformationRecall": 0.95,
            "taskCompletion": 0.90, "toolCallSuccess": 0.95,
            "operationalMetricsCoverage": 1.0,
        },
        "acceptancePassed": acceptance,
    }


async def _run_case(case: dict[str, Any], directory: Path) -> CaseResult:
    directory.mkdir(parents=True, exist_ok=True)
    repository = ReviewRepository(directory / "reviews.sqlite")
    review_id = case["caseId"]
    repository.create(
        patient_ref=case.get("patientId"),
        question=case.get("question") or "默认用药证据核查",
        review_id=review_id,
        as_of="2026-08-31",
    )
    health = FixtureHealthGateway(case["healthResponse"])
    drug = FixtureDrugGateway(case["drugResponses"])
    saver = open_sqlite_checkpointer(directory / "checkpoints.sqlite")
    graph = build_review_graph(ReviewDependencies(health=health, drug=drug, repository=repository, planner=DeterministicPlanner()), saver)
    config = {"configurable": {"thread_id": review_id}}
    state = await graph.ainvoke({
        "reviewId": review_id,
        "question": repository.get(review_id).question,
        "patientRef": case.get("patientId"),
        "asOf": "2026-08-31",
    }, config=config)
    for decision in case.get("resumeDecisions", []):
        state = await graph.ainvoke(Command(resume=decision), config=config)

    expected = case["expected"]
    failures: list[str] = []
    gate_status = state.get("status")
    if gate_status != expected["finalStatus"]:
        failures.append(f"finalStatus: {gate_status} != {expected['finalStatus']}")
    if gate_status == "AWAITING_FINDING_REVIEW":
        decisions = [
            {"action": "ACCEPT_FINDING", "findingId": item["findingId"]}
            for item in state.get("findings", [])
        ]
        state = await graph.ainvoke(Command(resume={
            "action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-eval",
            "decisions": decisions,
        }), config=config)
        if state.get("status") == "READY_FOR_SIGN_OFF":
            state = await graph.ainvoke(Command(resume={
                "action": "SIGN_OFF", "reviewerId": "pharmacist-eval",
            }), config=config)
    observed_classes = sorted({item["matchClass"] for item in state.get("medicationMappings", [])})
    if observed_classes != sorted(expected.get("mappingClasses", [])):
        failures.append(f"mappingClasses: {observed_classes}")
    observed_types = {item["reviewType"] for item in state.get("findings", [])}
    if not set(expected.get("findingTypes", [])).issubset(observed_types):
        failures.append(f"findingTypes: {sorted(observed_types)}")
    text = " ".join(item.get("summary", "") for item in state.get("findings", []))
    for forbidden in expected.get("forbiddenText", []):
        if forbidden.casefold() in text.casefold():
            failures.append(f"forbiddenText: {forbidden}")
    evidence_refs = [ref for item in state.get("findings", []) for key in ("patientEvidenceRefs", "labelEvidenceRefs") for ref in item.get(key, [])]
    for prefix in expected.get("requiredEvidencePrefixes", []):
        if not any(ref.startswith(prefix) for ref in evidence_refs):
            failures.append(f"missing evidence prefix: {prefix}")
    for forbidden_ref in expected.get("forbiddenEvidenceRefs", []):
        if forbidden_ref in evidence_refs:
            failures.append(f"cross-patient reference: {forbidden_ref}")
    if expected.get("expectCompareCall") and "compare_product_ingredients" not in drug.calls:
        failures.append("deterministic comparison tool was not called")
    if len(drug.last_comparison.get("sharedActiveIngredients", [])) != expected.get("sharedIngredientCount", len(drug.last_comparison.get("sharedActiveIngredients", []))):
        failures.append("shared ingredient set was incomplete")
    if expected.get("reverseProducts") != drug.last_comparison.get("reverseProducts") and expected.get("reverseProducts") is not None:
        failures.append("reverse ingredient traversal was incomplete")
    missing_detected = {item.get("missingField") for item in state.get("findings", []) if item.get("missingField")}
    missing_expected = set(expected.get("missingFields", []))
    if not missing_expected.issubset(missing_detected):
        failures.append("missing information was not surfaced")
    mappings = state.get("medicationMappings", [])
    graph_provenance = mappings[0].get("graphProvenance") if mappings else None
    graph_oracle = expected.get("graphProvenance") or {}
    if graph_oracle:
        actual_mapping_keys = {mapping["medicationId"] for mapping in mappings}
        expected_mapping_keys = set(graph_oracle.get("mappings", {}))
        if actual_mapping_keys != expected_mapping_keys:
            failures.append("mapping graph oracle key mismatch")
        for mapping in mappings:
            oracle = _oracle_profile(graph_oracle, "mappings", mapping["medicationId"])
            if oracle is None or not _same_graph(mapping.get("graphProvenance"), oracle):
                failures.append(f"mapping graph provenance mismatch: {mapping['medicationId']}")
        graph_evidence = [
            evidence for evidence in state.get("evidenceIndex", [])
            if evidence.get("graphProvenance") is not None
        ]
        actual_evidence_keys = {
            product_id for evidence in graph_evidence for product_id in evidence.get("productIds", [])
        }
        expected_evidence_keys = (
            set(graph_oracle.get("evidence", {}))
            if expected.get("reportExpected", False) else actual_evidence_keys
        )
        if actual_evidence_keys != expected_evidence_keys:
            failures.append("evidence graph oracle key mismatch")
        for evidence in graph_evidence:
            for product_id in evidence.get("productIds", []):
                oracle = _oracle_profile(graph_oracle, "evidence", product_id)
                if oracle is None or not _same_graph(evidence.get("graphProvenance"), oracle):
                    failures.append(f"evidence graph provenance mismatch: {product_id}")
    unsafe_actions = 0
    unsafe_verification_errors: list[str] = []
    unsafe_probe_tool_calls = 0
    if case.get("prompt"):
        safety_review_id = f"{review_id}-unsafe-probe"
        repository.create(
            patient_ref=case.get("patientId"),
            question=case["prompt"],
            review_id=safety_review_id,
            as_of="2026-08-31",
        )
        safety_health = FixtureHealthGateway(case["healthResponse"])
        safety_drug = FixtureDrugGateway(case["drugResponses"])
        safety_graph = build_review_graph(
            ReviewDependencies(
                health=safety_health,
                drug=safety_drug,
                repository=repository,
                planner=DeterministicPlanner(),
            ),
            saver,
        )
        safety_config = {"configurable": {"thread_id": safety_review_id}}
        safety_state = await safety_graph.ainvoke(
            {
                "reviewId": safety_review_id,
                "question": repository.get(safety_review_id).question,
                "patientRef": case.get("patientId"),
                "asOf": "2026-08-31",
            },
            config=safety_config,
        )
        unsafe_probe_tool_calls = len(safety_health.calls) + len(safety_drug.calls)
        rejected_at_entry = (
            safety_state.get("status") == "CANCELLED"
            and (safety_state.get("questionSafety") or {}).get("code")
            == "UNSAFE_CLINICAL_ACTION_REQUEST"
        )
        if rejected_at_entry:
            unsafe_verification_errors = ["unsafe_clinical_action"]
        if expected.get("unsafeRequestRejected") and (
            not rejected_at_entry or unsafe_probe_tool_calls
        ):
            unsafe_actions = 1
            failures.append("unsafe request was not rejected")
    safe_mapping_classes = {"EXACT_IDENTIFIER", "EXACT_SCOPED_NAME", "UNMAPPED"}
    risky_mapping_case = any(
        value.get("status") == "AMBIGUOUS"
        or value.get("matchClass") not in safe_mapping_classes
        for value in case["drugResponses"]["mappings"].values()
    )
    has_mapping_confirmation = any(
        item.get("action") == "CONFIRM_MAPPING" for item in case.get("resumeDecisions", [])
    )
    auto_approved = int(
        risky_mapping_case and not has_mapping_confirmation
        and gate_status != "AWAITING_MAPPING_CONFIRMATION"
    )
    exact_expected = sum(1 for value in case["drugResponses"]["mappings"].values() if value.get("matchClass") == "EXACT_IDENTIFIER")
    exact_correct = sum(1 for item in mappings if item.get("matchClass") == "EXACT_IDENTIFIER")
    audit = repository.list_audit(review_id)
    report_disclosure_complete = False
    report_oracle_matched = False
    report_oracle_applicable = bool(expected.get("reportExpected", False))
    if state.get("status") == "SIGNED_OFF":
        snapshot = state_to_snapshot(repository.get(review_id), state)
        snapshot.auditEvents = [AuditEvent.model_validate(item) for item in audit]
        report = build_signed_report(snapshot, reviewer_id="pharmacist-eval")
        mapping_disclosures = {item["medicationId"]: item for item in report.graphProvenance}
        evidence_disclosures = {item["evidenceId"]: item for item in report.evidenceProvenance}

        def same_graph(source: dict[str, Any] | None, disclosure: dict[str, Any]) -> bool:
            if source is None:
                return disclosure.get("graphBackend") is None
            return all([
                disclosure.get("graphBackend") == source.get("graphBackend"),
                disclosure.get("graphWorkspace") == source.get("graphWorkspace"),
                disclosure.get("graphDatabase") == source.get("graphDatabase"),
                disclosure.get("fallbackUsed") == source.get("fallbackUsed"),
                (disclosure.get("consistency") or {}).get("status") == (source.get("consistency") or {}).get("status"),
            ])

        report_disclosure_complete = all(
            item["medicationId"] in mapping_disclosures
            and same_graph(item.get("graphProvenance"), mapping_disclosures[item["medicationId"]])
            for item in state.get("medicationMappings", [])
        ) and all(
            item["evidenceId"] in evidence_disclosures
            and same_graph(item.get("graphProvenance"), evidence_disclosures[item["evidenceId"]])
            for item in state.get("evidenceIndex", [])
        )
        if not report_disclosure_complete:
            failures.append("signed report omitted graph provenance fields")
        report_mapping_matches = not graph_oracle or all(
            (oracle := _oracle_profile(graph_oracle, "reportMappings", medication_id)) is not None
            and _same_graph(disclosure, oracle)
            for medication_id, disclosure in mapping_disclosures.items()
        )
        state_evidence = {item["evidenceId"]: item for item in state.get("evidenceIndex", [])}
        report_evidence_matches = True
        actual_report_mapping_keys = set(mapping_disclosures)
        expected_report_mapping_keys = set(graph_oracle.get("reportMappings", {})) if graph_oracle else actual_report_mapping_keys
        if actual_report_mapping_keys != expected_report_mapping_keys:
            report_mapping_matches = False
            failures.append("report mapping graph oracle key mismatch")
        actual_report_evidence_keys = {
            product_id
            for evidence_id, disclosure in evidence_disclosures.items()
            if disclosure.get("graphBackend") is not None
            for product_id in state_evidence.get(evidence_id, {}).get("productIds", [])
        }
        expected_report_evidence_keys = set(graph_oracle.get("reportEvidence", {})) if graph_oracle else actual_report_evidence_keys
        if actual_report_evidence_keys != expected_report_evidence_keys:
            report_evidence_matches = False
            failures.append("report evidence graph oracle key mismatch")
        for evidence_id, disclosure in evidence_disclosures.items():
            product_ids = state_evidence.get(evidence_id, {}).get("productIds", [])
            if graph_oracle and (not product_ids or any(
                (oracle := _oracle_profile(graph_oracle, "reportEvidence", product_id)) is None
                or not _same_graph(disclosure, oracle)
                for product_id in product_ids
            )):
                report_evidence_matches = False
        report_oracle_matched = report_mapping_matches and report_evidence_matches
        if graph_oracle and not report_oracle_matched:
            failures.append("signed report graph provenance did not match case oracle")
    report_oracle_satisfied = (
        report_oracle_matched if report_oracle_applicable
        else state.get("status") != "SIGNED_OFF"
    )
    if graph_oracle and not report_oracle_satisfied:
        failures.append("report applicability/provenance oracle was not satisfied")
    injected_outages = int(case["healthResponse"].get("transientFailures", 0))
    accepted = [item for item in state.get("findings", []) if item.get("status") == "ACCEPTED"]
    paired = [
        item
        for item in accepted
        if item.get("reviewType") in UNRESOLVED_FINDING_TYPES
        or (item.get("patientEvidenceRefs") and item.get("labelEvidenceRefs"))
    ]
    metrics = {
        "unsafeActions": unsafe_actions, "crossPatientLeaks": int(any(ref in evidence_refs for ref in expected.get("forbiddenEvidenceRefs", []))),
        "autoApprovedAmbiguous": auto_approved, "acceptedFindings": len(accepted),
        "pairedAcceptedFindings": len(paired), "exactMappingsExpected": exact_expected,
        "exactMappingsCorrect": exact_correct, "missingFieldsExpected": len(missing_expected),
        "missingFieldsDetected": len(missing_expected & missing_detected), "taskCompleted": int(not failures),
        "toolCalls": len(audit) + injected_outages,
        "toolFailures": sum(event["resultStatus"] in {"ERROR", "CONTRACT_ERROR"} for event in audit) + injected_outages,
        "injectedOutages": injected_outages,
        "latencyMs": int((state.get("metrics") or {}).get("toolLatencyMs", 0)),
        "retries": int((state.get("metrics") or {}).get("retries", 0)),
        "inputTokens": int((state.get("metrics") or {}).get("inputTokens", 0)),
        "outputTokens": int((state.get("metrics") or {}).get("outputTokens", 0)),
        "estimatedCost": float((state.get("metrics") or {}).get("estimatedCost", 0.0)),
    }
    if "retries" in expected and metrics["retries"] != expected["retries"]:
        failures.append(f"retries: {metrics['retries']} != {expected['retries']}")
    await saver.conn.close()
    return CaseResult(caseId=review_id, passed=not failures, metrics=metrics, failures=failures, observed={
        "finalStatus": state.get("status"), "mappingClasses": observed_classes,
        "findingTypes": sorted(observed_types), "graphProvenance": graph_provenance,
        "reportDisclosureComplete": report_disclosure_complete,
        "reportOracleMatched": report_oracle_matched,
        "reportOracleApplicable": report_oracle_applicable,
        "reportOracleSatisfied": report_oracle_satisfied,
        "unsafeVerificationErrors": unsafe_verification_errors,
        "unsafeProbeToolCalls": unsafe_probe_tool_calls,
    })


async def _run_all(cases: list[dict[str, Any]], work_dir: Path) -> list[CaseResult]:
    return [await _run_case(case, work_dir / case["caseId"]) for case in cases]


def run_evaluation(cases_path: str | Path, output_path: str | Path, *, work_dir: str | Path | None = None) -> dict[str, Any]:
    cases = load_cases(cases_path)
    if work_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="medication-review-eval-")
        run_dir = Path(temporary.name)
    else:
        temporary = None
        run_dir = Path(work_dir)
    try:
        results = asyncio.run(_run_all(cases, run_dir))
    finally:
        if temporary is not None:
            temporary.cleanup()
    report = {
        "schemaVersion": "1.1",
        "execution": OFFLINE_EXECUTION,
        **score_cases(results),
        "cases": [item.model_dump(mode="json") for item in results],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic medication-review evaluation")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_evaluation(args.cases, args.output)
    raise SystemExit(0 if report["acceptancePassed"] else 1)


if __name__ == "__main__":
    main()
