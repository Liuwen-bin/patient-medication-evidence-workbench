from pathlib import Path

import pytest
from langgraph.types import Command

from medication_review_agent.gateways import ToolContractError
from medication_review_agent.models import ReviewSnapshot, ReviewStatus
from medication_review_agent.workflow import state_to_snapshot

from tests.fakes import (
    FakeDrugGateway,
    FakeHealthGateway,
    build_test_graph,
    envelope,
    health_context,
    mapped_response,
    standard_drug_responses,
)


MED1 = {"id": "med-1", "medication": "ARNICA", "identifiers": [{"system": "ndc", "code": "1"}], "strength": None, "dosage": None, "route": None, "evidenceRef": "FHIR:MedicationRequest/med-1"}
MED2 = {"id": "med-2", "medication": "METFORMIN", "identifiers": [], "strength": None, "dosage": None, "route": None, "evidenceRef": "FHIR:MedicationRequest/med-2"}


def test_patient_candidates_survive_snapshot_projection() -> None:
    existing = ReviewSnapshot(reviewId="review-1", status=ReviewStatus.RUNNING)
    projected = state_to_snapshot(existing, {
        "status": ReviewStatus.AWAITING_PATIENT_CONFIRMATION.value,
        "candidates": [{"id": "p1", "patientNumber": "P001"}],
    })
    assert projected.candidates == [{"id": "p1", "patientNumber": "P001"}]


@pytest.mark.asyncio
async def test_review_continues_with_mapped_and_unmapped_medications(tmp_path: Path) -> None:
    responses = standard_drug_responses({
        "ARNICA": mapped_response(),
        "METFORMIN": envelope("UNMAPPED", {"matchClass": "UNMAPPED", "selectedProductId": None, "candidates": [], "unmatchedFields": []}),
    })
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, MED2)), FakeDrugGateway(responses))
    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )
    assert result["status"] == "AWAITING_FINDING_REVIEW"
    assert {item["matchClass"] for item in result["medicationMappings"]} == {"EXACT_IDENTIFIER", "UNMAPPED"}
    assert any(item["reviewType"] == "EVIDENCE_GAP" for item in result["findings"])
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_mapping_interrupt_survives_process_restart_and_targets_selected_product(tmp_path: Path) -> None:
    ambiguous = envelope("AMBIGUOUS", {
        "matchClass": "AMBIGUOUS_NAME", "selectedProductId": None,
        "candidates": [{"productId": "DRUG_PRODUCT::1"}, {"productId": "DRUG_PRODUCT::2"}],
        "unmatchedFields": [],
    }, provenance={"graphBackend": "neo4j", "fallbackUsed": False, "consistency": {"status": "CONSISTENT"}})
    drug = FakeDrugGateway(standard_drug_responses({"ARNICA": ambiguous}))
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), drug)
    config = {"configurable": {"thread_id": "review-1"}}
    interrupted = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    assert interrupted["status"] == "AWAITING_MAPPING_CONFIRMATION"
    await graph.checkpointer.conn.close()

    resumed_graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), drug)
    resumed = await resumed_graph.ainvoke(Command(resume={
        "action": "CONFIRM_MAPPING", "medicationId": "med-1",
        "productId": "DRUG_PRODUCT::1", "reviewerId": "pharmacist-demo",
    }), config=config)
    assert resumed["status"] == "AWAITING_FINDING_REVIEW"
    fact_calls = [call for call in drug.calls if call[0] == "get_product_facts"]
    assert fact_calls == [("get_product_facts", "DRUG_PRODUCT::1")]
    await resumed_graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_snapshot_fallback_creates_gap_and_forces_human_review(tmp_path: Path) -> None:
    responses = standard_drug_responses({"ARNICA": mapped_response(backend="snapshot", fallback=True, consistency="UNAVAILABLE")})
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(responses))
    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )
    assert any("graph_fallback_used" in item.get("verificationWarnings", []) for item in result["findings"])
    assert any(item["requiresHumanReview"] for item in result["medicationMappings"])
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_finding_review_and_final_signoff_survive_restart(tmp_path: Path) -> None:
    drug = FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()}))
    config = {"configurable": {"thread_id": "review-1"}}
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), drug)
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    target = next(item for item in first["findings"] if item["reviewType"] != "EVIDENCE_GAP")
    finding_id = target["findingId"]
    await graph.checkpointer.conn.close()
    finding_reopened = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), drug)
    ready = await finding_reopened.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo",
        "decisions": [{"action": "ACCEPT_FINDING", "findingId": finding_id}],
    }), config=config)
    assert ready["status"] == "READY_FOR_SIGN_OFF"
    await finding_reopened.checkpointer.conn.close()
    reopened = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), drug)
    signed = await reopened.ainvoke(Command(resume={
        "action": "SIGN_OFF", "reviewerId": "pharmacist-demo",
    }), config=config)
    assert signed["status"] == "SIGNED_OFF"
    await reopened.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_patient_confirmation_resumes_with_selected_fhir_id(tmp_path: Path) -> None:
    class SequencedHealth:
        def __init__(self) -> None:
            self.responses = [
                envelope("AMBIGUOUS", {"patient": None, "candidates": [
                    {"id": "p1", "evidenceRef": "FHIR:Patient/p1"},
                    {"id": "p2", "evidenceRef": "FHIR:Patient/p2"},
                ]}),
                health_context(MED1),
            ]
            self.calls = []

        async def get_review_context(self, patient_id, as_of):
            self.calls.append((patient_id, as_of))
            return self.responses.pop(0)

    health = SequencedHealth()
    graph = build_test_graph(tmp_path, health, FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})))
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "Pat Lee", "asOf": "2026-08-31"}, config=config)
    assert first["status"] == "AWAITING_PATIENT_CONFIRMATION"
    await graph.checkpointer.conn.close()
    reopened = build_test_graph(tmp_path, health, FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})))
    resumed = await reopened.ainvoke(Command(resume={
        "action": "CONFIRM_PATIENT", "patientId": "p1", "reviewerId": "pharmacist-demo",
    }), config=config)
    assert resumed["status"] == "AWAITING_FINDING_REVIEW"
    assert health.calls[-1][0] == "p1"
    await reopened.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_transient_tool_failure_retries_twice_then_continues(tmp_path: Path, monkeypatch) -> None:
    class FlakyHealth:
        def __init__(self) -> None:
            self.calls = 0

        async def get_review_context(self, patient_id, as_of):
            self.calls += 1
            if self.calls < 3:
                raise TimeoutError("temporary")
            return health_context(MED1)

    async def no_delay(_seconds: float) -> None:
        return None

    monkeypatch.setattr("medication_review_agent.workflow.asyncio.sleep", no_delay)
    health = FlakyHealth()
    graph = build_test_graph(tmp_path, health, FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})))
    result = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config={"configurable": {"thread_id": "review-1"}})
    assert result["status"] == "AWAITING_FINDING_REVIEW"
    assert health.calls == 3
    assert result["metrics"]["retries"] == 2
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_schema_error_is_not_retried_and_blocks_review(tmp_path: Path) -> None:
    class InvalidHealth:
        def __init__(self) -> None:
            self.calls = 0

        async def get_review_context(self, patient_id, as_of):
            self.calls += 1
            raise ToolContractError("wrong schema")

    health = InvalidHealth()
    graph = build_test_graph(tmp_path, health, FakeDrugGateway({}))
    result = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config={"configurable": {"thread_id": "review-1"}})
    assert result["status"] == "BLOCKED_TOOL_ERROR"
    assert health.calls == 1
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_missing_health_fields_become_explicit_evidence_gaps(tmp_path: Path) -> None:
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1, missing=["allergies", "activeMedications.med-1.route"])),
        FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
    )
    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )
    gaps = [item for item in result["findings"] if item["reviewType"] == "EVIDENCE_GAP"]
    assert {item["missingField"] for item in gaps} >= {"allergies", "activeMedications.med-1.route"}
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_neo4j_set_comparison_builds_duplicate_ingredient_finding(tmp_path: Path) -> None:
    med2 = {**MED2, "medication": "ARNICA TWO", "identifiers": [{"system": "ndc", "code": "2"}]}
    responses = standard_drug_responses({"ARNICA": mapped_response(), "ARNICA TWO": mapped_response()})
    responses["compare"] = envelope("OK", {
        "products": ["DRUG_PRODUCT::1", "DRUG_PRODUCT::2"],
        "sharedActiveIngredients": [{"entityId": "INGREDIENT::ARNICA", "name": "ARNICA"}],
    }, refs=["SPL:doc-1#document"], provenance={
        "graphBackend": "neo4j", "graphWorkspace": "dailymed", "graphDatabase": "neo4j",
        "fallbackUsed": False, "consistency": {"status": "CONSISTENT"},
    })
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, med2)), FakeDrugGateway(responses))
    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )
    duplicates = [item for item in result["findings"] if item["reviewType"] == "DUPLICATE_ACTIVE_INGREDIENT"]
    assert duplicates[0]["graphProvenance"]["graphBackend"] == "neo4j"
    assert duplicates[0]["labelEvidenceRefs"] == ["SPL:doc-1#document"]
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_pharmacist_cannot_accept_non_gap_finding_without_paired_evidence(tmp_path: Path) -> None:
    responses = standard_drug_responses({"ARNICA": mapped_response()})
    responses["search"] = envelope("INSUFFICIENT_EVIDENCE", {"evidence": []})
    responses["validate"] = envelope("INSUFFICIENT_EVIDENCE", {"claims": [{
        "claimId": "placeholder", "valid": False, "errors": ["missing_label_evidence"],
    }]})
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), drug)
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config,
    )
    target = next(item for item in first["findings"] if item["reviewType"] != "EVIDENCE_GAP")
    finding_id = target["findingId"]
    result = await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo",
        "decisions": [{"action": "ACCEPT_FINDING", "findingId": finding_id}],
    }), config=config)
    assert result["status"] == "NEEDS_MORE_EVIDENCE"
    updated = next(item for item in result["findings"] if item["findingId"] == finding_id)
    assert updated["status"] == "NEEDS_MORE_EVIDENCE"
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_ok_fuzzy_mapping_still_requires_pharmacist_confirmation(tmp_path: Path) -> None:
    fuzzy = envelope("OK", {
        "matchClass": "FUZZY_CANDIDATE", "selectedProductId": "DRUG_PRODUCT::1",
        "candidates": [{"productId": "DRUG_PRODUCT::1"}], "unmatchedFields": [],
    }, provenance={"graphBackend": "neo4j", "fallbackUsed": False, "consistency": {"status": "CONSISTENT"}})
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(standard_drug_responses({"ARNICA": fuzzy})))
    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )
    assert result["status"] == "AWAITING_MAPPING_CONFIRMATION"
    assert result["medicationMappings"][0]["requiresHumanReview"] is True
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_scoped_name_with_unmatched_fields_requires_confirmation(tmp_path: Path) -> None:
    scoped = envelope("OK", {
        "matchClass": "EXACT_SCOPED_NAME", "selectedProductId": "DRUG_PRODUCT::1",
        "candidates": [{"productId": "DRUG_PRODUCT::1"}], "unmatchedFields": ["strength"],
    }, provenance={"graphBackend": "neo4j", "fallbackUsed": False, "consistency": {"status": "CONSISTENT"}})
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(standard_drug_responses({"ARNICA": scoped})))
    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )
    assert result["status"] == "AWAITING_MAPPING_CONFIRMATION"
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_retrieval_fallback_provenance_forces_gap_and_survives_finding(tmp_path: Path) -> None:
    responses = standard_drug_responses({"ARNICA": mapped_response()})
    responses["search"] = envelope("OK", {"evidence": [{
        "referenceId": "S1", "documentId": "doc-1", "sectionId": "warnings",
        "content": "fallback evidence", "evidenceRef": "SPL:doc-1#warnings",
    }]}, refs=["SPL:doc-1#warnings"], provenance={
        "graphBackend": "snapshot", "graphWorkspace": "dailymed", "graphDatabase": None,
        "fallbackUsed": True, "consistency": {"status": "UNAVAILABLE"},
    })
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(responses))
    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )
    assert any(item.get("graphProvenance", {}).get("graphBackend") == "snapshot" for item in result["evidenceIndex"])
    assert any("graph_fallback_used" in item.get("verificationWarnings", []) for item in result["findings"])
    assert any(item["reviewType"] == "EVIDENCE_GAP" and item.get("graphProvenance", {}).get("fallbackUsed") for item in result["findings"])
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_validator_error_blocks_review_instead_of_approving_empty_results(tmp_path: Path) -> None:
    responses = standard_drug_responses({"ARNICA": mapped_response()})
    responses["validate"] = envelope("ERROR", {}, errors=["validator unavailable"])
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(responses))
    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )
    assert result["status"] == "BLOCKED_TOOL_ERROR"
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_patient_confirmation_rejects_id_outside_candidates(tmp_path: Path) -> None:
    health = FakeHealthGateway(envelope("AMBIGUOUS", {"patient": None, "candidates": [
        {"id": "p1", "evidenceRef": "FHIR:Patient/p1"}, {"id": "p2", "evidenceRef": "FHIR:Patient/p2"},
    ]}))
    graph = build_test_graph(tmp_path, health, FakeDrugGateway({}))
    config = {"configurable": {"thread_id": "review-1"}}
    await graph.ainvoke({"reviewId": "review-1", "patientRef": "Pat Lee", "asOf": "2026-08-31"}, config=config)
    with pytest.raises(ValueError, match="candidate"):
        await graph.ainvoke(Command(resume={
            "action": "CONFIRM_PATIENT", "patientId": "p999", "reviewerId": "pharmacist-demo",
        }), config=config)
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_partial_finding_decisions_remain_resumable(tmp_path: Path) -> None:
    responses = standard_drug_responses({
        "ARNICA": mapped_response(),
        "METFORMIN": envelope("UNMAPPED", {"matchClass": "UNMAPPED", "selectedProductId": None, "candidates": [], "unmatchedFields": []}),
    })
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, MED2)), FakeDrugGateway(responses))
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    first_id, second_id = [item["findingId"] for item in first["findings"][:2]]
    partial = await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo",
        "decisions": [{"action": "REJECT_FINDING", "findingId": first_id}],
    }), config=config)
    assert partial["status"] == "AWAITING_FINDING_REVIEW"
    completed = await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo",
        "decisions": [{"action": "ACCEPT_FINDING", "findingId": second_id}],
    }), config=config)
    assert completed["status"] == "READY_FOR_SIGN_OFF"
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_request_more_evidence_reenters_retrieval(tmp_path: Path) -> None:
    drug = FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()}))
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), drug)
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo",
        "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": first["findings"][0]["findingId"]}],
    }), config=config)
    assert [name for name, _ in drug.calls].count("search_label_evidence") == 2
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_audit_uses_stable_sha256_identifiers(tmp_path: Path) -> None:
    repository_path = tmp_path / "reviews.sqlite"
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})))
    await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )
    from medication_review_agent.repository import ReviewRepository
    events = ReviewRepository(repository_path).list_audit("review-1")
    hashes = [value for event in events for key, value in event["argumentSummary"].items() if key.endswith("Hash")]
    assert hashes and all(len(value) == 64 and set(value) <= set("0123456789abcdef") for value in hashes)
    assert "med-1" not in str([event["argumentSummary"] for event in events])
    await graph.checkpointer.conn.close()


@pytest.mark.parametrize("tool_name,status", [
    ("facts", "UNMAPPED"),
    ("search", "UNMAPPED"),
    ("compare", "UNMAPPED"),
    ("validate", "UNMAPPED"),
])
@pytest.mark.asyncio
async def test_unexpected_non_ok_drug_envelopes_block_verification(tmp_path: Path, tool_name: str, status: str) -> None:
    med2 = {**MED2, "medication": "ARNICA TWO", "identifiers": [{"system": "ndc", "code": "2"}]}
    responses = standard_drug_responses({"ARNICA": mapped_response(), "ARNICA TWO": mapped_response()})
    responses[tool_name] = envelope(status, {}, errors=["injected non-OK response"])
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, med2)), FakeDrugGateway(responses))
    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )
    assert result["status"] == "BLOCKED_TOOL_ERROR"
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_resolver_error_cannot_supply_an_automatic_mapping(tmp_path: Path) -> None:
    failure = envelope("ERROR", {
        "matchClass": "EXACT_IDENTIFIER", "autoAcceptable": True,
        "selectedProductId": "DRUG_PRODUCT::1", "candidates": [], "unmatchedFields": [],
    }, errors=["resolver unavailable"])
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(standard_drug_responses({"ARNICA": failure})))
    result = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config={"configurable": {"thread_id": "review-1"}})
    assert result["status"] == "BLOCKED_TOOL_ERROR"
    assert not result.get("medicationMappings")
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_unmapped_resolver_discards_a_spurious_selected_product(tmp_path: Path) -> None:
    unmapped = envelope("UNMAPPED", {
        "matchClass": "EXACT_IDENTIFIER", "autoAcceptable": True,
        "selectedProductId": "DRUG_PRODUCT::1", "candidates": [], "unmatchedFields": [],
    })
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(standard_drug_responses({"ARNICA": unmapped})))
    result = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config={"configurable": {"thread_id": "review-1"}})
    assert result["medicationMappings"][0]["matchClass"] == "UNMAPPED"
    assert result["medicationMappings"][0]["selectedProductId"] is None
    assert result["status"] == "AWAITING_FINDING_REVIEW"
    await graph.checkpointer.conn.close()


@pytest.mark.parametrize("tool_name", ["facts", "compare"])
@pytest.mark.asyncio
async def test_insufficient_deterministic_evidence_becomes_a_reviewable_gap(tmp_path: Path, tool_name: str) -> None:
    med2 = {**MED2, "medication": "ARNICA TWO", "identifiers": [{"system": "ndc", "code": "2"}]}
    responses = standard_drug_responses({"ARNICA": mapped_response(), "ARNICA TWO": mapped_response()})
    responses[tool_name] = envelope("INSUFFICIENT_EVIDENCE", {}, errors=["source has no supported evidence"])
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, med2)), FakeDrugGateway(responses))
    result = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config={"configurable": {"thread_id": "review-1"}})
    assert result["status"] == "AWAITING_FINDING_REVIEW"
    assert any(
        item["reviewType"] == "EVIDENCE_GAP" and item.get("sourceTool") == {
            "facts": "get_product_facts", "compare": "compare_product_ingredients",
        }[tool_name]
        for item in result["findings"]
    )
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_scoped_name_requires_auto_acceptable_unique_candidate(tmp_path: Path) -> None:
    scoped = envelope("OK", {
        "matchClass": "EXACT_SCOPED_NAME", "autoAcceptable": False,
        "selectedProductId": "DRUG_PRODUCT::1",
        "candidates": [{"productId": "DRUG_PRODUCT::1"}, {"productId": "DRUG_PRODUCT::2"}],
        "unmatchedFields": [],
    }, provenance={"graphBackend": "neo4j", "fallbackUsed": False, "consistency": {"status": "CONSISTENT"}})
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(standard_drug_responses({"ARNICA": scoped})))
    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )
    assert result["status"] == "AWAITING_MAPPING_CONFIRMATION"
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_multiple_ambiguous_mappings_are_confirmed_one_at_a_time(tmp_path: Path) -> None:
    med2 = {**MED2, "medication": "ARNICA TWO"}
    provenance = {"graphBackend": "neo4j", "fallbackUsed": False, "consistency": {"status": "CONSISTENT"}}
    responses = standard_drug_responses({
        "ARNICA": envelope("AMBIGUOUS", {"matchClass": "AMBIGUOUS_NAME", "selectedProductId": None, "candidates": [{"productId": "DRUG_PRODUCT::1"}], "unmatchedFields": []}, provenance=provenance),
        "ARNICA TWO": envelope("AMBIGUOUS", {"matchClass": "AMBIGUOUS_NAME", "selectedProductId": None, "candidates": [{"productId": "DRUG_PRODUCT::2"}], "unmatchedFields": []}, provenance=provenance),
    })
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, med2)), FakeDrugGateway(responses))
    config = {"configurable": {"thread_id": "review-1"}}
    await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    second_gate = await graph.ainvoke(Command(resume={"action": "CONFIRM_MAPPING", "medicationId": "med-1", "productId": "DRUG_PRODUCT::1", "reviewerId": "pharmacist-demo"}), config=config)
    assert second_gate["status"] == "AWAITING_MAPPING_CONFIRMATION"
    completed = await graph.ainvoke(Command(resume={"action": "CONFIRM_MAPPING", "medicationId": "med-2", "productId": "DRUG_PRODUCT::2", "reviewerId": "pharmacist-demo"}), config=config)
    assert completed["status"] == "AWAITING_FINDING_REVIEW"
    assert all(not item["mappingConfirmationRequired"] for item in completed["medicationMappings"])
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_multi_product_label_references_remain_product_scoped(tmp_path: Path) -> None:
    med2 = {**MED2, "medication": "ARNICA TWO"}
    responses = standard_drug_responses({"ARNICA": mapped_response(), "ARNICA TWO": envelope("OK", {
        "matchClass": "EXACT_IDENTIFIER", "autoAcceptable": True, "selectedProductId": "DRUG_PRODUCT::2", "candidates": [], "unmatchedFields": [],
    }, provenance={"graphBackend": "neo4j", "fallbackUsed": False, "consistency": {"status": "CONSISTENT"}})})
    responses["search"] = [
        envelope("OK", {"evidence": [{"referenceId": "S1", "evidenceRef": "SPL:doc-1#warnings", "content": "one"}]}, refs=["SPL:doc-1#warnings"]),
        envelope("OK", {"evidence": [{"referenceId": "S2", "evidenceRef": "SPL:doc-2#warnings", "content": "two"}]}, refs=["SPL:doc-2#warnings"]),
    ]
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, med2)), FakeDrugGateway(responses))
    result = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config={"configurable": {"thread_id": "review-1"}})
    label_findings = {item["medicationIds"][0]: item for item in result["findings"] if item["reviewType"] == "LABEL_EVIDENCE_REVIEW"}
    assert label_findings["med-1"]["labelEvidenceRefs"] == ["SPL:doc-1#warnings"]
    assert label_findings["med-2"]["labelEvidenceRefs"] == ["SPL:doc-2#warnings"]
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_targeted_reinvestigation_preserves_other_finding_ids_and_decisions(tmp_path: Path) -> None:
    responses = standard_drug_responses({
        "ARNICA": mapped_response(),
        "METFORMIN": envelope("UNMAPPED", {"matchClass": "UNMAPPED", "selectedProductId": None, "candidates": [], "unmatchedFields": []}),
    })
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, MED2)), drug)
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    label = next(item for item in first["findings"] if item["reviewType"] == "LABEL_EVIDENCE_REVIEW")
    gap = next(item for item in first["findings"] if item["reviewType"] == "EVIDENCE_GAP")
    await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REJECT_FINDING", "findingId": gap["findingId"]}]}), config=config)
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": label["findingId"]}]}), config=config)
    preserved = next(item for item in updated["findings"] if item["findingId"] == gap["findingId"])
    assert preserved["status"] == "REJECTED"
    assert any(item["findingId"] == label["findingId"] for item in updated["findings"])
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_reinvestigation_refreshes_fallback_provenance_and_upserts_stable_gap(tmp_path: Path) -> None:
    consistent = {"graphBackend": "neo4j", "graphWorkspace": "dailymed", "graphDatabase": "neo4j", "fallbackUsed": False, "consistency": {"status": "CONSISTENT"}}
    fallback = {"graphBackend": "snapshot", "graphWorkspace": "dailymed", "graphDatabase": None, "fallbackUsed": True, "consistency": {"status": "UNAVAILABLE"}}
    responses = standard_drug_responses({"ARNICA": mapped_response()})
    responses["facts"] = [envelope("OK", {"product": {}}, provenance=consistent), envelope("OK", {"product": {}}, provenance=fallback)]
    responses["search"] = [
        envelope("OK", {"evidence": [{"referenceId": "S1", "evidenceRef": "SPL:doc-1#warnings", "content": "initial"}]}, refs=["SPL:doc-1#warnings"], provenance=consistent),
        envelope("OK", {"evidence": [{"referenceId": "S2", "evidenceRef": "SPL:doc-2#warnings", "content": "fallback"}]}, refs=["SPL:doc-2#warnings"], provenance=fallback),
    ]
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(responses))
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    target = next(item for item in first["findings"] if item["reviewType"] == "LABEL_EVIDENCE_REVIEW")
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": target["findingId"]}]}), config=config)
    refreshed = next(item for item in updated["findings"] if item["findingId"] == target["findingId"])
    assert refreshed["graphProvenance"]["fallbackUsed"] is True
    assert "graph_fallback_used" in refreshed["verificationWarnings"]
    gaps = [item for item in updated["findings"] if item["findingId"] == f"provenance-gap-{target['findingId']}"]
    assert len(gaps) == 1
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_reinvestigation_insufficient_facts_becomes_a_gap_not_a_tool_failure(tmp_path: Path) -> None:
    responses = standard_drug_responses({"ARNICA": mapped_response()})
    responses["facts"] = [
        envelope("OK", {"product": {"productId": "DRUG_PRODUCT::1"}}),
        envelope("INSUFFICIENT_EVIDENCE", {}, errors=["facts unavailable"]),
        envelope("OK", {"product": {"productId": "DRUG_PRODUCT::1"}}),
    ]
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(responses))
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    target = next(item for item in first["findings"] if item["reviewType"] == "LABEL_EVIDENCE_REVIEW")
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": target["findingId"]}]}), config=config)
    assert updated["status"] == "AWAITING_FINDING_REVIEW"
    assert any(
        item["reviewType"] == "EVIDENCE_GAP" and item.get("sourceTool") == "get_product_facts"
        for item in updated["findings"]
    )
    recovered = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": target["findingId"]}]}), config=config)
    assert not any(
        item["findingId"] == f"evidence-gap-get_product_facts-{target['findingId']}"
        for item in recovered["findings"]
    )
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_consistent_refresh_retires_stale_fallback_warning_and_gap(tmp_path: Path) -> None:
    consistent = {"graphBackend": "neo4j", "graphWorkspace": "dailymed", "graphDatabase": "neo4j", "fallbackUsed": False, "consistency": {"status": "CONSISTENT"}}
    fallback = {"graphBackend": "snapshot", "graphWorkspace": "dailymed", "graphDatabase": None, "fallbackUsed": True, "consistency": {"status": "UNAVAILABLE"}}
    responses = standard_drug_responses({"ARNICA": mapped_response()})
    responses["facts"] = [envelope("OK", {}, provenance=consistent), envelope("OK", {}, provenance=fallback), envelope("OK", {}, provenance=consistent)]
    responses["search"] = [
        envelope("OK", {"evidence": [{"evidenceRef": "SPL:doc-1#warnings"}]}, provenance=consistent),
        envelope("OK", {"evidence": [{"evidenceRef": "SPL:doc-2#warnings"}]}, provenance=fallback),
        envelope("OK", {"evidence": [{"evidenceRef": "SPL:doc-3#warnings"}]}, provenance=consistent),
    ]
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(responses))
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    target = next(item for item in first["findings"] if item["reviewType"] == "LABEL_EVIDENCE_REVIEW")
    fallback_state = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": target["findingId"]}]}), config=config)
    assert any(item["findingId"] == f"provenance-gap-{target['findingId']}" for item in fallback_state["findings"])
    recovered = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": target["findingId"]}]}), config=config)
    refreshed = next(item for item in recovered["findings"] if item["findingId"] == target["findingId"])
    assert refreshed["graphProvenance"]["graphBackend"] == "neo4j"
    assert "graph_fallback_used" not in refreshed["verificationWarnings"]
    assert not any(item["findingId"] == f"provenance-gap-{target['findingId']}" for item in recovered["findings"])
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_duplicate_ingredient_reinvestigation_reruns_comparison_and_retires_resolved_claim(tmp_path: Path) -> None:
    med2 = {**MED2, "medication": "ARNICA TWO", "identifiers": [{"system": "ndc", "code": "2"}]}
    responses = standard_drug_responses({"ARNICA": mapped_response(), "ARNICA TWO": mapped_response()})
    responses["compare"] = [
        envelope("OK", {"sharedActiveIngredients": [{"entityId": "INGREDIENT::ARNICA", "name": "ARNICA"}]}, refs=["SPL:doc-1#document"]),
        envelope("OK", {"sharedActiveIngredients": []}),
    ]
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, med2)), drug)
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    duplicate = next(item for item in first["findings"] if item["reviewType"] == "DUPLICATE_ACTIVE_INGREDIENT")
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": duplicate["findingId"]}]}), config=config)
    refreshed = next(item for item in updated["findings"] if item["findingId"] == duplicate["findingId"])
    assert [name for name, _ in drug.calls].count("compare_product_ingredients") == 2
    assert refreshed["status"] == "REJECTED"
    assert refreshed["sharedActiveIngredients"] == []
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_duplicate_comparison_insufficiency_blocks_parent_claim(tmp_path: Path) -> None:
    med2 = {**MED2, "medication": "ARNICA TWO", "identifiers": [{"system": "ndc", "code": "2"}]}
    responses = standard_drug_responses({"ARNICA": mapped_response(), "ARNICA TWO": mapped_response()})
    responses["compare"] = [
        envelope("OK", {"sharedActiveIngredients": [{"entityId": "INGREDIENT::ARNICA", "name": "ARNICA"}]}, refs=["SPL:doc-1#document"]),
        envelope("INSUFFICIENT_EVIDENCE", {}, errors=["comparison unavailable"]),
    ]
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, med2)), drug)
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    duplicate = next(item for item in first["findings"] if item["reviewType"] == "DUPLICATE_ACTIVE_INGREDIENT")
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": duplicate["findingId"]}]}), config=config)
    parent = next(item for item in updated["findings"] if item["findingId"] == duplicate["findingId"])
    assert parent["status"] == "NEEDS_MORE_EVIDENCE"
    assert "comparison_evidence_insufficient" in parent["verificationErrors"]
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_duplicate_comparison_fallback_provenance_is_authoritative(tmp_path: Path) -> None:
    med2 = {**MED2, "medication": "ARNICA TWO", "identifiers": [{"system": "ndc", "code": "2"}]}
    fallback = {"graphBackend": "snapshot", "graphWorkspace": "dailymed", "graphDatabase": None, "fallbackUsed": True, "consistency": {"status": "UNAVAILABLE"}}
    responses = standard_drug_responses({"ARNICA": mapped_response(), "ARNICA TWO": mapped_response()})
    responses["compare"] = [
        envelope("OK", {"sharedActiveIngredients": [{"entityId": "INGREDIENT::ARNICA", "name": "ARNICA"}]}, refs=["SPL:doc-1#document"]),
        envelope("OK", {"sharedActiveIngredients": [{"entityId": "INGREDIENT::ARNICA", "name": "ARNICA"}]}, refs=["SPL:doc-2#document"], provenance=fallback),
    ]
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, med2)), FakeDrugGateway(responses))
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    duplicate = next(item for item in first["findings"] if item["reviewType"] == "DUPLICATE_ACTIVE_INGREDIENT")
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": duplicate["findingId"]}]}), config=config)
    parent = next(item for item in updated["findings"] if item["findingId"] == duplicate["findingId"])
    assert parent["graphProvenance"]["fallbackUsed"] is True
    assert "graph_fallback_used" in parent["verificationWarnings"]
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_patient_context_gap_reinvestigation_recollects_health_context(tmp_path: Path) -> None:
    health = FakeHealthGateway(health_context(MED1, missing=["allergies"]))
    graph = build_test_graph(tmp_path, health, FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})))
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    gap = next(item for item in first["findings"] if item.get("missingField") == "allergies")
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": gap["findingId"]}]}), config=config)
    assert len(health.calls) == 2
    assert any(item["findingId"] == gap["findingId"] for item in updated["findings"])
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_resolved_patient_context_gap_is_retired_after_refresh(tmp_path: Path) -> None:
    class SequencedHealthGateway:
        def __init__(self) -> None:
            self.responses = [health_context(MED1, missing=["allergies"]), health_context(MED1)]
            self.calls: list[tuple[str | None, str | None]] = []

        async def get_review_context(self, patient_id: str | None, as_of: str | None):
            self.calls.append((patient_id, as_of))
            return self.responses.pop(0)

    health = SequencedHealthGateway()
    graph = build_test_graph(tmp_path, health, FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})))
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    gap = next(item for item in first["findings"] if item.get("missingField") == "allergies")
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": gap["findingId"]}]}), config=config)
    refreshed = next(item for item in updated["findings"] if item["findingId"] == gap["findingId"])
    assert refreshed["stillMissing"] is False
    assert refreshed["status"] == "REJECTED"
    assert refreshed["summary"].startswith("Resolved:")
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_context_refresh_reresolves_changed_medication_and_replaces_dependent_state(tmp_path: Path) -> None:
    changed = {**MED1, "medication": "METFORMIN", "identifiers": [{"system": "ndc", "code": "2"}]}
    class SequencedHealthGateway:
        def __init__(self) -> None:
            self.responses = [health_context(MED1, missing=["allergies"]), health_context(changed)]
        async def get_review_context(self, patient_id: str | None, as_of: str | None):
            return self.responses.pop(0)

    responses = standard_drug_responses({
        "ARNICA": mapped_response(),
        "METFORMIN": envelope("OK", {
            "matchClass": "EXACT_IDENTIFIER", "autoAcceptable": True,
            "selectedProductId": "DRUG_PRODUCT::2", "candidates": [], "unmatchedFields": [],
        }),
    })
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(tmp_path, SequencedHealthGateway(), drug)
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    gap = next(item for item in first["findings"] if item.get("missingField") == "allergies")
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": gap["findingId"]}]}), config=config)
    mapping = next(item for item in updated["medicationMappings"] if item["medicationId"] == "med-1")
    assert [name for name, _ in drug.calls].count("resolve_medication") == 2
    assert mapping["sourceName"] == "METFORMIN"
    assert mapping["selectedProductId"] == "DRUG_PRODUCT::2"
    assert all("DRUG_PRODUCT::1" not in item.get("productIds", []) for item in updated["evidenceIndex"])
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_partial_context_change_preserves_unaffected_finding_and_decision(tmp_path: Path) -> None:
    med2 = {**MED2, "medication": "ARNICA TWO", "identifiers": [{"system": "ndc", "code": "2"}]}
    changed_med1 = {**MED1, "medication": "METFORMIN", "identifiers": [{"system": "ndc", "code": "3"}]}
    class SequencedHealthGateway:
        def __init__(self) -> None:
            self.responses = [health_context(MED1, med2, missing=["allergies"]), health_context(changed_med1, med2)]
        async def get_review_context(self, patient_id: str | None, as_of: str | None):
            return self.responses.pop(0)

    responses = standard_drug_responses({
        "ARNICA": mapped_response(), "ARNICA TWO": mapped_response(),
        "METFORMIN": envelope("OK", {"matchClass": "EXACT_IDENTIFIER", "autoAcceptable": True, "selectedProductId": "DRUG_PRODUCT::3", "candidates": [], "unmatchedFields": []}),
    })
    graph = build_test_graph(tmp_path, SequencedHealthGateway(), FakeDrugGateway(responses))
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    unaffected = next(item for item in first["findings"] if item.get("medicationIds") == ["med-2"] and item["reviewType"] == "LABEL_EVIDENCE_REVIEW")
    gap = next(item for item in first["findings"] if item.get("missingField") == "allergies")
    await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REJECT_FINDING", "findingId": unaffected["findingId"]}]}), config=config)
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": gap["findingId"]}]}), config=config)
    preserved = next(item for item in updated["findings"] if item["findingId"] == unaffected["findingId"])
    assert preserved["status"] == "REJECTED"
    assert len([
        item for item in updated["findings"]
        if item.get("medicationIds") == ["med-2"] and item["reviewType"] == "LABEL_EVIDENCE_REVIEW"
    ]) == 1
    assert any(decision.get("findingId") == unaffected["findingId"] and decision["action"] == "REJECT_FINDING" for decision in updated["humanDecisions"])
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_initial_comparison_gap_reinvestigation_reruns_comparison_and_surfaces_duplicate(tmp_path: Path) -> None:
    med2 = {**MED2, "medication": "ARNICA TWO", "identifiers": [{"system": "ndc", "code": "2"}]}
    responses = standard_drug_responses({"ARNICA": mapped_response(), "ARNICA TWO": mapped_response()})
    responses["compare"] = [
        envelope("INSUFFICIENT_EVIDENCE", {}, errors=["comparison unavailable"]),
        envelope("OK", {"sharedActiveIngredients": [{"entityId": "INGREDIENT::ARNICA", "name": "ARNICA"}]}, refs=["SPL:comparison-doc#ingredients"]),
    ]
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, med2)), drug)
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    gap = next(item for item in first["findings"] if item.get("sourceTool") == "compare_product_ingredients")
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": gap["findingId"]}]}), config=config)
    assert [name for name, _ in drug.calls].count("compare_product_ingredients") == 2
    refreshed = next(item for item in updated["findings"] if item["findingId"] == gap["findingId"])
    assert refreshed["reviewType"] == "DUPLICATE_ACTIVE_INGREDIENT"
    assert refreshed["sharedActiveIngredients"]
    assert set(refreshed["medicationIds"]) == {"med-1", "med-2"}
    assert set(refreshed["patientEvidenceRefs"]) == {"FHIR:MedicationRequest/med-1", "FHIR:MedicationRequest/med-2"}
    assert "SPL:comparison-doc#ingredients" in refreshed["labelEvidenceRefs"]
    assert refreshed["status"] == "PENDING"
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_requesting_generated_facts_gap_uses_canonical_parent_and_retires_gap(tmp_path: Path) -> None:
    responses = standard_drug_responses({"ARNICA": mapped_response()})
    responses["facts"] = [
        envelope("OK", {}), envelope("INSUFFICIENT_EVIDENCE", {}, errors=["missing"]),
        envelope("OK", {}),
    ]
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), FakeDrugGateway(responses))
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    parent = next(item for item in first["findings"] if item["reviewType"] == "LABEL_EVIDENCE_REVIEW")
    with_gap = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": parent["findingId"]}]}), config=config)
    gap = next(item for item in with_gap["findings"] if item.get("sourceTool") == "get_product_facts")
    recovered = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": gap["findingId"]}]}), config=config)
    assert not any(item["findingId"] == gap["findingId"] for item in recovered["findings"])
    assert any(item["findingId"] == parent["findingId"] for item in recovered["findings"])
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_unmapped_gap_reinvestigation_reruns_mapping_and_reuses_finding_id(tmp_path: Path) -> None:
    responses = standard_drug_responses({
        "METFORMIN": [
            envelope("UNMAPPED", {"matchClass": "UNMAPPED", "selectedProductId": None, "candidates": [], "unmatchedFields": []}),
            mapped_response(),
        ],
    })
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED2)), drug)
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    gap = next(item for item in first["findings"] if item["reviewType"] == "EVIDENCE_GAP")
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": gap["findingId"]}]}), config=config)
    refreshed = next(item for item in updated["findings"] if item["findingId"] == gap["findingId"])
    assert [name for name, _ in drug.calls].count("resolve_medication") == 2
    assert refreshed["reviewType"] == "LABEL_EVIDENCE_REVIEW"
    assert refreshed["selectedProductIds"] == ["DRUG_PRODUCT::1"]
    await graph.checkpointer.conn.close()
