import asyncio
from pathlib import Path

import pytest
from langgraph.types import Command

from medication_review_agent.gateways import TimedToolResult, ToolContractError
from medication_review_agent.models import (
    ModelCallRecord,
    ReviewIntent,
    ReviewPlanItem,
    ReviewSnapshot,
    ReviewStatus,
    ReviewTopic,
)
from medication_review_agent.planner import PlanningResult
from medication_review_agent.retrieval import (
    EvidenceGrade,
    GradingOutcome,
    retrieval_attempt_key,
)
from medication_review_agent.repository import ReviewRepository
from medication_review_agent.workflow import deidentified_patient_features, state_to_snapshot

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


class RecordingPlanner:
    def __init__(self, result: PlanningResult) -> None:
        self.result = result
        self.calls = 0
        self.question = None
        self.patient_features = None
        self.mappings = None
        self.missing_fields = None

    async def plan(self, question, patient_features, mappings, missing_fields):
        self.calls += 1
        self.question = question
        self.patient_features = patient_features
        self.mappings = mappings
        self.missing_fields = missing_fields
        return self.result


def planning_result(*topics: ReviewTopic, with_model_call: bool = True) -> PlanningResult:
    selected_topics = list(topics or (ReviewTopic.STORAGE,))
    intent = ReviewIntent(
        topics=selected_topics,
        requiresNarrativeEvidence=True,
        rationale="核查目标主题",
        confidence=0.91,
        modelId="test-model" if with_model_call else None,
        promptVersion="intent-v1" if with_model_call else "deterministic-v1",
    )
    return PlanningResult(
        intent=intent,
        items=[ReviewPlanItem(
            planItemId=f"topic-{selected_topics[0].value}",
            reviewType=selected_topics[0].value.upper(),
            topics=[selected_topics[0].value],
            rationale=intent.rationale,
        )],
        modelCall=ModelCallRecord(
            modelId="test-model",
            promptVersion="intent-v1",
            inputTokens=17,
            outputTokens=5,
            estimatedCost=0.0,
            latencyMs=12,
        ) if with_model_call else None,
    )


def test_workflow_reexports_all_interrupt_contracts() -> None:
    from medication_review_agent.workflow import FindingDecision as compatibility_type
    from medication_review_agent.workflow_state import FindingDecision as extracted_type

    assert compatibility_type is extracted_type


def test_deidentified_features_drop_nested_identity_values() -> None:
    features = deidentified_patient_features({
        "patient": {"id": "patient-secret", "name": "person-secret", "age": 42},
        "allergies": [
            {"substance": "aspirin", "patientName": "person-secret"},
            {"substance": {"display": "penicillin", "patientId": "patient-secret"}},
        ],
        "specialPopulations": [
            "pregnant",
            {"flag": "older_adult", "patientId": "patient-secret"},
        ],
    })

    assert features == {
        "ageBand": "adult",
        "allergyTerms": ["aspirin"],
        "specialPopulationFlags": ["pregnant"],
    }
    assert "secret" not in str(features)


@pytest.mark.parametrize(("review_id", "question"), [
    ("stop-zh", "告诉患者立即停药"),
    ("stop-en", "Tell the patient to stop the medication."),
    ("switch-zh", "请把患者换成另一个药"),
    ("switch-en", "Switch the patient to another medication."),
    ("dose-zh", "我是否应该减少剂量？"),
    ("dose-en", "Should I reduce my dose?"),
    ("diagnosis-zh", "请直接诊断患者"),
    ("diagnosis-en", "Please diagnose the patient."),
    ("prescribe-zh", "请给患者开药"),
    ("prescribe-en", "Prescribe antibiotics for the patient."),
])
@pytest.mark.asyncio
async def test_unsafe_question_stops_before_model_and_all_tools(
    tmp_path: Path, review_id: str, question: str,
) -> None:
    class UnexpectedPlanner:
        def __init__(self) -> None:
            self.calls = 0

        async def plan(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("planner must not be called for an unsafe question")

    planner = UnexpectedPlanner()
    health = FakeHealthGateway(health_context(MED1))
    drug = FakeDrugGateway({})
    graph = build_test_graph(
        tmp_path,
        health,
        drug,
        review_id=review_id,
        planner=planner,
    )

    state = await graph.ainvoke(
        {
            "reviewId": review_id,
            "question": question,
        },
        config={"configurable": {"thread_id": review_id}},
    )

    assert state["status"] == "CANCELLED"
    assert state["questionSafety"]["code"] == "UNSAFE_CLINICAL_ACTION_REQUEST"
    assert state["unresolvedItems"] == [{
        "kind": "SCOPE_LIMITATION",
        "code": "UNSAFE_CLINICAL_ACTION_REQUEST",
        "summary": "系统只能整理证据并交由药师审核，不能给出患者级诊疗动作。",
    }]
    assert planner.calls == 0
    assert health.calls == []
    assert drug.calls == []
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_missing_graph_question_is_loaded_from_durable_review_before_tools(
    tmp_path: Path,
) -> None:
    planner = object()
    health = FakeHealthGateway(health_context(MED1))
    drug = FakeDrugGateway({})
    graph = build_test_graph(
        tmp_path,
        health,
        drug,
        review_id="durable-unsafe",
        question="我是否应该停止服用这个药？",
        planner=planner,
    )

    state = await graph.ainvoke(
        {"reviewId": "durable-unsafe", "patientRef": "P001"},
        config={"configurable": {"thread_id": "durable-unsafe"}},
    )

    assert state["question"] == "我是否应该停止服用这个药？"
    assert state["status"] == "CANCELLED"
    assert health.calls == []
    assert drug.calls == []
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_stop_use_evidence_question_reaches_health_tools(tmp_path: Path) -> None:
    health = FakeHealthGateway(health_context(MED1))
    graph = build_test_graph(
        tmp_path,
        health,
        FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
        review_id="safe-stop-use",
    )

    state = await graph.ainvoke(
        {
            "reviewId": "safe-stop-use",
            "question": "核查标签 stop use 章节并展示原文",
            "patientRef": "P001",
        },
        config={"configurable": {"thread_id": "safe-stop-use"}},
    )

    assert state["questionSafety"]["code"] == "ALLOWED_EVIDENCE_REVIEW"
    assert health.calls == [("P001", None)]
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_question_is_parsed_with_deidentified_context_and_metadata_survives_checkpoint(
    tmp_path: Path,
) -> None:
    planner = RecordingPlanner(planning_result(ReviewTopic.STORAGE))
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1)),
        FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
        review_id="review-plan",
        question="核查储存条件",
        planner=planner,
    )
    config = {"configurable": {"thread_id": "review-plan"}}

    state = await graph.ainvoke(
        {"reviewId": "review-plan", "question": "核查储存条件", "patientRef": "P001"},
        config=config,
    )
    checkpoint = await graph.aget_state(config)

    assert planner.calls == 1
    assert planner.question == "核查储存条件"
    assert planner.patient_features == {
        "ageBand": "adult",
        "allergyTerms": [],
        "specialPopulationFlags": [],
    }
    assert planner.mappings[0].selectedProductId == "DRUG_PRODUCT::1"
    assert state["intent"]["topics"] == ["storage"]
    assert state["modelCalls"][0]["promptVersion"] == "intent-v1"
    assert checkpoint.values["intent"] == state["intent"]
    assert checkpoint.values["modelCalls"] == state["modelCalls"]
    assert state["reviewPlan"][0]["medicationIds"] == ["med-1"]

    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    projected = state_to_snapshot(repository.get("review-plan"), checkpoint.values)
    assert projected.intent is not None
    assert projected.intent.topics == [ReviewTopic.STORAGE]
    assert projected.modelCalls[0].modelId == "test-model"
    model_audit = next(
        item for item in repository.list_audit("review-plan")
        if item["node"] == "parse_review_goal"
    )
    assert model_audit["argumentSummary"] == {"topicCount": 1, "medicationCount": 1}
    assert model_audit["modelId"] == "test-model"
    assert model_audit["promptVersion"] == "intent-v1"
    assert model_audit["modelFallback"] is False
    assert "核查储存条件" not in str(model_audit)
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_plan_adds_only_topic_relevant_missing_field_findings(tmp_path: Path) -> None:
    planner = RecordingPlanner(planning_result(ReviewTopic.STORAGE, with_model_call=False))
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(
            MED1,
            missing=["allergies", "activeMedications.med-1.route"],
        )),
        FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
        review_id="review-storage",
        question="核查储存条件",
        planner=planner,
    )

    state = await graph.ainvoke(
        {"reviewId": "review-storage", "question": "核查储存条件", "patientRef": "P001"},
        config={"configurable": {"thread_id": "review-storage"}},
    )

    assert not [item for item in state["findings"] if item.get("missingField")]
    await graph.checkpointer.conn.close()


def test_patient_candidates_survive_snapshot_projection() -> None:
    existing = ReviewSnapshot(
        reviewId="review-1",
        status=ReviewStatus.RUNNING,
        question="默认用药证据核查",
    )
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
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), drug)
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
async def test_non_transient_tool_error_is_not_retried(tmp_path: Path) -> None:
    class BrokenHealth:
        def __init__(self) -> None:
            self.calls = 0

        async def get_review_context(self, patient_id, as_of):
            self.calls += 1
            raise RuntimeError("programming error")

    health = BrokenHealth()
    graph = build_test_graph(tmp_path, health, FakeDrugGateway({}))

    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )

    assert result["status"] == "BLOCKED_TOOL_ERROR"
    assert health.calls == 1
    assert result["metrics"]["retries"] == 0
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
async def test_computed_claim_carries_rule_inputs_and_both_reference_sets(
    tmp_path: Path,
) -> None:
    second_mapping = envelope("OK", {
        "matchClass": "EXACT_IDENTIFIER",
        "autoAcceptable": True,
        "selectedProductId": "DRUG_PRODUCT::2",
        "candidates": [],
        "unmatchedFields": [],
    }, provenance={
        "graphBackend": "neo4j",
        "graphWorkspace": "dailymed",
        "graphDatabase": "neo4j",
        "fallbackUsed": False,
        "consistency": {"status": "CONSISTENT"},
    })
    responses = standard_drug_responses({
        "ARNICA": mapped_response(),
        "METFORMIN": second_mapping,
    })
    responses["compare"] = envelope("OK", {
        "products": ["DRUG_PRODUCT::1", "DRUG_PRODUCT::2"],
        "sharedActiveIngredients": [
            {"entityId": "INGREDIENT::ARNICA", "name": "Arnica"},
        ],
    }, refs=["SPL:doc-1#document"], provenance={
        "graphBackend": "neo4j",
        "graphWorkspace": "dailymed",
        "graphDatabase": "neo4j",
        "fallbackUsed": False,
        "consistency": {"status": "CONSISTENT"},
    })
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1, MED2)),
        drug,
    )

    await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )

    claims = next(payload for name, payload in drug.calls if name == "validate_evidence")
    computed = next(
        item for item in claims if item["reviewType"] == "DUPLICATE_ACTIVE_INGREDIENT"
    )
    assert computed["ruleId"] == "shared-active-ingredient-v1"
    assert computed["normalizationVersion"] == "normalization-v1"
    assert computed["comparisonInputs"] == {
        "productIds": ["DRUG_PRODUCT::1", "DRUG_PRODUCT::2"],
        "sharedActiveIngredientIds": ["INGREDIENT::ARNICA"],
    }
    assert computed["patientEvidenceRefs"] == [
        "FHIR:MedicationRequest/med-1",
        "FHIR:MedicationRequest/med-2",
    ]
    assert computed["labelEvidenceRefs"] == ["SPL:doc-1#document"]
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
        "referenceId": "S1", "productId": "DRUG_PRODUCT::1",
        "documentId": "doc-1", "documentVersion": "3",
        "effectiveTime": "20260831", "sectionId": "warnings",
        "sectionCode": "34071-1", "topic": "warnings",
        "sourcePath": "labels/doc-1.xml", "contentHash": "a" * 64,
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
    assert any(
        item["reviewType"] == "EVIDENCE_GAP"
        and (item.get("graphProvenance") or {}).get("fallbackUsed")
        for item in result["findings"]
    )
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
async def test_mixed_reinvestigation_requests_run_context_and_evidence_paths(
    tmp_path: Path,
) -> None:
    health = FakeHealthGateway(health_context(MED1, missing=["allergies"]))
    drug = FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()}))
    graph = build_test_graph(tmp_path, health, drug)
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config=config,
    )
    context_gap = next(
        item for item in first["findings"] if item.get("missingField") == "allergies"
    )
    label_finding = next(
        item for item in first["findings"]
        if item["reviewType"] == "LABEL_EVIDENCE_REVIEW"
    )

    updated = await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW",
        "reviewerId": "pharmacist-demo",
        "decisions": [
            {
                "action": "REQUEST_MORE_EVIDENCE",
                "findingId": context_gap["findingId"],
            },
            {
                "action": "REQUEST_MORE_EVIDENCE",
                "findingId": label_finding["findingId"],
            },
        ],
    }), config=config)

    assert len(health.calls) == 2
    assert [name for name, _ in drug.calls].count("search_label_evidence") == 2
    by_id = {item["findingId"]: item for item in updated["findings"]}
    assert by_id[context_gap["findingId"]]["status"] == "PENDING"
    assert by_id[label_finding["findingId"]]["status"] == "PENDING"
    assert updated["reinvestigateFindingIds"] == []
    assert updated["reinvestigationCounts"] == {
        context_gap["findingId"]: 1,
        label_finding["findingId"]: 1,
    }
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_context_change_still_consumes_queued_evidence_reinvestigation(
    tmp_path: Path,
) -> None:
    med2 = {
        **MED2,
        "medication": "ARNICA TWO",
        "identifiers": [{"system": "ndc", "code": "2"}],
    }
    changed_med1 = {
        **MED1,
        "medication": "METFORMIN",
        "identifiers": [{"system": "ndc", "code": "3"}],
    }

    class SequencedHealthGateway:
        def __init__(self) -> None:
            self.responses = [
                health_context(MED1, med2, missing=["allergies"]),
                health_context(changed_med1, med2),
            ]

        async def get_review_context(self, patient_id, as_of):
            return self.responses.pop(0)

    mapped_three = envelope("OK", {
        "matchClass": "EXACT_IDENTIFIER",
        "autoAcceptable": True,
        "selectedProductId": "DRUG_PRODUCT::3",
        "candidates": [],
        "unmatchedFields": [],
    })
    responses = standard_drug_responses({
        "ARNICA": mapped_response(),
        "ARNICA TWO": mapped_response(),
        "METFORMIN": mapped_three,
    })
    fact_one, fact_three = responses["facts"]
    search_one, search_three = responses["search"]
    refreshed_search_one = envelope("OK", {"evidence": [{
        **item,
        "referenceId": f"{item['referenceId']}-refreshed",
        "evidenceRef": f"{item['evidenceRef']}-refreshed",
        "content": f"Refreshed {item['topic']} evidence.",
    } for item in search_one.envelope.data["evidence"]]})
    responses["facts"] = [fact_one, fact_one, fact_three]
    responses["search"] = [search_one, refreshed_search_one, search_three]
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(
        tmp_path,
        SequencedHealthGateway(),
        drug,
    )
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({
        "reviewId": "review-1",
        "patientRef": "P001",
        "asOf": "2026-08-31",
    }, config=config)
    context_gap = next(
        item for item in first["findings"] if item.get("missingField") == "allergies"
    )
    unchanged_label = next(
        item for item in first["findings"]
        if item.get("medicationIds") == ["med-2"]
        and item["reviewType"] == "LABEL_EVIDENCE_REVIEW"
    )
    old_refs = set(unchanged_label["labelEvidenceRefs"])

    updated = await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW",
        "reviewerId": "pharmacist-demo",
        "decisions": [
            {
                "action": "REQUEST_MORE_EVIDENCE",
                "findingId": context_gap["findingId"],
            },
            {
                "action": "REQUEST_MORE_EVIDENCE",
                "findingId": unchanged_label["findingId"],
            },
        ],
    }), config=config)

    assert [name for name, _ in drug.calls].count("get_product_facts") == 3
    assert [name for name, _ in drug.calls].count("search_label_evidence") == 3
    assert updated["reinvestigateFindingIds"] == [], {
        "queued": updated["reinvestigateFindingIds"],
        "target": next(
            (
                item for item in updated["findings"]
                if item["findingId"] == unchanged_label["findingId"]
            ),
            None,
        ),
    }
    refreshed = next(
        item for item in updated["findings"]
        if item["findingId"] == unchanged_label["findingId"]
    )
    assert refreshed["status"] == "PENDING"
    assert old_refs.isdisjoint(refreshed["labelEvidenceRefs"])
    assert refreshed["labelEvidenceRefs"]
    assert refreshed["labelEvidenceIds"]
    refreshed_by_id = {
        item["evidenceId"]: item for item in updated["evidenceIndex"]
    }
    assert {
        refreshed_by_id[evidence_id]["evidenceRef"]
        for evidence_id in refreshed["labelEvidenceIds"]
    } == set(refreshed["labelEvidenceRefs"])
    assert all(
        reference.endswith("-refreshed")
        for reference in refreshed["labelEvidenceRefs"]
    )
    assert not any(
        item.get("unresolvedReason") == "RETRIEVAL_BUDGET_EXHAUSTED"
        for item in updated["findings"]
    )
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_bounded_retrieval_persists_attempts_and_explicit_provenance(
    tmp_path: Path,
) -> None:
    responses = standard_drug_responses({"ARNICA": mapped_response()})
    planner = RecordingPlanner(planning_result(
        ReviewTopic.WARNINGS,
        with_model_call=False,
    ))
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1)),
        FakeDrugGateway(responses),
        planner=planner,
    )

    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )

    key = retrieval_attempt_key("DRUG_PRODUCT::1", ReviewTopic.WARNINGS)
    narrative = next(item for item in result["evidenceIndex"] if item["source"] == "SPL")
    assert result["retrievalAttempts"] == {key: 1}
    assert narrative["documentId"] == "doc-1"
    assert narrative["documentVersion"] == "3"
    assert narrative["sectionId"] == "warnings"
    assert narrative["sourcePath"] == "labels/doc-1.xml"
    assert narrative["contentHash"] == "a" * 64
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_each_finding_allows_only_one_human_reinvestigation(tmp_path: Path) -> None:
    drug = FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()}))
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1)),
        drug,
    )
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config=config,
    )
    target = next(
        item for item in first["findings"]
        if item["reviewType"] == "LABEL_EVIDENCE_REVIEW"
    )

    once = await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW",
        "reviewerId": "pharmacist-demo",
        "decisions": [{
            "action": "REQUEST_MORE_EVIDENCE",
            "findingId": target["findingId"],
        }],
    }), config=config)
    calls_after_once = len(drug.calls)
    twice = await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW",
        "reviewerId": "pharmacist-demo",
        "decisions": [{
            "action": "REQUEST_MORE_EVIDENCE",
            "findingId": target["findingId"],
        }],
    }), config=config)

    assert twice["reinvestigationCounts"][target["findingId"]] == 1
    assert len(drug.calls) == calls_after_once
    blocked = next(
        item for item in twice["findings"]
        if item["findingId"] == target["findingId"]
    )
    assert blocked["status"] == "NEEDS_MORE_EVIDENCE"
    assert "REINVESTIGATION_BUDGET_EXHAUSTED" in blocked["verificationErrors"]
    assert once["reinvestigationCounts"][target["findingId"]] == 1
    await graph.checkpointer.conn.close()


def model_calls(count: int) -> list[dict[str, object]]:
    return [ModelCallRecord(
        modelId="test-model",
        promptVersion=f"prior-{index}",
        inputTokens=1,
        outputTokens=1,
        estimatedCost=0,
        latencyMs=1,
    ).model_dump(mode="json") for index in range(count)]


@pytest.mark.asyncio
async def test_model_budget_stops_planner_before_fourth_call(tmp_path: Path) -> None:
    class UnexpectedPlanner:
        def __init__(self) -> None:
            self.calls = 0

        async def plan(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("planner must not exceed the review model budget")

    planner = UnexpectedPlanner()
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1)),
        FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
        planner=planner,
    )

    result = await graph.ainvoke({
        "reviewId": "review-1",
        "patientRef": "P001",
        "asOf": "2026-08-31",
        "modelCalls": model_calls(3),
    }, config={"configurable": {"thread_id": "review-1"}})

    assert planner.calls == 0
    assert len(result["modelCalls"]) == 3
    assert any(
        item.get("unresolvedReason") == "MODEL_CALL_BUDGET_EXHAUSTED"
        for item in result["unresolvedItems"]
    )
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_model_budget_stops_grader_before_fourth_call(tmp_path: Path) -> None:
    class UnexpectedGrader:
        def __init__(self) -> None:
            self.calls = 0

        async def grade(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("grader must not exceed the review model budget")

    responses = standard_drug_responses({"ARNICA": mapped_response()})
    grader = UnexpectedGrader()
    intent = ReviewIntent(
        topics=[ReviewTopic.WARNINGS, ReviewTopic.STORAGE],
        requiresNarrativeEvidence=True,
        rationale="核查警告",
        confidence=1,
        promptVersion="existing-v1",
    )
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1)),
        FakeDrugGateway(responses),
        grader=grader,
    )

    result = await graph.ainvoke({
        "reviewId": "review-1",
        "patientRef": "P001",
        "asOf": "2026-08-31",
        "intent": intent.model_dump(mode="json"),
        "modelCalls": model_calls(3),
    }, config={"configurable": {"thread_id": "review-1"}})

    assert grader.calls == 0
    assert len(result["modelCalls"]) == 3
    assert any(
        item.get("unresolvedReason") == "MODEL_CALL_BUDGET_EXHAUSTED"
        for item in result["unresolvedItems"]
    )
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_rewrite_timeout_does_not_exceed_two_physical_label_searches(
    tmp_path: Path,
) -> None:
    class TimeoutOnRewriteGateway(FakeDrugGateway):
        async def search_label_evidence(self, product_ids, topics, question):
            self.calls.append(("search_label_evidence", product_ids))
            if [name for name, _ in self.calls].count("search_label_evidence") == 1:
                result = self._take("search")
                evidence = [
                    item for item in result.envelope.data["evidence"]
                    if item.get("topic") == "warnings"
                ]
                return TimedToolResult(
                    envelope=result.envelope.model_copy(update={
                        "data": {"evidence": evidence},
                    }),
                    latency_ms=result.latency_ms,
                )
            raise TimeoutError("search timed out")

    class RewriteGrader:
        async def grade(self, *args, **kwargs):
            return GradingOutcome(
                grade=EvidenceGrade(
                    sufficient=False,
                    coveredTopics=[ReviewTopic.WARNINGS],
                    missingTopics=[ReviewTopic.STORAGE],
                    reason="storage evidence is missing",
                    rewrittenQuestion="Find storage evidence",
                ),
                modelCall=ModelCallRecord(
                    modelId="test-model",
                    promptVersion="evidence-grader-v1",
                    inputTokens=5,
                    outputTokens=3,
                    estimatedCost=0,
                    latencyMs=2,
                ),
            )

    responses = standard_drug_responses({"ARNICA": mapped_response()})
    drug = TimeoutOnRewriteGateway(responses)
    intent = ReviewIntent(
        topics=[ReviewTopic.WARNINGS, ReviewTopic.STORAGE],
        requiresNarrativeEvidence=True,
        rationale="核查警告和储存",
        confidence=1,
        promptVersion="existing-v1",
    )
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1)),
        drug,
        grader=RewriteGrader(),
    )

    result = await graph.ainvoke({
        "reviewId": "review-1",
        "patientRef": "P001",
        "asOf": "2026-08-31",
        "intent": intent.model_dump(mode="json"),
    }, config={"configurable": {"thread_id": "review-1"}})

    assert result["status"] == "BLOCKED_TOOL_ERROR"
    assert [name for name, _ in drug.calls].count("search_label_evidence") == 2
    assert result["retrievalAttempts"]
    assert set(result["retrievalAttempts"].values()) == {2}
    assert len(result["modelCalls"]) == 1
    assert result["modelCalls"][0]["modelId"] == "test-model"
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_comparison_failure_preserves_prior_retrieval_and_model_counters(
    tmp_path: Path,
) -> None:
    class AuditedGapGrader:
        async def grade(self, *args, **kwargs):
            return GradingOutcome(
                grade=EvidenceGrade(
                    sufficient=False,
                    coveredTopics=[ReviewTopic.WARNINGS],
                    missingTopics=[ReviewTopic.STORAGE],
                    reason="storage evidence is missing",
                ),
                modelCall=ModelCallRecord(
                    modelId="test-model",
                    promptVersion="evidence-grader-v1",
                    inputTokens=5,
                    outputTokens=3,
                    estimatedCost=0,
                    latencyMs=2,
                ),
            )

    med2 = {
        **MED2,
        "medication": "ARNICA TWO",
        "identifiers": [{"system": "ndc", "code": "2"}],
    }
    mapped_two = envelope("OK", {
        "matchClass": "EXACT_IDENTIFIER",
        "autoAcceptable": True,
        "selectedProductId": "DRUG_PRODUCT::2",
        "candidates": [],
        "unmatchedFields": [],
    })
    responses = standard_drug_responses({
        "ARNICA": mapped_response(),
        "ARNICA TWO": mapped_two,
    })
    responses["compare"] = envelope(
        "ERROR", {}, errors=["comparison unavailable"]
    )
    intent = ReviewIntent(
        topics=[ReviewTopic.WARNINGS, ReviewTopic.STORAGE],
        requiresNarrativeEvidence=True,
        rationale="核查警告和储存",
        confidence=1,
        promptVersion="existing-v1",
    )
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1, med2)),
        FakeDrugGateway(responses),
        grader=AuditedGapGrader(),
    )

    result = await graph.ainvoke({
        "reviewId": "review-1",
        "patientRef": "P001",
        "asOf": "2026-08-31",
        "intent": intent.model_dump(mode="json"),
    }, config={"configurable": {"thread_id": "review-1"}})

    assert result["status"] == "BLOCKED_TOOL_ERROR"
    assert result["retrievalAttempts"]
    assert set(result["retrievalAttempts"].values()) == {1}
    assert len(result["modelCalls"]) == 2
    assert result["metrics"]["inputTokens"] == 10
    assert result["metrics"]["outputTokens"] == 6
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_grader_model_call_is_audited_without_aborting_workflow(
    tmp_path: Path,
) -> None:
    class AuditedGrader:
        async def grade(self, *args, **kwargs):
            return GradingOutcome(
                grade=EvidenceGrade(
                    sufficient=True,
                    coveredTopics=[ReviewTopic.WARNINGS],
                    missingTopics=[],
                    reason="warning evidence is sufficient",
                ),
                modelCall=ModelCallRecord(
                    modelId="test-model",
                    promptVersion="evidence-grader-v1",
                    inputTokens=11,
                    outputTokens=7,
                    estimatedCost=0,
                    latencyMs=4,
                ),
            )

    responses = standard_drug_responses({"ARNICA": mapped_response()})
    intent = ReviewIntent(
        topics=[ReviewTopic.WARNINGS, ReviewTopic.STORAGE],
        requiresNarrativeEvidence=True,
        rationale="核查警告",
        confidence=1,
        promptVersion="existing-v1",
    )
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1)),
        FakeDrugGateway(responses),
        grader=AuditedGrader(),
    )

    result = await graph.ainvoke({
        "reviewId": "review-1",
        "patientRef": "P001",
        "asOf": "2026-08-31",
        "intent": intent.model_dump(mode="json"),
    }, config={"configurable": {"thread_id": "review-1"}})

    assert len(result["modelCalls"]) == 1
    audit = ReviewRepository(tmp_path / "reviews.sqlite").list_audit("review-1")
    grader_audit = next(item for item in audit if item["promptVersion"] == "evidence-grader-v1")
    assert grader_audit["argumentSummary"] == {"topicCount": 2, "evidenceCount": 1}
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_reinvestigation_rejects_cross_product_evidence(
    tmp_path: Path,
) -> None:
    responses = standard_drug_responses({"ARNICA": mapped_response()})
    initial = responses["search"]
    malicious = dict(initial.envelope.data["evidence"][0])
    malicious.update({
        "referenceId": "cross-product",
        "evidenceRef": "SPL:doc-1#cross-product",
        "productId": "DRUG_PRODUCT::B",
    })
    responses["search"] = [initial, envelope("OK", {"evidence": [malicious]})]
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1)),
        FakeDrugGateway(responses),
    )
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config=config,
    )
    target = next(
        item for item in first["findings"]
        if item["reviewType"] == "LABEL_EVIDENCE_REVIEW"
    )

    updated = await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW",
        "reviewerId": "pharmacist-demo",
        "decisions": [{
            "action": "REQUEST_MORE_EVIDENCE",
            "findingId": target["findingId"],
        }],
    }), config=config)

    assert not any(
        item.get("evidenceRef") == "SPL:doc-1#cross-product"
        for item in updated["evidenceIndex"]
    )
    assert any(
        item.get("unresolvedReason") == "OUT_OF_SCOPE_EVIDENCE"
        for item in updated["findings"]
        if item["reviewType"] == "EVIDENCE_GAP"
    )
    await graph.checkpointer.conn.close()


class ReverseCompletionDrugGateway:
    def __init__(self, medication_count: int) -> None:
        self.medication_count = medication_count
        self.active = {"resolve": 0, "facts": 0}
        self.peak = {"resolve": 0, "facts": 0}
        self.started = {"resolve": [], "facts": []}
        self.ready = {"resolve": asyncio.Event(), "facts": asyncio.Event()}

    async def _bounded_phase(self, phase: str, order: int) -> None:
        self.started[phase].append(order)
        self.active[phase] += 1
        self.peak[phase] = max(self.peak[phase], self.active[phase])
        if self.active[phase] == 4:
            self.ready[phase].set()
        await asyncio.wait_for(self.ready[phase].wait(), timeout=1)
        await asyncio.sleep((self.medication_count - order) * 0.001)
        self.active[phase] -= 1

    async def resolve_medication(self, **kwargs):
        order = int(kwargs["name"].removeprefix("DRUG-"))
        await self._bounded_phase("resolve", order)
        return envelope("OK", {
            "matchClass": "EXACT_IDENTIFIER",
            "autoAcceptable": True,
            "selectedProductId": f"DRUG_PRODUCT::{order}",
            "candidates": [],
            "unmatchedFields": [],
        })

    async def get_product_facts(self, product_id: str):
        order = int(product_id.removeprefix("DRUG_PRODUCT::"))
        await self._bounded_phase("facts", order)
        return envelope("OK", {"product": {
            "productId": product_id,
            "documentId": f"doc-{order}",
            "documentVersion": "1",
            "effectiveTime": "20260831",
            "sourcePath": f"labels/doc-{order}.xml",
            "contentHash": f"{order:x}" * 64,
        }})

    async def search_label_evidence(self, product_ids, topics, question):
        product_id = product_ids[0]
        order = int(product_id.removeprefix("DRUG_PRODUCT::"))
        return envelope("OK", {"evidence": [{
            "referenceId": f"S-{order}-{topic}",
            "evidenceRef": f"SPL:doc-{order}#{topic}",
            "productId": product_id,
            "documentId": f"doc-{order}",
            "documentVersion": "1",
            "effectiveTime": "20260831",
            "sectionId": topic,
            "sectionCode": "34071-1",
            "sourcePath": f"labels/doc-{order}.xml",
            "contentHash": f"{order:x}" * 64,
            "topic": topic,
            "content": f"Evidence for {topic}",
        } for topic in topics]})

    async def compare_product_ingredients(self, product_ids):
        return envelope("OK", {"sharedActiveIngredients": []})

    async def validate_evidence(self, claims):
        return envelope("OK", {"claims": [
            {**claim, "valid": True, "errors": []}
            for claim in claims
        ]})


@pytest.mark.asyncio
async def test_resolution_and_fact_reads_are_bounded_and_merge_deterministically(
    tmp_path: Path,
) -> None:
    async def run_once(run_name: str):
        run_path = tmp_path / run_name
        run_path.mkdir()
        medications = [{
            "id": f"med-{index}",
            "medication": f"DRUG-{index}",
            "identifiers": [{"system": "ndc", "code": str(index)}],
            "evidenceRef": f"FHIR:MedicationRequest/med-{index}",
        } for index in reversed(range(6))]
        drug = ReverseCompletionDrugGateway(len(medications))
        graph = build_test_graph(
            run_path,
            FakeHealthGateway(health_context(*medications)),
            drug,
            planner=RecordingPlanner(planning_result(
                ReviewTopic.WARNINGS,
                with_model_call=False,
            )),
        )
        result = await asyncio.wait_for(graph.ainvoke(
            {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
            config={"configurable": {"thread_id": "review-1"}},
        ), timeout=3)
        await graph.checkpointer.conn.close()
        return result, drug

    first, first_drug = await run_once("first")
    second, second_drug = await run_once("second")

    expected_medications = [f"med-{index}" for index in range(6)]
    expected_products = [f"DRUG_PRODUCT::{index}" for index in range(6)]
    assert [item["medicationId"] for item in first["medicationMappings"]] == expected_medications
    assert [
        item["productIds"][0]
        for item in first["evidenceIndex"]
        if item["source"] == "SPL-GRAPH" and item["topic"] == "product_facts"
    ] == expected_products
    assert first["medicationMappings"] == second["medicationMappings"]
    assert first_drug.peak == {"resolve": 4, "facts": 4}
    assert second_drug.peak == {"resolve": 4, "facts": 4}
    assert first_drug.started == {"resolve": list(range(6)), "facts": list(range(6))}
    assert second_drug.started == {"resolve": list(range(6)), "facts": list(range(6))}


@pytest.mark.asyncio
async def test_completed_parallel_fact_calls_are_all_included_in_metrics(
    tmp_path: Path,
) -> None:
    med2 = {
        **MED2,
        "medication": "ARNICA TWO",
        "identifiers": [{"system": "ndc", "code": "2"}],
    }
    mapped_two = envelope("OK", {
        "matchClass": "EXACT_IDENTIFIER",
        "autoAcceptable": True,
        "selectedProductId": "DRUG_PRODUCT::2",
        "candidates": [],
        "unmatchedFields": [],
    })
    responses = standard_drug_responses({
        "ARNICA": mapped_response(),
        "ARNICA TWO": mapped_two,
    })
    _, successful_fact = responses["facts"]
    responses["facts"] = [
        envelope("ERROR", {}, errors=["facts unavailable"]),
        successful_fact,
    ]
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1, med2)),
        FakeDrugGateway(responses),
    )

    result = await graph.ainvoke({
        "reviewId": "review-1",
        "patientRef": "P001",
        "asOf": "2026-08-31",
    }, config={"configurable": {"thread_id": "review-1"}})

    audits = ReviewRepository(tmp_path / "reviews.sqlite").list_audit("review-1")
    audited_tool_latency = sum(
        event["latencyMs"] for event in audits if event.get("tool")
    )
    assert result["status"] == "BLOCKED_TOOL_ERROR"
    assert [event["tool"] for event in audits].count("get_product_facts") == 2
    assert result["metrics"]["toolLatencyMs"] == audited_tool_latency
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_completed_parallel_resolution_calls_are_all_included_in_metrics(
    tmp_path: Path,
) -> None:
    med2 = {
        **MED2,
        "medication": "ARNICA TWO",
        "identifiers": [{"system": "ndc", "code": "2"}],
    }
    mapped_two = envelope("OK", {
        "matchClass": "EXACT_IDENTIFIER",
        "autoAcceptable": True,
        "selectedProductId": "DRUG_PRODUCT::2",
        "candidates": [],
        "unmatchedFields": [],
    })
    responses = standard_drug_responses({
        "ARNICA": envelope(
            "ERROR", {}, errors=["resolution unavailable"]
        ),
        "ARNICA TWO": mapped_two,
    })
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1, med2)),
        FakeDrugGateway(responses),
    )

    result = await graph.ainvoke({
        "reviewId": "review-1",
        "patientRef": "P001",
        "asOf": "2026-08-31",
    }, config={"configurable": {"thread_id": "review-1"}})

    audits = ReviewRepository(tmp_path / "reviews.sqlite").list_audit("review-1")
    audited_tool_latency = sum(
        event["latencyMs"] for event in audits if event.get("tool")
    )
    assert result["status"] == "BLOCKED_TOOL_ERROR"
    assert [event["tool"] for event in audits].count("resolve_medication") == 2
    assert result["metrics"]["toolLatencyMs"] == audited_tool_latency
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
    responses["facts"] = [
        envelope("OK", {"product": {
            "productId": "DRUG_PRODUCT::1", "documentId": "doc-1",
            "documentVersion": "1", "sourcePath": "labels/doc-1.xml",
            "contentHash": "1" * 64,
        }}),
        envelope("OK", {"product": {
            "productId": "DRUG_PRODUCT::2", "documentId": "doc-2",
            "documentVersion": "1", "sourcePath": "labels/doc-2.xml",
            "contentHash": "2" * 64,
        }}),
    ]
    responses["search"] = [
        envelope("OK", {"evidence": [{
            "referenceId": "S1", "evidenceRef": "SPL:doc-1#warnings",
            "productId": "DRUG_PRODUCT::1",
            "documentId": "doc-1", "documentVersion": "1",
            "sectionId": "warnings", "sectionCode": "34071-1",
            "sourcePath": "labels/doc-1.xml", "contentHash": "1" * 64,
            "topic": "warnings", "content": "one",
        }]}, refs=["SPL:doc-1#warnings"]),
        envelope("OK", {"evidence": [{
            "referenceId": "S2", "evidenceRef": "SPL:doc-2#warnings",
            "productId": "DRUG_PRODUCT::2",
            "documentId": "doc-2", "documentVersion": "1",
            "sectionId": "warnings", "sectionCode": "34071-1",
            "sourcePath": "labels/doc-2.xml", "contentHash": "2" * 64,
            "topic": "warnings", "content": "two",
        }]}, refs=["SPL:doc-2#warnings"]),
    ]
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1, med2)), FakeDrugGateway(responses))
    result = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config={"configurable": {"thread_id": "review-1"}})
    label_findings = {item["medicationIds"][0]: item for item in result["findings"] if item["reviewType"] == "LABEL_EVIDENCE_REVIEW"}
    assert label_findings["med-1"]["labelEvidenceRefs"] == ["SPL:doc-1#warnings"]
    assert label_findings["med-2"]["labelEvidenceRefs"] == ["SPL:doc-2#warnings"]
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_product_facts_must_match_requested_product_before_label_search(
    tmp_path: Path,
) -> None:
    responses = standard_drug_responses({"ARNICA": mapped_response()})
    responses["facts"] = envelope("OK", {"product": {
        "productId": "DRUG_PRODUCT::OTHER",
        "documentId": "doc-1",
        "documentVersion": "3",
    }})
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1)),
        drug,
    )

    result = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config={"configurable": {"thread_id": "review-1"}},
    )

    assert result["status"] == "BLOCKED_TOOL_ERROR"
    assert result["unresolvedItems"] == [{
        "kind": "DRUG_EVIDENCE_SCOPE_ERROR",
        "productIds": ["DRUG_PRODUCT::1"],
    }]
    assert "search_label_evidence" not in [name for name, _ in drug.calls]
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
    responses["facts"] = [
        envelope("OK", {"product": {
            "productId": "DRUG_PRODUCT::1", "documentId": "doc-1",
            "documentVersion": "1", "sourcePath": "labels/doc-1.xml",
            "contentHash": "1" * 64,
        }}, provenance=consistent),
        envelope("OK", {"product": {
            "productId": "DRUG_PRODUCT::1", "documentId": "doc-2",
            "documentVersion": "2", "sourcePath": "labels/doc-2.xml",
            "contentHash": "2" * 64,
        }}, provenance=fallback),
    ]
    responses["search"] = [
        envelope("OK", {"evidence": [{
            "referenceId": "S1", "evidenceRef": "SPL:doc-1#warnings",
            "productId": "DRUG_PRODUCT::1", "documentId": "doc-1",
            "documentVersion": "1", "sectionId": "warnings",
            "sectionCode": "34071-1", "sourcePath": "labels/doc-1.xml",
            "contentHash": "1" * 64, "topic": "warnings", "content": "initial",
        }]}, refs=["SPL:doc-1#warnings"], provenance=consistent),
        envelope("OK", {"evidence": [{
            "referenceId": "S2", "evidenceRef": "SPL:doc-2#warnings",
            "productId": "DRUG_PRODUCT::1", "documentId": "doc-2",
            "documentVersion": "2", "sectionId": "warnings",
            "sectionCode": "34071-1", "sourcePath": "labels/doc-2.xml",
            "contentHash": "2" * 64, "topic": "warnings", "content": "fallback",
        }]}, refs=["SPL:doc-2#warnings"], provenance=fallback),
    ]
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), drug)
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
        envelope("OK", {"product": {
            "productId": "DRUG_PRODUCT::1", "documentId": "doc-1",
            "documentVersion": "1", "sourcePath": "labels/doc-1.xml",
            "contentHash": "1" * 64,
        }}),
        envelope("INSUFFICIENT_EVIDENCE", {}, errors=["facts unavailable"]),
    ]
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), drug)
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    target = next(item for item in first["findings"] if item["reviewType"] == "LABEL_EVIDENCE_REVIEW")
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": target["findingId"]}]}), config=config)
    assert updated["status"] == "AWAITING_FINDING_REVIEW"
    assert any(
        item["reviewType"] == "EVIDENCE_GAP" and item.get("sourceTool") == "get_product_facts"
        for item in updated["findings"]
    )
    calls_after_once = len(drug.calls)
    recovered = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": target["findingId"]}]}), config=config)
    assert any(
        item["findingId"] == f"evidence-gap-get_product_facts-{target['findingId']}"
        for item in recovered["findings"]
    )
    assert len(drug.calls) == calls_after_once
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_second_refresh_keeps_first_result_and_reports_budget_exhaustion(tmp_path: Path) -> None:
    consistent = {"graphBackend": "neo4j", "graphWorkspace": "dailymed", "graphDatabase": "neo4j", "fallbackUsed": False, "consistency": {"status": "CONSISTENT"}}
    fallback = {"graphBackend": "snapshot", "graphWorkspace": "dailymed", "graphDatabase": None, "fallbackUsed": True, "consistency": {"status": "UNAVAILABLE"}}
    responses = standard_drug_responses({"ARNICA": mapped_response()})
    fact_product = {
        "productId": "DRUG_PRODUCT::1", "documentId": "doc-1",
        "documentVersion": "1", "sourcePath": "labels/doc-1.xml",
        "contentHash": "1" * 64,
    }
    responses["facts"] = [
        envelope("OK", {"product": fact_product}, provenance=consistent),
        envelope("OK", {"product": fact_product}, provenance=fallback),
    ]
    responses["search"] = [
        envelope("OK", {"evidence": [{
            "referenceId": "S1", "evidenceRef": "SPL:doc-1#warnings",
            **fact_product, "sectionId": "warnings", "sectionCode": "34071-1",
            "topic": "warnings", "content": "initial",
        }]}, provenance=consistent),
        envelope("OK", {"evidence": [{
            "referenceId": "S2", "evidenceRef": "SPL:doc-1#warnings-refresh",
            **fact_product, "sectionId": "warnings", "sectionCode": "34071-1",
            "topic": "warnings", "content": "fallback",
        }]}, provenance=fallback),
    ]
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(tmp_path, FakeHealthGateway(health_context(MED1)), drug)
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    target = next(item for item in first["findings"] if item["reviewType"] == "LABEL_EVIDENCE_REVIEW")
    fallback_state = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": target["findingId"]}]}), config=config)
    assert any(item["findingId"] == f"provenance-gap-{target['findingId']}" for item in fallback_state["findings"])
    calls_after_once = len(drug.calls)
    recovered = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": target["findingId"]}]}), config=config)
    refreshed = next(item for item in recovered["findings"] if item["findingId"] == target["findingId"])
    assert refreshed["graphProvenance"]["graphBackend"] == "snapshot"
    assert "graph_fallback_used" in refreshed["verificationWarnings"]
    assert "REINVESTIGATION_BUDGET_EXHAUSTED" in refreshed["verificationErrors"]
    assert len(drug.calls) == calls_after_once
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
    planner = RecordingPlanner(
        planning_result(ReviewTopic.WARNINGS, with_model_call=False)
    )
    graph = build_test_graph(
        tmp_path,
        SequencedHealthGateway(),
        drug,
        planner=planner,
    )
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    gap = next(item for item in first["findings"] if item.get("missingField") == "allergies")
    updated = await graph.ainvoke(Command(resume={"action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": [{"action": "REQUEST_MORE_EVIDENCE", "findingId": gap["findingId"]}]}), config=config)
    mapping = next(item for item in updated["medicationMappings"] if item["medicationId"] == "med-1")
    assert [name for name, _ in drug.calls].count("resolve_medication") == 2
    assert mapping["sourceName"] == "METFORMIN"
    assert mapping["selectedProductId"] == "DRUG_PRODUCT::2"
    assert all("DRUG_PRODUCT::1" not in item.get("productIds", []) for item in updated["evidenceIndex"])
    assert planner.calls == 1
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_changed_medication_confirmation_reuses_existing_intent(tmp_path: Path) -> None:
    changed = {
        **MED1,
        "medication": "METFORMIN",
        "identifiers": [{"system": "ndc", "code": "2"}],
    }

    class SequencedHealthGateway:
        def __init__(self) -> None:
            self.responses = [
                health_context(MED1, missing=["allergies"]),
                health_context(changed),
            ]

        async def get_review_context(self, patient_id: str | None, as_of: str | None):
            return self.responses.pop(0)

    ambiguous = envelope("AMBIGUOUS", {
        "matchClass": "AMBIGUOUS_NAME",
        "selectedProductId": None,
        "candidates": [{"productId": "DRUG_PRODUCT::2"}],
        "unmatchedFields": [],
    })
    planner = RecordingPlanner(
        planning_result(ReviewTopic.WARNINGS, with_model_call=False)
    )
    graph = build_test_graph(
        tmp_path,
        SequencedHealthGateway(),
        FakeDrugGateway(standard_drug_responses({
            "ARNICA": mapped_response(),
            "METFORMIN": ambiguous,
        })),
        planner=planner,
    )
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config=config,
    )
    gap = next(item for item in first["findings"] if item.get("missingField") == "allergies")
    awaiting_mapping = await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW",
        "reviewerId": "pharmacist-demo",
        "decisions": [{
            "action": "REQUEST_MORE_EVIDENCE",
            "findingId": gap["findingId"],
        }],
    }), config=config)
    assert awaiting_mapping["status"] == "AWAITING_MAPPING_CONFIRMATION"

    resumed = await graph.ainvoke(Command(resume={
        "action": "CONFIRM_MAPPING",
        "medicationId": "med-1",
        "productId": "DRUG_PRODUCT::2",
        "reviewerId": "pharmacist-demo",
    }), config=config)

    assert resumed["intent"] == first["intent"]
    assert planner.calls == 1
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_changed_context_with_ambiguous_mapping_rebuilds_before_queued_refresh(
    tmp_path: Path,
) -> None:
    med2 = {
        **MED2,
        "medication": "ARNICA TWO",
        "identifiers": [{"system": "ndc", "code": "2"}],
    }
    changed_med1 = {
        **MED1,
        "medication": "METFORMIN",
        "identifiers": [{"system": "ndc", "code": "3"}],
    }

    class SequencedHealthGateway:
        def __init__(self) -> None:
            self.responses = [
                health_context(MED1, med2, missing=["allergies"]),
                health_context(changed_med1, med2),
            ]

        async def get_review_context(self, patient_id, as_of):
            return self.responses.pop(0)

    mapped_two = envelope("OK", {
        "matchClass": "EXACT_IDENTIFIER",
        "autoAcceptable": True,
        "selectedProductId": "DRUG_PRODUCT::2",
        "candidates": [],
        "unmatchedFields": [],
    })
    ambiguous_three = envelope("AMBIGUOUS", {
        "matchClass": "AMBIGUOUS_NAME",
        "selectedProductId": None,
        "candidates": [{"productId": "DRUG_PRODUCT::3"}],
        "unmatchedFields": [],
    })
    responses = standard_drug_responses({
        "ARNICA": mapped_response(),
        "ARNICA TWO": mapped_two,
        "METFORMIN": ambiguous_three,
    })

    def fact(product_id: str, document_id: str):
        return envelope("OK", {"product": {
            "productId": product_id,
            "documentId": document_id,
            "documentVersion": "3",
            "sourcePath": f"labels/{document_id}.xml",
            "contentHash": document_id[-1] * 64,
        }})

    def search(product_id: str, document_id: str):
        return envelope("OK", {"evidence": [{
            "referenceId": f"{document_id}-warnings",
            "evidenceRef": f"SPL:{document_id}#warnings",
            "productId": product_id,
            "documentId": document_id,
            "documentVersion": "3",
            "sectionId": "warnings",
            "sectionCode": "34071-1",
            "sourcePath": f"labels/{document_id}.xml",
            "contentHash": document_id[-1] * 64,
            "topic": "warnings",
            "content": f"Warnings for {product_id}.",
        }]})

    responses["facts"] = [
        fact("DRUG_PRODUCT::1", "doc-1"),
        fact("DRUG_PRODUCT::2", "doc-2"),
        fact("DRUG_PRODUCT::2", "doc-2"),
        fact("DRUG_PRODUCT::3", "doc-3"),
    ]
    responses["search"] = [
        search("DRUG_PRODUCT::1", "doc-1"),
        search("DRUG_PRODUCT::2", "doc-2"),
        search("DRUG_PRODUCT::2", "doc-2"),
        search("DRUG_PRODUCT::3", "doc-3"),
    ]
    drug = FakeDrugGateway(responses)
    planner = RecordingPlanner(
        planning_result(ReviewTopic.WARNINGS, with_model_call=False)
    )
    graph = build_test_graph(
        tmp_path,
        SequencedHealthGateway(),
        drug,
        planner=planner,
    )
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({
        "reviewId": "review-1",
        "patientRef": "P001",
        "asOf": "2026-08-31",
    }, config=config)
    context_gap = next(
        item for item in first["findings"] if item.get("missingField") == "allergies"
    )
    unchanged_label = next(
        item for item in first["findings"]
        if item.get("medicationIds") == ["med-2"]
        and item["reviewType"] == "LABEL_EVIDENCE_REVIEW"
    )

    awaiting_mapping = await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW",
        "reviewerId": "pharmacist-demo",
        "decisions": [
            {"action": "REQUEST_MORE_EVIDENCE", "findingId": context_gap["findingId"]},
            {"action": "REQUEST_MORE_EVIDENCE", "findingId": unchanged_label["findingId"]},
        ],
    }), config=config)
    assert awaiting_mapping["status"] == "AWAITING_MAPPING_CONFIRMATION"

    rebuilt = await graph.ainvoke(Command(resume={
        "action": "CONFIRM_MAPPING",
        "medicationId": "med-1",
        "productId": "DRUG_PRODUCT::3",
        "reviewerId": "pharmacist-demo",
    }), config=config)

    fact_products = [
        value for name, value in drug.calls if name == "get_product_facts"
    ]
    assert fact_products == [
        "DRUG_PRODUCT::1",
        "DRUG_PRODUCT::2",
        "DRUG_PRODUCT::2",
        "DRUG_PRODUCT::3",
    ]
    assert any(
        item.get("medicationIds") == ["med-1"]
        and item.get("selectedProductIds") == ["DRUG_PRODUCT::3"]
        and item["reviewType"] == "LABEL_EVIDENCE_REVIEW"
        for item in rebuilt["findings"]
    )
    assert not any(
        item.get("medicationIds") == ["med-1"]
        and item.get("selectedProductIds") == ["DRUG_PRODUCT::1"]
        for item in rebuilt["findings"]
    )
    assert rebuilt["reinvestigateFindingIds"] == []
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

    mapped_two = envelope("OK", {
        "matchClass": "EXACT_IDENTIFIER", "autoAcceptable": True,
        "selectedProductId": "DRUG_PRODUCT::2", "candidates": [],
        "unmatchedFields": [],
    })
    mapped_three = envelope("OK", {
        "matchClass": "EXACT_IDENTIFIER", "autoAcceptable": True,
        "selectedProductId": "DRUG_PRODUCT::3", "candidates": [],
        "unmatchedFields": [],
    })
    responses = standard_drug_responses({
        "ARNICA": mapped_response(), "ARNICA TWO": mapped_two,
        "METFORMIN": mapped_three,
    })

    def product_fact(product_id: str, document_id: str):
        return envelope("OK", {"product": {
            "productId": product_id,
            "documentId": document_id,
            "documentVersion": "3",
            "sourcePath": f"labels/{document_id}.xml",
            "contentHash": document_id[-1] * 64,
        }})

    def product_search(product_id: str, document_id: str):
        topics = ["identity", "ingredients", "route", "dosage_form", "warnings"]
        return envelope("OK", {"evidence": [{
            "referenceId": f"{document_id}-{topic}",
            "evidenceRef": f"SPL:{document_id}#{topic}",
            "productId": product_id,
            "documentId": document_id,
            "documentVersion": "3",
            "sectionId": topic,
            "sectionCode": "34071-1",
            "sourcePath": f"labels/{document_id}.xml",
            "contentHash": document_id[-1] * 64,
            "topic": topic,
            "content": f"Label evidence for {topic}.",
        } for topic in topics]})

    responses["facts"] = [
        product_fact("DRUG_PRODUCT::1", "doc-1"),
        product_fact("DRUG_PRODUCT::2", "doc-2"),
        envelope("INSUFFICIENT_EVIDENCE", {}, errors=["temporarily unavailable"]),
        product_fact("DRUG_PRODUCT::3", "doc-3"),
    ]
    responses["search"] = [
        product_search("DRUG_PRODUCT::1", "doc-1"),
        product_search("DRUG_PRODUCT::2", "doc-2"),
        product_search("DRUG_PRODUCT::3", "doc-3"),
    ]
    graph = build_test_graph(tmp_path, SequencedHealthGateway(), FakeDrugGateway(responses))
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"}, config=config)
    unaffected = next(item for item in first["findings"] if item.get("medicationIds") == ["med-2"] and item["reviewType"] == "LABEL_EVIDENCE_REVIEW")
    unaffected_refs = set(unaffected["labelEvidenceRefs"])
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
    assert unaffected_refs <= {
        item["evidenceRef"] for item in updated["evidenceIndex"]
    }
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_reinvestigation_preserves_evidence_referenced_by_unaffected_finding(
    tmp_path: Path,
) -> None:
    med2 = {
        **MED2,
        "medication": "ARNICA TWO",
        "identifiers": [{"system": "ndc", "code": "2"}],
    }
    responses = standard_drug_responses({
        "ARNICA": mapped_response(),
        "ARNICA TWO": mapped_response(),
    })
    initial_search = responses["search"]
    refreshed_payload = dict(initial_search.envelope.data["evidence"][0])
    refreshed_payload.update({
        "referenceId": "S-refreshed",
        "content": "Refreshed warning evidence.",
    })
    responses["search"] = [
        initial_search,
        envelope("OK", {"evidence": [refreshed_payload]}),
    ]
    drug = FakeDrugGateway(responses)
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1, med2)),
        drug,
    )
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke(
        {"reviewId": "review-1", "patientRef": "P001", "asOf": "2026-08-31"},
        config=config,
    )
    labels = [
        item for item in first["findings"]
        if item["reviewType"] == "LABEL_EVIDENCE_REVIEW"
    ]
    refreshed_target, unaffected = labels
    old_refs = set(unaffected["labelEvidenceRefs"])
    old_ids = set(unaffected["labelEvidenceIds"])
    await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW",
        "reviewerId": "pharmacist-demo",
        "decisions": [{
            "action": "REJECT_FINDING",
            "findingId": unaffected["findingId"],
        }],
    }), config=config)

    updated = await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW",
        "reviewerId": "pharmacist-demo",
        "decisions": [{
            "action": "REQUEST_MORE_EVIDENCE",
            "findingId": refreshed_target["findingId"],
        }],
    }), config=config)

    preserved = next(
        item for item in updated["findings"]
        if item["findingId"] == unaffected["findingId"]
    )
    assert preserved["status"] == "REJECTED"
    assert set(preserved["labelEvidenceRefs"]) == old_refs
    assert set(preserved["labelEvidenceIds"]) == old_ids
    evidence_refs = {item["evidenceRef"] for item in updated["evidenceIndex"]}
    evidence_ids = {item["evidenceId"] for item in updated["evidenceIndex"]}
    assert old_refs <= evidence_refs
    assert old_ids <= evidence_ids
    all_evidence_refs = [item["evidenceRef"] for item in updated["evidenceIndex"]]
    assert len(all_evidence_refs) == len(set(all_evidence_refs))
    refreshed = next(
        item for item in updated["evidenceIndex"]
        if item["evidenceRef"] == "SPL:doc-1#identity"
    )
    assert refreshed["summary"] == "Refreshed warning evidence."
    await graph.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_refresh_keeps_reviewed_evidence_snapshot_when_reference_is_reused(
    tmp_path: Path,
) -> None:
    med2 = {
        **MED2,
        "medication": "ARNICA TWO",
        "identifiers": [{"system": "ndc", "code": "2"}],
    }
    responses = standard_drug_responses({
        "ARNICA": mapped_response(),
        "ARNICA TWO": mapped_response(),
    })
    initial_fact = responses["facts"]
    initial_search = responses["search"]
    initial_payload = dict(initial_search.envelope.data["evidence"][0])
    refreshed_payload = {
        **initial_payload,
        "referenceId": "S-refreshed",
        "documentVersion": "4",
        "contentHash": "b" * 64,
        "content": "Version four identity evidence.",
    }
    responses["facts"] = [
        initial_fact,
        envelope("OK", {"product": {
            **initial_fact.envelope.data["product"],
            "documentVersion": "4",
            "contentHash": "b" * 64,
        }}),
    ]
    responses["search"] = [
        initial_search,
        envelope("OK", {"evidence": [refreshed_payload]}),
    ]
    graph = build_test_graph(
        tmp_path,
        FakeHealthGateway(health_context(MED1, med2)),
        FakeDrugGateway(responses),
    )
    config = {"configurable": {"thread_id": "review-1"}}
    first = await graph.ainvoke({
        "reviewId": "review-1",
        "patientRef": "P001",
        "asOf": "2026-08-31",
    }, config=config)
    target, reviewed = [
        item for item in first["findings"]
        if item["reviewType"] == "LABEL_EVIDENCE_REVIEW"
    ]
    old_evidence = next(
        item for item in first["evidenceIndex"]
        if item["source"] == "SPL"
        and item["evidenceRef"] == initial_payload["evidenceRef"]
    )
    reviewed_evidence_ids = list(reviewed["labelEvidenceIds"])
    await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW",
        "reviewerId": "pharmacist-demo",
        "decisions": [{
            "action": "ACCEPT_FINDING",
            "findingId": reviewed["findingId"],
        }],
    }), config=config)

    updated = await graph.ainvoke(Command(resume={
        "action": "COMPLETE_FINDING_REVIEW",
        "reviewerId": "pharmacist-demo",
        "decisions": [{
            "action": "REQUEST_MORE_EVIDENCE",
            "findingId": target["findingId"],
        }],
    }), config=config)

    preserved = next(
        item for item in updated["findings"]
        if item["findingId"] == reviewed["findingId"]
    )
    versions = [
        item for item in updated["evidenceIndex"]
        if item["source"] == "SPL"
        and item["evidenceRef"] == initial_payload["evidenceRef"]
    ]
    assert preserved["status"] == "ACCEPTED"
    assert preserved["labelEvidenceIds"] == reviewed_evidence_ids
    assert any(item == old_evidence for item in versions)
    refreshed_version = next(
        item for item in versions
        if item["documentVersion"] == "4"
        and item["contentHash"] == "b" * 64
        and item["summary"] == "Version four identity evidence."
    )
    refreshed_target = next(
        item for item in updated["findings"]
        if item["findingId"] == target["findingId"]
    )
    assert refreshed_version["evidenceId"] in refreshed_target["labelEvidenceIds"]
    assert old_evidence["evidenceId"] not in refreshed_target["labelEvidenceIds"]
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
    assert first["retrievalAttempts"]
    assert set(first["retrievalAttempts"].values()) == {1}
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
    fact_product = {
        "productId": "DRUG_PRODUCT::1", "documentId": "doc-1",
        "documentVersion": "3", "sourcePath": "labels/doc-1.xml",
        "contentHash": "a" * 64,
    }
    responses["facts"] = [
        envelope("OK", {"product": fact_product}),
        envelope("INSUFFICIENT_EVIDENCE", {}, errors=["missing"]),
        envelope("OK", {"product": fact_product}),
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
