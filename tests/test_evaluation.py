import json
from pathlib import Path

from medication_review_agent.evaluation import CaseResult, load_cases, run_evaluation, score_cases


CASES = Path(__file__).parents[1] / "evaluation" / "cases.jsonl"


def test_metrics_count_safety_failures_as_zero_tolerance() -> None:
    report = score_cases([
        CaseResult(caseId="safe", passed=True, metrics={"unsafeActions": 0}),
        CaseResult(caseId="unsafe", passed=False, metrics={"unsafeActions": 1}),
    ])
    assert report["unsafeActionCases"] == 1
    assert report["acceptancePassed"] is False


def test_empty_evaluation_suite_cannot_pass() -> None:
    report = score_cases([])
    assert report["acceptancePassed"] is False


def test_every_run_requires_latency_and_cost_fields() -> None:
    report = score_cases([CaseResult(caseId="missing-metrics", passed=True, metrics={"latencyMs": 10})])
    assert report["runsWithCompleteOperationalMetrics"] == 0


def test_fixture_contains_all_fifteen_stable_cases() -> None:
    cases = load_cases(CASES)
    assert [item["caseId"] for item in cases] == [
        "exact_identifier_mapping", "exact_scoped_name_mapping",
        "ambiguous_product_variant", "fuzzy_candidate_not_approved",
        "unmapped_conventional_medication", "duplicate_active_ingredient",
        "neo4j_reverse_ingredient_completeness", "neo4j_snapshot_consistency",
        "neo4j_unavailable_snapshot_fallback", "missing_allergy_history",
        "pregnancy_label_evidence", "duplicate_patient_name",
        "cross_patient_isolation", "transient_mcp_retry",
        "unsafe_stop_or_dose_request",
    ]
    fuzzy = next(item for item in cases if item["caseId"] == "fuzzy_candidate_not_approved")
    assert fuzzy["drugResponses"]["mappings"]["ARNCA"]["status"] == "OK"
    required_oracle_sections = {"profiles", "mappings", "evidence", "reportMappings", "reportEvidence"}
    assert all(
        required_oracle_sections <= set(item["expected"]["graphProvenance"])
        and item["expected"]["graphProvenance"]["profiles"]
        and item["expected"]["graphProvenance"]["mappings"]
        and (
            not item["expected"].get("reportExpected")
            or all(item["expected"]["graphProvenance"][section] for section in {"evidence", "reportMappings", "reportEvidence"})
        )
        for item in cases
        if any(mapping.get("graph") for mapping in item["drugResponses"]["mappings"].values())
    )
    assert all(
        isinstance(item["expected"].get("reportExpected"), bool)
        for item in cases
        if any(mapping.get("graph") for mapping in item["drugResponses"]["mappings"].values())
    )


def test_missing_evidence_provenance_fails_independent_oracle(tmp_path: Path) -> None:
    case = load_cases(CASES)[0]
    case = json.loads(json.dumps(case))
    case["caseId"] = "missing-evidence-provenance"
    case["drugResponses"]["omitEvidenceProvenance"] = True
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    report = run_evaluation(cases_path, tmp_path / "report.json", work_dir=tmp_path / "runs")
    assert report["acceptancePassed"] is False
    assert any("evidence graph oracle key mismatch" in failure for failure in report["cases"][0]["failures"])


def test_unemitted_expected_graph_oracle_key_fails_completeness(tmp_path: Path) -> None:
    case = json.loads(json.dumps(load_cases(CASES)[0]))
    case["caseId"] = "missing-expected-evidence-item"
    case["expected"]["graphProvenance"]["evidence"]["DRUG_PRODUCT::MISSING"] = "p1"
    case["expected"]["graphProvenance"]["reportEvidence"]["DRUG_PRODUCT::MISSING"] = "p1"
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    report = run_evaluation(cases_path, tmp_path / "report.json", work_dir=tmp_path / "runs")
    assert report["acceptancePassed"] is False
    assert any("evidence graph oracle key mismatch" in failure for failure in report["cases"][0]["failures"])


def test_full_synthetic_evaluation_records_graph_and_operational_metrics(tmp_path: Path) -> None:
    report = run_evaluation(CASES, tmp_path / "report.json", work_dir=tmp_path / "runs")
    assert len(report["cases"]) == 15
    assert report["acceptancePassed"] is True
    fallback = next(item for item in report["cases"] if item["caseId"] == "neo4j_unavailable_snapshot_fallback")
    assert fallback["observed"]["graphProvenance"]["fallbackUsed"] is True
    assert fallback["observed"]["reportDisclosureComplete"] is True
    assert fallback["observed"]["reportOracleMatched"] is True
    unsafe = next(item for item in report["cases"] if item["caseId"] == "unsafe_stop_or_dose_request")
    assert "unsafe_clinical_action" in unsafe["observed"]["unsafeVerificationErrors"]
    fuzzy = next(item for item in report["cases"] if item["caseId"] == "fuzzy_candidate_not_approved")
    assert fuzzy["observed"]["reportOracleApplicable"] is False
    assert fuzzy["observed"]["reportOracleSatisfied"] is True
    assert report["summary"]["acceptedFindingCount"] > 0
    assert all(
        item["observed"]["finalStatus"] == "SIGNED_OFF"
        for item in report["cases"]
        if item["caseId"] != "fuzzy_candidate_not_approved"
    )
    assert report["runsWithCompleteOperationalMetrics"] == 15
