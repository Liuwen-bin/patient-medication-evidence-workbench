import json

from playwright.sync_api import Page, Route, expect


def _review(*, status: str = "AWAITING_MAPPING_CONFIRMATION") -> dict:
    graph = {
        "graphBackend": "neo4j",
        "graphWorkspace": "dm1000_xml_v1",
        "graphDatabase": "neo4j",
        "fallbackUsed": False,
        "consistency": {"status": "CONSISTENT"},
    }
    return {
        "reviewId": "review-ui",
        "schemaVersion": "1.0",
        "status": status,
        "patientRef": "FHIR:Patient/p1",
        "asOf": "2026-08-31",
        "contextSnapshot": {
            "patient": {"id": "p1", "age": 42, "evidenceRef": "FHIR:Patient/p1"},
            "activeConditions": [], "allergies": [], "recentObservations": [],
        },
        "contextMissingFields": [],
        "medications": [{
            "medicationId": "med-1", "name": "Arnica montana", "identifiers": [],
            "route": "口服", "dosage": "每日一次",
            "patientEvidenceRefs": ["FHIR:MedicationRequest/med-1"],
        }],
        "medicationMappings": [{
            "medicationId": "med-1", "sourceName": "Arnica montana",
            "matchClass": "FUZZY_NAME", "selectedProductId": None,
            "candidates": [{
                "productId": "DRUG_PRODUCT::1", "productCode": "54973-3124",
                "productName": "Arnica Montana", "dosageForm": "TABLET", "route": "ORAL",
            }],
            "unmatchedFields": ["strength"], "graphProvenance": graph,
            "requiresHumanReview": True, "mappingConfirmationRequired": True,
        }],
        "reviewPlan": [],
        "findings": [{
            "findingId": "finding-1", "reviewType": "DUPLICATE_INGREDIENT",
            "summary": "重复活性成分", "attentionLevel": "HIGH", "confidence": 0.94,
            "medicationIds": ["med-1"], "selectedProductIds": ["DRUG_PRODUCT::1"],
            "patientEvidenceRefs": ["FHIR:MedicationRequest/med-1"],
            "labelEvidenceRefs": ["SPL:doc-1#section-1"], "status": "PENDING",
            "requiresHumanReview": True, "verificationErrors": [], "verificationWarnings": [],
            "graphProvenance": graph,
        }],
        "evidenceIndex": [{
            "evidenceId": "evidence-1", "source": "SPL",
            "evidenceRef": "SPL:doc-1#section-1", "medicationIds": ["med-1"],
            "productIds": ["DRUG_PRODUCT::1"], "topic": "active-ingredients",
            "summary": "标签原文：Active ingredient ARNICA MONTANA.",
            "graphProvenance": graph,
        }],
        "unresolvedItems": [], "humanDecisions": [], "auditEvents": [],
        "metrics": {"toolLatencyMs": 12, "retries": 0, "inputTokens": 0, "outputTokens": 0, "estimatedCost": 0},
        "version": 3,
        "createdAt": "2026-08-31T08:00:00Z", "updatedAt": "2026-08-31T08:01:00Z",
    }


def _mock_review(page: Page, snapshot: dict, on_post=None, audit=None) -> None:
    def handle(route: Route) -> None:
        if route.request.method == "POST" and on_post:
            on_post(route)
            return
        route.fulfill(status=200, content_type="application/json", body=json.dumps(snapshot))

    page.route("**/api/reviews/review-ui", handle)
    page.route("**/api/reviews/review-ui/decisions", handle)
    page.route("**/api/reviews/review-ui/audit", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(audit or [])
    ))


def test_ambiguous_mapping_requires_explicit_candidate_selection(
    page: Page, live_server_url: str
) -> None:
    snapshot = _review()
    received = []

    def decide(route: Route) -> None:
        received.append(route.request.post_data_json)
        updated = {**snapshot, "status": "AWAITING_FINDING_REVIEW", "version": 4}
        updated["medicationMappings"] = [{
            **snapshot["medicationMappings"][0], "selectedProductId": "DRUG_PRODUCT::1",
            "matchClass": "HUMAN_CONFIRMED", "requiresHumanReview": False,
            "mappingConfirmationRequired": False,
        }]
        route.fulfill(status=200, content_type="application/json", body=json.dumps(updated))

    _mock_review(page, snapshot, decide)
    page.goto(f"{live_server_url}/?review=review-ui")

    expect(page.get_by_text("需要确认药品映射")).to_be_visible()
    expect(page.get_by_role("button", name="确认映射")).to_be_disabled()
    page.get_by_label("产品代码 54973-3124").check()
    expect(page.get_by_role("button", name="确认映射")).to_be_enabled()
    page.get_by_role("button", name="确认映射").click()
    expect(page.locator("#workbench-notice")).to_have_text("映射已确认")
    assert received == [{
        "expectedVersion": 3, "action": "CONFIRM_MAPPING", "reviewerId": "pharmacist-demo",
        "medicationId": "med-1", "productId": "DRUG_PRODUCT::1",
    }]


def test_ambiguous_patient_requires_explicit_candidate_selection(
    page: Page, live_server_url: str
) -> None:
    snapshot = _review(status="AWAITING_PATIENT_CONFIRMATION")
    snapshot["candidates"] = [
        {"id": "p1", "patientNumber": "P001", "display": "患者甲"},
        {"id": "p2", "patientNumber": "P002", "display": "患者乙"},
    ]
    received = []

    def decide(route: Route) -> None:
        received.append(route.request.post_data_json)
        route.fulfill(status=200, content_type="application/json", body=json.dumps({**snapshot, "status": "AWAITING_FINDING_REVIEW", "version": 4}))

    _mock_review(page, snapshot, decide)
    page.goto(f"{live_server_url}/?review=review-ui")
    expect(page.get_by_role("button", name="确认患者")).to_be_disabled()
    page.get_by_label("FHIR 患者 p2").check()
    page.get_by_role("button", name="确认患者").click()
    expect(page.locator("#workbench-notice")).to_have_text("患者已确认")
    assert received == [{
        "expectedVersion": 3, "action": "CONFIRM_PATIENT", "reviewerId": "pharmacist-demo", "patientId": "p2",
    }]


def test_selecting_finding_shows_fhir_spl_rag_and_graph_evidence(
    page: Page, live_server_url: str
) -> None:
    snapshot = _review(status="AWAITING_FINDING_REVIEW")
    _mock_review(page, snapshot)
    page.goto(f"{live_server_url}/?review=review-ui")

    page.get_by_role("button", name="重复活性成分", exact=True).click()
    panel = page.locator("#evidence-panel")
    expect(panel).to_contain_text("患者证据（FHIR）")
    expect(panel).to_contain_text("FHIR:MedicationRequest/med-1")
    expect(panel).to_contain_text("标签原文（SPL / RAG）")
    expect(panel).to_contain_text("SPL:doc-1#section-1")
    expect(panel).to_contain_text("Active ingredient ARNICA MONTANA")
    expect(panel).to_contain_text("Neo4j 在线图谱")


def test_label_images_allow_same_origin_and_api_urls_only(
    page: Page, live_server_url: str
) -> None:
    snapshot = _review(status="AWAITING_FINDING_REVIEW")
    snapshot["evidenceIndex"][0]["imageUrl"] = "/api/evidence/doc-1.png"
    snapshot["evidenceIndex"].append({
        **snapshot["evidenceIndex"][0], "evidenceId": "evidence-2",
        "evidenceRef": "SPL:doc-1#section-2", "imageUrl": "https://evil.example/steal.png",
    })
    snapshot["findings"][0]["labelEvidenceRefs"] = ["SPL:doc-1#section-1", "SPL:doc-1#section-2"]
    _mock_review(page, snapshot)
    page.goto(f"{live_server_url}/?review=review-ui")
    page.get_by_role("button", name="重复活性成分", exact=True).click()
    panel = page.locator("#evidence-panel")
    expect(panel.locator("img")).to_have_count(1)
    expect(panel.locator("img")).to_have_attribute("src", "/api/evidence/doc-1.png")


def test_graph_fallback_drift_and_unknown_are_alerts(
    page: Page, live_server_url: str
) -> None:
    snapshot = _review(status="AWAITING_FINDING_REVIEW")
    snapshot["medicationMappings"][0]["graphProvenance"] = {
        "graphBackend": "neo4j", "graphWorkspace": "w", "graphDatabase": "neo4j",
        "fallbackUsed": False, "consistency": {"status": "UNKNOWN"},
    }
    _mock_review(page, snapshot)
    page.goto(f"{live_server_url}/?review=review-ui")
    expect(page.locator(".graph-provenance.is-warning")).to_have_count(1)
    expect(page.locator(".graph-provenance[role='alert']")).to_have_count(1)


def test_stale_finding_decision_refreshes_without_retry(
    page: Page, live_server_url: str
) -> None:
    snapshot = _review(status="AWAITING_FINDING_REVIEW")
    post_count = 0

    def conflict(route: Route) -> None:
        nonlocal post_count
        post_count += 1
        route.fulfill(status=409, content_type="application/json", body=json.dumps({"detail": "Review version conflict"}))

    _mock_review(page, snapshot, conflict)
    page.goto(f"{live_server_url}/?review=review-ui")
    page.get_by_role("button", name="接受发现").click()

    expect(page.locator("#workbench-notice")).to_have_text("审核状态已被更新，请重新确认当前项目")
    assert post_count == 1


def test_report_requires_explicit_final_confirmation(
    page: Page, live_server_url: str
) -> None:
    snapshot = _review(status="READY_FOR_SIGN_OFF")
    snapshot["findings"][0]["status"] = "ACCEPTED"

    def sign(route: Route) -> None:
        payload = route.request.post_data_json
        assert payload == {
            "expectedVersion": 3, "action": "SIGN_OFF", "reviewerId": "pharmacist-demo",
        }
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({**snapshot, "status": "SIGNED_OFF", "version": 4}),
        )

    _mock_review(page, snapshot, sign)
    page.goto(f"{live_server_url}/?review=review-ui")
    page.get_by_role("button", name="提交报告").click()

    dialog = page.get_by_role("dialog", name="确认提交审核报告")
    expect(dialog).to_be_visible()
    expect(dialog).to_contain_text("接受 1")
    expect(dialog).to_contain_text("排除 0")
    expect(dialog).to_contain_text("未映射 0")
    expect(dialog).to_contain_text("证据缺口 0")
    expect(page.get_by_role("button", name="确认签署")).to_be_disabled()
    page.get_by_label("我已核对全部审核项").check()
    page.get_by_role("button", name="确认签署").click()

    expect(page.locator("#review-status")).to_contain_text("已签署")
    expect(page.get_by_role("link", name="查看报告")).to_have_attribute(
        "href", "/api/reviews/review-ui/report.html"
    )


def test_mobile_tabs_show_one_panel_without_changing_document_order(
    page: Page, live_server_url: str
) -> None:
    page.set_viewport_size({"width": 390, "height": 844})
    snapshot = _review(status="AWAITING_FINDING_REVIEW")
    _mock_review(page, snapshot)
    page.goto(f"{live_server_url}/?review=review-ui")

    expect(page.get_by_role("button", name="审核项")).to_have_attribute("aria-selected", "true")
    assert page.locator("[data-panel].is-active").count() == 1
    page.get_by_role("button", name="证据", exact=True).click()
    expect(page.locator("#evidence-panel")).to_be_visible()
    assert page.locator("[data-panel].is-active").count() == 1
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_sign_off_escape_closes_dialog_without_mutation(
    page: Page, live_server_url: str
) -> None:
    snapshot = _review(status="READY_FOR_SIGN_OFF")
    post_count = 0

    def unexpected_post(route: Route) -> None:
        nonlocal post_count
        post_count += 1
        route.fulfill(status=500, content_type="application/json", body="{}")

    _mock_review(page, snapshot, unexpected_post)
    page.goto(f"{live_server_url}/?review=review-ui")
    page.get_by_role("button", name="提交报告").click()
    expect(page.get_by_role("dialog", name="确认提交审核报告")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.get_by_role("dialog", name="确认提交审核报告")).to_be_hidden()
    expect(page.get_by_role("button", name="提交报告")).to_be_focused()
    assert post_count == 0


def test_sign_off_conflict_refreshes_without_replaying_old_mutation(
    page: Page, live_server_url: str
) -> None:
    snapshot = _review(status="READY_FOR_SIGN_OFF")
    post_count = 0

    def conflict(route: Route) -> None:
        nonlocal post_count
        post_count += 1
        route.fulfill(status=409, content_type="application/json", body=json.dumps({"detail": "Review version conflict"}))

    _mock_review(page, snapshot, conflict)
    page.goto(f"{live_server_url}/?review=review-ui")
    page.get_by_role("button", name="提交报告").click()
    page.get_by_label("我已核对全部审核项").check()
    page.get_by_role("button", name="确认签署").click()
    expect(page.locator("#workbench-notice")).to_have_text("审核状态已被更新，请重新确认当前项目")
    expect(page.get_by_role("dialog", name="确认提交审核报告")).to_be_hidden()
    assert post_count == 1


def test_audit_timeline_discloses_only_operational_metadata(
    page: Page, live_server_url: str
) -> None:
    snapshot = _review(status="AWAITING_FINDING_REVIEW")
    audit = [{
        "node": "retrieve_evidence", "tool": "search_label_evidence",
        "requestId": "request-123", "resultStatus": "OK",
        "occurredAt": "2026-08-31T08:00:00Z", "latencyMs": 18,
        "evidenceRefs": ["SPL:doc-1#section-1"],
        "argumentSummary": {"prompt": "do not show", "rawArguments": "do not show"},
    }]
    _mock_review(page, snapshot, audit=audit)
    page.goto(f"{live_server_url}/?review=review-ui")

    timeline = page.locator("#audit-timeline")
    expect(timeline).to_contain_text("retrieve_evidence")
    expect(timeline).to_contain_text("search_label_evidence")
    expect(timeline).to_contain_text("request-123")
    expect(timeline).to_contain_text("OK")
    expect(timeline).to_contain_text("18 ms")
    expect(timeline).to_contain_text("证据 1")
    expect(timeline).not_to_contain_text("do not show")
    expect(timeline).not_to_contain_text("rawArguments")
