from playwright.sync_api import Page, expect


def test_empty_workbench_is_actionable(page: Page, live_server_url: str) -> None:
    page.goto(live_server_url)

    expect(page.get_by_label("患者编号或 FHIR ID")).to_be_visible()
    expect(page.get_by_role("button", name="开始复核")).to_be_enabled()
    expect(page.locator("#patient-panel")).to_contain_text("尚未载入患者")
    expect(page.locator("#review-queue")).to_contain_text("尚无审核任务")


def test_existing_review_renders_context_queue_and_graph_provenance(
    page: Page, live_server_url: str
) -> None:
    page.goto(f"{live_server_url}/?review=review-1")

    expect(page.locator("#review-status")).to_contain_text("进行中")
    expect(page.locator("#patient-panel")).to_contain_text("FHIR:Patient/p1")
    expect(page.locator("#patient-panel")).to_contain_text("资料缺失")
    expect(page.locator("#patient-panel")).to_contain_text("pregnancyStatus")
    expect(page.locator("#review-queue")).to_contain_text("Arnica montana")
    expect(page.locator("#review-queue")).to_contain_text("DRUG_PRODUCT::1")
    expect(page.locator("#review-queue")).to_contain_text("Neo4j 在线图谱")
    expect(page.locator("#review-queue")).to_contain_text("图谱一致")
    expect(page.locator("#review-queue")).to_contain_text("核对标签注意事项")
