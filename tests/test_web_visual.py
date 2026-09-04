from pathlib import Path

import pytest
from playwright.sync_api import Page

from tests.test_web_decisions import _mock_review, _review


@pytest.mark.parametrize("viewport", [(1440, 900), (1024, 768), (900, 700), (390, 844), (360, 800)])
def test_workbench_viewports_have_stable_panels_and_no_overflow(
    page: Page, live_server_url: str, viewport: tuple[int, int]
) -> None:
    page.set_viewport_size({"width": viewport[0], "height": viewport[1]})
    page.goto(live_server_url)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    for selector in (".topbar",):
        box = page.locator(selector).bounding_box()
        assert box is not None
        assert box["width"] > 0 and box["height"] > 0
    active = page.locator("[data-panel].is-active")
    if viewport[0] <= 899:
        assert active.count() == 1
        box = active.bounding_box()
        assert box is not None
        assert box["width"] > 0 and box["height"] > 0
    else:
        assert page.locator("#patient-panel").is_visible()
        assert page.locator("#review-queue").is_visible()
        assert page.locator("#evidence-panel").is_visible()
        boxes = [page.locator(selector).bounding_box() for selector in ("#patient-panel", "#review-queue", "#evidence-panel")]
        assert all(box is not None for box in boxes)
        for index, left in enumerate(boxes):
            for right in boxes[index + 1:]:
                assert left["x"] + left["width"] <= right["x"] or right["x"] + right["width"] <= left["x"]


def test_workbench_has_no_page_errors_or_failed_same_origin_requests(
    page: Page, live_server_url: str
) -> None:
    page_errors = []
    console_errors = []
    failed_requests = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    page.on("requestfailed", lambda request: failed_requests.append(request.url) if request.url.startswith(live_server_url) else None)
    page.goto(live_server_url)
    assert page_errors == []
    assert console_errors == []
    assert failed_requests == []


def test_capture_workbench_screenshots(page: Page, live_server_url: str) -> None:
    artifact_dir = Path(__file__).parents[1] / "artifacts" / "screenshots"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    page.set_viewport_size({"width": 1440, "height": 900})

    ambiguous = _review(status="AWAITING_MAPPING_CONFIRMATION")
    _mock_review(page, ambiguous)
    page.goto(f"{live_server_url}/?review=review-ui")
    page.screenshot(path=str(artifact_dir / "01-ambiguous-product.png"), full_page=True)

    page.unroute_all()
    evidence = _review(status="AWAITING_FINDING_REVIEW")
    _mock_review(page, evidence)
    page.goto(f"{live_server_url}/?review=review-ui")
    page.get_by_role("button", name="重复活性成分", exact=True).click()
    page.screenshot(path=str(artifact_dir / "02-finding-evidence.png"), full_page=True)

    page.unroute_all()
    preview = _review(status="SIGNED_OFF")
    preview["findings"][0]["status"] = "ACCEPTED"
    preview["writebackStatus"] = "PREPARED"
    preview["writebackJob"] = {
        "jobId": "writeback-review-ui-3",
        "reviewVersion": 3,
        "expectedVersion": 3,
        "bundleHash": "a" * 64,
        "resources": [
            {"resourceType": "DetectedIssue", "id": "mr-di-route", "summary": "Verified route mismatch"},
            {"resourceType": "Task", "id": "mr-task-pregnancy", "summary": "Pregnancy status missing"},
        ],
        "warnings": ["Synthetic data only"],
        "blockedFindings": [{"findingId": "gap-1", "reason": "MISSING_PAIRED_EVIDENCE"}],
    }
    _mock_review(page, preview)
    page.goto(f"{live_server_url}/?review=review-ui")
    page.screenshot(path=str(artifact_dir / "03-writeback-preview.png"), full_page=True)
