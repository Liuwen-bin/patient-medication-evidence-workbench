# Pharmacist Workbench Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有报告导向的末端操作改造成药师完成 Finding 审核、预览 FHIR 资源、明确确认并查看写回结果的完整工作台闭环。

**Architecture:** FastAPI 用独立的 `/complete`、`/writeback/prepare` 和 `/writeback/commit` handler 管理审核与写回状态，继续使用 API key、reviewer binding、review lock 和乐观版本。前端保持原生安全 DOM 渲染，桌面三栏显示患者/队列/证据，移动端用三个 tab；报告入口降为已完成 review 的次要导出操作。

**Tech Stack:** FastAPI、Pydantic 2、SQLite、原生 HTML/CSS/JavaScript、pytest、Playwright

**Spec:** `docs/superpowers/specs/2026-09-04-patient-medication-evidence-workbench-design.md`

## Global Constraints

- 主业务按钮文案为“完成审核”，不得继续把“提交报告”作为流程终点。
- 工作台创建 review 时必须输入 `question`，顶部同时保留 patientId、asOf、reviewerId 和开始/恢复操作。
- 桌面为患者上下文、核查队列、证据详情三栏；移动端保持相同 DOM 顺序并一次只显示一个 tab。
- 每个动态字段使用 `textContent`、`createElement`、属性白名单或现有安全 URL helper，不使用 `innerHTML` 渲染 API 内容。
- MedicationRequest 统一显示为“活动用药医嘱”，并在可见区域说明这不等于患者实际服药。
- 写回 preview 必须显示患者、资源类型/ID/数量、Finding/MedicationRequest/证据对应关系、Task 原因、reviewer、reviewVersion、bundleHash、warnings 和 blocked Findings。
- commit 必须携带当前 API `expectedVersion`、preview 中的 `bundleHash`、`reviewerId` 和字面值 `confirmed=true`。
- HTTP 409 后只刷新最新 snapshot，不自动重放完成、Finding 或写回操作。
- 1440x900、1024x768、390x844 和 360x800 均不得出现页面水平溢出、控件遮挡或文本超出按钮。
- JSON/HTML 报告只作为次要导出入口，不阻塞完成审核或写回。

---

## File Structure

| File | Responsibility |
|---|---|
| `src/medication_review_agent/api.py` | 完成、prepare、commit、查询写回状态和统一 reviewer guard |
| `src/medication_review_agent/web/index.html` | 顶部核查输入、三栏容器、写回 preview/confirm dialog、移动 tab |
| `src/medication_review_agent/web/api-client.js` | typed request helper 与 409 refresh 错误类型 |
| `src/medication_review_agent/web/app.js` | 页面状态、事件处理和主业务命令 |
| `src/medication_review_agent/web/render.js` | snapshot、Finding、evidence、preview、result 的安全 DOM 投影 |
| `src/medication_review_agent/web/styles.css` | 稳定三栏、写回表格/dialog、响应式 tab 和 focus 状态 |
| `tests/test_api.py` | endpoint 状态、身份、版本、hash 和失败持久化 |
| `tests/test_web_decisions.py` | 完成/prepare/commit、dialog、冲突刷新和报告次级入口 |
| `tests/test_web_render.py` | 完整数据投影和安全 DOM 文案 |
| `tests/test_web_visual.py` | 四个 viewport 的尺寸、溢出、截图和页面错误 |

### Task 1: Complete Review And Writeback API Lifecycle

**Files:**
- Modify: `src/medication_review_agent/api.py:72`
- Modify: `src/medication_review_agent/models.py`
- Modify: `src/medication_review_agent/repository.py:28`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `WritebackCoordinator` and schema 1.1 `ReviewSnapshot` from plans 01-02.
- Produces: `POST /api/reviews/{reviewId}/complete`, `POST /api/reviews/{reviewId}/writeback/prepare`, `POST /api/reviews/{reviewId}/writeback/commit`, `GET /api/reviews/{reviewId}/writeback`.

- [ ] **Step 1: Write failing completion endpoint tests**

```python
def test_complete_review_resumes_final_human_interrupt(client: TestClient) -> None:
    snapshot = review_ready_for_completion(client)
    response = client.post(
        f"/api/reviews/{snapshot['reviewId']}/complete",
        headers=reviewer_headers("pharmacist-1"),
        json={"expectedVersion": snapshot["version"], "reviewerId": "pharmacist-1"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "SIGNED_OFF"
    assert response.json()["writebackStatus"] == "NOT_REQUESTED"


def test_complete_rejects_pending_finding(client: TestClient) -> None:
    snapshot = review_with_pending_finding(client)
    response = client.post(
        f"/api/reviews/{snapshot['reviewId']}/complete",
        headers=reviewer_headers("pharmacist-1"),
        json={"expectedVersion": snapshot["version"], "reviewerId": "pharmacist-1"},
    )
    assert response.status_code == 409
```

- [ ] **Step 2: Add strict request models and reusable reviewer guard**

```python
class CompleteReviewRequest(BaseModel):
    expectedVersion: int = Field(ge=0)
    reviewerId: str = Field(min_length=1, max_length=100)


class PrepareWritebackRequest(CompleteReviewRequest):
    pass


class CommitWritebackRequest(CompleteReviewRequest):
    bundleHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmed: Literal[True]


def require_reviewer(body_reviewer_id: str, header_reviewer_id: str | None) -> None:
    if configured_key and not configured_reviewer_id:
        raise HTTPException(503, "REVIEW_API_REVIEWER_ID is required for reviewer decisions")
    if (
        not header_reviewer_id
        or header_reviewer_id != body_reviewer_id
        or (configured_reviewer_id and header_reviewer_id != configured_reviewer_id)
    ):
        raise HTTPException(403, "Reviewer identity does not match configured reviewer")
```

Use this same guard for `/decisions`, `/complete`, `/writeback/prepare` and `/writeback/commit`.

- [ ] **Step 3: Implement `/complete` as the existing final interrupt resume**

Resume the graph with `Command(resume={"action": "SIGN_OFF", "reviewerId": body.reviewerId})` through the existing mutation journal. Accept only `READY_FOR_SIGN_OFF`; preserve `SIGNED_OFF` as the domain value but expose UI copy “审核已完成”. Remove final sign-off handling from the generic `/decisions` accepted-state list.

- [ ] **Step 4: Write failing prepare endpoint tests**

```python
def test_prepare_persists_preview_and_advances_writeback_state(client: TestClient) -> None:
    snapshot = signed_review(client)
    response = client.post(
        f"/api/reviews/{snapshot['reviewId']}/writeback/prepare",
        headers=reviewer_headers("pharmacist-1"),
        json={"expectedVersion": snapshot["version"], "reviewerId": "pharmacist-1"},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["writebackStatus"] == "PREPARED"
    assert body["writebackJob"]["bundleHash"] == "a" * 64


def test_prepare_rejects_unsigned_or_stale_review(client: TestClient) -> None:
    snapshot = running_review(client)
    response = client.post(
        f"/api/reviews/{snapshot['reviewId']}/writeback/prepare",
        headers=reviewer_headers("pharmacist-1"),
        json={"expectedVersion": snapshot["version"] - 1, "reviewerId": "pharmacist-1"},
    )
    assert response.status_code == 409
```

- [ ] **Step 5: Implement prepare with one optimistic save**

Under `review_lock`, reload the snapshot, validate `expectedVersion`, `SIGNED_OFF`, reviewer identity and `writebackStatus in {NOT_REQUESTED, FAILED}`. Call `WritebackCoordinator.prepare`; on success set `writebackStatus=PREPARED`, store the returned job, clear `writebackError`, and save with the request version. On `WritebackError`, set `writebackStatus=FAILED`, store `WritebackFailure(code, message, retryable)`, save once, then return HTTP 422 for policy errors or 502 for retryable upstream errors.

- [ ] **Step 6: Write failing commit, idempotency and conflict tests**

```python
def test_commit_requires_current_version_hash_reviewer_and_true(client: TestClient) -> None:
    snapshot = prepared_review(client)
    response = client.post(
        f"/api/reviews/{snapshot['reviewId']}/writeback/commit",
        headers=reviewer_headers("pharmacist-1"),
        json={
            "expectedVersion": snapshot["version"],
            "reviewerId": "pharmacist-1",
            "bundleHash": snapshot["writebackJob"]["bundleHash"],
            "confirmed": True,
        },
    )
    assert response.status_code == 200
    assert response.json()["writebackStatus"] == "COMMITTED"


def test_writeback_409_never_calls_health_mcp(client: TestClient, fake_health) -> None:
    snapshot = prepared_review(client)
    response = client.post(
        f"/api/reviews/{snapshot['reviewId']}/writeback/commit",
        headers=reviewer_headers("pharmacist-1"),
        json={"expectedVersion": 0, "reviewerId": "pharmacist-1", "bundleHash": "b" * 64, "confirmed": True},
    )
    assert response.status_code == 409
    assert fake_health.commit_calls == 0
```

- [ ] **Step 7: Implement commit state semantics**

The API body's `expectedVersion` is the current ReviewRepository optimistic version. The value sent to Health MCP is `snapshot.writebackJob.expectedVersion`, frozen during preview; never substitute the post-preview repository version. Require the body hash to equal the stored job hash before calling MCP.

On success, store the commit result, set `COMMITTED`, clear `writebackError`, append a redacted audit event, and save once. On failure, preserve the same job/hash, set `FAILED`, store the typed failure and save once so retry calls can reuse the immutable preview.

- [ ] **Step 8: Implement writeback query response**

```python
@app.get("/api/reviews/{review_id}/writeback")
def read_writeback(review_id: str):
    snapshot = get_review(review_id)
    return {
        "reviewId": snapshot.reviewId,
        "reviewVersion": snapshot.version,
        "status": snapshot.writebackStatus,
        "job": snapshot.writebackJob,
        "error": snapshot.writebackError,
    }
```

- [ ] **Step 9: Run API tests and commit**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/test_api.py tests/test_writeback.py -q -p pytest_asyncio.plugin
git add src/medication_review_agent/api.py src/medication_review_agent/models.py src/medication_review_agent/repository.py tests/test_api.py
git commit -m "feat: expose guarded review completion and writeback APIs"
```

Expected: PASS; stale version/hash/reviewer requests never reach Health MCP; failed commit keeps the completed review and immutable preview.

### Task 2: Capture Review Goal And Rename The Primary Completion Action

**Files:**
- Modify: `src/medication_review_agent/web/index.html`
- Modify: `src/medication_review_agent/web/app.js`
- Modify: `src/medication_review_agent/web/api-client.js`
- Modify: `tests/test_web_render.py`
- Modify: `tests/test_web_decisions.py`

**Interfaces:**
- Consumes: create API requiring `{patientId, asOf, question}` and complete API from Task 1.
- Produces: required `#review-question` field, `completeReview(reviewId, expectedVersion, reviewerId)`, and a primary “完成审核” command.

- [ ] **Step 1: Write failing creation-form tests**

```python
def test_start_review_sends_question_and_uses_medication_order_copy(page: Page, live_server_url: str) -> None:
    page.goto(live_server_url)
    page.fill("#patient-id", "demo-001")
    page.fill("#review-question", "核查成分、途径和标签警告")
    page.fill("#reviewer-id", "pharmacist-1")
    with page.expect_request("**/api/reviews") as request:
        page.click("#start-review")
    assert request.value.post_data_json["question"] == "核查成分、途径和标签警告"
    assert "活动用药医嘱" in page.locator("body").inner_text()
```

- [ ] **Step 2: Add the compact top-bar control**

Add a label/input pair with `id="review-question"`, `required`, `maxlength="500"`, and a default demonstration value “核查活动用药医嘱的成分、途径、剂型和标签警告”. Keep the label visible and allow it to wrap at mobile widths.

- [ ] **Step 3: Update API client commands**

```javascript
export function createReview(payload) {
  return requestJson("/api/reviews", { method: "POST", body: payload });
}

export function completeReview(reviewId, payload) {
  return requestJson(`/api/reviews/${encodeURIComponent(reviewId)}/complete`, {
    method: "POST",
    body: payload,
  });
}
```

The generic decision endpoint continues to handle patient, mapping and Finding decisions only.

- [ ] **Step 4: Replace report-first completion copy**

When state is `READY_FOR_SIGN_OFF`, render the primary button text “完成审核”. When state is `SIGNED_OFF`, render “审核已完成” and expose “导出 JSON”/“导出 HTML” as secondary links. The button handler posts `expectedVersion` and `reviewerId` to `/complete`.

- [ ] **Step 5: Test escape and 409 behavior**

Keep the existing confirmation dialog escape behavior. On 409, close the dialog, fetch the latest review once, render it, and show “状态已更新，请重新确认”; do not replay the previous completion request.

- [ ] **Step 6: Run focused browser tests and commit**

```powershell
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD -ErrorAction SilentlyContinue
python -m pytest tests/test_web_render.py tests/test_web_decisions.py -q
git add src/medication_review_agent/web/index.html src/medication_review_agent/web/app.js src/medication_review_agent/web/api-client.js tests/test_web_render.py tests/test_web_decisions.py
git commit -m "feat: make review completion the primary workbench action"
```

Expected: PASS; start requests include question; report export is not presented before completion.

### Task 3: Render Writeback Preview And Explicit Confirmation

**Files:**
- Modify: `src/medication_review_agent/web/index.html`
- Modify: `src/medication_review_agent/web/app.js`
- Modify: `src/medication_review_agent/web/api-client.js`
- Modify: `src/medication_review_agent/web/render.js`
- Modify: `src/medication_review_agent/web/styles.css`
- Modify: `tests/test_web_render.py`
- Modify: `tests/test_web_decisions.py`

**Interfaces:**
- Consumes: schema 1.1 fields `writebackStatus`, `writebackJob`, `writebackError` and API endpoints from Task 1.
- Produces: prepare command, preview table, modal confirmation, commit result and retry affordance.

- [ ] **Step 1: Write failing complete-preview-commit test**

```python
def test_signed_review_previews_then_confirms_writeback(page: Page, live_server_url: str) -> None:
    posts: list[tuple[str, dict]] = []
    mock_writeback_lifecycle(page, posts)
    page.goto(live_server_url)
    page.click("#prepare-writeback")
    expect(page.locator("#writeback-preview")).to_contain_text("DetectedIssue")
    expect(page.locator("#writeback-preview")).to_contain_text("Task")
    expect(page.locator("#writeback-preview")).to_contain_text("bundleHash")
    page.click("#open-writeback-confirmation")
    page.click("#confirm-writeback")
    assert posts[-1][0].endswith("/writeback/commit")
    assert posts[-1][1]["confirmed"] is True
    expect(page.locator("#writeback-result")).to_contain_text("写回完成")
```

- [ ] **Step 2: Add API client methods**

```javascript
export function prepareWriteback(reviewId, payload) {
  return requestJson(`/api/reviews/${encodeURIComponent(reviewId)}/writeback/prepare`, {
    method: "POST",
    body: payload,
  });
}

export function commitWriteback(reviewId, payload) {
  return requestJson(`/api/reviews/${encodeURIComponent(reviewId)}/writeback/commit`, {
    method: "POST",
    body: payload,
  });
}
```

- [ ] **Step 3: Build preview DOM with stable sections**

Create unframed sections for summary, resources, blocked Findings, warnings and hash metadata. Each resource row shows resource type, ID, source Finding/unresolved ID, implicated MedicationRequest and evidence refs. Use `document.createElement`, `textContent`, `append`, and the existing safe link helper; do not concatenate HTML strings.

```javascript
export function renderWritebackPreview(container, job) {
  container.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = `FHIR 写回预览 · ${job.resources.length} 个资源`;
  container.append(heading, buildWritebackMetadata(job), buildResourceTable(job.resources));
}
```

- [ ] **Step 4: Add a high-friction confirmation dialog**

The dialog displays patientRef, reviewerId, resource count, exact bundleHash and “只新增资源，不修改原始临床记录”. The commit button remains disabled until the pharmacist checks a native checkbox `#writeback-confirmed`. Closing by Escape or cancel sends no request.

- [ ] **Step 5: Handle retryable and non-retryable failures**

For `writebackStatus=FAILED`, render the stored machine-readable code and message. Show “重试写回” only when `writebackError.retryable=true` and `writebackJob` is present; retry uses the stored hash and current snapshot version. A validation failure without a job offers “重新生成预览” after the review state is corrected.

- [ ] **Step 6: Add XSS regression assertions**

```python
def test_writeback_preview_renders_untrusted_summary_as_text(page: Page, live_server_url: str) -> None:
    mock_preview(page, summary='<img src=x onerror="window.previewPwned=true">')
    page.goto(live_server_url)
    expect(page.locator("#writeback-preview")).to_contain_text("<img src=x")
    assert page.locator("#writeback-preview img").count() == 0
    assert page.evaluate("window.previewPwned") is None
```

- [ ] **Step 7: Run browser interaction tests and commit**

```powershell
python -m pytest tests/test_web_render.py tests/test_web_decisions.py -q
git add src/medication_review_agent/web/index.html src/medication_review_agent/web/app.js src/medication_review_agent/web/api-client.js src/medication_review_agent/web/render.js src/medication_review_agent/web/styles.css tests/test_web_render.py tests/test_web_decisions.py
git commit -m "feat: add FHIR preview and confirmation workflow"
```

Expected: PASS; Escape causes no mutation; commit is impossible before checking confirmation; injected markup is visible only as text.

### Task 4: Stabilize Three-Column And Mobile Layouts

**Files:**
- Modify: `src/medication_review_agent/web/styles.css`
- Modify: `src/medication_review_agent/web/render.js`
- Modify: `tests/test_web_visual.py`

**Interfaces:**
- Consumes: all context, Finding, evidence and writeback states from Tasks 1-3.
- Produces: stable panel sizes at 1440x900/1024x768 and tabbed views at 390x844/360x800.

- [ ] **Step 1: Extend visual fixtures with long real-shaped content**

Use a snapshot containing two medication orders, one ambiguous product list, three Findings, long SPL evidence, graph provenance warnings, one DetectedIssue, one Task and one blocked Finding. Do not use an empty-state screenshot for acceptance.

- [ ] **Step 2: Assert dimensions and overflow at all viewports**

```python
@pytest.mark.parametrize("viewport", [
    {"width": 1440, "height": 900},
    {"width": 1024, "height": 768},
    {"width": 390, "height": 844},
    {"width": 360, "height": 800},
])
def test_complete_workbench_has_no_horizontal_overflow(page: Page, live_server_url: str, viewport: dict) -> None:
    page.set_viewport_size(viewport)
    mock_complete_review(page)
    page.goto(live_server_url)
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    assert overflow <= 1
```

- [ ] **Step 3: Lock desktop grid and narrow content constraints**

Use `grid-template-columns: minmax(240px, 0.8fr) minmax(360px, 1.3fr) minmax(300px, 1fr)` above 900px. Apply `min-width: 0` to every grid child, `overflow-wrap: anywhere` to IDs/hashes/refs, fixed minimum heights to tabs/buttons, and scroll only within evidence or long-list panels.

- [ ] **Step 4: Preserve DOM order for mobile tabs**

At `max-width: 899px`, switch the workspace to one column. The tab buttons toggle `[hidden]`/`aria-selected` for `patient-panel`, `review-panel`, and `evidence-panel`; do not move DOM nodes. When a user selects a Finding on mobile, switch to the evidence tab and put focus on its heading.

- [ ] **Step 5: Capture the three required portfolio states**

Update `test_capture_workbench_screenshots` to write:

```text
artifacts/screenshots/01-ambiguous-product.png
artifacts/screenshots/02-finding-evidence.png
artifacts/screenshots/03-writeback-preview.png
```

Each screenshot uses synthetic, non-empty API fixtures and a 1440x900 viewport.

- [ ] **Step 6: Run browser and visual tests**

```powershell
python -m pytest tests/test_web_render.py tests/test_web_decisions.py tests/test_web_visual.py -q
```

Expected: PASS; no page errors, failed same-origin requests, clipped actions, overlapping text or horizontal overflow.

- [ ] **Step 7: Commit responsive completion UI**

```powershell
git add src/medication_review_agent/web/styles.css src/medication_review_agent/web/render.js tests/test_web_visual.py artifacts/screenshots/01-ambiguous-product.png artifacts/screenshots/02-finding-evidence.png artifacts/screenshots/03-writeback-preview.png
git commit -m "test: verify complete workbench across viewports"
```

## Plan-Level Verification

- [ ] Run core/API tests in a process with plugin autoload disabled; expected PASS.
- [ ] Run browser tests in a separate normal process; expected PASS.
- [ ] Exercise a stale completion and stale commit; expected one GET refresh and no automatic retry POST.
- [ ] Inspect preview DOM with an HTML-like summary; expected literal text and no injected element.
- [ ] Confirm report links are absent before `SIGNED_OFF`, secondary after completion, and never required for writeback.
- [ ] Confirm the UI can complete patient ambiguity, product ambiguity, Finding decisions, one evidence request and writeback confirmation without using a chat interface.
