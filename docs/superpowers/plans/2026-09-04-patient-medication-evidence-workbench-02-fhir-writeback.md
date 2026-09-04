# FHIR Writeback Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为药师已完成的核查生成可预览、可确认、事务化且幂等的 FHIR `DetectedIssue`、`Task` 和 `Provenance` 写回。

**Architecture:** Health Record MCP 内新增 `ReviewWritebackService`，preview 将严格白名单输入转为 transaction Bundle 并持久化不可变 job，但不触碰 `fhir_resources`；commit 校验 reviewer、version、hash 和 `confirmed=true` 后在一个 `BEGIN IMMEDIATE` 事务中只执行 insert。Medication Review Agent 负责从已验证的 snapshot 构造最小 payload，并通过同一 Health MCP gateway 调用两个写回工具；LLM 对象和 planner 均无法获得该 gateway 方法。

**Tech Stack:** Python 3.11、Pydantic 2、SQLite、FastMCP、FastAPI、pytest、pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-09-04-patient-medication-evidence-workbench-design.md`

## Global Constraints

- Health Record MCP 根目录为 `C:/Users/Administrator/Desktop/mcp/health-record-mcp/Agent`；Agent 根目录为当前 `medication_review_agent` 仓库。
- 不增加第三个 MCP 服务；只在 Health Record MCP 暴露 `validate_medication_review_writeback` 和 `commit_medication_review_writeback`。
- preview 不写 `fhir_resources`；commit 只新增，不允许 update、delete 或覆盖同 ID 不同内容的资源。
- 白名单资源固定为 `DetectedIssue`、`Task`、`Provenance`；首版不在 writeback Bundle 中生成 `DocumentReference`。
- `PRODUCT_UNMAPPED`、`PATIENT_FIELD_MISSING`、`LABEL_EVIDENCE_MISSING`、`PRODUCT_AMBIGUOUS`、`GRAPH_PROVENANCE_WARNING` 不得转换成临床 `DetectedIssue`。
- 只有状态为 `ACCEPTED`、验证错误为空且同时包含 FHIR/SPL 引用的 route/form/ingredient Finding 可转换成 `DetectedIssue`。
- 确定性 ID 输入分别为 `reviewId + findingId + schemaVersion`、`reviewId + unresolvedItemId + schemaVersion`、`reviewId + reviewVersion + bundleHash`。
- 数据库对 `(review_id, review_version)` 唯一；相同 hash 重试返回原结果，不同 hash 返回冲突。
- commit 失败不撤销药师决定，不调用 LLM，并允许使用相同 job/hash 幂等重试。
- 每个资源都标记 `urn:medication-review:agent-assisted`、`urn:medication-review:pharmacist-reviewed` 和 `urn:medication-review:data-kind|synthetic`。
- 审计不保存 API key、完整患者 payload、prompt、chain-of-thought、SQL、Cypher 或 raw tool arguments。

---

## Cross-Repository File Structure

| Repository | File | Responsibility |
|---|---|---|
| Health MCP | `mcp/writeback_service.py` | 输入 schema、FHIR builder、preview job、transaction commit、幂等 |
| Health MCP | `mcp/ehr_service.py` | `asOf` 过滤、MedicationReference 解引用和完整活动用药医嘱字段 |
| Health MCP | `mcp/mcp_server.py` | 注册两个狭窄的写回工具，不加入 LLM agent 工具集 |
| Health MCP | `tests/test_writeback_service.py` | 资源映射、白名单、患者归属、preview、rollback 和幂等 |
| Health MCP | `tests/test_mcp_server.py` | 工具名称和参数契约 |
| Review Agent | `src/medication_review_agent/models.py` | `WritebackJob` 和请求/结果模型 |
| Review Agent | `src/medication_review_agent/gateways.py` | 两个 Health MCP 写回调用方法 |
| Review Agent | `src/medication_review_agent/writeback.py` | 从 snapshot 构造最小 payload，协调 prepare/commit |
| Review Agent | `tests/test_gateways.py` | MCP 参数和 envelope 契约 |
| Review Agent | `tests/test_writeback.py` | 资格过滤、隐私投影和 coordinator 状态结果 |

## Frozen Cross-Repository Contract

Preview request:

```json
{
  "schemaVersion": "1.0",
  "reviewSchemaVersion": "1.1",
  "reviewId": "review-001",
  "reviewVersion": 7,
  "patientRef": "Patient/demo-001",
  "reviewerId": "pharmacist-001",
  "findings": [],
  "unresolvedItems": [],
  "agent": {
    "name": "medication-review-agent",
    "version": "0.1.0",
    "modelIds": ["configured-model"]
  },
  "syntheticData": true
}
```

Preview success `ToolEnvelope.data`:

```json
{
  "jobId": "writeback-review-001-7",
  "reviewId": "review-001",
  "reviewVersion": 7,
  "expectedVersion": 7,
  "patientRef": "Patient/demo-001",
  "bundleHash": "lowercase-sha256",
  "resources": [
    {"resourceType": "DetectedIssue", "id": "mr-di-lowercasehash"},
    {"resourceType": "Provenance", "id": "mr-prov-lowercasehash"}
  ],
  "warnings": [],
  "blockedFindings": []
}
```

Commit request:

```json
{
  "jobId": "writeback-review-001-7",
  "bundleHash": "lowercase-sha256",
  "expectedVersion": 7,
  "confirmed": true
}
```

Commit success `ToolEnvelope.data`:

```json
{
  "jobId": "writeback-review-001-7",
  "committed": true,
  "idempotentReplay": false,
  "bundleHash": "lowercase-sha256",
  "created": ["DetectedIssue/mr-di-lowercasehash", "Provenance/mr-prov-lowercasehash"]
}
```

### Task 1: Isolate The Health MCP Git Boundary

**Files:**
- Modify in Health MCP: `.gitignore`
- Verify only in Health MCP: all source and test files selected for the baseline commit

**Interfaces:**
- Consumes: `C:/Users/Administrator/Desktop/mcp/health-record-mcp/Agent`, which currently resolves to the parent `health-record-mcp` Git root and is largely untracked there.
- Produces: an independent Git root at the exact `Agent` directory and a user-owned GitHub repository `https://github.com/Liuwen-bin/patient-health-record-mcp`.

- [ ] **Step 1: Prove the current root is unsafe for an Agent commit**

```powershell
Set-Location C:\Users\Administrator\Desktop\mcp\health-record-mcp\Agent
$resolved = git rev-parse --show-toplevel
if ($resolved -eq (Get-Location).Path) { throw "Expected the pre-isolation root to be the parent repository" }
git status --short
```

Expected: top-level resolves to `C:/Users/Administrator/Desktop/mcp/health-record-mcp`, confirming that a parent-level `git add` would include unrelated user changes.

- [ ] **Step 2: Complete the Health MCP ignore rules before initialization**

Ensure `.gitignore` contains:

```gitignore
.env
.venv/
.codegraph/
.pytest_cache/
__pycache__/
*.py[cod]
data/
uploads/inbox/
uploads/processing/
uploads/archived/
uploads/failed/
uploads/reports/
*.sqlite
*.sqlite-*
*.tar.gz
*.zip
```

Keep `.env.example`, source, tests, README, requirements and synthetic CSV examples tracked.

- [ ] **Step 3: Initialize exactly the Agent directory**

```powershell
Set-Location C:\Users\Administrator\Desktop\mcp\health-record-mcp\Agent
git init -b main
$resolved = git rev-parse --show-toplevel
if ($resolved -ne (Get-Location).Path) { throw "Health MCP Git root is not isolated" }
git status --short
```

Expected: the top-level equals the `Agent` directory and ignored data, secrets, archives and `.codegraph` are absent from status.

- [ ] **Step 4: Scan the exact baseline before committing**

```powershell
git add .gitignore .env.example .dockerignore Dockerfile LICENSE.txt README.md main.py openai-mcp-tool.json requirements.txt mcp tests
git diff --cached --name-only
git diff --cached --check
git diff --cached | Select-String -Pattern '(api[_-]?key|authorization|bearer).{0,4}[A-Za-z0-9_-]{20,}'
```

Expected: staged files are limited to Health MCP source/config/docs/tests; the whitespace and credential scans have no findings.

- [ ] **Step 5: Commit and publish the isolated baseline**

```powershell
git commit -m "chore: establish patient health record MCP baseline"
gh repo view Liuwen-bin/patient-health-record-mcp *> $null
if ($LASTEXITCODE -ne 0) {
  gh repo create Liuwen-bin/patient-health-record-mcp --public --source . --remote origin --push
} else {
  git remote add origin https://github.com/Liuwen-bin/patient-health-record-mcp.git
  git push -u origin main
}
```

Expected: `origin` belongs to `Liuwen-bin`, `main` is synchronized, and no commit is made to `jmandel/health-record-mcp`. The Drug MCP remains read-only for this one-week scope because its current Git root resolves to `C:/Users/Administrator`; do not commit from that root.

### Task 2: Complete The Patient-Scoped Medication Read Contract

**Files:**
- Modify in Health MCP: `mcp/ehr_service.py:435`
- Modify in Health MCP: `tests/test_ehr_service.py`
- Modify: `src/medication_review_agent/models.py:74`
- Modify: `src/medication_review_agent/workflow.py:210`
- Modify: `tests/test_workflow.py`

**Interfaces:**
- Consumes: `get_medication_review_context(patientId: str | None, asOf: str | None)` and stored Patient/MedicationRequest/Medication/Observation resources.
- Produces: patient-scoped medications with `medicationReference`, dereferenced evidence, identifiers, `strength`/`strengthSource`, route text/coding, dosage-form text/coding, dosage/timing, authored/effective period and all evidence refs; observations and future-authored orders obey inclusive `asOf` date semantics.

- [ ] **Step 1: Write failing `asOf` tests in Health MCP**

```python
def test_review_context_excludes_future_orders_and_observations(service: EHRService) -> None:
    result = service.get_medication_review_context("demo-1", "2026-09-04")
    assert {item["id"] for item in result["data"]["activeMedications"]} == {"med-before"}
    assert {item["id"] for item in result["data"]["recentObservations"]} == {"obs-before"}
    assert any("current active status" in warning for warning in result["warnings"])
```

Populate `med-before.authoredOn="2026-09-04T09:00:00+08:00"`, `med-after.authoredOn="2026-09-05T09:00:00+08:00"`, and matching observations on each date. `asOf` includes all valid events through the end of the named calendar date; it does not claim historical reconstruction of MedicationRequest status.

- [ ] **Step 2: Write failing MedicationReference and coding tests**

```python
def test_review_context_dereferences_medication_and_preserves_codings(service: EHRService) -> None:
    result = service.get_medication_review_context("demo-1", "2026-09-04")
    medication = result["data"]["activeMedications"][0]
    assert medication["medicationReference"] == "Medication/medication-1"
    assert medication["identifiers"] == [{"system": "urn:ndc", "code": "0001", "display": "Drug A"}]
    assert medication["strength"] == "10 mg"
    assert medication["strengthSource"] == "Medication.code.text"
    assert medication["routeCodings"][0]["code"] == "PO"
    assert medication["dosageFormCodings"][0]["code"] == "TAB"
    assert medication["evidenceRefs"] == [
        "FHIR:Medication/medication-1",
        "FHIR:MedicationRequest/med-request-1",
    ]
```

- [ ] **Step 3: Implement normalized timestamp filtering**

Add `_occurs_on_or_before(resource: dict[str, Any], as_of: date) -> bool`. Parse ISO date/datetime values from `_resource_date`; compare timezone-aware values by their recorded calendar date and return `False` for a valid future value. A malformed non-empty date returns an `ERROR` envelope naming the resource reference instead of treating it as current.

Apply it to Observation and MedicationRequest. For MedicationRequest, preserve the explicit warning: “Active medication orders reflect current stored status; historical status at asOf cannot be reconstructed from this dataset.”

- [ ] **Step 4: Dereference Medication resources within the selected patient context**

Build a dictionary of `Medication/{id}` resources, resolve only the reference carried by the selected patient's MedicationRequest, and merge code/identifier/form/ingredient display fields without scanning by free-text name. Emit both FHIR refs and keep the request as the owning patient-scoped fact.

```python
reference = (request.get("medicationReference") or {}).get("reference")
medication_resource = medications_by_reference.get(reference)
request_concept = request.get("medicationCodeableConcept") or {}
medication_concept = (medication_resource or {}).get("code") or {}
concept = request_concept if _concept_text(request_concept) != "Unknown" else medication_concept
```

- [ ] **Step 5: Extend the Agent medication model compatibly**

Keep existing scalar `strength`, `route`, and `dosageForm` fields so current mapping calls do not break. Add `strengthSource: str | None`, `routeCodings: list[dict[str, str]]`, `dosageFormCodings: list[dict[str, str]]`, `medicationReference: str | None`, `medicationEvidenceRefs: list[str]`, `authoredOn: str | None`, and `effectivePeriod: dict[str, str] | None`. Normalize `patientEvidenceRefs` from all returned `evidenceRefs`, not only the MedicationRequest ref.

- [ ] **Step 6: Run both repositories' read-contract tests**

```powershell
Set-Location C:\Users\Administrator\Desktop\mcp\health-record-mcp\Agent
python -m pytest tests/test_ehr_service.py -q

Set-Location C:\Users\Administrator\Downloads\dm_spl_release_homeopathic\homeopathic\medication_review_agent
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/test_models.py tests/test_workflow.py -q -p pytest_asyncio.plugin
```

Expected: PASS; future facts are excluded, MedicationReference is resolved deterministically, and existing direct-concept medication fixtures still pass.

- [ ] **Step 7: Commit the read contract in each repository**

```powershell
Set-Location C:\Users\Administrator\Desktop\mcp\health-record-mcp\Agent
git add mcp/ehr_service.py tests/test_ehr_service.py
git commit -m "feat: complete medication review context contract"

Set-Location C:\Users\Administrator\Downloads\dm_spl_release_homeopathic\homeopathic\medication_review_agent
git add src/medication_review_agent/models.py src/medication_review_agent/workflow.py tests/test_workflow.py
git commit -m "feat: preserve referenced medication evidence"
```

### Task 3: Pure FHIR Resource Builder And Input Policy

**Files:**
- Create in Health MCP: `mcp/writeback_service.py`
- Create in Health MCP: `tests/test_writeback_service.py`

**Interfaces:**
- Consumes: `ReviewWritebackPayload` with completed review data only.
- Produces: `build_writeback_bundle(payload: ReviewWritebackPayload) -> BuiltWriteback`, where `BuiltWriteback.bundle` is a FHIR R4 `Bundle(type="transaction")`, `resources` is a sorted preview list, and `blockedFindings` lists excluded item IDs and machine-readable reasons.

- [ ] **Step 1: Write failing mapping and deterministic-ID tests**

```python
def test_builder_maps_verified_issue_gap_and_provenance() -> None:
    payload = valid_payload(
        findings=[accepted_route_mismatch()],
        unresolved=[missing_patient_field()],
    )
    first = build_writeback_bundle(payload)
    second = build_writeback_bundle(payload)
    types = [entry["resource"]["resourceType"] for entry in first.bundle["entry"]]
    assert types == ["DetectedIssue", "Task", "Provenance"]
    assert first.bundle == second.bundle
    assert first.bundleHash == second.bundleHash


def test_unmapped_and_missing_evidence_never_become_detected_issue() -> None:
    payload = valid_payload(findings=[accepted_evidence_gap()], unresolved=[])
    built = build_writeback_bundle(payload)
    assert all(
        entry["resource"]["resourceType"] != "DetectedIssue"
        for entry in built.bundle["entry"]
    )
```

- [ ] **Step 2: Run focused tests and verify red**

```powershell
Set-Location C:\Users\Administrator\Desktop\mcp\health-record-mcp\Agent
python -m pytest tests/test_writeback_service.py::test_builder_maps_verified_issue_gap_and_provenance tests/test_writeback_service.py::test_unmapped_and_missing_evidence_never_become_detected_issue -q
```

Expected: FAIL because `writeback_service.py` does not exist.

- [ ] **Step 3: Define strict preview payload models**

```python
DETECTED_ISSUE_TYPES = frozenset({
    "ROUTE_MISMATCH",
    "DOSAGE_FORM_MISMATCH",
    "INGREDIENT_ALLERGY_NAME_MATCH",
    "SHARED_ACTIVE_INGREDIENT",
})
TASK_TYPES = frozenset({
    "PRODUCT_AMBIGUOUS",
    "PRODUCT_UNMAPPED",
    "PATIENT_FIELD_MISSING",
    "LABEL_EVIDENCE_MISSING",
    "GRAPH_PROVENANCE_WARNING",
})


class WritebackFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findingId: str
    reviewType: str
    summary: str = Field(min_length=1, max_length=1000)
    medicationIds: list[str] = Field(default_factory=list)
    selectedProductIds: list[str] = Field(default_factory=list)
    patientEvidenceRefs: list[str] = Field(default_factory=list)
    labelEvidenceRefs: list[str] = Field(default_factory=list)
    status: Literal["ACCEPTED"]
    verificationErrors: list[str] = Field(default_factory=list, max_length=0)


class WritebackUnresolvedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unresolvedItemId: str
    kind: str
    summary: str = Field(min_length=1, max_length=1000)
    medicationIds: list[str] = Field(default_factory=list)
    evidenceRefs: list[str] = Field(default_factory=list)


class ReviewWritebackPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schemaVersion: Literal["1.0"]
    reviewSchemaVersion: Literal["1.1"]
    reviewId: str
    reviewVersion: int = Field(ge=0)
    patientRef: str = Field(pattern=r"^Patient/[A-Za-z0-9\-.]{1,64}$")
    reviewerId: str = Field(min_length=1, max_length=100)
    findings: list[WritebackFinding]
    unresolvedItems: list[WritebackUnresolvedItem]
    agent: AgentAttribution
    syntheticData: Literal[True]
```

Reject payloads with zero findings and zero unresolved items, duplicate IDs, patient references outside the selected patient, unsupported evidence-reference prefixes, or medication references not present in `patientEvidenceRefs`.

- [ ] **Step 4: Implement canonical hashing and resource IDs**

```python
def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def resource_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:40]
    return f"mr-{prefix}-{digest}"


def bundle_hash(bundle: dict[str, Any]) -> str:
    normalized = copy.deepcopy(bundle)
    for entry in normalized.get("entry", []):
        resource = entry.get("resource") or {}
        if resource.get("resourceType") == "Provenance":
            resource.pop("id", None)
    return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()
```

Sort findings and unresolved items by their IDs before building entries. Build Provenance without an ID, compute `bundleHash` from the entire canonical Bundle while omitting only that derived Provenance ID, then assign `Provenance.id = resource_id("prov", reviewId, str(reviewVersion), bundleHash)`. Verification repeats the same normalization, so all clinical content and provenance fields are hash-covered without creating a circular hash dependency. Do not place `bundleHash` inside the Bundle itself.

- [ ] **Step 5: Build the exact FHIR mapping**

For an eligible Finding, create `DetectedIssue` with:

```python
{
    "resourceType": "DetectedIssue",
    "id": resource_id("di", payload.reviewId, finding.findingId, payload.reviewSchemaVersion),
    "meta": {"tag": WRITEBACK_TAGS},
    "identifier": [{"system": "urn:medication-review:finding", "value": finding.findingId}],
    "status": "final",
    "code": {"coding": [{"system": "urn:medication-review:finding-type", "code": finding.reviewType}]},
    "detail": finding.summary,
    "subject": {"reference": payload.patientRef},
    "implicated": [{"reference": normalize_medication_reference(value)} for value in sorted(finding.medicationIds)],
    "evidence": [{
        "detail": [{"display": ref} for ref in sorted(finding.patientEvidenceRefs + finding.labelEvidenceRefs)]
    }],
}
```

For every unresolved item in `TASK_TYPES`, create a `Task(status="requested", intent="order", for=patientRef, description=summary)` with identifier system `urn:medication-review:unresolved-item`. Build one Provenance targeting every DetectedIssue/Task, with recorded timestamp, reviewer agent, software agent, model entity identifiers and source evidence entities. No patient name or full source payload is copied.

- [ ] **Step 6: Add rejection tests for writeback-ineligible Findings**

```python
@pytest.mark.parametrize("mutation, reason", [
    ({"status": "PENDING"}, "FINDING_NOT_ACCEPTED"),
    ({"verificationErrors": ["invalid_reference"]}, "FINDING_NOT_VERIFIED"),
    ({"patientEvidenceRefs": []}, "MISSING_PAIRED_EVIDENCE"),
    ({"labelEvidenceRefs": []}, "MISSING_PAIRED_EVIDENCE"),
    ({"reviewType": "LABEL_EVIDENCE_REVIEW"}, "NOT_A_WRITEBACK_ISSUE"),
])
def test_ineligible_finding_is_blocked_not_promoted(mutation: dict[str, Any], reason: str) -> None:
    finding = accepted_route_mismatch().model_copy(update=mutation)
    built = build_writeback_bundle(valid_payload(findings=[finding]))
    assert built.blockedFindings == [{"findingId": finding.findingId, "reason": reason}]
    assert not any(entry["resource"]["resourceType"] == "DetectedIssue" for entry in built.bundle["entry"])
```

- [ ] **Step 7: Run builder tests**

```powershell
python -m pytest tests/test_writeback_service.py -q -k 'builder or ineligible or deterministic'
```

Expected: PASS; Bundle entry order and hash remain stable across runs.

- [ ] **Step 8: Commit the pure builder in the Health MCP repository**

```powershell
git add mcp/writeback_service.py tests/test_writeback_service.py
git commit -m "feat: build deterministic medication review bundles"
```

### Task 4: Persist Immutable Preview Jobs Without FHIR Mutation

**Files:**
- Modify in Health MCP: `mcp/writeback_service.py`
- Modify in Health MCP: `tests/test_writeback_service.py`

**Interfaces:**
- Consumes: `ReviewWritebackPayload` and current `fhir_resources` patient/MedicationRequest rows.
- Produces: `ReviewWritebackService.validate_medication_review_writeback(payload: dict[str, Any]) -> dict[str, Any]` returning ToolEnvelope `1.0`; creates one immutable `medication_review_writeback_jobs` row per `(review_id, review_version)`.

- [ ] **Step 1: Write a failing zero-mutation preview test**

```python
def test_preview_persists_job_but_does_not_write_fhir_resources(db_path: Path) -> None:
    service = ReviewWritebackService(db_path)
    before = fetch_resource_bytes(db_path)
    result = service.validate_medication_review_writeback(valid_payload_dict())
    after = fetch_resource_bytes(db_path)
    assert result["status"] == "OK"
    assert result["data"]["jobId"] == "writeback-review-001-7"
    assert before == after
    assert fetch_job(db_path, "review-001", 7)["status"] == "prepared"
```

- [ ] **Step 2: Add the writeback job schema**

```sql
CREATE TABLE IF NOT EXISTS medication_review_writeback_jobs (
    job_id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL,
    review_version INTEGER NOT NULL,
    patient_ref TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    bundle_hash TEXT NOT NULL,
    expected_version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('prepared', 'committed', 'failed')),
    bundle_json TEXT NOT NULL,
    preview_json TEXT NOT NULL,
    result_json TEXT,
    created_at TEXT NOT NULL,
    committed_at TEXT,
    UNIQUE(review_id, review_version)
);
```

Initialize this table in `ReviewWritebackService.__init__` using a normal read/write SQLite connection with foreign keys and `busy_timeout=10000`.

- [ ] **Step 3: Validate patient ownership against stored FHIR**

Read the exact Patient row and every referenced MedicationRequest row. Require each MedicationRequest `subject.reference` to equal `payload.patientRef`. Require every `FHIR:` evidence reference to identify an existing resource belonging to the same patient. Treat missing rows and mismatches as `ERROR`, never as empty evidence.

```python
def belongs_to_patient(resource: dict[str, Any], patient_ref: str) -> bool:
    if resource.get("resourceType") == "Patient":
        return f"Patient/{resource.get('id')}" == patient_ref
    subject = resource.get("subject") or resource.get("patient") or {}
    return subject.get("reference") == patient_ref
```

- [ ] **Step 4: Persist or replay the immutable preview**

Use `job_id = f"writeback-{reviewId}-{reviewVersion}"`. If no row exists, insert the canonical payload hash, Bundle and preview. If a row exists with the same payload hash and bundle hash, return the stored preview. If the same review/version has different content, return ToolEnvelope `ERROR` with `data.error.code="WRITEBACK_VERSION_CONFLICT"`, `retryable=false`.

- [ ] **Step 5: Test patient and version conflicts**

```python
def test_preview_rejects_cross_patient_medication_reference(db_path: Path) -> None:
    payload = valid_payload_dict()
    payload["findings"][0]["medicationIds"] = ["MedicationRequest/other-patient"]
    result = ReviewWritebackService(db_path).validate_medication_review_writeback(payload)
    assert result["status"] == "ERROR"
    assert result["data"]["error"]["code"] == "PATIENT_SCOPE_VIOLATION"


def test_same_review_version_with_different_bundle_is_conflict(db_path: Path) -> None:
    service = ReviewWritebackService(db_path)
    assert service.validate_medication_review_writeback(valid_payload_dict())["status"] == "OK"
    changed = valid_payload_dict()
    changed["findings"][0]["summary"] = "different verified summary"
    conflict = service.validate_medication_review_writeback(changed)
    assert conflict["data"]["error"]["code"] == "WRITEBACK_VERSION_CONFLICT"
```

- [ ] **Step 6: Run preview tests and commit**

```powershell
python -m pytest tests/test_writeback_service.py -q -k 'preview or patient or version'
git add mcp/writeback_service.py tests/test_writeback_service.py
git commit -m "feat: persist immutable writeback previews"
```

Expected: tests PASS; preview creates only a job row and leaves every `fhir_resources.json` byte unchanged.

### Task 5: Transactional Commit, Rollback, And Idempotency

**Files:**
- Modify in Health MCP: `mcp/writeback_service.py`
- Modify in Health MCP: `tests/test_writeback_service.py`

**Interfaces:**
- Consumes: `jobId: str`, `bundleHash: str`, `expectedVersion: int`, `confirmed: bool`.
- Produces: `ReviewWritebackService.commit_medication_review_writeback(job_id: str, bundle_hash: str, expected_version: int, confirmed: bool) -> dict[str, Any]`.

- [ ] **Step 1: Write failing confirmation and hash tests**

```python
@pytest.mark.parametrize("confirmed", [False, None])
def test_commit_requires_literal_true(db_path: Path, confirmed: bool | None) -> None:
    service, preview = prepared_service(db_path)
    result = service.commit_medication_review_writeback(
        preview["jobId"], preview["bundleHash"], preview["expectedVersion"], confirmed
    )
    assert result["status"] == "ERROR"
    assert count_writeback_resources(db_path) == 0


def test_commit_rejects_hash_or_version_mismatch(db_path: Path) -> None:
    service, preview = prepared_service(db_path)
    wrong = service.commit_medication_review_writeback(preview["jobId"], "0" * 64, 999, True)
    assert wrong["data"]["error"]["code"] == "WRITEBACK_CONFIRMATION_MISMATCH"
    assert count_writeback_resources(db_path) == 0
```

- [ ] **Step 2: Implement one insert-only SQLite transaction**

```python
connection.execute("BEGIN IMMEDIATE")
for entry in bundle["entry"]:
    resource = entry["resource"]
    existing = connection.execute(
        "SELECT json FROM fhir_resources WHERE resource_type = ? AND resource_id = ?",
        (resource["resourceType"], resource["id"]),
    ).fetchone()
    encoded = canonical_json(resource)
    if existing is not None and existing["json"] != encoded:
        raise WritebackConflict("FHIR_RESOURCE_CONFLICT")
    if existing is None:
        connection.execute(
            "INSERT INTO fhir_resources(resource_type, resource_id, json) VALUES (?, ?, ?)",
            (resource["resourceType"], resource["id"], encoded),
        )
connection.execute(
    "UPDATE medication_review_writeback_jobs SET status='committed', result_json=?, committed_at=? WHERE job_id=? AND status='prepared'",
    (canonical_json(result), now, job_id),
)
connection.commit()
```

On any exception, call `connection.rollback()`, keep the job retryable as `prepared` for transient database failures, and return an `ERROR` envelope with `retryable` set from an explicit error-code map.

- [ ] **Step 3: Write rollback and source immutability tests**

```python
def test_failed_transaction_rolls_back_all_new_resources(db_path: Path, monkeypatch) -> None:
    service, preview = prepared_service(db_path, findings=[accepted_route_mismatch()], unresolved=[missing_patient_field()])
    monkeypatch.setattr(service, "_insert_resource", fail_on_second_insert())
    result = service.commit_medication_review_writeback(preview["jobId"], preview["bundleHash"], 7, True)
    assert result["status"] == "ERROR"
    assert count_writeback_resources(db_path) == 0
    assert fetch_job(db_path, "review-001", 7)["status"] == "prepared"


def test_commit_never_changes_original_clinical_resources(db_path: Path) -> None:
    before = fetch_original_resource_bytes(db_path)
    service, preview = prepared_service(db_path)
    result = service.commit_medication_review_writeback(preview["jobId"], preview["bundleHash"], 7, True)
    assert result["status"] == "OK"
    assert fetch_original_resource_bytes(db_path) == before
```

- [ ] **Step 4: Write and implement idempotent replay**

```python
def test_same_commit_is_idempotent(db_path: Path) -> None:
    service, preview = prepared_service(db_path)
    first = service.commit_medication_review_writeback(preview["jobId"], preview["bundleHash"], 7, True)
    count = count_writeback_resources(db_path)
    second = service.commit_medication_review_writeback(preview["jobId"], preview["bundleHash"], 7, True)
    assert second["status"] == "OK"
    assert second["data"]["created"] == first["data"]["created"]
    assert second["data"]["idempotentReplay"] is True
    assert count_writeback_resources(db_path) == count
```

When job status is `committed`, return the stored `result_json` after setting `idempotentReplay=true`; do not begin a write transaction.

- [ ] **Step 5: Run the complete service suite and commit**

```powershell
python -m pytest tests/test_writeback_service.py -q
git add mcp/writeback_service.py tests/test_writeback_service.py
git commit -m "feat: commit FHIR writeback transactionally"
```

Expected: PASS; rollback leaves zero new resources; repeated commit leaves counts unchanged.

### Task 6: Expose Two MCP Tools And Keep Them Out Of The Model Toolset

**Files:**
- Modify in Health MCP: `mcp/mcp_server.py:40`
- Modify in Health MCP: `mcp/llm_agent.py:39`
- Modify in Health MCP: `tests/test_mcp_server.py`
- Create in Health MCP: `tests/test_llm_agent_tools.py`

**Interfaces:**
- Consumes: `ReviewWritebackService` from Tasks 1-3.
- Produces: MCP tools `validate_medication_review_writeback(payload)` and `commit_medication_review_writeback(jobId, bundleHash, expectedVersion, confirmed)`.

- [ ] **Step 1: Write failing MCP registration tests**

```python
@pytest.mark.asyncio
async def test_server_exposes_narrow_writeback_tools() -> None:
    server = create_mcp_server(fake_ehr, writeback_service=fake_writeback)
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert "validate_medication_review_writeback" in tools
    assert "commit_medication_review_writeback" in tools


def test_model_tool_list_excludes_writeback() -> None:
    names = {tool.name for tool in build_tools(fake_ehr)}
    assert "validate_medication_review_writeback" not in names
    assert "commit_medication_review_writeback" not in names
```

- [ ] **Step 2: Inject the service into `create_mcp_server`**

```python
def create_mcp_server(
    service: EHRService,
    *,
    import_service: ImportService | None = None,
    writeback_service: ReviewWritebackService | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
) -> FastMCP:
    writeback_service = writeback_service or ReviewWritebackService(service.db_path)
```

- [ ] **Step 3: Register exact tool signatures**

```python
@mcp.tool()
def validate_medication_review_writeback(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and preview an already pharmacist-reviewed writeback; does not modify FHIR resources."""
    return writeback_service.validate_medication_review_writeback(payload)


@mcp.tool()
def commit_medication_review_writeback(
    jobId: str,
    bundleHash: str,
    expectedVersion: int,
    confirmed: bool,
) -> dict[str, Any]:
    """Commit one previously previewed bundle when every confirmation field matches."""
    return writeback_service.commit_medication_review_writeback(
        jobId, bundleHash, expectedVersion, confirmed
    )
```

Do not add either function to `mcp/llm_agent.py:build_tools`.

- [ ] **Step 4: Run MCP and existing Health tests**

```powershell
python -m pytest tests/test_mcp_server.py tests/test_llm_agent_tools.py tests/test_writeback_service.py tests/test_ehr_service.py tests/test_import_service.py -q
```

Expected: PASS; FastMCP exposes both calls while the model-facing tool set excludes them.

- [ ] **Step 5: Commit MCP exposure**

```powershell
git add mcp/mcp_server.py tests/test_mcp_server.py tests/test_llm_agent_tools.py
git commit -m "feat: expose guarded medication review writeback tools"
```

### Task 7: Add Typed Health Gateway Calls In The Review Agent

**Files:**
- Modify: `src/medication_review_agent/gateways.py:87`
- Modify: `src/medication_review_agent/models.py`
- Modify: `tests/test_gateways.py`

**Interfaces:**
- Consumes: the frozen MCP request/response contract above.
- Produces: `HealthRecordGateway.validate_writeback(payload: dict[str, Any]) -> TimedToolResult` and `HealthRecordGateway.commit_writeback(job_id: str, bundle_hash: str, expected_version: int, confirmed: bool) -> TimedToolResult`.

- [ ] **Step 1: Write failing exact-argument tests**

```python
@pytest.mark.asyncio
async def test_health_gateway_calls_preview_with_payload() -> None:
    caller = FakeCaller({"validate_medication_review_writeback": envelope(data={"jobId": "job-1"})})
    gateway = HealthRecordGateway(caller)
    await gateway.validate_writeback({"schemaVersion": "1.0", "reviewId": "review-1"})
    assert caller.calls == [("validate_medication_review_writeback", {
        "payload": {"schemaVersion": "1.0", "reviewId": "review-1"}
    })]


@pytest.mark.asyncio
async def test_health_gateway_commit_preserves_confirmation_fields() -> None:
    caller = FakeCaller({"commit_medication_review_writeback": envelope()})
    gateway = HealthRecordGateway(caller)
    await gateway.commit_writeback("job-1", "a" * 64, 7, True)
    assert caller.calls[-1] == ("commit_medication_review_writeback", {
        "jobId": "job-1", "bundleHash": "a" * 64, "expectedVersion": 7, "confirmed": True,
    })
```

- [ ] **Step 2: Implement typed gateway methods**

```python
async def validate_writeback(self, payload: dict[str, Any]) -> TimedToolResult:
    return await self._call("validate_medication_review_writeback", {"payload": payload})


async def commit_writeback(
    self, job_id: str, bundle_hash: str, expected_version: int, confirmed: bool,
) -> TimedToolResult:
    return await self._call("commit_medication_review_writeback", {
        "jobId": job_id,
        "bundleHash": bundle_hash,
        "expectedVersion": expected_version,
        "confirmed": confirmed,
    })
```

- [ ] **Step 3: Run gateway tests and commit**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/test_gateways.py -q -p pytest_asyncio.plugin
git add src/medication_review_agent/gateways.py src/medication_review_agent/models.py tests/test_gateways.py
git commit -m "feat: add typed Health MCP writeback gateway"
```

Expected: PASS; ToolEnvelope validation rejects malformed preview or commit responses.

### Task 8: Build A Minimal Writeback Payload And Coordinator

**Files:**
- Create: `src/medication_review_agent/writeback.py`
- Create: `tests/test_writeback.py`
- Modify: `src/medication_review_agent/workflow.py:82`

**Interfaces:**
- Consumes: `ReviewSnapshot` with `status=SIGNED_OFF`, `reviewer_id`, and `HealthRecordGateway`.
- Produces: `build_writeback_payload(snapshot: ReviewSnapshot, reviewer_id: str) -> dict[str, Any]`, `WritebackCoordinator.prepare(snapshot, reviewer_id) -> WritebackJob`, `WritebackCoordinator.commit(job, confirmed) -> dict[str, Any]`.

- [ ] **Step 1: Write failing eligibility and data-minimization tests**

```python
def test_payload_contains_only_completed_verified_review_data() -> None:
    snapshot = signed_snapshot(
        findings=[accepted_verified(), rejected_finding(), pending_finding()],
        context={"patient": {"name": "不应发送", "patientNumber": "secret"}},
    )
    payload = build_writeback_payload(snapshot, "pharmacist-1")
    encoded = json.dumps(payload, ensure_ascii=False)
    assert [item["findingId"] for item in payload["findings"]] == ["accepted-1"]
    assert "不应发送" not in encoded
    assert "secret" not in encoded
    assert "contextSnapshot" not in encoded
```

- [ ] **Step 2: Implement stable unresolved item IDs**

```python
def unresolved_item_id(review_id: str, item: dict[str, Any]) -> str:
    source = "|".join([
        review_id,
        str(item.get("kind") or "UNKNOWN"),
        ",".join(sorted(item.get("medicationIds") or [])),
        str(item.get("summary") or ""),
    ])
    return "unresolved-" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]
```

Normalize `FHIR:Patient/id` to `Patient/id`. Include accepted Findings only, plus unresolved items and accepted evidence-gap Findings converted into unresolved entries. Include unique model IDs from `snapshot.modelCalls`, never token prompts or raw context.

Set payload `schemaVersion="1.0"` for the Health MCP writeback contract and copy `snapshot.schemaVersion` into `reviewSchemaVersion`; resource IDs use the latter so a future ReviewState schema migration cannot be confused with the independent MCP envelope/payload version.

- [ ] **Step 3: Write failing coordinator state tests**

```python
@pytest.mark.asyncio
async def test_prepare_requires_signed_review_and_returns_typed_job() -> None:
    gateway = FakeHealthWritebackGateway(preview_envelope())
    coordinator = WritebackCoordinator(gateway)
    with pytest.raises(WritebackStateError):
        await coordinator.prepare(running_snapshot(), "pharmacist-1")
    job = await coordinator.prepare(signed_snapshot(), "pharmacist-1")
    assert job.jobId == "writeback-review-1-7"
    assert job.reviewVersion == 7


@pytest.mark.asyncio
async def test_commit_sends_stored_job_values_only() -> None:
    gateway = FakeHealthWritebackGateway(commit_envelope())
    result = await WritebackCoordinator(gateway).commit(prepared_job(), confirmed=True)
    assert result["committed"] is True
    assert gateway.commit_args == ("writeback-review-1-7", "a" * 64, 7, True)
```

- [ ] **Step 4: Implement coordinator error mapping**

Map non-OK ToolEnvelope values to `WritebackError(code, retryable, message)` without using an empty response as success. `prepare` validates response data with `WritebackJob`; `commit` validates job ID and hash equality in the response. Neither method mutates the repository; API handlers in plan 03 own optimistic persistence.

- [ ] **Step 5: Keep writeback capability outside planner construction**

`ReviewDependencies` continues to hold `health`, `drug`, `repository`, and `planner`; do not pass the Health gateway into `StructuredLLMPlanner`, `EvidenceGrader`, prompts, or model tool bindings. Construct `WritebackCoordinator(dependencies.health)` only inside deterministic API writeback handlers.

- [ ] **Step 6: Run Agent writeback tests and commit**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/test_writeback.py tests/test_gateways.py tests/test_models.py -q -p pytest_asyncio.plugin
git add src/medication_review_agent/writeback.py src/medication_review_agent/workflow.py tests/test_writeback.py
git commit -m "feat: coordinate deterministic review writeback"
```

Expected: PASS; payload is minimal, only eligible Findings are included, and no model-facing object receives commit capability.

## Cross-Repository Verification

- [ ] In Health MCP, run `python -m pytest tests/test_writeback_service.py tests/test_mcp_server.py tests/test_llm_agent_tools.py -q`; expected PASS.
- [ ] In Review Agent, run core tests with plugin autoload disabled; expected PASS.
- [ ] Hash all original Patient, MedicationRequest, Condition, AllergyIntolerance and Observation JSON before commit and after two identical commits; expected byte-for-byte equality.
- [ ] Query `fhir_resources` after two commits; expected exactly one row per previewed resource ID.
- [ ] Change the bundle hash while keeping review/version constant; expected `WRITEBACK_CONFIRMATION_MISMATCH` or `WRITEBACK_VERSION_CONFLICT` and zero new rows.
- [ ] Search both repositories for writeback tool names in model tool builders; expected no model-facing registration.
