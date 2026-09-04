# Patient Medication Evidence Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在一周内把现有稳定的离线用药核查原型交付为可连接真实 LLM、两个 MCP 服务并经药师确认后幂等写回 FHIR 的求职作品集。

**Architecture:** 保留单个 LangGraph 受控状态机，由确定性节点掌管患者、产品、引用、循环和写回边界，仅把意图规划、一次语义覆盖判断和可选摘要交给 LLM。Health Record MCP 与 Drug Evidence MCP 保持独立，Medication Review FastAPI 统一持有 ReviewState、人工中断和写回状态。

**Tech Stack:** Python 3.11、FastAPI、Pydantic 2、LangGraph 1.x、SQLite、FastMCP、langchain-openai、Neo4j、Milvus/LightRAG、原生 HTML/CSS/JavaScript、pytest、Playwright、GitHub Actions

**Spec:** `docs/superpowers/specs/2026-09-04-patient-medication-evidence-workbench-design.md`

## Global Constraints

- 首版只有一个受控 LangGraph Agent，明确不增加多智能体。
- Health Record MCP 和 Drug Evidence MCP 不互相调用、不共享数据库。
- 模型只负责意图/主题规划、最多一次正文语义覆盖判断和可选摘要，不获得任何写回工具。
- 每条活动用药医嘱必须调用一次 `resolve_medication`，歧义、模糊和字段冲突均不得自动选择。
- 主题白名单固定为 `identity`、`ingredients`、`route`、`dosage_form`、`warnings`、`dosage`、`storage`、`indications`、`pregnancy`、`stop_use`、`images`。
- 每产品每主题最多两次正文检索，每项 Finding 最多一次人工补证据循环，每次 review 最多三次 LLM 调用。
- 非 `EVIDENCE_GAP` Finding 只有同时具备有效 FHIR 与 SPL 引用才允许接受。
- FHIR 写回只新增 `DetectedIssue`、`Task`、`Provenance`；`DocumentReference` 仅作为可选导出，绝不修改原始临床资源。
- 写回必须经过 preview、药师确认和 commit，使用确定性 ID、`reviewId + reviewVersion` 幂等键、`bundleHash` 与事务。
- DailyMed MCP 和 LLM 不接收患者姓名、患者编号、完整 Patient ID、完整 FHIR payload 或原始附件。
- MedicationRequest 在 UI 和输出中统一称为“活动用药医嘱”，不得声称患者实际正在服药。
- 当前仓库依赖版本范围保持 `pyproject.toml` 中的 Python `>=3.11`、FastAPI `>=0.115,<1.0`、Pydantic `>=2.8,<3.0`、LangGraph `>=1.0,<2.0`。
- API 保持单 worker；mutation 继续要求 API key、配置的 reviewer identity 和乐观版本校验。
- 所有病例均为合成数据；离线 Fixture 指标与真实在线 LLM/MCP 指标必须分文件、分说明展示。

---

## 当前基线

- 独立 Git 根：`medication_review_agent/.git`。
- GitHub：`https://github.com/Liuwen-bin/patient-medication-evidence-workbench`。
- 基线提交：`8c29dec chore: establish medication review agent baseline`。
- 已有能力：双 MCP gateway、持久化 LangGraph interrupt、SQLite checkpoint、乐观并发、mutation journal、15 个 Fixture 案例、药师工作台、浏览器和视觉测试。
- 已知缺口：生产仍使用 `DeterministicPlanner`；ReviewState 还没有 `question`、`intent`、`writebackStatus`、`writebackJob`；Health Record MCP 没有 review writeback 工具；报告仍占用最终主操作；在线评测未与离线回归分离。

## 子计划与依赖

| 顺序 | 子计划 | 独立可验收结果 | 前置 |
|---|---|---|---|
| 1 | `2026-09-04-patient-medication-evidence-workbench-01-llm-orchestration.md` | schema 1.1、真实结构化规划、受控二次检索、确定性降级 | 当前基线 |
| 2 | `2026-09-04-patient-medication-evidence-workbench-02-fhir-writeback.md` | Health MCP preview/commit、FHIR 资源构造、事务与幂等 | schema 1.1 |
| 3 | `2026-09-04-patient-medication-evidence-workbench-03-workbench.md` | 完成审核、写回预览、确认提交的端到端 UI/API | 写回契约 |
| 4 | `2026-09-04-patient-medication-evidence-workbench-04-evaluation-portfolio.md` | 分离 CI、五例在线评测、README、图、截图和演示材料 | 前三项 |

```text
schema 1.1 + LLM planning
          |
          +--> bounded evidence retrieval
          |
          +--> FHIR writeback contract --> workbench closure
                                         |
                                         +--> online evaluation + portfolio
```

### Task 1: Day 1 基线冻结与执行入口

**Files:**
- Modify: `README.md`
- Create: `docs/baseline/2026-09-04-baseline.md`
- Test: `tests/test_models.py`
- Test: `tests/test_api.py`
- Test: `tests/test_web_render.py`
- Test: `tests/test_web_decisions.py`
- Test: `tests/test_web_visual.py`

**Interfaces:**
- Consumes: Git commit `8c29dec`、现有 pytest 配置和 Fixture gateway。
- Produces: 两条互不共享 Python 进程的基线测试命令，以及一份明确标注离线性质的基线记录。

- [ ] **Step 1: 记录环境与当前提交**

Run:

```powershell
git rev-parse HEAD
python --version
python -m pip freeze | Select-String 'fastapi|pydantic|langgraph|langchain-openai|pytest|playwright'
```

Expected: HEAD 以 `8c29dec` 为基线，Python 为 3.11 或更高，列出的依赖均已安装。

- [ ] **Step 2: 运行 core 基线**

Run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests --ignore=tests/test_web_render.py --ignore=tests/test_web_decisions.py --ignore=tests/test_web_visual.py -q -p pytest_asyncio.plugin
```

Expected: core 测试全部通过；结果不得被描述为真实 MCP 或真实模型性能。

- [ ] **Step 3: 运行 browser 基线**

Run:

```powershell
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD -ErrorAction SilentlyContinue
python -m pytest tests/test_web_render.py tests/test_web_decisions.py tests/test_web_visual.py -q
```

Expected: 20 个 browser/visual 测试通过，无 Playwright 与 asyncio 嵌套事件循环错误。

- [ ] **Step 4: 写入基线文档和 README 测试入口**

在 `docs/baseline/2026-09-04-baseline.md` 记录提交 SHA、Python 版本、两条测试命令、通过数量、运行日期和“Fixture 不等于在线结果”的声明。将 README 的单条 `pytest tests -q` 替换为上面的 core/browser 两条命令。

- [ ] **Step 5: 提交基线记录**

```powershell
git add README.md docs/baseline/2026-09-04-baseline.md
git commit -m "docs: freeze executable project baseline"
```

Expected: 提交只包含基线文档和 README 命令，不包含 `.env`、SQLite 或截图缓存。

### Task 2: Day 2-3 交付 LLM 编排和受控检索

**Files:**
- Plan: `docs/superpowers/plans/2026-09-04-patient-medication-evidence-workbench-01-llm-orchestration.md`
- Modify: `src/medication_review_agent/models.py`
- Modify: `src/medication_review_agent/planner.py`
- Create: `src/medication_review_agent/model_config.py`
- Create: `src/medication_review_agent/retrieval.py`
- Modify: `src/medication_review_agent/workflow.py`
- Modify: `src/medication_review_agent/api.py`

**Interfaces:**
- Consumes: `ReviewPlanner.plan(...)`、`DrugEvidenceGateway.search_label_evidence(...)`、现有 audit/metrics。
- Produces: schema `1.1` ReviewSnapshot、`StructuredLLMPlanner`、`FallbackReviewPlanner`、`BoundedEvidenceRetriever`。

- [ ] **Step 1: 执行子计划 01 的 schema 与配置任务**

Run: 按子计划 01 的 Task 1-2 逐个运行单测并提交。

Expected: 旧 `1.0` 快照可确定性迁移，未知主版本被拒绝，模型密钥不进入 snapshot、audit 或 Git。

- [ ] **Step 2: 执行子计划 01 的 planner 与 retrieval 任务**

Run: 按子计划 01 的 Task 3-6 逐个运行单测并提交。

Expected: 真实配置启用时使用结构化模型；schema、超时、越界主题或模型失败时显式 fallback；任何一次 review 不超过三次模型调用。

- [ ] **Step 3: 运行 LLM 编排回归**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/test_models.py tests/test_repository.py tests/test_planner.py tests/test_retrieval.py tests/test_workflow.py tests/test_api.py -q -p pytest_asyncio.plugin
```

Expected: 全部通过，并覆盖 prompt injection 不改变主题、productId、循环预算或工具权限。

### Task 3: Day 4 交付 FHIR 写回契约

**Files:**
- Plan: `docs/superpowers/plans/2026-09-04-patient-medication-evidence-workbench-02-fhir-writeback.md`
- Create in Health MCP: `mcp/writeback_service.py`
- Modify in Health MCP: `mcp/mcp_server.py`
- Create in Health MCP: `tests/test_writeback_service.py`
- Modify: `src/medication_review_agent/gateways.py`
- Modify: `src/medication_review_agent/models.py`

**Interfaces:**
- Consumes: 已完成 review 的 accepted Findings、unresolved items、reviewer identity 和 review version。
- Produces: `validate_medication_review_writeback(payload)` 与 `commit_medication_review_writeback(jobId, bundleHash, expectedVersion, confirmed)`。

- [ ] **Step 1: 执行 Health MCP 的预览和事务提交任务**

Run: 按子计划 02 的 Task 1-6，在 `health-record-mcp/Agent` 中逐个完成测试和提交。

Expected: preview 对 `fhir_resources` 零写入；commit 只新增白名单资源；失败事务回滚；相同 hash 重试返回原结果，不重复插入。

- [ ] **Step 2: 执行 Agent gateway 和状态接线任务**

Run: 按子计划 02 的 Task 7-8，在当前仓库逐个完成测试和提交。

Expected: 模型对象与 planner 永远拿不到写回 gateway；准备失败不创建 job；提交失败保留 `SIGNED_OFF` 并把 `writebackStatus` 设为 `FAILED`。

- [ ] **Step 3: 运行跨仓库契约测试**

```powershell
Set-Location C:\Users\Administrator\Desktop\mcp\health-record-mcp\Agent
python -m pytest tests/test_writeback_service.py tests/test_mcp_server.py -q

Set-Location C:\Users\Administrator\Downloads\dm_spl_release_homeopathic\homeopathic\medication_review_agent
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/test_gateways.py tests/test_writeback.py tests/test_api.py -q -p pytest_asyncio.plugin
```

Expected: 两个仓库的契约测试均通过，工具名和字段名完全一致。

### Task 4: Day 5 交付药师工作台闭环

**Files:**
- Plan: `docs/superpowers/plans/2026-09-04-patient-medication-evidence-workbench-03-workbench.md`
- Modify: `src/medication_review_agent/api.py`
- Modify: `src/medication_review_agent/web/index.html`
- Modify: `src/medication_review_agent/web/app.js`
- Modify: `src/medication_review_agent/web/render.js`
- Modify: `src/medication_review_agent/web/styles.css`
- Modify: `tests/test_web_decisions.py`
- Modify: `tests/test_web_visual.py`

**Interfaces:**
- Consumes: `POST /complete`、`POST /writeback/prepare`、`POST /writeback/commit` 和 schema 1.1 snapshot。
- Produces: 从输入核查目标到药师完成审核、预览资源、明确确认、查看提交结果的任务型 UI。

- [ ] **Step 1: 完成 API 闭环**

Run: 按子计划 03 的 Task 1 执行测试和提交。

Expected: 完成审核与报告导出解耦，prepare/commit 均受 API key、reviewer、版本和状态保护。

- [ ] **Step 2: 完成桌面与移动端交互**

Run: 按子计划 03 的 Task 2-4 执行 Playwright 测试和提交。

Expected: 桌面三栏、移动三 tab 均可完成主流程；动态内容只通过安全 DOM API 渲染；360x800 起无水平溢出和控件遮挡。

### Task 5: Day 6-7 交付评测和求职叙事

**Files:**
- Plan: `docs/superpowers/plans/2026-09-04-patient-medication-evidence-workbench-04-evaluation-portfolio.md`
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/live-integration.yml`
- Create: `evaluation/online-cases.jsonl`
- Create: `scripts/run-live-evaluation.ps1`
- Create: `docs/portfolio/architecture.md`
- Create: `docs/portfolio/demo-script.md`
- Create: `docs/portfolio/tradeoffs-and-failures.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: 完整 UI/API、真实 Health MCP、Neo4j、Milvus/LightRAG、Drug MCP 和配置模型。
- Produces: `offline-regression-report.json`、`online-integration-report.json`、可复现实验命令、架构图、状态图、三张完整流程截图和讲解脚本。

- [ ] **Step 1: 分离 CI 与评测入口**

Run: 按子计划 04 的 Task 1-2 执行测试和提交。

Expected: PR 自动运行 core/browser 两个 job；live integration 只手动触发且不在日志泄露密钥。

- [ ] **Step 2: 运行五个真实在线合成病例**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-live-evaluation.ps1
```

Expected: 生成 `artifacts/evaluation/online-integration-report.json`，包含真实延迟、token、成本、重试、模型 ID、各节点轨迹和失败原因；完成率至少 80%，所有零容忍安全指标为 0。

- [ ] **Step 3: 完成作品集材料和最终验证**

Run: 按子计划 04 的 Task 3-5 执行文档检查、截图和全量验证。

Expected: README 能让新环境按明确顺序启动三个服务；三分钟演示覆盖正常、歧义、补证据、预览、提交和幂等重试；十分钟讲解能说明需求、边界、架构、实现、测试、失败和取舍。

- [ ] **Step 4: 推送完成版本**

```powershell
git status --short
git log --oneline --decorate -12
git push origin main
```

Expected: 工作区干净，`origin/main` 包含所有小步提交，GitHub 中没有 `.env`、数据库、缓存、真实患者数据或未脱敏日志。

## 每日停止线

| Day | 必须可演示 | 不通过时的范围处理 |
|---|---|---|
| 1 | 两组基线测试和准确的离线声明 | 不开始模型接线，先恢复绿色基线 |
| 2 | question -> 结构化 intent/plan，失败可降级 | 暂停可选摘要，不削弱 schema 与安全门 |
| 3 | 每产品/主题最多两次检索，补证据最多一次 | 保留确定性检索，关闭可选 grader 模型调用 |
| 4 | preview 零写入，commit 幂等且只新增 | 不进入 UI，先通过原资源字节不变测试 |
| 5 | UI 完成审核和写回确认闭环 | 可选报告样式让位于主业务闭环 |
| 6 | 五例在线报告可复现且失败可解释 | 不伪造成功指标，保留失败证据和复盘 |
| 7 | README、图、截图、演示脚本与 CI | 不增加新功能，只修阻塞演示的问题 |

## 完成定义

- [ ] 设计规格的 20 条验收标准均能对应到自动测试、在线评测断言或人工演示证据。
- [ ] `git ls-files` 不包含 `.env`、`*.sqlite`、`__pycache__`、`.pytest_cache`、真实 PHI 或 API key。
- [ ] core 与 browser 分进程执行并全部通过。
- [ ] 五个在线病例真实经过两个 MCP、配置模型、Neo4j 和 Milvus/LightRAG。
- [ ] 相同写回请求连续提交两次，第二次返回相同资源 ID 且数据库总数不增加。
- [ ] 提交前后 Patient、MedicationRequest、Condition、AllergyIntolerance、Observation 的 JSON 字节完全一致。
- [ ] 工作台显示“活动用药医嘱”、Agent 辅助、药师审核、合成数据和失败/降级状态。
- [ ] README 明确区分离线回归结果与在线结果，并说明本项目不提供诊断或处方建议。
