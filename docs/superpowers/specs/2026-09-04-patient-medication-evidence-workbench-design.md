# 患者用药证据核查工作台设计规格

状态：已确认，待实施
日期：2026-09-04  
项目定位：AI Agent / LLM 应用工程师与 Python 后端工程师求职作品集  
主要用户：门诊复诊前执行用药核查的药师或医学审核人员  
核心代码：`medication_review_agent`  
依赖系统：`health-record-mcp/Agent`、`dailymed_lightrag`

本文件是 `medication_review_agent` 独立作品集仓库的权威设计规格。

本规格整合并取代以下文档中与当前产品方向重复或冲突的部分：

- `docs/superpowers/specs/2026-08-31-medication-review-agent-design.md`；
- `health-record-mcp/Agent/docs/superpowers/specs/2026-09-04-patient-medication-evidence-agent-design.md`；
- `docs/superpowers/progress/2026-08-31-medication-review-agent-handoff.md` 中已经过时的进度结论。

## 1. 项目摘要

本项目实现一个面向药师的患者用药证据核查工作台。

药师在患者复诊前选择患者并输入本次核查目标。系统通过 Health Record MCP
读取患者范围内的 FHIR 病历，提取活动用药医嘱、过敏、活动疾病、近期观察结果、
特殊人群信息和资料缺失；随后通过 DailyMed Drug Evidence MCP 将每条用药医嘱
映射到确定的药品产品，查询结构化产品事实和 SPL 标签正文，并生成带双侧证据的
待核查项。

系统不会自动作出临床决定。患者歧义、药品歧义、核查项判断和 FHIR 写回均设置
持久化 Human-in-the-loop 中断。药师完成审核后，系统把确认的问题写成
`DetectedIssue`，把仍需补充或澄清的信息写成 `Task`，并用 `Provenance` 记录来源、
系统和审核人员。完整报告只是可选导出物，不是主要业务对象。

项目重点展示：

- 双 MCP 服务编排；
- LangGraph 显式状态机和持久化中断恢复；
- LLM 规划、受控 Agentic RAG 与结构化输出；
- 确定性产品映射、安全校验和证据对齐；
- FHIR 数据建模与人工确认后的幂等事务写回；
- 离线回归评测、真实在线评测、可观测性和故障降级；
- FastAPI、Pydantic、SQLite、异步编程、浏览器测试和 CI。

## 2. 求职作品集叙事

项目需要让面试官在短时间内理解以下完整过程：

```text
真实问题：患者病历和药品标签分散，人工核查耗时且容易遗漏来源
  -> 基线问题：自由 Tool Calling 无法稳定保证患者、产品和证据范围
  -> 架构决策：一个受控 Agent + 两个 MCP + 确定性安全节点
  -> LLM 职责：理解目标、规划主题、评估正文覆盖、组织已验证摘要
  -> 程序职责：患者隔离、产品解析、集合运算、引用校验、状态迁移
  -> 人工职责：处理歧义、审核 Finding、批准写回
  -> 工程保障：checkpoint、幂等、乐观并发、审计、重试和降级
  -> 结果验证：合成病例、工具轨迹、真实在线评测和可复现实验
  -> 已知边界：不是诊断、处方、相互作用数据库或自动临床决策系统
```

面试演示以一条正常路径、一条歧义路径和一条服务降级路径为主。复杂的 outbox、
租约和图谱一致性审计保留为可靠性深挖内容，不占据三分钟主演示的中心位置。

## 3. 业务场景

### 3.1 场景名称

门诊复诊前患者用药证据核查。

### 3.2 业务痛点

患者病历记录了用药医嘱和临床上下文，但不包含完整、可检索的药品标签事实；
DailyMed 提供权威标签和产品关系，但不知道具体患者记录了什么。药师需要在两个
信息域之间手工完成药名确认、产品变体选择、标签查找、证据比对和记录回填。

单独使用任一系统都无法完成患者范围内、来源可追踪的核查：

- Health Record MCP 回答“病历记录了什么”；
- DailyMed Drug Evidence MCP 回答“已确认产品的标签声明了什么”；
- Medication Review Agent 负责“以什么顺序调用、何时暂停、如何对齐和如何交给人”。

### 3.3 核心用户故事

作为药师，我希望在患者复诊前选择一个患者并说明核查目标，让系统整理患者当前的
活动用药医嘱，将药物映射到 DailyMed 产品，展示病历证据与标签证据，标记需要我
判断的问题，并在我确认后把结构化结果写回病历，从而减少重复查找且保留完整审计。

### 3.4 支持的核查项

首版只生成以下类型：

- `PRODUCT_AMBIGUOUS`：药品存在多个可兼容产品；
- `PRODUCT_UNMAPPED`：无法映射到当前 DailyMed 数据集；
- `PATIENT_FIELD_MISSING`：核查需要的病历字段未记录；
- `ROUTE_MISMATCH`：病历给药途径与标签产品途径不一致；
- `DOSAGE_FORM_MISMATCH`：病历剂型与产品剂型不一致；
- `INGREDIENT_ALLERGY_NAME_MATCH`：过敏物质名称与产品成分规范化后相同；
- `SHARED_ACTIVE_INGREDIENT`：多个已确认产品共享活性成分；
- `LABEL_EVIDENCE_REVIEW`：与核查主题相关的标签原文需要药师查看；
- `LABEL_EVIDENCE_MISSING`：标签正文或引用不足；
- `GRAPH_PROVENANCE_WARNING`：图谱使用 fallback、发生漂移或一致性未知。

这些都是核查信号，不是诊断、禁忌结论或停药建议。

## 4. 产品边界

### 4.1 首版目标

1. 通过药师工作台选择唯一患者并输入自然语言核查目标。
2. 从 Health Record MCP 获取患者范围内的结构化用药上下文。
3. 对每条活动用药医嘱执行产品解析，不猜测歧义候选。
4. 优先查询确定性产品事实，仅对正文主题执行受控 Agentic RAG。
5. 生成可追踪到 FHIR 与 DailyMed 的 Finding。
6. 在所有高风险决策点暂停并等待药师输入。
7. 支持进程重启后的中断恢复与并发版本校验。
8. 药师完成审核后预览并确认结构化 FHIR 写回。
9. 分别输出离线确定性评测和真实在线 LLM 评测。
10. 提供可复现的演示步骤、截图、架构说明和结果复盘。

### 4.2 非目标

首版不实现：

- 诊断、处方、停药、换药或剂量调整；
- 完整药物相互作用检查；
- CYP450、Beers Criteria 或临床指南推理；
- 自动判断患者适合或不适合某个药物；
- 将 MedicationRequest 当作患者实际服药或依从性证据；
- 自动修改 `Patient`、`MedicationRequest`、`Condition`、
  `AllergyIntolerance` 或 `Observation`；
- 真实患者数据接入；
- 由模型直接调用 FHIR 提交工具；
- 无限制 ReAct 循环；
- 多智能体编排；
- 生产级多租户、跨机构权限或公网部署；
- 为展示技术名词而复制已有 LightRAG 或 MCP 业务逻辑。

## 5. 现有实现基线与复用范围

### 5.1 已完成并直接复用

`health-record-mcp/Agent` 已具备：

- FHIR SQLite 读取和患者范围隔离；
- `get_medication_review_context(patientId, asOf?)`；
- 稳定 FHIR 引用和明确缺失字段；
- FastMCP、CSV 预览/确认导入模式及相应测试。

`dailymed_lightrag` 已具备：

- `resolve_medication`、`get_product_facts`、`search_label_evidence`、
  `compare_product_ingredients`、`validate_evidence` 五个 MCP 工具；
- Neo4j 确定性产品图谱；
- Milvus + LightRAG SPL 正文检索；
- snapshot fallback、图谱一致性和 provenance；
- 产品范围内的引用验证。

`medication_review_agent` 已具备：

- Pydantic 状态、工具封装、Finding、证据和审计模型；
- 双 MCP 类型化 HTTP Gateway；
- LangGraph 主流程和 SQLite durable checkpoint；
- 患者、映射、Finding 和签署中断；
- 重试、乐观并发、单进程租约、mutation journal/outbox 和恢复；
- FastAPI 工作台接口、HTML 工作台和可选报告；
- 15 个 Fixture 评测病例和浏览器/视觉测试。

### 5.2 需要改造而不是重写

- 用 `StructuredLLMPlanner` 替代生产环境中的纯 `DeterministicPlanner`，并保留后者作为
  测试基线和模型故障降级；
- 将自然语言核查目标加入 ReviewState 和规划接口；
- 增加受限 Evidence Grader 和一次查询改写；
- 将 `workflow.py` 中超过一千行的嵌套节点按职责拆分；
- 增加写回预览、确认、事务提交和写回审计；
- 将当前 Fixture 评测与真实在线评测明确分开；
- 修正单命令运行 pytest 时 Playwright 与 asyncio 插件的事件循环冲突；
- 重新生成有完整业务数据的作品集截图和演示记录。

### 5.3 不能直接作为真实结果宣传

当前 `artifacts/evaluation/report.json` 使用 Fixture Health/Drug Gateway 和
`DeterministicPlanner`。其中 15/15、零 token 和毫秒级延迟只证明离线状态机与规则
回归正确，不能表述为真实 MCP、Neo4j、Milvus 或 LLM 的在线准确率和性能。

## 6. 架构决策

### 6.1 一个受控 Agent

系统使用一个 LangGraph Agent，不采用多智能体。两个数据域存在明确强顺序：先确认
患者，再读取用药，再确认产品，最后检索产品范围内的标签证据。多智能体会增加状态
同步和证据污染风险，首版没有可测收益。

### 6.2 两个 MCP 服务

```text
Pharmacist Workbench
        |
Medication Review FastAPI
        |
LangGraph Orchestrator + Review Store
        |---------------- Health Record MCP
        |                  - patient-scoped FHIR context
        |                  - writeback preview
        |                  - confirmed transactional writeback
        |
        `---------------- Drug Evidence MCP
                           - product resolution
                           - Neo4j structured facts
                           - Milvus/LightRAG label evidence
                           - citation validation
```

两个 MCP 不互相调用、不共享数据库。编排器持有统一 ReviewState，并为不同节点绑定
不同能力。模型只接触规划和文本证据能力，不拥有 FHIR 写回工具。

### 6.3 规则先行，模型受限

确定性代码负责：

- 患者唯一性和患者范围隔离；
- 工具响应 schema 校验；
- 产品候选是否可自动接受；
- 产品 ID 是否属于当前候选集；
- 途径、剂型和成分集合比较；
- Missing Information 传播；
- 引用存在性、患者/产品/文档范围；
- 工具预算、循环次数和状态迁移；
- 药师身份、乐观并发和写回幂等；
- 禁止性医疗动作和写回资源白名单。

LLM 只负责：

- 将药师自然语言目标解析为受控意图和主题；
- 根据已确认产品选择允许的 SPL 标签主题；
- 在第一次正文证据不足时评估语义覆盖并改写一次查询；
- 根据已验证 Claim 生成工作台摘要。

### 6.4 报告是投影，不是领域核心

核心领域对象是 `ReviewSession`、`Finding`、`EvidencePair`、
`PharmacistDecision` 和 `WritebackJob`。JSON/HTML 报告由最终 ReviewSnapshot 派生，
只在导出或归档时生成，不阻塞 Finding 审核和 FHIR 写回。

## 7. 端到端业务流程

```text
START
  -> create_review
  -> safety_gate
       unsafe request -> explain_scope -> END
  -> parse_review_goal              [LLM structured output]
  -> collect_review_context         [Health Record MCP]
       no/multiple patient -> HUMAN INTERRUPT
  -> validate_context               [deterministic]
  -> normalize_medications          [deterministic]
  -> resolve_medications            [Drug Evidence MCP]
       ambiguous/fuzzy -> HUMAN INTERRUPT
       unmapped -> record gap and continue
  -> plan_review                    [LLM with deterministic fallback]
  -> retrieve_structured_facts      [Drug Evidence MCP]
  -> retrieve_label_evidence?       [bounded Agentic RAG]
       insufficient and attempts < 2 -> rewrite once -> retrieve
       still insufficient -> record gap
  -> build_findings                 [deterministic alignment]
  -> validate_findings              [local + Drug Evidence MCP]
  -> pharmacist_review              [HUMAN INTERRUPT]
       request more evidence -> targeted retrieval, at most once/finding
  -> complete_review
  -> prepare_writeback              [Health Record MCP, no mutation]
  -> confirm_writeback              [HUMAN INTERRUPT]
  -> commit_writeback               [Health Record MCP transaction]
  -> END
```

状态图控制必须调用的顺序。LLM 不得决定是否跳过患者确认、产品解析、引用验证或写回
确认。

## 8. Human-in-the-loop 设计

### 8.1 中断点

1. `AWAITING_PATIENT_CONFIRMATION`：姓名重复或患者不唯一；
2. `AWAITING_MAPPING_CONFIRMATION`：产品歧义、模糊候选或字段冲突；
3. `AWAITING_FINDING_REVIEW`：药师逐项接受、排除或要求补证据；
4. `AWAITING_WRITEBACK_CONFIRMATION`：展示 FHIR Bundle 预览并等待最终提交。

### 8.2 中断与恢复契约

- `reviewId` 同时作为 LangGraph `thread_id`；
- 中断前将完整可序列化状态写入 SQLite checkpointer；
- Resume payload 使用 Pydantic 严格校验，不传自由消息历史；
- 患者或产品只能从中断时保存的候选集中选择；
- 每个 mutation 携带 `expectedVersion`；
- HTTP 409 后刷新状态，不自动重放旧决定；
- 进程重启后从 checkpoint 恢复，不重复已成功工具调用；
- 每项补证据最多一次，正文检索每产品每主题最多两次；
- 所有人工决定记录 reviewer、时间、原建议和最终决定。

### 8.3 Human-in-the-loop 不是无限循环

系统允许“药师要求补证据 -> 定向检索 -> 回到同一 Finding”的闭环，但循环上限由
程序控制。达到上限仍不足时，Finding 保持 `NEEDS_MORE_EVIDENCE`，并允许药师排除、
保留为 Task 或结束本次审核。模型不能继续自行尝试新工具。

## 9. ReviewState 与状态模型

建议核心状态：

```json
{
  "reviewId": "review-uuid",
  "schemaVersion": "1.1",
  "status": "RUNNING",
  "writebackStatus": "NOT_REQUESTED",
  "question": "核查该患者当前用药信息和标签证据",
  "intent": {
    "type": "MEDICATION_EVIDENCE_REVIEW",
    "topics": ["ingredients", "warnings"],
    "confidence": 0.96,
    "modelId": "configured-model",
    "promptVersion": "intent-v1"
  },
  "patientRef": "Patient/demo-001",
  "asOf": "2026-09-04",
  "contextSnapshot": {},
  "contextMissingFields": [],
  "medications": [],
  "medicationMappings": [],
  "reviewPlan": [],
  "evidenceIndex": [],
  "findings": [],
  "humanDecisions": [],
  "writebackJob": null,
  "auditEvents": [],
  "metrics": {},
  "version": 0
}
```

### 9.1 Review 状态

- `CREATED`；
- `RUNNING`；
- `AWAITING_PATIENT_CONFIRMATION`；
- `AWAITING_MAPPING_CONFIRMATION`；
- `AWAITING_FINDING_REVIEW`；
- `NEEDS_MORE_EVIDENCE`；
- `BLOCKED_TOOL_ERROR`；
- `READY_FOR_COMPLETION`；
- `SIGNED_OFF`：保留现有枚举，语义改为“药师已完成审核”，不代表报告是主业务；
- `CANCELLED`。

### 9.2 Writeback 状态

- `NOT_REQUESTED`；
- `PREPARING`；
- `PREPARED`；
- `COMMITTING`；
- `COMMITTED`；
- `FAILED`。

Review 与 Writeback 分开建模。写回失败不会撤销已经完成的药师审核，也不会触发重新
调用 LLM；系统使用相同 job、bundleHash 和幂等键安全重试。

当前 `ReviewSnapshot` schema 为 `1.0`。新增 `question`、`intent`、
`writebackStatus` 和 `writebackJob` 后将 ReviewState schema 升为 `1.1`；MCP 通用响应
封装继续使用独立的 `1.0`，两者不能混为同一个版本号。Repository 读取 `1.0` 快照时
执行确定性内存迁移，为新字段填入默认值，并在下一次成功 mutation 时保存为 `1.1`。
未知主版本必须拒绝加载，不能静默丢弃字段。

## 10. LLM 设计

### 10.1 模型配置

本地可复用 `LightRAG/.env` 中的 OpenAI-compatible 模型配置。Agent 通过显式路径
读取，不复制或提交密钥：

```text
AGENT_MODEL_ENV_PATH=C:/Users/Administrator/Downloads/dm_spl_release_homeopathic/homeopathic/LightRAG/.env

AGENT_LLM_BASE_URL <- QUERY_LLM_BINDING_HOST 或 LLM_BINDING_HOST
AGENT_LLM_API_KEY  <- QUERY_LLM_BINDING_API_KEY 或 LLM_BINDING_API_KEY
AGENT_LLM_MODEL    <- QUERY_LLM_MODEL 或 LLM_MODEL
```

当前可用模型示例为 `Qwen3.5-122B-A10B-FP8`。生产代码不硬编码模型名，审计事件保存
实际 model ID、prompt version、token、延迟和估算成本。

### 10.2 StructuredLLMPlanner

输入：

- 药师问题；
- 去标识化患者特征；
- 药物别名和已确认产品；
- 可用主题枚举；
- missingFields。

输出必须符合：

```json
{
  "intent": "MEDICATION_EVIDENCE_REVIEW",
  "topics": ["ingredients", "warnings"],
  "requiresNarrativeEvidence": true,
  "rationale": "需要核对成分并查看标签警告",
  "confidence": 0.94
}
```

主题仅允许：`identity`、`ingredients`、`route`、`dosage_form`、`warnings`、
`dosage`、`storage`、`indications`、`pregnancy`、`stop_use` 和 `images`。

模型输出解析失败、超时或包含越界主题时，使用现有 `DeterministicPlanner`，记录
`modelFallback=true` 和失败原因，但不把错误解释为没有相关证据。

### 10.3 Evidence Grader

优先做确定性覆盖检查：

- 返回产品是否为当前已确认 productId；
- 文档和章节是否在允许范围；
- 引用是否存在；
- 正文是否非空；
- 请求主题是否命中。

只有语义覆盖无法由字段判断时才调用模型，输出 `sufficient`、`coveredTopics`、
`missingTopics` 和 `reason`。不足时只允许改写一次查询，不能更换 productId 或扩大
文档范围。

### 10.4 Response Composer

模型只接收已验证 Claim、公开产品信息和去标识化引用，不接收完整 FHIR、原始附件、
API 密钥或写回工具。生成结果仅用于工作台摘要；结构化 Finding、状态、证据列表和
FHIR 写回 payload 均由程序生成。

每次复核最多三次模型调用：一次规划、一次可选 grader/rewrite、一次可选摘要。

## 11. MCP 工具边界

### 11.1 通用响应封装

```json
{
  "schemaVersion": "1.0",
  "status": "OK",
  "data": {},
  "evidenceRefs": [],
  "warnings": [],
  "errors": [],
  "provenance": {},
  "requestId": "request-uuid"
}
```

工具状态为 `OK`、`AMBIGUOUS`、`UNMAPPED`、`INSUFFICIENT_EVIDENCE` 和 `ERROR`。
编排层状态与工具状态使用不同枚举，并通过显式映射表转换。

错误建议增加机器可判定字段：

```json
{
  "code": "UPSTREAM_TIMEOUT",
  "message": "Drug Evidence MCP timed out",
  "retryable": true,
  "details": {}
}
```

### 11.2 Health Record MCP 读取工具

主工具为：

```text
get_medication_review_context(patientId, asOf?)
```

输出中的 MedicationRequest 必须表述为“活动用药医嘱”，不能直接表述为患者正在服药。
每条药物应包含：

- 原始名称和规范化名称；
- identifier system/code/display；
- strength 原始值和解析来源；
- route text/coding；
- dosage form text/coding；
- dosage/timing；
- MedicationReference 及解引用后的 Medication 证据；
- authoredOn/effective period；
- MedicationRequest 和 Medication evidenceRefs。

`asOf` 必须有明确语义：Observation 只包含时点之前的结果；未来 authoredOn 的医嘱
不得进入上下文。若当前数据没有 MedicationRequest 状态历史，系统必须明确只能判断
“查询时状态为 active”，不能声称准确还原历史时点的医嘱状态。

资料完整性按本次 intent/topic 判断。与当前问题无关的 `specialPopulations` 缺失不应
自动把储存类问题降级为整体 `INSUFFICIENT_EVIDENCE`。

### 11.3 Drug Evidence MCP

继续复用：

```text
resolve_medication
get_product_facts
search_label_evidence
compare_product_ingredients
validate_evidence
```

所有标签正文检索必须绑定已确认的 productId 和 SPL document version。来源至少保存：

- productId；
- setId/documentId；
- SPL version/effectiveTime；
- section ID/code；
- source path；
- content hash；
- graph backend/workspace/database；
- fallback 与 consistency 状态。

### 11.4 Health Record MCP 写回工具

不增加第三个 MCP Server。Health Record MCP 增加一个内部
`ReviewWritebackService`，仅暴露两个狭窄工具：

```text
validate_medication_review_writeback(payload)
commit_medication_review_writeback(jobId, bundleHash, expectedVersion, confirmed, reviewerId)
```

第一个工具只校验和生成 Bundle 预览，不写数据库。第二个工具要求
`confirmed=true`，校验提交人与预览审核人一致，并在一个 SQLite 事务中提交。模型工具列表不包含这两个工具；只有
FastAPI 确定性 writeback handler 可以调用。

## 12. Finding 与证据模型

### 12.1 Finding

```json
{
  "findingId": "finding-001",
  "reviewType": "ROUTE_MISMATCH",
  "summary": "病历途径与产品标签途径不一致，需药师核查",
  "attentionLevel": "REVIEW",
  "confidence": 1.0,
  "medicationIds": ["MedicationRequest/med-001"],
  "selectedProductIds": ["DRUG_PRODUCT::example"],
  "patientEvidenceRefs": ["FHIR:MedicationRequest/med-001"],
  "labelEvidenceRefs": ["SPL-GRAPH:DRUG_PRODUCT::example/route/ORAL"],
  "status": "PENDING",
  "requiresHumanReview": true,
  "ruleId": "route-mismatch-v1"
}
```

### 12.2 验证矩阵

| Claim/Finding 类型 | 必须验证 |
|---|---|
| Patient fact | FHIR 引用存在、属于选定患者、满足时点 |
| Label fact | 引用属于已确认产品和同一 SPL 版本 |
| Computed comparison | ruleId、规范化版本、输入值和双方引用 |
| Review signal | 匹配依据、原始值、规范化值和置信度 |
| Missing information | Health MCP 的查询覆盖范围，而不是简单搜索无结果 |

非 `EVIDENCE_GAP` Finding 只有同时拥有有效患者证据和标签证据才允许药师接受。
Finding 可以被拒绝或保留为待补证据，但验证失败的内容不能自动写回为临床问题。

## 13. FHIR 写回设计

### 13.1 写回原则

- 只新增资源，不修改原始临床资源；
- 只有药师完成所有 Finding 决策后才能准备写回；
- 准备和提交分成两个明确步骤；
- 模型永远没有提交能力；
- Bundle 必须通过资源类型、患者归属、引用和内容白名单校验；
- 写回使用事务、确定性资源 ID 和幂等键；
- 写回失败不撤销药师决定，可安全重试；
- 所有资源显著标记为 Agent 辅助、药师审核和合成演示数据。

### 13.2 资源映射

| 审核结果 | FHIR 资源 | 规则 |
|---|---|---|
| 药师确认的途径/剂型/成分核查问题 | `DetectedIssue` | 每个 accepted Finding 一个 |
| 产品歧义、缺少字段或仍需补证据 | `Task` | 表示待人工补录或澄清 |
| 操作者、时间、Agent、模型和来源 | `Provenance` | target 指向本次新增资源 |
| 完整结构化审核结果 | 可选 `DocumentReference` | 仅在用户选择导出/归档时生成 |

`PRODUCT_UNMAPPED`、`PATIENT_FIELD_MISSING` 和 `LABEL_EVIDENCE_MISSING` 不写成
临床 `DetectedIssue`，而写成 `Task` 或只保留在 ReviewState 中。

### 13.3 确定性 ID 与幂等

```text
DetectedIssue ID = hash(reviewId + findingId + schemaVersion)
Task ID          = hash(reviewId + unresolvedItemId + schemaVersion)
Provenance ID    = hash(reviewId + reviewVersion + bundleHash)
Writeback key    = reviewId + reviewVersion
```

数据库对 `(reviewId, reviewVersion)` 建立唯一约束。重复提交相同 bundleHash 返回原提交
结果；相同版本但不同 hash 返回冲突，不允许静默覆盖。

### 13.4 写回预览

工作台在确认前显示：

- 目标患者；
- 将新增的资源类型、ID 和数量；
- 每个 DetectedIssue 对应的 Finding、MedicationRequest 和证据；
- 每个 Task 的待办原因；
- reviewerId、reviewVersion 和 bundleHash；
- 警告及不能写回的 Finding。

## 14. 工具预算、重试和并发

固定总预算 12 与多药场景不兼容，改为分类预算：

```text
Health context:          1 次，患者澄清后允许 1 次重取
Medication resolution:  每条活动用药 1 次
Product facts:           每个已确认产品 1 次
Narrative retrieval:     每产品每主题最多 2 次
Ingredient comparison:  每次 review 最多 1 次批量比较
Evidence validation:     每次阶段最多 1 次批量验证
LLM calls:               每次 review 最多 3 次
Writeback validation:    每个 review version 1 次
Writeback commit:        幂等重试，不重复写资源
```

只对 timeout、connection reset、429 和显式 retryable 错误使用带抖动的指数退避。
Schema 错误、患者歧义、产品歧义和验证失败不能重试为成功。

多个药品的产品解析和结构化事实查询可以使用有界异步并发，但状态合并必须按
MedicationRequest ID 确定性排序。单 Agent 不代表所有 I/O 必须串行。

## 15. 安全、隐私和信任边界

### 15.1 数据最小化

DailyMed MCP 和 LLM 不接收患者姓名、患者编号、完整 FHIR Patient ID、完整病历或
原始附件。编排层为模型创建去标识化的 PatientFeatures 和 MedicationFeatures；
FHIR 引用只在确定性状态与验证节点内保存。

### 15.2 不可信内容

用户问题、FHIR 文本、附件、SPL 正文和检索片段全部作为不可信数据处理。它们不能
修改系统指令、工具列表、productId 范围、循环预算或写回权限。模型提示中明确标记
证据区，输出必须经过结构化 schema 和本地策略校验。

### 15.3 医疗边界

系统可以整理事实、比较字段、指出名称匹配和信息缺失，并建议药师进一步核查。
系统不能输出患者必须停药、换药、改剂量、确认过敏或不存在相互作用等结论。

### 15.4 写回权限

读取和写回能力在代码层分离。模型只能使用只读工具。写回 handler 要求 API key、
reviewer identity、completed review、expectedVersion、confirmed=true 和匹配的
bundleHash。审计不得记录 API key、完整患者 payload、prompt、chain-of-thought、SQL、
Cypher 或 raw tool arguments。

## 16. 错误与降级

| 场景 | 行为 |
|---|---|
| Health Record MCP 失败 | 不调用 DailyMed，不生成患者结论 |
| 患者不唯一 | 持久化中断，展示候选 |
| 某药无法映射 | 记录 gap，继续其他药物 |
| 产品歧义/模糊 | 持久化中断，禁止预选 |
| Neo4j 不可用且允许 snapshot | 显式 fallback，生成 provenance warning |
| Neo4j 健康但产品不存在 | 保持 UNMAPPED，不回退 snapshot 猜测 |
| Milvus/LightRAG 失败 | 保留结构化事实，正文标记不可用 |
| LLM planner 失败 | 使用 DeterministicPlanner，记录 fallback |
| 引用验证失败 | Finding 不可接受或降级为证据不足 |
| 超出检索预算 | 停止调用，保留已验证结果 |
| 写回校验失败 | 不生成 Bundle job，显示具体错误 |
| 写回提交失败 | Review 保持完成，writebackStatus=FAILED，可幂等重试 |

任何服务错误都不能表达为“没有相关事实”。

## 17. 可观测性

每个请求记录结构化事件：

```json
{
  "reviewId": "review-uuid",
  "requestId": "request-uuid",
  "node": "retrieve_label_evidence",
  "tool": "search_label_evidence",
  "resultStatus": "OK",
  "latencyMs": 412,
  "retryCount": 0,
  "evidenceCount": 3,
  "modelId": null,
  "promptVersion": null,
  "inputTokens": 0,
  "outputTokens": 0,
  "estimatedCost": 0,
  "modelFallback": false,
  "occurredAt": "2026-09-04T12:00:00Z"
}
```

请求级汇总包括节点轨迹、工具成功率、各工具延迟、检索次数、候选数量、Finding
接受/排除/补证据数量、LLM token/cost、最终状态和写回结果。

## 18. 评测设计

### 18.1 离线确定性回归

继续使用现有 15 个 Fixture 案例，验证：

- 患者范围隔离；
- 映射类别和候选安全门；
- Finding 规则；
- 引用配对；
- 中断恢复；
- retry/fallback；
- 并发版本冲突；
- 禁止性结论；
- 写回资源映射与幂等。

报告命名为 `offline-regression-report.json`，明确标注没有真实网络、模型或数据库延迟。

### 18.2 真实在线评测

至少选择 5 个稳定合成病例，真实连接：

```text
Health Record MCP
  -> LangGraph
  -> configured LLM
  -> Neo4j
  -> Milvus/LightRAG
  -> Drug Evidence MCP
  -> Pharmacist decision fixture
  -> writeback preview
```

在线病例：

1. 单患者单药完整路径；
2. 同名产品多变体，需要人工选择；
3. 过敏名称与成分名称匹配；
4. 一个映射成功、一个 UNMAPPED；
5. 正文检索不足或服务降级。

真实写回评测默认提交到隔离的合成 SQLite 数据库，提交前保存基线，验证新增资源和
原始资源未改变。

### 18.3 指标

| 指标 | 首版阈值 |
|---|---:|
| 跨患者证据污染 | 0 |
| 歧义或模糊产品自动接受 | 0 |
| 禁止性医疗结论 | 0 |
| accepted Finding 引用有效率 | 100% |
| exact identifier mapping accuracy | 100% |
| missing information recall | >= 95% |
| 写回原始临床资源修改数 | 0 |
| 幂等重试新增重复资源 | 0 |
| 在线任务完成率 | >= 80%，失败必须可解释 |
| 运行指标覆盖率 | 100% |

LLM-as-judge 仅评价摘要清晰度和正文语义覆盖。患者选择、产品 ID、引用、Finding、
状态迁移、安全结论和写回结果全部由程序断言。

## 19. 测试与 CI

测试分成两个独立 job，避免 Playwright 与 pytest-asyncio 在同一进程产生嵌套事件
循环：

```text
core:
  unit + contract + workflow + repository + api + evaluation

browser:
  web render + decisions + accessibility + visual viewports

live-integration:
  manual/scheduled, requires Health MCP + Neo4j + Milvus + Drug MCP + LLM
```

必须覆盖：

- 所有 LangGraph 条件边；
- 每个人工中断的暂停、非法 resume、合法 resume 和进程重启；
- 只有当前候选可以被确认；
- 补证据循环次数上限；
- LLM schema 失败和 deterministic fallback；
- prompt injection 不改变产品范围与工具权限；
- writeback preview 不修改数据库；
- commit 要求确认、身份、版本和 hash；
- transaction rollback、幂等提交和冲突；
- 原始 FHIR 资源在写回前后字节级不变；
- 1440x900、1024x768、390x844、360x800 无布局溢出。

## 20. 药师工作台

工作台采用任务型布局，不以聊天为主：

- 顶部：患者查询、asOf、核查目标、审核人员、开始/恢复；
- 左栏：患者上下文、活动用药医嘱、缺失信息和审计时间线；
- 中栏：产品映射、Finding 队列、状态和药师操作；
- 右栏：FHIR 证据、SPL 原文、图谱来源和验证结果；
- 完成审核后：写回预览、确认写回、写回结果；
- 次要入口：针对当前 Finding 的自然语言补证据请求；
- 可选操作：导出 JSON/HTML 审核报告。

桌面使用三栏，移动端使用患者、核查项、证据三个 tab。API 内容通过安全 DOM API
渲染，不使用 `innerHTML` 注入。

## 21. API 草案

```text
POST /api/reviews
POST /api/reviews/{reviewId}/run
GET  /api/reviews/{reviewId}
POST /api/reviews/{reviewId}/decisions
POST /api/reviews/{reviewId}/complete
POST /api/reviews/{reviewId}/writeback/prepare
POST /api/reviews/{reviewId}/writeback/commit
GET  /api/reviews/{reviewId}/writeback
GET  /api/reviews/{reviewId}/audit
GET  /api/reviews/{reviewId}/report.json   optional
GET  /api/reviews/{reviewId}/report.html  optional
```

`prepare` 返回资源预览和 bundleHash；`commit` 必须携带 `expectedVersion`、
`bundleHash`、`reviewerId` 和 `confirmed=true`。

## 22. 一周交付范围

### Day 1：基线与工程整理

- 将项目置于独立 Git 仓库，避免用户目录级 Git；
- 固定依赖和两类 pytest/CI 命令；
- 记录当前离线测试与在线服务基线；
- 将本规格设为唯一权威设计。

### Day 2：真实 LLM 规划

- 扩展 review request/question 和 planner protocol；
- 实现 StructuredLLMPlanner；
- 接入 `LightRAG/.env` 模型配置；
- 增加 schema、超时、fallback 和 token/cost 测试。

### Day 3：受控正文检索

- 实现 Evidence Grader；
- 实现一次查询改写和产品/文档范围约束；
- 增加注入、越界主题和循环预算测试。

### Day 4：FHIR 写回

- 实现 ReviewWritebackService；
- 生成 DetectedIssue、Task 和 Provenance；
- 实现 preview/commit、事务、幂等和 rollback 测试。

### Day 5：工作台闭环

- 将“提交报告”调整为“完成审核”；
- 增加写回预览和确认界面；
- 保留报告为可选导出；
- 完成浏览器、可访问性和视觉测试。

### Day 6：真实端到端与评测

- 启动 Health MCP、Neo4j、Milvus/LightRAG 和 Drug MCP；
- 运行五个在线合成病例；
- 输出真实 latency、token、cost、retry 和错误；
- 对比离线回归与在线结果。

### Day 7：作品集交付

- README：问题、架构、运行、演示、评测和限制；
- Mermaid/PNG 架构图和状态图；
- 三个有数据的关键流程截图；
- 三分钟演示脚本和十分钟技术讲解；
- 一页设计取舍、失败案例和后续路线图；
- 全量验证和只读代码审查。

### 明确推迟

- 多智能体；
- 复杂患者聊天机器人；
- 真实 PHI；
- 公网部署；
- PDF 编辑器；
- 临床指南和相互作用数据库；
- 多 worker 分布式 mutation；
- 为已有可靠性模块继续增加新抽象。

## 23. 演示脚本

### 三分钟主流程

1. 输入合成患者 ID 和“核查当前用药的成分、途径及标签警告”；
2. 展示 Agent 读取 FHIR 并解析两个药品；
3. 一个产品唯一映射，另一个产品触发人工选择；
4. 展示 FHIR 与 SPL/图谱配对证据；
5. 药师接受一个核查项、排除一个、对一个要求补证据；
6. LangGraph 从 checkpoint 定向恢复，最多补检索一次；
7. 药师完成审核，预览 DetectedIssue/Task/Provenance；
8. 确认写回并展示幂等重试没有重复资源；
9. 打开审计和在线评测结果。

### 面试深挖问题

- 为什么不使用完全自由的 ReAct？
- 为什么结构化图查询和向量检索不能互换？
- 为什么不能自动选择模糊产品？
- Human interrupt 如何跨进程恢复？
- 为什么 Review 状态与 Writeback 状态分离？
- 模型输出如何避免进入临床写回？
- Fixture 评测和真实在线评测有什么区别？
- 如何证明没有跨患者、跨产品证据污染？

## 24. 验收标准

首版完成必须同时满足：

1. 工作台能够连接两个真实 MCP 服务；
2. 自然语言核查目标由真实 LLM 解析为结构化计划；
3. LLM 失败时确定性 planner 可以降级并明确记录；
4. 每个患者相关流程先确定唯一患者；
5. 每条活动用药医嘱经过 resolve_medication；
6. 歧义、模糊和未映射产品不被自动猜测；
7. 标签检索始终绑定确认的产品和文档版本；
8. 正文检索最多两次，补证据循环最多一次；
9. 非 evidence-gap Finding 同时拥有有效 FHIR 与 DailyMed 引用；
10. 缺失数据不被解释为阴性、正常或不存在；
11. 每个人工中断可在进程重启后恢复；
12. 工作台完成患者、映射、Finding、补证据和写回确认闭环；
13. FHIR 写回只创建白名单资源，不修改原始临床资源；
14. 重复提交不产生重复资源；
15. 离线回归和在线评测分别生成、分别说明；
16. 安全阈值全部通过；
17. README 提供一键或明确分步启动与验证命令；
18. 演示截图包含真实合成数据，不使用空状态代替完整流程；
19. 项目存在独立 Git 根，不再解析到用户目录；
20. 文档明确展示限制、失败案例和可测的后续演进条件。

## 25. 主要风险与缓解

| 风险 | 缓解 |
|---|---|
| MedicationRequest 被误称为实际服药 | UI 和 Claim 使用“活动用药医嘱” |
| asOf 不能还原历史状态 | 明确时点语义与数据限制，不做超出数据的声明 |
| LLM 生成越界主题或医疗结论 | 结构化枚举、规则校验和 deterministic fallback |
| 标签正文提示注入 | 证据作为不可信数据，节点工具固定，输出 schema 校验 |
| 产品歧义被误选 | 候选集约束和持久化人工中断 |
| 向量证据跨产品污染 | productId + document version 强制过滤 |
| 图谱不可用 | 显式 fallback、provenance warning 和 partial result |
| 模型接触患者身份 | 去标识化 DTO，确定性节点保存真实引用 |
| 写回模型幻觉 | 只从药师接受且已验证的 Finding 生成资源 |
| 重复或并发写回 | expectedVersion、bundleHash、唯一约束和事务 |
| Fixture 指标被误解为线上能力 | 离线/在线报告分离并标注数据来源 |
| 项目技术点太多讲不清 | 主演示只保留一条闭环，复杂可靠性能力放附录 |
| Git 根错误导致误提交用户文件 | 初始化独立仓库前禁止 git add/commit |

## 26. 后续演进条件

只有满足可测条件时才增加多智能体：

- 单患者平均超过 10 种药且上下文成本成为主要问题；
- 新增至少两个独立知识域，如相互作用库和临床指南；
- 并行标签检索的延迟成为 P95 主要组成；
- 单 Agent 工具选择准确率在结构化约束后仍无法达标；
- 不同子任务需要不同模型或独立安全策略。

即使拆分，患者选择、产品候选安全门、集合运算、药师中断和 FHIR 写回仍保留为
确定性控制节点。
