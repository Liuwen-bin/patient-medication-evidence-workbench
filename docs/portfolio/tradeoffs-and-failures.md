# 关键取舍与失败复盘

## 设计取舍

| 决策 | 选择 | 代价 |
|---|---|---|
| Agent 形态 | 单个受控 LangGraph | 少了多 Agent 展示性，但状态和权限更容易证明 |
| 推理分工 | 规则负责身份/安全/写回，LLM 负责结构化语义 | 需要维护 schema 与确定性规则 |
| 知识检索 | Neo4j 结构化事实 + Milvus/LightRAG 原文 | 两套存储需要一致性和可用性观测 |
| 人工复核 | 三类持久化中断 + 一次补证据 | 流程比聊天更严格，但可以恢复和审计 |
| 写回 | preview/commit 两阶段、只新增 | 操作多一步，换来明确确认、事务和幂等 |
| 评测 | Fixture CI 与真实在线报告分离 | 在线依赖较重，但不会把模拟性能当真实能力 |

## 失败 1：模型上游连接被重置

- 输入条件：2026-09-05、提交 `9bdab35` 的真实五例运行使用外部模型配置；独立最小调用和五次规划都在读取响应时被上游重置连接。
- 可见症状：每例都到达 `SIGNED_OFF` 和写回预览，但 `modelCalls` 的 token/usage 不可用且 `modelFallback=true`。
- 机器码：`MODEL_UPSTREAM_ERROR`。
- 安全行为：切换 `DeterministicPlanner`，不扩大 patient/product scope、主题、工具权限或循环预算；业务链路完成 5/5，但报告强制 `realModel=false`、`acceptancePassed=false`。
- 回归测试：`tests/test_planner.py` 的 schema、上游异常与 fallback 用例；`tests/test_online_evaluation.py` 的真实组件和接受条件用例。
- 状态：显式降级和真实性标记已实现；端点恢复后已重新运行，旧结果仍只证明安全降级，不能替代后续真实模型证据。

## 失败 1B：端点不兼容 function calling

- 输入条件：2026-09-05、模型端点恢复后，提交 `3cc503a` 的真实五例运行使用 LangChain `with_structured_output(..., method="function_calling")`。
- 可见症状：调用产生 input/output token 并以 `stop` 结束，但没有 tool call、正文为空、`parsed=None`；五例均记录 `modelFallback=true`。
- 机器码：`MODEL_SCHEMA_ERROR`。
- 安全行为：不消费不可解析输出，回退到确定性规划器；五例业务闭环完成，但报告保持 `realModel=false`、`acceptancePassed=false`。
- 根因与修复：同一无患者数据的最小 schema 对照证明该端点的 `json_schema` 与 `json_mode` 可解析，而 `function_calling` 不产生工具调用。规划器和证据 grader 改用 `json_schema`。
- 回归测试：`tests/test_planner.py::test_structured_planner_returns_allowlisted_topics_only` 与 `tests/test_retrieval.py::test_structured_grader_returns_schema_and_model_audit` 固定结构化输出方法。
- 状态：已修复。提交 `978f9b3` 的五例运行全部使用真实模型、无 fallback，`realModel=true`、`acceptancePassed=true`。

## 失败 2：MCP session 与检索存储生命周期不匹配

- 输入条件：早期本机运行中 Milvus `19530` 未监听；依赖启动后，首个 MCP session 关闭又提前执行共享 LightRAG 的 `finalize_storages()`。
- 可见症状：后续 session 复用已终止对象，Neo4j driver 为空，五例在依赖边界超时。
- 机器码：历史运行记录 `DEPENDENCY_TIMEOUT` 与 `MCP_SESSION_STORAGE_LIFECYCLE_MISMATCH`。
- 安全行为：该次验收失败且没有写回；没有用 Fixture 替换。降级用例还要求可审计的 Milvus 故障证明，不能把服务异常解释成“没有证据”。
- 回归测试：`tests/test_gateways.py` 的持久 session/失败重连用例、`tests/test_dailymed_compat.py` 的兼容与 provenance 用例，以及 launcher 的隔离/清理契约。
- 状态：已在本项目兼容层修复，未修改 `dailymed_lightrag`。最新真实运行 `realDatabases=true`，五例均完成；故障注入例也通过。

## 失败 3：陈旧版本和写回冲突

- 输入条件：客户端用旧 `expectedVersion` 提交药师决定，或用不匹配的 `bundleHash`/reviewer 请求写回。
- 可见症状：API 返回 409，gateway 在任何写入前被阻断。
- 机器码：`STALE_REVIEW_VERSION`；Health 写回冲突使用明确的版本/hash 冲突码。
- 安全行为：普通决策最多在一次 GET 刷新后重试；写回 commit 只发送一次 POST，随后仅 GET 刷新并返回 `STALE_REVIEW_VERSION`，绝不重放 POST。Review 保留原状态，源 FHIR 不变。
- 回归测试：`tests/test_api.py::test_stale_decision_returns_conflict`、`test_writeback_rejects_stale_version_hash_and_reviewer_before_gateway`，以及 Health MCP 的事务/幂等测试。
- 状态：已修复并作为并发契约保留。多 worker 仍是显式非目标，需共享锁后才能开放。

## 没有被美化的结果

离线 15/15 只能说明确定性规则与 Fixture 合同回归通过。提交 `978f9b3` 的最新真实在线运行在
真实模型、MCP 和数据库链路上完成 5/5，五次规划均无 fallback，聚合安全和质量指标达标，
`realModel=true`、`realDatabases=true`、`acceptancePassed=true`。此前连接重置、schema 不兼容、
检索生命周期和陈旧版本冲突仍作为独立证据保留，不能因为最终成功而删除失败过程。
