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

- 输入条件：2026-09-05、提交 `f358ee1` 的真实五例运行使用外部模型配置；独立最小调用和五次规划都在读取响应时被上游重置连接。
- 可见症状：每例都到达 `SIGNED_OFF` 和写回预览，但 `modelCalls` 的 token/usage 不可用且 `modelFallback=true`。
- 机器码：`MODEL_UPSTREAM_ERROR`。
- 安全行为：切换 `DeterministicPlanner`，不扩大 patient/product scope、主题、工具权限或循环预算；业务链路完成 5/5，但报告强制 `realModel=false`、`acceptancePassed=false`。
- 回归测试：`tests/test_planner.py` 的 schema、上游异常与 fallback 用例；`tests/test_online_evaluation.py` 的真实组件和接受条件用例。
- 状态：显式降级和真实性标记已实现；上游端点仍是外部限制，恢复后必须重跑，不能用这次 5/5 证明模型质量。

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
- 安全行为：runner 只允许一次 GET 刷新后重试；第二次冲突失败。Review 保留原状态，源 FHIR 不变。
- 回归测试：`tests/test_api.py::test_stale_decision_returns_conflict`、`test_writeback_rejects_stale_version_hash_and_reviewer_before_gateway`，以及 Health MCP 的事务/幂等测试。
- 状态：已修复并作为并发契约保留。多 worker 仍是显式非目标，需共享锁后才能开放。

## 没有被美化的结果

离线 15/15 只能说明确定性规则与 Fixture 合同回归通过。最新真实在线运行在真实 MCP/数据库
链路上业务完成 5/5，聚合安全和质量指标达标，说明生命周期修复与降级路径有效；但五次模型
规划全部回退，`realModel=false`，因此完整在线验收仍为 false。面试时必须同时讲清“业务闭环
通过”和“模型质量未验收”，不能把二者合并成一个成功结论。
