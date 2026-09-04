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

## 失败 1：模型 schema 不合格

- 输入条件：模型返回超出主题白名单、缺字段或无法通过 `ReviewIntent` schema 的结构化输出。
- 可见症状：primary planner 结果被拒绝，audit/modelCalls 记录失败而不是使用半合法字段。
- 机器码：`MODEL_SCHEMA_INVALID` 或对应 planner failure code。
- 安全行为：切换 `DeterministicPlanner`；不扩大主题、不改变 patient/product scope，也不增加工具权限。
- 回归测试：`tests/test_planner.py` 的结构化输出、白名单和 fallback 用例；`tests/test_workflow.py` 的 prompt injection 用例。
- 状态：已修复为显式降级机制。限制是降级计划更保守，不代表模型语义质量达标。

## 失败 2：Milvus/LightRAG 不可用

- 输入条件：2026-09-05 本机真实在线五例，Health MCP、Drug MCP 和 Review API 启动；Milvus `19530` 未监听。
- 可见症状：Drug MCP 初始化 collection 时连接失败，5 个 case 各在约 120 秒超时，模型未被调用。
- 机器码：原始 runner 记录 `UNEXPECTED_ONLINE_ERROR`；复盘后归一化为 `DEPENDENCY_TIMEOUT`，依赖原因为 `MILVUS_UNAVAILABLE`。
- 安全行为：在线验收 `false`、task completion/metrics coverage 为 0；没有生成 Finding 结论，没有写回，也没有用 Fixture 替换结果。
- 回归测试：`tests/test_online_evaluation.py::test_online_runner_labels_dependency_timeouts` 和 launcher 失败报告/进程清理契约。
- 状态：环境限制仍存在。恢复 Milvus 后必须重跑五例，不能把当前报告改写为成功。

## 失败 3：陈旧版本和写回冲突

- 输入条件：客户端用旧 `expectedVersion` 提交药师决定，或用不匹配的 `bundleHash`/reviewer 请求写回。
- 可见症状：API 返回 409，gateway 在任何写入前被阻断。
- 机器码：`STALE_REVIEW_VERSION`；Health 写回冲突使用明确的版本/hash 冲突码。
- 安全行为：runner 只允许一次 GET 刷新后重试；第二次冲突失败。Review 保留原状态，源 FHIR 不变。
- 回归测试：`tests/test_api.py::test_stale_decision_returns_conflict`、`test_writeback_rejects_stale_version_hash_and_reviewer_before_gateway`，以及 Health MCP 的事务/幂等测试。
- 状态：已修复并作为并发契约保留。多 worker 仍是显式非目标，需共享锁后才能开放。

## 没有被美化的结果

离线 15/15 只能说明确定性规则与 Fixture 合同回归通过。真实在线 0/5 表明当前机器未满足
Milvus 运行前提，因此无法声称真实模型、完整数据库链路、在线延迟或在线写回达到阈值。公开
摘要保留两者，面试演示可以用 Fixture 展示交互，但必须同步说明在线失败证据。
