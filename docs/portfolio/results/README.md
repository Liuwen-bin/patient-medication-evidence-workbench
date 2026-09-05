# Evaluation Results

这里仅保存可公开的聚合摘要。原始在线运行目录位于
`artifacts/live-runs/<timestamp>/`，包含隔离 SQLite、进程日志和完整脱敏报告，已被
`.gitignore` 排除。

## 两种结果不可混用

- `offline-regression-summary.json` 来自 15 个固定合成 Fixture，用于确定性回归；它不经过网络、真实模型或真实数据库。
- `online-integration-summary.json` 来自真实 HTTP/MCP 启动流程；失败也会保留，不能用离线结果替换。

## 手动在线工作流

`.github/workflows/live-integration.yml` 只有 `workflow_dispatch`，只能运行在私有的
`self-hosted/windows/medication-review-live` runner。runner 必须能够访问两个本地仓库、
Neo4j、Milvus 和模型配置。任一真实依赖缺失时 job 失败并上传
`online-integration-report.json`，不会切换到 Fixture。

`commit_synthetic=true` 只允许写入本次运行目录中的 Health SQLite 副本。launcher 会在
运行前后校验源数据库 SHA-256 与修改时间，并且只终止自己启动的进程。无论 preview 还是
commit 模式，五个服务端口只要有一个已占用就拒绝启动，避免把外部进程误记为本次运行。

在线安全指标不使用成功默认值：case 文件提供患者、产品映射和缺失字段 oracle，FHIR 引用
归属从隔离 Health 数据库校验，SPL 引用绑定从 `evidenceIndex` 校验，源资源变化和语义重复
写回从数据库前后快照计算；缺少任何观测都会降低 `metricsCoverage`。

公开摘要只保留 case ID、聚合指标、执行模式和机器可读失败码，不包含患者姓名、完整
Patient ID、原始证据、服务端点、prompt、隐藏推理或凭据。

## 最新在线结论

提交 `66d7ff0` 的运行在真实 Health/Drug MCP 与数据库上完成 5/5 个 case，每个 case 都独立
观察到两个 MCP，聚合指标全部达到阈值；模型上游连接被重置，五次规划均显式回退。因此该
运行可证明真实业务链路和安全降级，但 `realModel=false`、`acceptancePassed=false`，报告的
`acceptanceFailureCodes` 为 `MODEL_UPSTREAM_ERROR`，不能作为真实模型质量通过的证据。
