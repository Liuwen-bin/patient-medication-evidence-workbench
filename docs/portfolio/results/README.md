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
运行前后校验源数据库 SHA-256 与修改时间，并且只终止自己启动的进程。

公开摘要只保留 case ID、聚合指标、执行模式和机器可读失败码，不包含患者姓名、完整
Patient ID、原始证据、服务端点、prompt、隐藏推理或凭据。
