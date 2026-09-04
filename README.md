# Medication Review Agent

这是一个面向药师复核的合成数据 MVP。它通过 Health Record MCP 获取患者侧 FHIR 证据，通过 Drug Evidence MCP 获取 DailyMed SPL、Neo4j 知识图谱和 LightRAG 证据，再由 LangGraph 管理可恢复的人工确认与最终签署。

它不是临床决策系统，不诊断、开药、停药、换药或调整剂量，也不会写回电子病历。仓库提供的患者和评估记录均为合成数据。

## 安装

```powershell
Set-Location C:\Users\Administrator\Downloads\dm_spl_release_homeopathic\homeopathic\medication_review_agent
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

## 本地启动顺序

终端 1，启动只读 Health Record MCP：

```powershell
Set-Location C:\Users\Administrator\Desktop\mcp\health-record-mcp\Agent
python mcp/mcp_server.py --transport http --host 127.0.0.1 --port 8000
```

终端 2，启动 Drug Evidence MCP。该服务以 Neo4j 为在线主图，Milvus/LightRAG 检索标签章节；显式配置时才允许 snapshot fallback：

```powershell
Set-Location C:\Users\Administrator\Downloads\dm_spl_release_homeopathic\homeopathic\dailymed_lightrag
dailymed-drug-mcp --transport http --host 127.0.0.1 --port 8010
```

终端 3，启动 review API：

```powershell
Set-Location C:\Users\Administrator\Downloads\dm_spl_release_homeopathic\homeopathic\medication_review_agent
medication-review-api
```

The API intentionally supports one worker only. Review-scoped mutation locks coordinate the
repository version check, LangGraph checkpoint resume, and repository save; multiple worker
processes would bypass that in-process consistency boundary and are rejected at startup by an
OS-level lease on the checkpoint store. Uvicorn/Gunicorn multi-worker flags are unsupported;
the second process fails its lifespan startup even if `REVIEW_API_WORKERS` is not set.

API 默认监听 `http://127.0.0.1:8020`。所有 mutation 默认要求非空 `REVIEW_API_KEY`；药师决策还要求 `X-Reviewer-Id`、payload `reviewerId` 与 `REVIEW_API_REVIEWER_ID` 三者一致，把共享凭据绑定到配置的药师身份。仅显式设置 `ALLOW_INSECURE_LOCAL_MUTATIONS=true` 才可在本地关闭 mutation 鉴权；非回环绑定仍必须同时设置 `ALLOW_REMOTE_API=true` 和 API key。远程明文 MCP 默认被拒绝。

## 测试和评估

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests `
  --ignore=tests/test_web_render.py `
  --ignore=tests/test_web_decisions.py `
  --ignore=tests/test_web_visual.py `
  -q -p pytest_asyncio.plugin
```

浏览器与视觉测试需要在一个新的 PowerShell 进程中运行：

```powershell
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD -ErrorAction SilentlyContinue
python -m pytest `
  tests/test_web_render.py `
  tests/test_web_decisions.py `
  tests/test_web_visual.py -q
```

离线评估：

```powershell
medication-review-evaluate --cases evaluation/cases.jsonl --output artifacts/evaluation/report.json
```

评估包含 15 个稳定合成案例，覆盖精确/歧义/模糊/未映射、Neo4j 反向成分遍历与集合比较、snapshot 一致性和 fallback、患者隔离、缺失病史、人工恢复、瞬时重试以及禁止的停药/剂量请求。输出记录安全阈值、映射与缺失信息指标、工具成功率、延迟、重试、token 和估算成本。

上述测试和评估均使用合成数据与 Fixture gateway，只用于确定性回归，不代表真实 LLM、MCP、Neo4j、Milvus 或 DailyMed 在线链路的质量和性能。基线详情见 `docs/baseline/2026-09-04-baseline.md`。

## 关键约束

- graphBackend、graphWorkspace、graphDatabase、fallbackUsed 和 consistency 会随 mapping、evidence、finding、checkpoint 和报告保留。
- snapshot fallback、DRIFT 或 UNAVAILABLE 会形成显式 evidence gap 并强制药师复核。
- 非 evidence-gap finding 只有同时具备 FHIR 与 SPL 引用才可接受。
- 审计只保存节点、工具、状态、引用、延迟、重试和成本元数据，不保存原始患者 payload 或隐藏推理。
