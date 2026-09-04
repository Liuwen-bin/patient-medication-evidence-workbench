# 演示与面试讲解脚本

## 三分钟主流程

### 0:00-0:25 业务问题

打开工作台，指向顶部的患者、审核日期和核查目标。说明：药师真正需要的是可追溯的复核闭环，
不是聊天报告；系统只称“活动用药医嘱”，不会声称患者实际服药。

### 0:25-0:50 架构边界

打开架构图。说明：一个受控 LangGraph 连接两个独立 MCP；LLM 只做结构化规划和有限语义
判断，患者/产品选择、规则、引用验证及写回均是确定性路径，模型没有写回工具。

### 0:50-1:35 患者与产品中断

1. 输入合成患者 `FHIR:Patient/p1`、核查日期和“核查活动用药医嘱的成分、途径、剂型和标签警告”。
2. 点击“开始复核”，展示患者证据与活动用药医嘱。
3. 当 `Arnica montana` 出现 `FUZZY_NAME` 候选时，指出系统没有预选。
4. 选择当前候选并点击“确认映射”。说明：resume 只接受当前候选集中的 product ID，并携带 `expectedVersion`。

预期画面：状态从“等待药品映射确认”进入“等待审核”，产品来源和图谱 provenance 保留。

### 1:35-2:10 Finding 与一次补证据

1. 选中 Finding，右栏同时显示 `FHIR:MedicationRequest/...` 与 `SPL:...#section`。
2. 对一个 Finding 点击“补充证据”；状态从 checkpoint 定向返回证据节点。
3. 恢复后接受该 Finding，排除另一个 Finding。

说明：每产品/主题最多两次正文检索，每个 Finding 最多一次人工补证据；第二次请求会被预算
拒绝。非 evidence-gap Finding 没有 FHIR 与 SPL 双引用时不可接受。

### 2:10-2:40 写回预览与幂等

1. 点击“完成审核”，确认状态为 `SIGNED_OFF`。
2. 点击“生成写回预览”，展示患者、reviewer、reviewVersion、bundleHash 和将新增的资源表。
3. 勾选高摩擦确认，点击“确认写回”；随后以相同 job/hash 再提交一次。

说明：preview 零写入；commit 只新增白名单 FHIR 资源。第二次返回相同资源 ID，资源总数不变。

### 2:40-3:00 审计与结果

展示审计时间线和评测摘要。说明：离线 15/15 是 Fixture 回归；真实在线五例已发起，但当前
Drug MCP 的 MCP session 与 LightRAG storage 生命周期不匹配，导致 5/5 超时；模型调用也只
记录到上游失败与 fallback，没有成功观测，结果明确不通过。这证明失败路径也可观测，且没有
用离线结果冒充在线质量。

## 十分钟技术叙事

### 0:00-1:00 需求与非目标

目标是药师证据复核，不是诊断或处方。输入自然语言只是选择核查主题，不能授权停药、换药、
改剂量或完整相互作用判断。报告被降为可选投影，主业务是 ReviewState、人工决定和安全写回。

### 1:00-2:00 为什么拒绝自由 ReAct 和多 Agent

自由 ReAct 很难给出稳定候选约束、预算和副作用证明；多 Agent 会增加交接状态和评测面，却
没有独立知识域收益。首版用一个状态机，把模型放在适合语义判断的两个窄位置。对应实现：
`planner.py`、`retrieval.py`、`workflow.py`。

### 2:00-3:00 Schema 与恢复

`ReviewSnapshot` schema 1.1 保存 question、intent、mapping、Finding、证据、模型调用、检索
次数、writeback 状态和 version。LangGraph checkpoint 与 ReviewRepository 分离，通过 mutation
journal 处理“checkpoint 已前进但业务快照未保存”的崩溃窗口。对应测试：`test_repository.py`、
`test_api.py`、`test_workflow.py`。

### 3:00-4:00 两个 MCP 的职责

Health MCP 保持患者范围和 FHIR 语义；Drug MCP 保持产品、SPL、Neo4j 与 Milvus 语义。两者
不互调。每条医嘱必须解析产品，正文检索必须绑定确认产品和文档版本，不能用向量相似度替代
结构化身份事实。

### 4:00-5:00 隐私与提示注入

模型只看到去标识化特征；DailyMed MCP 也不接收患者身份。用户问题、FHIR/SPL 文本均作为
不可信数据，不能修改主题白名单、product ID 范围、循环次数或工具权限。schema/策略失败时
显式 fallback，并在 audit 中记录 failure code 而非 prompt。

### 5:00-6:00 Human-in-the-loop

患者、产品、Finding 是三类持久化中断。resume 同时校验当前状态、候选集、reviewer 和版本。
“补证据”不是无限对话，而是带 Finding ID 的一次性定向返回；这让过程可恢复、可测试、可解释。

### 6:00-7:00 FHIR 事务与幂等

只从已接受且通过引用验证的 Finding 构建资源。prepare 冻结 Bundle 并计算 hash；commit 校验
确认、身份、版本和 hash，在 Health MCP 单事务提交。确定性资源 ID 与唯一写回键让网络重试
返回同一结果；原 Patient、MedicationRequest 等资源做字节级不变验证。

### 7:00-8:00 测试与评测

测试按 core/browser 分进程。离线 15 case 测规则、安全门、故障注入和指标，适合 CI；在线
5 case 真实连接服务、模型和数据库，失败也落机器码。LLM-as-judge 只适合摘要清晰度，患者、
产品、引用、状态和写回由程序断言。

### 8:00-9:00 真实失败

第一次真实运行暴露 Milvus 未启动；启动 Docker 中的 Milvus、Neo4j、etcd 和 MinIO 后重跑，
首个 MCP session 已成功连接数据库并调用工具，但关闭时 finalize 共享 LightRAG；后续 session
复用已终止对象，`Neo4jDrugGraphRepository` 因 `_driver` 为空失败。模型端点同时出现
`APIConnectionError/httpx.ReadError`。五例仍在 120 秒边界失败并归入 `DEPENDENCY_TIMEOUT`；
公开结果保持不通过。详见 `tradeoffs-and-failures.md`。

### 9:00-10:00 演进条件

先统一 Drug MCP session 与 LightRAG storage 生命周期、恢复模型端点并达到在线完成率、覆盖率
和安全阈值，再谈部署。多 worker 需要共享锁与幂等协调；多 Agent 只有在新增至少两个独立知识域或十种以上
用药导致可测上下文/延迟瓶颈时成立。

## 八个常见追问

1. 为什么不用自由 ReAct？医疗状态和副作用需要稳定、可证明的候选与预算边界。
2. 图查询和向量检索为什么不能互换？图负责产品身份/成分集合，向量负责标签段落召回；相似不等于身份。
3. 为什么模糊产品不能自动选？错误产品会污染之后全部 SPL 证据，必须在最早边界人工确认。
4. Human interrupt 如何跨进程恢复？ReviewRepository 保存版本化快照，LangGraph SQLite 保存 checkpoint，resume 使用 review ID。
5. Review 与 Writeback 为什么分离？药师决定是领域事实，外部写入是可失败、可重试的副作用。
6. 如何防止模型结果写进病历？模型不持有写回 gateway；Bundle 只从验证后的结构化 Finding 确定性生成。
7. Fixture 与在线评测有什么区别？Fixture 测逻辑可重复性；在线评测才测真实模型、网络、Neo4j/Milvus 和 MCP 性能。
8. 如何证明没有跨患者/产品污染？患者引用归属、确认 product ID、SPL version/filter 和 accepted 引用矩阵都有程序断言。
