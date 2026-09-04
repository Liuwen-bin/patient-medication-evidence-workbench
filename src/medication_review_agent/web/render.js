const STATUS_LABELS = {
  CREATED: "待开始",
  RUNNING: "进行中",
  AWAITING_PATIENT_CONFIRMATION: "等待患者确认",
  AWAITING_MAPPING_CONFIRMATION: "等待药品映射确认",
  AWAITING_FINDING_REVIEW: "等待审核",
  NEEDS_MORE_EVIDENCE: "等待补充证据",
  BLOCKED_TOOL_ERROR: "工具异常，审核阻塞",
  READY_FOR_SIGN_OFF: "待签署",
  SIGNED_OFF: "已签署",
  CANCELLED: "已取消",
};

function element(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined) node.textContent = String(options.text);
  if (options.type) node.type = options.type;
  if (options.id) node.id = options.id;
  if (options.attrs) {
    Object.entries(options.attrs).forEach(([name, value]) => {
      if (value !== null && value !== undefined) node.setAttribute(name, String(value));
    });
  }
  children.filter(Boolean).forEach((child) => node.append(child));
  return node;
}

function clear(node) {
  node.replaceChildren();
}

function allowedImageUrl(value) {
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const url = new URL(value, window.location.href);
    if (url.origin !== window.location.origin) return null;
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    if (!url.pathname.startsWith("/api/")) return null;
    return value;
  } catch {
    return null;
  }
}

function heading(text, level = 2) {
  return element(`h${level}`, { text });
}

function definitionList(items) {
  const list = element("dl", { className: "fact-list" });
  items.forEach(([label, value]) => {
    list.append(
      element("dt", { text: label }),
      element("dd", { text: value === null || value === undefined || value === "" ? "未记录" : value }),
    );
  });
  return list;
}

function graphLabels(provenance) {
  if (!provenance) return [];
  const consistency = provenance.consistency?.status || "UNKNOWN";
  const backend = provenance.fallbackUsed || provenance.graphBackend === "snapshot"
    ? "离线图谱回退"
    : provenance.graphBackend === "neo4j" ? "Neo4j 在线图谱" : `图谱：${provenance.graphBackend || "未记录"}`;
  const consistencyLabel = consistency === "CONSISTENT"
    ? "图谱一致"
    : ["DRIFT", "DRIFTED", "INCONSISTENT"].includes(consistency) ? "图谱漂移" : "一致性未检查";
  return [backend, consistencyLabel];
}

export function graphProvenanceBlock(provenance) {
  if (!provenance) return null;
  const warning = provenance.fallbackUsed
    || provenance.consistency?.status !== "CONSISTENT";
  const block = element("div", {
    className: `graph-provenance${warning ? " is-warning" : ""}`,
    attrs: warning ? { role: "alert" } : {},
  });
  const badges = element("div", { className: "badge-row" });
  graphLabels(provenance).forEach((label) => badges.append(element("span", { className: "badge", text: label })));
  block.append(
    badges,
    definitionList([
      ["后端", provenance.graphBackend],
      ["工作区", provenance.graphWorkspace],
      ["数据库", provenance.graphDatabase],
      ["检查时间", provenance.consistency?.checkedAt],
    ]),
  );
  return block;
}

export function renderStatus(review, statusNode) {
  statusNode.textContent = review ? (STATUS_LABELS[review.status] || review.status) : "未开始";
}

function signOffCounts(review) {
  const findings = review?.findings || [];
  const mappings = review?.medicationMappings || [];
  return {
    accepted: findings.filter((finding) => finding.status === "ACCEPTED").length,
    rejected: findings.filter((finding) => finding.status === "REJECTED").length,
    unmapped: mappings.filter((mapping) => mapping.matchClass === "UNMAPPED").length,
    evidenceGaps: findings.filter((finding) => finding.reviewType === "EVIDENCE_GAP").length,
  };
}

export function renderSignOffDialog(review, onConfirm, onClose) {
  const counts = signOffCounts(review);
  const titleId = "sign-off-dialog-title";
  const dialog = element("div", {
    className: "sign-off-dialog",
    attrs: {
      role: "dialog",
      "aria-modal": "true",
      "aria-labelledby": titleId,
    },
  });
  const title = element("h2", { id: titleId, text: "确认提交审核报告" });
  const summary = element("p", {
    className: "dialog-guidance",
    text: "提交后将生成签署报告，请确认已逐项核对当前审核结果。",
  });
  const countList = element("div", { className: "sign-off-counts", attrs: { "aria-label": "审核计数" } });
  [["接受", counts.accepted], ["排除", counts.rejected], ["未映射", counts.unmapped], ["证据缺口", counts.evidenceGaps]].forEach(([label, count]) => {
    countList.append(element("div", { className: "sign-off-count", text: `${label} ${count}` }));
  });
  const reviewerLabel = element("label", { attrs: { for: "sign-off-reviewer-id" }, text: "审核药师 ID" });
  const reviewerInput = element("input", {
    id: "sign-off-reviewer-id",
    attrs: { required: "", autocomplete: "username" },
  });
  reviewerInput.value = document.querySelector("#reviewer-id")?.value.trim() || "";
  const confirmationLabel = element("label", { className: "dialog-checkbox" });
  const confirmation = element("input", { type: "checkbox", id: "sign-off-confirmation" });
  confirmationLabel.append(confirmation, element("span", { text: "我已核对全部审核项" }));
  const actions = element("div", { className: "dialog-actions" });
  const cancel = element("button", { type: "button", text: "取消" });
  const confirm = element("button", { className: "primary", type: "button", text: "确认签署" });
  confirm.disabled = true;
  const updateConfirmState = () => {
    confirm.disabled = !confirmation.checked || !reviewerInput.value.trim();
  };
  reviewerInput.addEventListener("input", updateConfirmState);
  confirmation.addEventListener("change", updateConfirmState);
  cancel.addEventListener("click", onClose);
  confirm.addEventListener("click", () => {
    if (!confirm.disabled) onConfirm(reviewerInput.value.trim());
  });
  actions.append(cancel, confirm);
  dialog.append(title, summary, countList, reviewerLabel, reviewerInput, confirmationLabel, actions);
  return dialog;
}

export function renderPatientContext(review, target) {
  clear(target);
  target.append(heading("患者上下文"));
  if (!review) {
    target.append(element("p", { className: "empty-state", text: "尚未载入患者。输入患者编号后开始复核。" }));
    return;
  }
  const context = review.contextSnapshot || {};
  const patient = context.patient || {};
  target.append(definitionList([
    ["复核编号", review.reviewId],
    ["患者引用", review.patientRef],
    ["FHIR 证据", patient.evidenceRef],
    ["年龄", patient.age],
    ["审核日期", review.asOf],
  ]));
  const missing = review.contextMissingFields || context.missingFields || [];
  if (missing.length) {
    const section = element("section", { className: "missing-data", attrs: { "aria-label": "资料缺失" } });
    section.append(heading("资料缺失", 3));
    const list = element("ul");
    missing.forEach((item) => list.append(element("li", { text: `${item}：未记录` })));
    section.append(list);
    target.append(section);
  }
  const collections = [
    ["活动诊断", context.activeConditions],
    ["过敏记录", context.allergies],
    ["近期观察", context.recentObservations],
    ["特殊人群信息", context.specialPopulations],
  ];
  collections.forEach(([label, values]) => {
    const count = Array.isArray(values) ? values.length : 0;
    target.append(element("p", { className: "context-count", text: `${label}：${count ? `${count} 条` : "未记录"}` }));
  });
}

export function renderAuditTimeline(events, target) {
  target.querySelector("#audit-timeline")?.remove();
  const section = element("section", { id: "audit-timeline", className: "audit-timeline", attrs: { "aria-label": "审计时间线" } });
  section.append(heading("审计时间线", 3));
  if (!events?.length) {
    section.append(element("p", { className: "empty-state", text: "尚无审计事件。" }));
    target.append(section);
    return;
  }
  const list = element("ol");
  events.forEach((event) => {
    const item = element("li", { className: "audit-event" });
    const occurredAt = event.occurredAt ? new Date(event.occurredAt).toLocaleString("zh-CN") : "时间未记录";
    const evidenceCount = Array.isArray(event.evidenceRefs) ? event.evidenceRefs.length : 0;
    item.append(element("strong", { text: `${event.node || "节点未记录"} · ${event.tool || "工具未记录"}` }));
    item.append(element("span", { text: `${event.resultStatus || "状态未记录"} · ${event.requestId || "request ID 未记录"}` }));
    item.append(element("span", { text: `${occurredAt} · ${event.latencyMs == null ? "延迟未记录" : `${event.latencyMs} ms`} · 证据 ${evidenceCount}` }));
    list.append(item);
  });
  section.append(list);
  target.append(section);
}

function medicationRow(medication, mapping) {
  const row = element("article", { className: "medication-row" });
  row.append(heading(medication.name || "未命名药品", 3));
  row.append(definitionList([
    ["剂量", medication.dosage || medication.strength],
    ["剂型", medication.dosageForm],
    ["给药途径", medication.route],
    ["映射类别", mapping?.matchClass],
    ["产品代码", mapping?.selectedProductId],
    ["患者证据", (medication.patientEvidenceRefs || []).join("；")],
  ]));
  const graph = graphProvenanceBlock(mapping?.graphProvenance);
  if (graph) row.append(graph);
  return row;
}

function renderMappingConfirmation(review, mapping, onDecision) {
  const section = element("section", { className: "decision-panel mapping-decision" });
  section.append(heading("需要确认药品映射", 3));
  section.append(element("p", {
    className: "decision-guidance",
    text: `原始药品：${mapping.sourceName}。模糊或不完整映射不会自动采用，请人工选择。`,
  }));
  const group = element("fieldset");
  group.append(element("legend", { text: "候选药品" }));
  (mapping.candidates || []).forEach((candidate) => {
    const code = candidate.productCode || candidate.productId || "未记录";
    const input = element("input", {
      type: "radio",
      attrs: {
        name: `mapping-${mapping.medicationId}`,
        value: candidate.productId,
        "aria-label": `产品代码 ${code}`,
      },
    });
    const label = element("label", { className: "candidate-option" }, [
      input,
      element("span", { text: candidate.productName || candidate.productId || "未命名产品" }),
      element("small", { text: `产品代码 ${code} · ${candidate.dosageForm || "剂型未记录"} · ${candidate.route || "途径未记录"}` }),
    ]);
    group.append(label);
  });
  if ((mapping.unmatchedFields || []).length) {
    section.append(element("p", { className: "warning-text", text: `未匹配字段：${mapping.unmatchedFields.join("、")}` }));
  }
  const confirm = element("button", { className: "primary", type: "button", text: "确认映射" });
  confirm.disabled = true;
  group.addEventListener("change", () => {
    confirm.disabled = !group.querySelector("input:checked");
  });
  confirm.addEventListener("click", () => {
    const selected = group.querySelector("input:checked");
    if (selected) onDecision("CONFIRM_MAPPING", { medicationId: mapping.medicationId, productId: selected.value });
  });
  section.append(group, confirm);
  return section;
}

function renderPatientConfirmation(review, onDecision) {
  const section = element("section", { className: "decision-panel patient-decision" });
  section.append(heading("需要确认患者身份", 3));
  section.append(element("p", { className: "decision-guidance", text: "存在多个匹配的患者记录，请选择准确的 FHIR 患者。" }));
  const group = element("fieldset");
  group.append(element("legend", { text: "候选患者" }));
  (review.candidates || []).forEach((candidate) => {
    const id = candidate.id || candidate.patientId;
    if (!id) return;
    const input = element("input", {
      type: "radio",
      attrs: { name: "patient-candidate", value: id, "aria-label": `FHIR 患者 ${id}` },
    });
    const label = element("label", { className: "candidate-option" }, [
      input,
      element("span", { text: candidate.display || candidate.name || id }),
      element("small", { text: `FHIR 患者 ${id} · 患者编号 ${candidate.patientNumber || "未记录"}` }),
    ]);
    group.append(label);
  });
  const confirm = element("button", { className: "primary", type: "button", text: "确认患者" });
  confirm.disabled = true;
  group.addEventListener("change", () => { confirm.disabled = !group.querySelector("input:checked"); });
  confirm.addEventListener("click", () => {
    const selected = group.querySelector("input:checked");
    if (selected) onDecision("CONFIRM_PATIENT", { patientId: selected.value });
  });
  section.append(group, confirm);
  return section;
}

function renderFindingActions(finding, onDecision) {
  const actions = element("div", { className: "finding-actions interactive-actions" });
  [
    ["接受发现", "ACCEPT_FINDING"],
    ["排除发现", "REJECT_FINDING"],
    ["补充证据", "REQUEST_MORE_EVIDENCE"],
  ].forEach(([label, action]) => {
    const button = element("button", { type: "button", text: label });
    if (action === "ACCEPT_FINDING") button.classList.add("primary");
    button.disabled = finding.status !== "PENDING" && finding.status !== "NEEDS_MORE_EVIDENCE";
    button.addEventListener("click", () => onDecision(action, { findingId: finding.findingId }));
    actions.append(button);
  });
  return actions;
}

export function renderReviewQueue(review, target, selectedFindingId, onSelectFinding, onDecision) {
  clear(target);
  target.append(heading("审核任务队列"));
  if (!review) {
    target.append(element("p", { className: "empty-state", text: "尚无审核任务。" }));
    return;
  }
  const medications = review.medications || [];
  if (medications.length) {
    const medicationSection = element("section", { className: "queue-section" });
    medicationSection.append(heading("当前用药", 3));
    medications.forEach((medication) => {
      const mapping = (review.medicationMappings || []).find((item) => item.medicationId === medication.medicationId);
      medicationSection.append(medicationRow(medication, mapping));
    });
    target.append(medicationSection);
  }
  if (review.status === "AWAITING_MAPPING_CONFIRMATION") {
    (review.medicationMappings || [])
      .filter((mapping) => mapping.mappingConfirmationRequired || (mapping.requiresHumanReview && mapping.candidates?.length))
      .forEach((mapping) => target.append(renderMappingConfirmation(review, mapping, onDecision)));
  }
  if (review.status === "AWAITING_PATIENT_CONFIRMATION") {
    target.append(renderPatientConfirmation(review, onDecision));
  }
  const findingSection = element("section", { className: "queue-section" });
  findingSection.append(heading("待核对发现", 3));
  const findings = review.findings || [];
  if (!findings.length) {
    findingSection.append(element("p", { className: "empty-state", text: "尚无审核发现。" }));
  }
  findings.forEach((finding) => {
    const button = element("button", {
      className: "finding-row",
      type: "button",
      attrs: {
        "aria-current": finding.findingId === selectedFindingId ? "true" : "false",
        "aria-label": finding.summary,
      },
    }, [
      element("span", { className: "finding-title", text: finding.summary }),
      element("span", { className: "finding-meta", text: `${finding.attentionLevel} · ${finding.status}` }),
    ]);
    button.addEventListener("click", () => onSelectFinding(finding.findingId));
    findingSection.append(button);
    if (finding.findingId === selectedFindingId && ["AWAITING_FINDING_REVIEW", "NEEDS_MORE_EVIDENCE"].includes(review.status)) {
      findingSection.append(renderFindingActions(finding, onDecision));
    }
  });
  target.append(findingSection);
}

export function renderEvidence(review, target, selectedFindingId) {
  clear(target);
  target.append(heading("证据核对"));
  if (!review || !selectedFindingId) {
    target.append(element("p", { className: "empty-state", text: "选择审核项以查看患者证据与药品标签原文。" }));
    return;
  }
  const finding = (review.findings || []).find((item) => item.findingId === selectedFindingId);
  if (!finding) return;
  target.append(element("p", { className: "evidence-summary", text: finding.summary }));
  const patientSection = element("section", { className: "evidence-section" });
  patientSection.append(heading("患者证据（FHIR）", 3));
  (finding.patientEvidenceRefs || []).forEach((ref) => patientSection.append(element("p", { className: "evidence-ref", text: ref })));
  if (!(finding.patientEvidenceRefs || []).length) patientSection.append(element("p", { text: "未记录患者证据" }));
  const labelSection = element("section", { className: "evidence-section" });
  labelSection.append(heading("标签原文（SPL / RAG）", 3));
  const evidenceItems = (review.evidenceIndex || []).filter((item) =>
    (finding.labelEvidenceRefs || []).includes(item.evidenceRef));
  (finding.labelEvidenceRefs || []).forEach((ref) => {
    const item = evidenceItems.find((candidate) => candidate.evidenceRef === ref);
    labelSection.append(
      element("p", { className: "evidence-ref", text: ref }),
      element("blockquote", { text: item?.summary || "标签原文摘要未记录" }),
    );
    const imageUrl = allowedImageUrl(item?.imageUrl);
    if (imageUrl) {
      labelSection.append(element("img", {
        className: "label-evidence-image",
        attrs: { src: imageUrl, alt: `标签证据图片 ${ref}`, loading: "lazy" },
      }));
    }
  });
  if (!(finding.labelEvidenceRefs || []).length) labelSection.append(element("p", { text: "未记录标签证据" }));
  const graph = graphProvenanceBlock(finding.graphProvenance || evidenceItems[0]?.graphProvenance);
  if (graph) labelSection.prepend(graph);
  target.append(patientSection, labelSection);
}
