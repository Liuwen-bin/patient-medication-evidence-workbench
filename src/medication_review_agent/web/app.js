import {
  commitWriteback,
  completeReview,
  createReview,
  getAudit,
  getReview,
  prepareWriteback,
  runReview,
  sendDecision,
} from "./api-client.js";
import {
  renderAuditTimeline,
  renderEvidence,
  renderPatientContext,
  renderReviewQueue,
  renderSignOffDialog,
  renderStatus,
  renderWritebackDialog,
  renderWritebackPreview,
  renderWritebackResult,
} from "./render.js";

const state = {
  review: null,
  selectedFindingId: null,
  activeTab: "patient",
  pending: false,
  notice: null,
  audit: [],
};

const nodes = {
  form: document.querySelector("#patient-form"),
  patientId: document.querySelector("#patient-id"),
  asOf: document.querySelector("#as-of"),
  question: document.querySelector("#review-question"),
  run: document.querySelector("#start-review"),
  sign: document.querySelector("#complete-review"),
  prepare: document.querySelector("#prepare-writeback"),
  commit: document.querySelector("#open-writeback-confirmation"),
  exports: document.querySelector("#report-exports"),
  status: document.querySelector("#review-status"),
  patient: document.querySelector("#patient-panel"),
  queue: document.querySelector("#review-queue"),
  evidence: document.querySelector("#evidence-panel"),
  evidenceDetail: document.querySelector("#evidence-detail"),
  writebackPreview: document.querySelector("#writeback-preview"),
  writebackResult: document.querySelector("#writeback-result"),
  live: document.querySelector("#live-region"),
  dialogRoot: document.querySelector("#dialog-root"),
};
let dialogOpener = null;

const awaitingHuman = new Set([
  "AWAITING_PATIENT_CONFIRMATION",
  "AWAITING_MAPPING_CONFIRMATION",
  "AWAITING_FINDING_REVIEW",
  "NEEDS_MORE_EVIDENCE",
  "READY_FOR_SIGN_OFF",
]);

function setActiveTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.setAttribute("aria-selected", button.dataset.tab === tab ? "true" : "false");
  });
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.panel === tab);
  });
}

function announce(message) {
  nodes.live.textContent = message;
}

function notice(message, tone = "info") {
  state.notice = { message, tone };
  announce(message);
}

function reviewerId() {
  return document.querySelector("#reviewer-id").value.trim();
}

async function decide(action, details = {}) {
  if (!state.review || state.pending) return;
  if (!reviewerId()) {
    document.querySelector("#reviewer-id").reportValidity();
    return;
  }
  state.pending = true;
  render();
  try {
    state.review = await sendDecision(state.review.reviewId, {
      expectedVersion: state.review.version,
      action,
      reviewerId: reviewerId(),
      ...details,
    });
    if (action === "CONFIRM_MAPPING") notice("映射已确认");
    else if (action === "CONFIRM_PATIENT") notice("患者已确认");
    else announce("审核决定已保存");
  } catch (error) {
    if (error.status === 409) {
      state.review = await getReview(state.review.reviewId);
      notice("审核状态已被更新，请重新确认当前项目", "error");
    } else {
      notice(`操作失败：${error.message}`, "error");
    }
  } finally {
    state.pending = false;
    render();
  }
}

function closeDialog() {
  nodes.dialogRoot.replaceChildren();
  dialogOpener?.focus();
  dialogOpener = null;
}

async function signOff(reviewer) {
  if (!state.review || state.pending) return;
  state.pending = true;
  render();
  try {
    state.review = await completeReview(state.review.reviewId, {
      expectedVersion: state.review.version,
      reviewerId: reviewer,
    });
    closeDialog();
    notice("审核已完成");
  } catch (error) {
    if (error.status === 409) {
      closeDialog();
      state.review = await getReview(state.review.reviewId);
      notice("状态已更新，请重新确认", "error");
    } else {
      notice(`操作失败：${error.message}`, "error");
    }
  } finally {
    state.pending = false;
    render();
  }
}

function openSignOffDialog() {
  if (!state.review || state.review.status !== "READY_FOR_SIGN_OFF" || state.pending) return;
  const dialog = renderSignOffDialog(state.review, signOff, closeDialog);
  dialogOpener = nodes.sign;
  nodes.dialogRoot.replaceChildren(dialog);
  const reviewerInput = dialog.querySelector("#sign-off-reviewer-id");
  reviewerInput?.focus();
  dialog.addEventListener("keydown", (event) => {
    if (event.key !== "Tab") return;
    const focusables = [...dialog.querySelectorAll("button, input")].filter((node) => !node.disabled);
    if (!focusables.length) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });
}

async function prepareCurrentWriteback() {
  if (!state.review || state.pending || !reviewerId()) return;
  state.pending = true;
  render();
  try {
    state.review = await prepareWriteback(state.review.reviewId, {
      expectedVersion: state.review.version,
      reviewerId: reviewerId(),
    });
    notice("FHIR 写回预览已生成");
    if (window.matchMedia("(max-width: 899px)").matches) setActiveTab("evidence");
  } catch (error) {
    if (error.status === 409) {
      state.review = await getReview(state.review.reviewId);
      notice("状态已更新，请重新确认", "error");
    } else {
      notice(`生成预览失败：${error.message}`, "error");
    }
  } finally {
    state.pending = false;
    render();
  }
}

async function commitCurrentWriteback() {
  if (!state.review?.writebackJob || state.pending) return;
  state.pending = true;
  render();
  try {
    state.review = await commitWriteback(state.review.reviewId, {
      expectedVersion: state.review.version,
      reviewerId: reviewerId(),
      bundleHash: state.review.writebackJob.bundleHash,
      confirmed: true,
    });
    closeDialog();
    notice("FHIR 写回完成");
  } catch (error) {
    closeDialog();
    if (error.status === 409) {
      state.review = await getReview(state.review.reviewId);
      notice("状态已更新，请重新确认", "error");
    } else {
      try {
        state.review = await getReview(state.review.reviewId);
      } catch {
        // Keep the last durable snapshot when refresh is unavailable.
      }
      notice(`写回失败：${error.message}`, "error");
    }
  } finally {
    state.pending = false;
    render();
  }
}

function openWritebackDialog() {
  if (!state.review?.writebackJob || state.pending || !reviewerId()) return;
  const dialog = renderWritebackDialog(
    state.review,
    reviewerId(),
    commitCurrentWriteback,
    closeDialog,
  );
  dialogOpener = nodes.commit;
  nodes.dialogRoot.replaceChildren(dialog);
  dialog.querySelector("#writeback-confirmed")?.focus();
}

function render() {
  renderStatus(state.review, nodes.status);
  renderPatientContext(state.review, nodes.patient);
  renderAuditTimeline(state.audit, nodes.patient);
  renderReviewQueue(state.review, nodes.queue, state.selectedFindingId, (findingId) => {
    state.selectedFindingId = findingId;
    render();
    if (window.matchMedia("(max-width: 899px)").matches) {
      setActiveTab("evidence");
      const evidenceHeading = nodes.evidenceDetail.querySelector("h2");
      evidenceHeading?.setAttribute("tabindex", "-1");
      evidenceHeading?.focus();
    }
  }, decide);
  renderEvidence(state.review, nodes.evidenceDetail, state.selectedFindingId);
  renderWritebackPreview(nodes.writebackPreview, state.review);
  renderWritebackResult(nodes.writebackResult, state.review);
  if (state.notice) {
    const node = document.createElement("p");
    node.id = "workbench-notice";
    node.className = `workbench-notice ${state.notice.tone}`;
    node.setAttribute("role", state.notice.tone === "error" ? "alert" : "status");
    node.textContent = state.notice.message;
    nodes.queue.prepend(node);
  }
  nodes.run.disabled = state.pending;
  nodes.sign.disabled = state.pending || state.review?.status !== "READY_FOR_SIGN_OFF";
  nodes.sign.textContent = state.review?.status === "SIGNED_OFF" ? "审核已完成" : "完成审核";
  const canPrepare = state.review?.status === "SIGNED_OFF" && (
    state.review.writebackStatus === "NOT_REQUESTED"
    || (state.review.writebackStatus === "FAILED" && !state.review.writebackJob)
  );
  nodes.prepare.disabled = state.pending || !canPrepare;
  nodes.prepare.textContent = state.review?.writebackStatus === "FAILED" ? "重新生成预览" : "生成写回预览";
  const canCommit = state.review?.status === "SIGNED_OFF"
    && Boolean(state.review.writebackJob)
    && (state.review.writebackStatus === "PREPARED"
      || (state.review.writebackStatus === "FAILED" && state.review.writebackError?.retryable));
  nodes.commit.disabled = state.pending || !canCommit;
  nodes.commit.textContent = state.review?.writebackStatus === "FAILED" ? "重试写回" : "确认写回";
  nodes.exports.replaceChildren();
  if (state.review?.status === "SIGNED_OFF") {
    [["导出 JSON", "report.json"], ["导出 HTML", "report.html"]].forEach(([label, file]) => {
      const link = document.createElement("a");
      link.textContent = label;
      link.href = `/api/reviews/${encodeURIComponent(state.review.reviewId)}/${file}`;
      nodes.exports.append(link);
    });
  }
}

async function loadReview(reviewId) {
  state.pending = true;
  render();
  try {
    state.review = await getReview(reviewId);
    try {
      state.audit = await getAudit(reviewId);
    } catch (error) {
      state.audit = [];
      notice(`审计时间线暂不可用：${error.message}`, "error");
    }
    nodes.patientId.value = state.review.patientRef || "";
    nodes.asOf.value = state.review.asOf || "";
    nodes.question.value = state.review.question || nodes.question.value;
    state.selectedFindingId = state.review.findings?.[0]?.findingId || null;
    if (awaitingHuman.has(state.review.status) && window.matchMedia("(max-width: 899px)").matches) {
      setActiveTab("queue");
    }
    announce("复核状态已载入");
  } catch (error) {
    announce(`无法载入复核：${error.message}`);
  } finally {
    state.pending = false;
    render();
  }
}

nodes.run.addEventListener("click", async () => {
  if (!nodes.form.reportValidity()) return;
  state.pending = true;
  render();
  try {
    let review = state.review;
    if (!review) {
      review = await createReview({
        patientId: nodes.patientId.value.trim(),
        asOf: nodes.asOf.value,
        question: nodes.question.value.trim(),
      });
    }
    state.review = await runReview(review.reviewId);
    state.selectedFindingId = state.review.findings?.[0]?.findingId || null;
    history.replaceState(null, "", `?review=${encodeURIComponent(state.review.reviewId)}`);
    announce("复核已运行，请核对当前审核项");
    if (awaitingHuman.has(state.review.status) && window.matchMedia("(max-width: 899px)").matches) setActiveTab("queue");
  } catch (error) {
    announce(`运行失败：${error.message}`);
  } finally {
    state.pending = false;
    render();
  }
});

document.querySelectorAll("[data-tab]").forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});

nodes.sign.addEventListener("click", openSignOffDialog);
nodes.prepare.addEventListener("click", prepareCurrentWriteback);
nodes.commit.addEventListener("click", openWritebackDialog);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && nodes.dialogRoot.firstElementChild) closeDialog();
});

if (!nodes.asOf.value) nodes.asOf.value = new Date().toISOString().slice(0, 10);
render();
const reviewId = new URLSearchParams(window.location.search).get("review");
if (reviewId) loadReview(reviewId);
