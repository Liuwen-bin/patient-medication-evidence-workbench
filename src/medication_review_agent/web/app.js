import { createReview, getAudit, getReview, runReview, sendDecision } from "./api-client.js";
import { renderAuditTimeline, renderEvidence, renderPatientContext, renderReviewQueue, renderSignOffDialog, renderStatus } from "./render.js";

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
  run: document.querySelector("#run-review"),
  sign: document.querySelector("#sign-report"),
  status: document.querySelector("#review-status"),
  patient: document.querySelector("#patient-panel"),
  queue: document.querySelector("#review-queue"),
  evidence: document.querySelector("#evidence-panel"),
  live: document.querySelector("#live-region"),
  dialogRoot: document.querySelector("#dialog-root"),
};
let signOffOpener = null;

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

function closeSignOffDialog() {
  nodes.dialogRoot.replaceChildren();
  signOffOpener?.focus();
  signOffOpener = null;
}

async function signOff(reviewer) {
  if (!state.review || state.pending) return;
  state.pending = true;
  render();
  try {
    state.review = await sendDecision(state.review.reviewId, {
      expectedVersion: state.review.version,
      action: "SIGN_OFF",
      reviewerId: reviewer,
    });
    closeSignOffDialog();
    notice("报告已签署");
  } catch (error) {
    if (error.status === 409) {
      closeSignOffDialog();
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

function openSignOffDialog() {
  if (!state.review || state.review.status !== "READY_FOR_SIGN_OFF" || state.pending) return;
  const dialog = renderSignOffDialog(state.review, signOff, closeSignOffDialog);
  signOffOpener = nodes.sign;
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

function render() {
  renderStatus(state.review, nodes.status);
  renderPatientContext(state.review, nodes.patient);
  renderAuditTimeline(state.audit, nodes.patient);
  renderReviewQueue(state.review, nodes.queue, state.selectedFindingId, (findingId) => {
    state.selectedFindingId = findingId;
    render();
    if (window.matchMedia("(max-width: 920px)").matches) setActiveTab("evidence");
  }, decide);
  renderEvidence(state.review, nodes.evidence, state.selectedFindingId);
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
  let reportLink = document.querySelector("#signed-report-link");
  if (state.review?.status === "SIGNED_OFF") {
    if (!reportLink) {
      reportLink = document.createElement("a");
      reportLink.id = "signed-report-link";
      reportLink.textContent = "查看报告";
      nodes.sign.insertAdjacentElement("afterend", reportLink);
    }
    reportLink.href = `/api/reviews/${encodeURIComponent(state.review.reviewId)}/report.html`;
  } else {
    reportLink?.remove();
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
    state.selectedFindingId = state.review.findings?.[0]?.findingId || null;
    if (awaitingHuman.has(state.review.status) && window.matchMedia("(max-width: 920px)").matches) {
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
      review = await createReview({ patientId: nodes.patientId.value.trim(), asOf: nodes.asOf.value });
    }
    state.review = await runReview(review.reviewId);
    state.selectedFindingId = state.review.findings?.[0]?.findingId || null;
    history.replaceState(null, "", `?review=${encodeURIComponent(state.review.reviewId)}`);
    announce("复核已运行，请核对当前审核项");
    if (awaitingHuman.has(state.review.status) && window.matchMedia("(max-width: 920px)").matches) setActiveTab("queue");
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
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && nodes.dialogRoot.firstElementChild) closeSignOffDialog();
});

if (!nodes.asOf.value) nodes.asOf.value = new Date().toISOString().slice(0, 10);
render();
const reviewId = new URLSearchParams(window.location.search).get("review");
if (reviewId) loadReview(reviewId);
