export async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const message = typeof body === "object" && body
      ? body.detail || body.message
      : null;
    const error = new Error(message || `HTTP ${response.status}`);
    error.status = response.status;
    error.body = body;
    throw error;
  }
  return body;
}

export const createReview = (payload) => request("/api/reviews", {
  method: "POST",
  body: JSON.stringify(payload),
});

export const runReview = (reviewId) => request(`/api/reviews/${encodeURIComponent(reviewId)}/run`, {
  method: "POST",
});

export const getReview = (reviewId) => request(`/api/reviews/${encodeURIComponent(reviewId)}`);

export const sendDecision = (reviewId, payload) => request(
  `/api/reviews/${encodeURIComponent(reviewId)}/decisions`,
  {
    method: "POST",
    headers: { "x-reviewer-id": payload.reviewerId },
    body: JSON.stringify(payload),
  },
);

export const completeReview = (reviewId, payload) => request(
  `/api/reviews/${encodeURIComponent(reviewId)}/complete`,
  {
    method: "POST",
    headers: { "x-reviewer-id": payload.reviewerId },
    body: JSON.stringify(payload),
  },
);

export const prepareWriteback = (reviewId, payload) => request(
  `/api/reviews/${encodeURIComponent(reviewId)}/writeback/prepare`,
  {
    method: "POST",
    headers: { "x-reviewer-id": payload.reviewerId },
    body: JSON.stringify(payload),
  },
);

export const commitWriteback = (reviewId, payload) => request(
  `/api/reviews/${encodeURIComponent(reviewId)}/writeback/commit`,
  {
    method: "POST",
    headers: { "x-reviewer-id": payload.reviewerId },
    body: JSON.stringify(payload),
  },
);

export const getAudit = (reviewId) => request(`/api/reviews/${encodeURIComponent(reviewId)}/audit`);
