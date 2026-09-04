from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from .gateways import DrugEvidenceGateway, HealthRecordGateway, ToolContractError
from .planner import build_planner_from_env
from .models import (
    AuditEvent,
    ReviewStatus,
    WritebackFailure,
    WritebackStatus,
)
from .report import ReportNotSigned, build_signed_report, render_report_html, render_report_json
from .repository import ReviewNotFound, ReviewRepository, ReviewVersionConflict
from .retrieval import build_grader_from_env
from .safety import QuestionSafetyDecision, evaluate_review_question
from .workflow import ReviewDependencies, build_review_graph, open_sqlite_checkpointer, state_to_snapshot
from .writeback import WritebackCoordinator, WritebackError, WritebackStateError


WEB_DIR = Path(__file__).with_name("web")


class ApiProcessLease:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.path.stat().st_size == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("Medication review API requires a single API process per checkpoint store") from exc
        self.handle = handle

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


class CreateReviewRequest(BaseModel):
    patientId: str | None = None
    asOf: str | None = None
    question: str = Field(min_length=3, max_length=500)

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class DecisionRequest(BaseModel):
    expectedVersion: int = Field(ge=0)
    action: str
    reviewerId: str
    patientId: str | None = None
    medicationId: str | None = None
    productId: str | None = None
    findingId: str | None = None
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    note: str | None = None


class CancelRequest(BaseModel):
    expectedVersion: int = Field(ge=0)


class CompleteReviewRequest(BaseModel):
    expectedVersion: int = Field(ge=0)
    reviewerId: str = Field(min_length=1, max_length=100)


class PrepareWritebackRequest(CompleteReviewRequest):
    pass


class CommitWritebackRequest(CompleteReviewRequest):
    bundleHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmed: Literal[True]


def validate_bind_settings(host: str, *, allow_remote: bool, api_key: str) -> None:
    if host in {"127.0.0.1", "localhost", "::1"}:
        return
    if not allow_remote or not api_key.strip():
        raise ValueError("Non-loopback API binding requires ALLOW_REMOTE_API=true and REVIEW_API_KEY")


def validate_worker_count(workers: int) -> None:
    if workers != 1:
        raise ValueError("Medication review API requires a single worker for checkpoint consistency")


def _decision_payload(body: DecisionRequest) -> dict[str, Any]:
    if body.action in {"ACCEPT_FINDING", "REJECT_FINDING", "REQUEST_MORE_EVIDENCE"}:
        if not body.findingId:
            raise HTTPException(422, "findingId is required")
        return {
            "action": "COMPLETE_FINDING_REVIEW", "reviewerId": body.reviewerId,
            "decisions": [{"action": body.action, "findingId": body.findingId, "note": body.note}],
        }
    payload = body.model_dump(exclude={"expectedVersion"}, exclude_none=True)
    return payload


def create_app(
    dependencies: ReviewDependencies, *, checkpointer: AsyncSqliteSaver | None = None,
    checkpoint_path: str | Path | None = None,
) -> FastAPI:
    owned_checkpointer = checkpointer is None
    lease_path = Path(checkpoint_path or "data/review-checkpoints.sqlite").with_suffix(".api.lock")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        lease = ApiProcessLease(lease_path)
        lease.acquire()
        saver = checkpointer or open_sqlite_checkpointer(checkpoint_path or "data/review-checkpoints.sqlite")
        app.state.checkpointer = saver
        app.state.graph = build_review_graph(dependencies, saver)
        try:
            yield
        finally:
            if owned_checkpointer:
                await saver.conn.close()
            lease.release()

    app = FastAPI(title="Medication Review Agent", version="0.1.0", lifespan=lifespan)
    review_locks: dict[str, asyncio.Lock] = {}
    review_locks_guard = asyncio.Lock()
    configured_key = os.getenv("REVIEW_API_KEY", "")
    configured_reviewer_id = os.getenv("REVIEW_API_REVIEWER_ID", "")
    allow_insecure_mutations = os.getenv("ALLOW_INSECURE_LOCAL_MUTATIONS", "false").lower() in {"1", "true", "yes", "on"}

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        is_mutation = request.method in {"POST", "PUT", "PATCH", "DELETE"}
        if is_mutation and not configured_key and not allow_insecure_mutations:
            return Response(status_code=503, content="Mutation API requires REVIEW_API_KEY")
        if configured_key and request.url.path != "/api/health" and request.headers.get("x-api-key") != configured_key:
            return Response(status_code=401, content="Unauthorized")
        return await call_next(request)

    def get_review(review_id: str):
        try:
            return dependencies.repository.get(review_id)
        except ReviewNotFound as exc:
            raise HTTPException(404, "Review not found") from exc

    def require_reviewer(
        body_reviewer_id: str, header_reviewer_id: str | None
    ) -> None:
        if configured_key and not configured_reviewer_id:
            raise HTTPException(
                503, "REVIEW_API_REVIEWER_ID is required for reviewer decisions"
            )
        if (
            not header_reviewer_id
            or header_reviewer_id != body_reviewer_id
            or (
                configured_reviewer_id
                and header_reviewer_id != configured_reviewer_id
            )
        ):
            raise HTTPException(
                403, "Reviewer identity does not match configured reviewer"
            )

    def persist(review_id: str, graph_state: dict[str, Any], expected_version: int):
        existing = get_review(review_id)
        projected = state_to_snapshot(existing, graph_state)
        try:
            return dependencies.repository.save(projected, expected_version=expected_version)
        except ReviewVersionConflict as exc:
            raise HTTPException(409, "Review version conflict") from exc

    def cancel_unsafe_resume(snapshot, decision: QuestionSafetyDecision):
        cancelled = snapshot.model_copy(deep=True)
        cancelled.status = ReviewStatus.CANCELLED
        cancelled.unresolvedItems = [{
            "kind": "SCOPE_LIMITATION",
            "code": decision.code,
            "summary": decision.explanation,
        }]
        try:
            return dependencies.repository.save(
                cancelled, expected_version=snapshot.version,
            )
        except ReviewVersionConflict as exc:
            raise HTTPException(409, "Review version conflict") from exc

    def mutation_identity(review_id: str, expected_version: int, payload: dict[str, Any]) -> tuple[str, str]:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        mutation_id = hashlib.sha256(
            f"{review_id}:{expected_version}:{fingerprint}".encode("utf-8")
        ).hexdigest()
        return mutation_id, fingerprint

    async def review_lock(review_id: str) -> asyncio.Lock:
        async with review_locks_guard:
            return review_locks.setdefault(review_id, asyncio.Lock())

    async def checkpoint_backup(review_id: str) -> dict[str, list[tuple[Any, ...]]]:
        saver = app.state.checkpointer
        await saver.setup()
        async with saver.lock:
            checkpoints = await (await saver.conn.execute(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ?",
                (review_id,),
            )).fetchall()
            writes = await (await saver.conn.execute(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value FROM writes WHERE thread_id = ?",
                (review_id,),
            )).fetchall()
        return {
            "checkpoints": [tuple(row) for row in checkpoints],
            "writes": [tuple(row) for row in writes],
        }

    async def restore_checkpoint(review_id: str, backup: dict[str, list[tuple[Any, ...]]]) -> None:
        saver = app.state.checkpointer
        async with saver.lock:
            await saver.conn.execute("DELETE FROM writes WHERE thread_id = ?", (review_id,))
            await saver.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (review_id,))
            if backup["checkpoints"]:
                await saver.conn.executemany(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    backup["checkpoints"],
                )
            if backup["writes"]:
                await saver.conn.executemany(
                    "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    backup["writes"],
                )
            await saver.conn.commit()

    @staticmethod
    def encode_backup(backup: dict[str, list[tuple[Any, ...]]]) -> bytes:
        return base64.b64encode(pickle.dumps(backup, protocol=pickle.HIGHEST_PROTOCOL))

    @staticmethod
    def decode_backup(payload: bytes) -> dict[str, list[tuple[Any, ...]]]:
        return pickle.loads(base64.b64decode(payload))

    async def recover_pending(review_id: str) -> None:
        for mutation in dependencies.repository.pending_mutations(review_id):
            if mutation["state"] == "CHECKPOINT_ADVANCED":
                dependencies.repository.commit_mutation(mutation["mutation_id"])
            else:
                await restore_checkpoint(review_id, decode_backup(mutation["checkpoint_backup"]))
                dependencies.repository.abort_prepared_mutation(mutation["mutation_id"])

    async def checkpoint_exists(review_id: str) -> bool:
        saver = app.state.checkpointer
        await saver.setup()
        async with saver.lock:
            row = await (await saver.conn.execute(
                "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1", (review_id,)
            )).fetchone()
        return row is not None

    async def mutate(
        review_id: str, *, expected_version: int, payload: dict[str, Any], command: Any,
    ):
        mutation_id, fingerprint = mutation_identity(review_id, expected_version, payload)
        await recover_pending(review_id)
        existing = dependencies.repository.get_mutation(mutation_id)
        if existing and existing["state"] == "COMMITTED":
            return get_review(review_id)
        snapshot = get_review(review_id)
        if snapshot.version != expected_version:
            raise HTTPException(409, "Review version conflict")
        backup = await checkpoint_backup(review_id)
        dependencies.repository.prepare_mutation(
            mutation_id=mutation_id, review_id=review_id,
            expected_version=expected_version, action_fingerprint=fingerprint,
            checkpoint_backup=encode_backup(backup),
        )
        try:
            result = await app.state.graph.ainvoke(
                command, config={"configurable": {"thread_id": review_id}},
            )
            projected = state_to_snapshot(snapshot, result)
            dependencies.repository.mark_checkpoint_advanced(mutation_id, projected)
            return dependencies.repository.commit_mutation(mutation_id)
        except (ValueError, TypeError):
            await restore_checkpoint(review_id, backup)
            dependencies.repository.abort_prepared_mutation(mutation_id)
            raise
        except BaseException:
            journal = dependencies.repository.get_mutation(mutation_id)
            if journal and journal["state"] == "PREPARED":
                await restore_checkpoint(review_id, backup)
                dependencies.repository.abort_prepared_mutation(mutation_id)
            raise

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "healthy", "schemaVersion": "1.0"}

    @app.post("/api/reviews", status_code=status.HTTP_201_CREATED)
    def create_review(body: CreateReviewRequest):
        return dependencies.repository.create(
            patient_ref=body.patientId,
            as_of=body.asOf,
            question=body.question,
        )

    @app.post("/api/reviews/{review_id}/run")
    async def run_review(review_id: str):
        lock = await review_lock(review_id)
        async with lock:
            payload_snapshot = get_review(review_id)
            payload = {
                "action": "RUN", "patientRef": payload_snapshot.patientRef,
                "asOf": payload_snapshot.asOf, "question": payload_snapshot.question,
            }
            mutation_id, _ = mutation_identity(review_id, payload_snapshot.version, payload)
            owned_before_recovery = dependencies.repository.get_mutation(mutation_id)
            await recover_pending(review_id)
            if owned_before_recovery and owned_before_recovery["state"] == "CHECKPOINT_ADVANCED":
                return get_review(review_id)
            snapshot = get_review(review_id)
            if snapshot.status.value != "CREATED":
                raise HTTPException(409, "Review has already started")
            if await checkpoint_exists(review_id):
                raise HTTPException(409, "Created review has an unowned checkpoint")
            payload = {
                "action": "RUN", "patientRef": snapshot.patientRef,
                "asOf": snapshot.asOf, "question": snapshot.question,
            }
            return await mutate(
                review_id, expected_version=snapshot.version, payload=payload,
                command={"reviewId": review_id, "patientRef": snapshot.patientRef,
                         "asOf": snapshot.asOf, "question": snapshot.question,
                         "mutationId": mutation_identity(review_id, snapshot.version, payload)[0]},
            )

    @app.get("/api/reviews/{review_id}")
    def read_review(review_id: str):
        return get_review(review_id)

    @app.post("/api/reviews/{review_id}/decisions")
    async def decide(review_id: str, body: DecisionRequest, x_reviewer_id: str | None = Header(default=None)):
        require_reviewer(body.reviewerId, x_reviewer_id)
        lock = await review_lock(review_id)
        async with lock:
            try:
                payload = _decision_payload(body)
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, str(exc)) from exc
            mutation_id, _ = mutation_identity(
                review_id, body.expectedVersion, payload,
            )
            owned_before_recovery = dependencies.repository.get_mutation(mutation_id)
            await recover_pending(review_id)
            if (
                owned_before_recovery
                and owned_before_recovery["state"] == "CHECKPOINT_ADVANCED"
            ):
                return get_review(review_id)
            snapshot = get_review(review_id)
            if snapshot.version != body.expectedVersion:
                raise HTTPException(409, "Review version conflict")
            if snapshot.status.value not in {
                "AWAITING_PATIENT_CONFIRMATION", "AWAITING_MAPPING_CONFIRMATION",
                "AWAITING_FINDING_REVIEW", "NEEDS_MORE_EVIDENCE",
            }:
                raise HTTPException(409, "Review is not awaiting a decision")
            question_safety = evaluate_review_question(snapshot.question)
            if not question_safety.allowed:
                return cancel_unsafe_resume(snapshot, question_safety)
            try:
                return await mutate(
                    review_id, expected_version=body.expectedVersion, payload=payload,
                    command=Command(resume=payload, update={
                        "mutationId": mutation_id,
                        "question": snapshot.question,
                        "questionSafety": question_safety.model_dump(mode="json"),
                    }),
                )
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/reviews/{review_id}/complete")
    async def complete_review(
        review_id: str,
        body: CompleteReviewRequest,
        x_reviewer_id: str | None = Header(default=None),
    ):
        require_reviewer(body.reviewerId, x_reviewer_id)
        lock = await review_lock(review_id)
        async with lock:
            await recover_pending(review_id)
            snapshot = get_review(review_id)
            if snapshot.version != body.expectedVersion:
                raise HTTPException(409, "Review version conflict")
            if snapshot.status != ReviewStatus.READY_FOR_SIGN_OFF:
                raise HTTPException(409, "Review is not ready for completion")
            payload = {"action": "SIGN_OFF", "reviewerId": body.reviewerId}
            try:
                return await mutate(
                    review_id,
                    expected_version=body.expectedVersion,
                    payload=payload,
                    command=Command(resume=payload),
                )
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/reviews/{review_id}/writeback/prepare")
    async def prepare_writeback(
        review_id: str,
        body: PrepareWritebackRequest,
        x_reviewer_id: str | None = Header(default=None),
    ):
        require_reviewer(body.reviewerId, x_reviewer_id)
        lock = await review_lock(review_id)
        async with lock:
            await recover_pending(review_id)
            snapshot = get_review(review_id)
            if snapshot.version != body.expectedVersion:
                raise HTTPException(409, "Review version conflict")
            if snapshot.status != ReviewStatus.SIGNED_OFF:
                raise HTTPException(409, "Review must be completed before writeback")
            if snapshot.writebackStatus not in {
                WritebackStatus.NOT_REQUESTED,
                WritebackStatus.FAILED,
            }:
                raise HTTPException(409, "Writeback preview already exists")
            coordinator = WritebackCoordinator(dependencies.health)
            try:
                job = await coordinator.prepare(snapshot, body.reviewerId)
            except WritebackStateError as exc:
                raise HTTPException(409, str(exc)) from exc
            except (WritebackError, ToolContractError) as exc:
                failure = (
                    WritebackFailure(
                        code=exc.code,
                        message=str(exc),
                        retryable=exc.retryable,
                    )
                    if isinstance(exc, WritebackError)
                    else WritebackFailure(
                        code="HEALTH_MCP_CONTRACT_ERROR",
                        message=str(exc),
                        retryable=True,
                    )
                )
                failed = snapshot.model_copy(deep=True)
                failed.writebackStatus = WritebackStatus.FAILED
                failed.writebackError = failure
                try:
                    dependencies.repository.save(
                        failed, expected_version=body.expectedVersion
                    )
                except ReviewVersionConflict as conflict:
                    raise HTTPException(409, "Review version conflict") from conflict
                raise HTTPException(502 if failure.retryable else 422, failure.message)
            prepared = snapshot.model_copy(deep=True)
            prepared.writebackStatus = WritebackStatus.PREPARED
            prepared.writebackJob = job
            prepared.writebackError = None
            try:
                return dependencies.repository.save(
                    prepared, expected_version=body.expectedVersion
                )
            except ReviewVersionConflict as exc:
                raise HTTPException(409, "Review version conflict") from exc

    @app.post("/api/reviews/{review_id}/writeback/commit")
    async def commit_writeback(
        review_id: str,
        body: CommitWritebackRequest,
        x_reviewer_id: str | None = Header(default=None),
    ):
        require_reviewer(body.reviewerId, x_reviewer_id)
        lock = await review_lock(review_id)
        async with lock:
            await recover_pending(review_id)
            snapshot = get_review(review_id)
            if snapshot.version != body.expectedVersion:
                raise HTTPException(409, "Review version conflict")
            if snapshot.status != ReviewStatus.SIGNED_OFF:
                raise HTTPException(409, "Review must remain completed for writeback")
            if snapshot.writebackJob is None or snapshot.writebackStatus not in {
                WritebackStatus.PREPARED,
                WritebackStatus.FAILED,
                WritebackStatus.COMMITTED,
            }:
                raise HTTPException(409, "Writeback preview is not prepared")
            if body.bundleHash != snapshot.writebackJob.bundleHash:
                raise HTTPException(409, "Writeback bundle hash conflict")
            coordinator = WritebackCoordinator(dependencies.health)
            try:
                result = await coordinator.commit(
                    snapshot.writebackJob, confirmed=body.confirmed
                )
            except (WritebackError, ToolContractError) as exc:
                failure = (
                    WritebackFailure(
                        code=exc.code,
                        message=str(exc),
                        retryable=exc.retryable,
                    )
                    if isinstance(exc, WritebackError)
                    else WritebackFailure(
                        code="HEALTH_MCP_CONTRACT_ERROR",
                        message=str(exc),
                        retryable=True,
                    )
                )
                failed = snapshot.model_copy(deep=True)
                failed.writebackStatus = WritebackStatus.FAILED
                failed.writebackError = failure
                try:
                    dependencies.repository.save(
                        failed, expected_version=body.expectedVersion
                    )
                except ReviewVersionConflict as conflict:
                    raise HTTPException(409, "Review version conflict") from conflict
                raise HTTPException(502 if failure.retryable else 422, failure.message)
            committed = snapshot.model_copy(deep=True)
            committed.writebackStatus = WritebackStatus.COMMITTED
            committed.writebackJob.result = result
            committed.writebackError = None
            try:
                saved = dependencies.repository.save(
                    committed, expected_version=body.expectedVersion
                )
            except ReviewVersionConflict as exc:
                raise HTTPException(409, "Review version conflict") from exc
            dependencies.repository.append_audit(
                review_id,
                node="commit_writeback",
                tool="commit_medication_review_writeback",
                request_id=None,
                result_status="OK",
                argument_summary={"evidenceCount": len(result.get("created") or [])},
                evidence_refs=[],
                latency_ms=0,
            )
            return saved

    @app.get("/api/reviews/{review_id}/writeback")
    def read_writeback(review_id: str):
        snapshot = get_review(review_id)
        return {
            "reviewId": snapshot.reviewId,
            "reviewVersion": snapshot.version,
            "status": snapshot.writebackStatus,
            "job": snapshot.writebackJob,
            "error": snapshot.writebackError,
        }

    @app.post("/api/reviews/{review_id}/cancel")
    async def cancel(review_id: str, body: CancelRequest, x_reviewer_id: str | None = Header(default=None)):
        if configured_key and not configured_reviewer_id:
            raise HTTPException(503, "REVIEW_API_REVIEWER_ID is required for reviewer decisions")
        if not x_reviewer_id or (configured_reviewer_id and x_reviewer_id != configured_reviewer_id):
            raise HTTPException(403, "Reviewer identity header is required")
        lock = await review_lock(review_id)
        async with lock:
            await recover_pending(review_id)
            snapshot = get_review(review_id)
            expected = body.expectedVersion
            if expected != snapshot.version:
                raise HTTPException(409, "Review version conflict")
            if snapshot.status in {ReviewStatus.SIGNED_OFF, ReviewStatus.CANCELLED}:
                raise HTTPException(409, "Terminal review cannot be cancelled")
            snapshot.status = ReviewStatus.CANCELLED
            try:
                return dependencies.repository.save(snapshot, expected_version=expected)
            except ReviewVersionConflict as exc:
                raise HTTPException(409, "Review version conflict") from exc

    @app.get("/api/reviews/{review_id}/audit")
    def audit(review_id: str):
        get_review(review_id)
        return dependencies.repository.list_audit(review_id)

    def signed_report(review_id: str):
        snapshot = get_review(review_id)
        snapshot.auditEvents = [AuditEvent.model_validate(item) for item in dependencies.repository.list_audit(review_id)]
        reviewer = next((item.reviewerId for item in reversed(snapshot.humanDecisions) if item.action == "SIGN_OFF"), None)
        try:
            return build_signed_report(snapshot, reviewer)
        except ReportNotSigned as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/reviews/{review_id}/report.json")
    def report_json(review_id: str):
        return Response(render_report_json(signed_report(review_id)), media_type="application/json")

    @app.get("/api/reviews/{review_id}/report.html")
    def report_html(review_id: str):
        return Response(render_report_html(signed_report(review_id)), media_type="text/html")

    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="workbench-assets")

    @app.get("/", include_in_schema=False)
    def workbench():
        return FileResponse(WEB_DIR / "index.html", media_type="text/html")

    return app


def build_app_from_env() -> FastAPI:
    validate_worker_count(int(os.getenv("REVIEW_API_WORKERS", "1")))
    repository = ReviewRepository(os.getenv("REVIEW_DB_PATH", "data/reviews.sqlite"))
    dependencies = ReviewDependencies(
        health=HealthRecordGateway.from_env(), drug=DrugEvidenceGateway.from_env(),
        repository=repository, planner=build_planner_from_env(),
        grader=build_grader_from_env(),
    )
    return create_app(dependencies, checkpoint_path=os.getenv("REVIEW_CHECKPOINT_DB", "data/review-checkpoints.sqlite"))


def main() -> None:
    host = os.getenv("REVIEW_API_HOST", "127.0.0.1")
    port = int(os.getenv("REVIEW_API_PORT", "8020"))
    allow_remote = os.getenv("ALLOW_REMOTE_API", "false").lower() in {"1", "true", "yes", "on"}
    workers = int(os.getenv("REVIEW_API_WORKERS", "1"))
    validate_bind_settings(host, allow_remote=allow_remote, api_key=os.getenv("REVIEW_API_KEY", ""))
    validate_worker_count(workers)
    uvicorn.run(build_app_from_env(), host=host, port=port, workers=workers)
