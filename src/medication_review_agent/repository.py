from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from .models import AuditEvent, ReviewSnapshot, ReviewStatus


class ReviewVersionConflict(RuntimeError):
    pass


class ReviewNotFound(KeyError):
    pass


class AuditRedactionError(ValueError):
    pass


_HASH_SUMMARY_KEYS = {"patientIdHash", "medicationIdHash", "productIdHash"}
_COUNT_SUMMARY_KEYS = {"productCount", "topicCount", "claimCount", "medicationCount"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_argument_summary(value: Any) -> None:
    if not isinstance(value, dict):
        raise AuditRedactionError("Audit argument summary must be an object")
    for key, child in value.items():
        if key in _HASH_SUMMARY_KEYS:
            if not isinstance(child, str) or not _SHA256_RE.fullmatch(child):
                raise AuditRedactionError(f"Audit hash must be lowercase SHA-256: {key}")
        elif key in _COUNT_SUMMARY_KEYS:
            if not isinstance(child, int) or isinstance(child, bool) or child < 0:
                raise AuditRedactionError(f"Audit count must be a non-negative integer: {key}")
        else:
            raise AuditRedactionError(f"Audit argument key is not allowlisted: {key}")


class ReviewRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS reviews (
                  review_id TEXT PRIMARY KEY,
                  version INTEGER NOT NULL,
                  status TEXT NOT NULL,
                  patient_ref TEXT,
                  snapshot_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  review_id TEXT NOT NULL,
                  mutation_id TEXT,
                  audit_slot TEXT,
                  event_key TEXT,
                  committed INTEGER NOT NULL DEFAULT 1,
                  occurred_at TEXT NOT NULL,
                  event_json TEXT NOT NULL,
                  FOREIGN KEY(review_id) REFERENCES reviews(review_id)
                );
                CREATE TABLE IF NOT EXISTS mutation_journal (
                  mutation_id TEXT PRIMARY KEY,
                  review_id TEXT NOT NULL,
                  expected_version INTEGER NOT NULL,
                  action_fingerprint TEXT NOT NULL,
                  state TEXT NOT NULL CHECK(state IN ('PREPARED', 'CHECKPOINT_ADVANCED', 'COMMITTED')),
                  checkpoint_backup BLOB NOT NULL,
                  projected_snapshot_json TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(review_id) REFERENCES reviews(review_id)
                );
            """)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(audit_events)")}
            for name, definition in (
                ("mutation_id", "TEXT"), ("event_key", "TEXT"),
                ("audit_slot", "TEXT"),
                ("committed", "INTEGER NOT NULL DEFAULT 1"),
            ):
                if name not in columns:
                    connection.execute(f"ALTER TABLE audit_events ADD COLUMN {name} {definition}")
            connection.execute("DROP INDEX IF EXISTS audit_events_mutation_key")
            connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS audit_events_mutation_slot
                ON audit_events(review_id, mutation_id, audit_slot)
                WHERE mutation_id IS NOT NULL AND audit_slot IS NOT NULL""")

    def create(
        self, patient_ref: str | None, *, review_id: str | None = None,
        as_of: str | None = None,
    ) -> ReviewSnapshot:
        now = datetime.now(UTC)
        snapshot = ReviewSnapshot(
            reviewId=review_id or str(uuid4()), status=ReviewStatus.CREATED,
            patientRef=patient_ref, asOf=as_of, createdAt=now, updatedAt=now,
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO reviews(review_id, version, status, patient_ref, snapshot_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (snapshot.reviewId, 0, snapshot.status.value, snapshot.patientRef,
                 snapshot.model_dump_json(), now.isoformat(), now.isoformat()),
            )
        return snapshot

    def get(self, review_id: str) -> ReviewSnapshot:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM reviews WHERE review_id = ?", (review_id,)
            ).fetchone()
        if row is None:
            raise ReviewNotFound(review_id)
        return ReviewSnapshot.model_validate_json(row["snapshot_json"])

    def save(self, snapshot: ReviewSnapshot, *, expected_version: int) -> ReviewSnapshot:
        updated = snapshot.model_copy(deep=True)
        updated.version = expected_version + 1
        updated.updatedAt = datetime.now(UTC)
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE reviews
                   SET version = ?, status = ?, patient_ref = ?, snapshot_json = ?, updated_at = ?
                   WHERE review_id = ? AND version = ?""",
                (updated.version, updated.status.value, updated.patientRef,
                 updated.model_dump_json(), updated.updatedAt.isoformat(),
                 updated.reviewId, expected_version),
            )
            if cursor.rowcount != 1:
                raise ReviewVersionConflict(
                    f"Review {updated.reviewId} is not at version {expected_version}"
                )
        return updated

    def append_audit(
        self, review_id: str, *, node: str, tool: str | None,
        request_id: str | None, result_status: str,
        argument_summary: dict[str, Any], evidence_refs: list[str], latency_ms: int,
        retry_count: int = 0, model_id: str | None = None,
        prompt_version: str | None = None, input_tokens: int = 0,
        output_tokens: int = 0, estimated_cost: float = 0.0,
        mutation_id: str | None = None, audit_slot: str | None = None,
    ) -> AuditEvent:
        _validate_argument_summary(argument_summary)
        audit_slot = audit_slot or node
        event = AuditEvent(
            mutationId=mutation_id,
            auditSlot=audit_slot,
            node=node, tool=tool, requestId=request_id, resultStatus=result_status,
            argumentSummary=argument_summary, evidenceRefs=evidence_refs,
            latencyMs=latency_ms, retryCount=retry_count, modelId=model_id,
            promptVersion=prompt_version, inputTokens=input_tokens,
            outputTokens=output_tokens, estimatedCost=estimated_cost,
        )
        event_key = hashlib.sha256((audit_slot or "").encode("utf-8")).hexdigest()
        try:
            with self._connect() as connection:
                staged = bool(mutation_id and connection.execute(
                    "SELECT 1 FROM mutation_journal WHERE mutation_id = ? AND state != 'COMMITTED'",
                    (mutation_id,),
                ).fetchone())
                connection.execute(
                    """INSERT INTO audit_events
                       (review_id, mutation_id, audit_slot, event_key, committed, occurred_at, event_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(review_id, mutation_id, audit_slot) WHERE mutation_id IS NOT NULL AND audit_slot IS NOT NULL
                       DO NOTHING""",
                    (review_id, mutation_id, audit_slot, event_key, 0 if staged else 1,
                     event.occurredAt.isoformat(), event.model_dump_json()),
                )
                if mutation_id:
                    existing = connection.execute(
                        "SELECT event_json FROM audit_events WHERE review_id = ? AND mutation_id = ? AND audit_slot = ?",
                        (review_id, mutation_id, audit_slot),
                    ).fetchone()
                    event = AuditEvent.model_validate_json(existing["event_json"])
        except sqlite3.IntegrityError as exc:
            raise ReviewNotFound(review_id) from exc
        return event

    def list_audit(self, review_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event_json FROM audit_events WHERE review_id = ? AND committed = 1 ORDER BY event_id",
                (review_id,),
            ).fetchall()
        return [json.loads(row["event_json"]) for row in rows]

    def prepare_mutation(
        self, *, mutation_id: str, review_id: str, expected_version: int,
        action_fingerprint: str, checkpoint_backup: bytes,
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO mutation_journal
                   (mutation_id, review_id, expected_version, action_fingerprint, state,
                    checkpoint_backup, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'PREPARED', ?, ?, ?)""",
                (mutation_id, review_id, expected_version, action_fingerprint,
                 checkpoint_backup, now, now),
            )
            row = connection.execute(
                "SELECT * FROM mutation_journal WHERE mutation_id = ?", (mutation_id,)
            ).fetchone()
        return dict(row)

    def get_mutation(self, mutation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM mutation_journal WHERE mutation_id = ?", (mutation_id,)
            ).fetchone()
        return dict(row) if row else None

    def pending_mutations(self, review_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM mutation_journal WHERE review_id = ? AND state != 'COMMITTED' ORDER BY created_at",
                (review_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_checkpoint_advanced(self, mutation_id: str, snapshot: ReviewSnapshot) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE mutation_journal SET state = 'CHECKPOINT_ADVANCED',
                   projected_snapshot_json = ?, updated_at = ?
                   WHERE mutation_id = ? AND state = 'PREPARED'""",
                (snapshot.model_dump_json(), datetime.now(UTC).isoformat(), mutation_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Mutation {mutation_id} is not PREPARED")

    def abort_prepared_mutation(self, mutation_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM audit_events WHERE mutation_id = ? AND committed = 0", (mutation_id,)
            )
            connection.execute(
                "DELETE FROM mutation_journal WHERE mutation_id = ? AND state != 'COMMITTED'", (mutation_id,)
            )

    def commit_mutation(self, mutation_id: str) -> ReviewSnapshot:
        row = self.get_mutation(mutation_id)
        if row is None:
            raise RuntimeError(f"Mutation {mutation_id} not found")
        if row["state"] == "COMMITTED":
            return self.get(row["review_id"])
        if row["state"] != "CHECKPOINT_ADVANCED" or not row["projected_snapshot_json"]:
            raise RuntimeError(f"Mutation {mutation_id} has not advanced its checkpoint")
        projected = ReviewSnapshot.model_validate_json(row["projected_snapshot_json"])
        current = self.get(row["review_id"])
        if current.version == row["expected_version"]:
            projected = self.save(projected, expected_version=row["expected_version"])
        elif current.version == row["expected_version"] + 1:
            projected = current
        else:
            raise ReviewVersionConflict(
                f"Review {projected.reviewId} is not recoverable at version {current.version}"
            )
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM mutation_journal WHERE mutation_id = ?", (mutation_id,)
            ).fetchone()
            connection.execute(
                "UPDATE audit_events SET committed = 1 WHERE mutation_id = ?", (mutation_id,)
            )
            connection.execute(
                "UPDATE mutation_journal SET state = 'COMMITTED', updated_at = ? WHERE mutation_id = ?",
                (datetime.now(UTC).isoformat(), mutation_id),
            )
        return projected
