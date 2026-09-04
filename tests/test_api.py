from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from medication_review_agent.api import validate_bind_settings, validate_worker_count
from medication_review_agent.api import build_app_from_env, create_app
from medication_review_agent.planner import DeterministicPlanner
from medication_review_agent.models import ReviewStatus
from medication_review_agent.repository import ReviewRepository
from medication_review_agent.workflow import ReviewDependencies
from tests.fakes import FakeDrugGateway, FakeHealthGateway, health_context, mapped_response, standard_drug_responses
from tests.test_workflow import MED1


def test_create_run_and_read_review(client: TestClient) -> None:
    created = client.post("/api/reviews", json={"patientId": "P001", "asOf": "2026-08-31"})
    assert created.status_code == 201
    review_id = created.json()["reviewId"]
    run = client.post(f"/api/reviews/{review_id}/run")
    assert run.status_code == 200
    current = client.get(f"/api/reviews/{review_id}").json()
    assert current["status"] in {"AWAITING_MAPPING_CONFIRMATION", "AWAITING_FINDING_REVIEW"}
    assert current["version"] == 1


def test_stale_decision_returns_conflict(client: TestClient) -> None:
    response = client.post("/api/reviews/review-1/decisions", json={
        "expectedVersion": 0, "action": "REJECT_FINDING",
        "findingId": "f1", "reviewerId": "pharmacist-demo",
    })
    assert response.status_code == 409


def test_unknown_review_returns_not_found(client: TestClient) -> None:
    assert client.get("/api/reviews/missing").status_code == 404


def test_unsigned_report_returns_conflict(client: TestClient) -> None:
    assert client.get("/api/reviews/review-1/report.json").status_code == 409


def test_audit_endpoint_returns_source_linked_events(client: TestClient) -> None:
    created = client.post("/api/reviews", json={"patientId": "P001", "asOf": "2026-08-31"}).json()
    client.post(f"/api/reviews/{created['reviewId']}/run")
    events = client.get(f"/api/reviews/{created['reviewId']}/audit").json()
    assert events
    assert any(event["evidenceRefs"] for event in events)


def test_non_loopback_bind_requires_remote_flag_and_api_key() -> None:
    for allow, key in [(False, "secret"), (True, "")]:
        try:
            validate_bind_settings("0.0.0.0", allow_remote=allow, api_key=key)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe non-loopback binding was accepted")
    validate_bind_settings("0.0.0.0", allow_remote=True, api_key="secret")


def test_api_rejects_multiple_workers_because_checkpoint_lock_is_process_local() -> None:
    validate_worker_count(1)
    try:
        validate_worker_count(2)
    except ValueError as exc:
        assert "single worker" in str(exc)
    else:
        raise AssertionError("multi-worker API deployment was accepted")


def test_mutation_requires_api_key(client: TestClient) -> None:
    response = client.post("/api/reviews", json={"patientId": "P001"}, headers={"x-api-key": ""})
    assert response.status_code == 401


def test_reviewer_header_must_match_decision_identity(client: TestClient) -> None:
    response = client.post("/api/reviews/review-1/decisions", json={
        "expectedVersion": 1, "action": "REJECT_FINDING",
        "findingId": "f1", "reviewerId": "different-reviewer",
    })
    assert response.status_code == 403


def test_self_asserted_reviewer_cannot_override_api_key_identity(client: TestClient) -> None:
    response = client.post("/api/reviews/review-1/decisions", headers={
        "x-api-key": "test-secret", "x-reviewer-id": "intruder",
    }, json={
        "expectedVersion": 1, "action": "REJECT_FINDING",
        "findingId": "f1", "reviewerId": "intruder",
    })
    assert response.status_code == 403


def test_decision_before_review_run_returns_conflict(client: TestClient) -> None:
    created = client.post("/api/reviews", json={"patientId": "P001"}).json()
    response = client.post(f"/api/reviews/{created['reviewId']}/decisions", json={
        "expectedVersion": 0, "action": "REJECT_FINDING",
        "findingId": "f1", "reviewerId": "pharmacist-demo",
    })
    assert response.status_code == 409


def test_concurrent_decisions_do_not_let_loser_overwrite_checkpoint(client: TestClient) -> None:
    created = client.post("/api/reviews", json={"patientId": "P001", "asOf": "2026-08-31"}).json()
    review_id = created["reviewId"]
    run = client.post(f"/api/reviews/{review_id}/run").json()
    finding_ids = [item["findingId"] for item in run["findings"][:2]]
    assert finding_ids

    def decide(finding_id: str):
        return client.post(f"/api/reviews/{review_id}/decisions", json={
            "expectedVersion": 1, "action": "REJECT_FINDING",
            "findingId": finding_id, "reviewerId": "pharmacist-demo",
        })

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(decide, [finding_ids[0], finding_ids[-1]]))
    assert sorted(response.status_code for response in responses) == [200, 409]
    winner = next(response.json() for response in responses if response.status_code == 200)
    durable = client.get(f"/api/reviews/{review_id}").json()
    assert durable["humanDecisions"] == winner["humanDecisions"]

    pending = next(item for item in durable["findings"] if item["status"] == "PENDING")
    followup = client.post(f"/api/reviews/{review_id}/decisions", json={
        "expectedVersion": durable["version"], "action": "REJECT_FINDING",
        "findingId": pending["findingId"], "reviewerId": "pharmacist-demo",
    })
    assert followup.status_code == 200
    followup_actions = [
        (item.get("action"), item.get("findingId"))
        for item in followup.json()["humanDecisions"]
    ]
    winner_actions = [
        (item.get("action"), item.get("findingId"))
        for item in winner["humanDecisions"]
    ]
    assert all(action in followup_actions for action in winner_actions)


def test_cancel_requires_observed_version_and_persists_cancelled_enum(client: TestClient) -> None:
    created = client.post("/api/reviews", json={"patientId": "P001"}).json()
    review_id = created["reviewId"]
    assert client.post(f"/api/reviews/{review_id}/cancel", json={}).status_code == 422
    cancelled = client.post(f"/api/reviews/{review_id}/cancel", json={"expectedVersion": 0})
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "CANCELLED"
    assert client.post(f"/api/reviews/{review_id}/cancel", json={"expectedVersion": 0}).status_code == 409


def test_repository_save_failure_restores_checkpoint_for_safe_retry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_API_KEY", "test-secret")
    monkeypatch.setenv("REVIEW_API_REVIEWER_ID", "pharmacist-demo")
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    dependencies = ReviewDependencies(
        health=FakeHealthGateway(health_context(MED1)),
        drug=FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
        repository=repository, planner=DeterministicPlanner(),
    )
    app = create_app(dependencies, checkpoint_path=tmp_path / "checkpoints.sqlite")
    with TestClient(app, raise_server_exceptions=False) as isolated:
        isolated.headers.update({"x-api-key": "test-secret", "x-reviewer-id": "pharmacist-demo"})
        created = isolated.post("/api/reviews", json={"patientId": "P001"}).json()
        review_id = created["reviewId"]
        run = isolated.post(f"/api/reviews/{review_id}/run").json()
        finding_id = run["findings"][0]["findingId"]
        original_save = repository.save
        attempts = 0

        def fail_once(snapshot, *, expected_version):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("injected repository failure")
            return original_save(snapshot, expected_version=expected_version)

        monkeypatch.setattr(repository, "save", fail_once)
        payload = {"expectedVersion": 1, "action": "REJECT_FINDING", "findingId": finding_id, "reviewerId": "pharmacist-demo"}
        assert isolated.post(f"/api/reviews/{review_id}/decisions", json=payload).status_code == 500
        durable = isolated.get(f"/api/reviews/{review_id}").json()
        assert durable["version"] == 1
        retry = isolated.post(f"/api/reviews/{review_id}/decisions", json=payload)
        assert retry.status_code == 200
        assert any(item.get("findingId") == finding_id and item["action"] == "REJECT_FINDING" for item in retry.json()["humanDecisions"])


def test_signed_review_cannot_be_cancelled(client: TestClient) -> None:
    created = client.post("/api/reviews", json={"patientId": "P001"}).json()
    review_id = created["reviewId"]
    current = client.post(f"/api/reviews/{review_id}/run").json()
    decisions = [{"action": "ACCEPT_FINDING", "findingId": item["findingId"]} for item in current["findings"]]
    ready = client.post(f"/api/reviews/{review_id}/decisions", json={"expectedVersion": 1, "action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo", "decisions": decisions}).json()
    signed = client.post(f"/api/reviews/{review_id}/decisions", json={"expectedVersion": ready["version"], "action": "SIGN_OFF", "reviewerId": "pharmacist-demo"}).json()
    assert signed["status"] == "SIGNED_OFF"
    assert client.post(f"/api/reviews/{review_id}/cancel", json={"expectedVersion": signed["version"]}).status_code == 409


def test_app_factory_rejects_multi_worker_environment(monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_API_WORKERS", "2")
    try:
        build_app_from_env()
    except ValueError as exc:
        assert "single worker" in str(exc)
    else:
        raise AssertionError("ASGI factory bypassed single-worker validation")


def test_second_app_process_owner_cannot_share_checkpoint_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_API_KEY", "test-secret")
    monkeypatch.setenv("REVIEW_API_REVIEWER_ID", "pharmacist-demo")
    checkpoint_path = tmp_path / "shared-checkpoints.sqlite"
    dependencies = ReviewDependencies(
        health=FakeHealthGateway(health_context(MED1)),
        drug=FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
        repository=ReviewRepository(tmp_path / "reviews.sqlite"),
        planner=DeterministicPlanner(),
    )
    first = create_app(dependencies, checkpoint_path=checkpoint_path)
    second = create_app(dependencies, checkpoint_path=checkpoint_path)
    with TestClient(first):
        with pytest.raises(RuntimeError, match="single API process"):
            with TestClient(second):
                pass


def test_blocked_review_cannot_restart_in_same_checkpoint(client: TestClient) -> None:
    created = client.post("/api/reviews", json={"patientId": "P001"}).json()
    review_id = created["reviewId"]
    repository = client.app.state.graph  # confirms the app lifecycle is active
    assert repository is not None
    response = client.post(f"/api/reviews/{review_id}/run")
    assert response.status_code == 200
    # CREATED is the only restartable state; forcing any existing review through /run is rejected.
    assert client.post(f"/api/reviews/{review_id}/run").status_code == 409


def test_created_review_with_stale_checkpoint_is_rejected_without_importing_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_API_KEY", "test-secret")
    monkeypatch.setenv("REVIEW_API_REVIEWER_ID", "pharmacist-demo")
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    created = repository.create(patient_ref="P001", review_id="review-stale")
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    dependencies = ReviewDependencies(
        health=FakeHealthGateway(health_context(MED1)),
        drug=FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
        repository=repository, planner=DeterministicPlanner(),
    )
    app = create_app(dependencies, checkpoint_path=checkpoint_path)
    with TestClient(app) as isolated:
        isolated.headers.update({"x-api-key": "test-secret", "x-reviewer-id": "pharmacist-demo"})
        import sqlite3
        isolated.portal.call(app.state.checkpointer.setup)
        with sqlite3.connect(checkpoint_path) as connection:
            connection.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (created.reviewId, "", "stale", "msgpack", b"stale", b"stale"),
            )
        response = isolated.post(f"/api/reviews/{created.reviewId}/run")
        assert response.status_code == 409
        assert repository.get(created.reviewId).status.value == "CREATED"


def test_failed_decision_retry_does_not_duplicate_audit_events(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_API_KEY", "test-secret")
    monkeypatch.setenv("REVIEW_API_REVIEWER_ID", "pharmacist-demo")
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    dependencies = ReviewDependencies(
        health=FakeHealthGateway(health_context(MED1)),
        drug=FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
        repository=repository, planner=DeterministicPlanner(),
    )
    app = create_app(dependencies, checkpoint_path=tmp_path / "checkpoints.sqlite")
    with TestClient(app, raise_server_exceptions=False) as isolated:
        isolated.headers.update({"x-api-key": "test-secret", "x-reviewer-id": "pharmacist-demo"})
        created = isolated.post("/api/reviews", json={"patientId": "P001"}).json()
        review_id = created["reviewId"]
        run = isolated.post(f"/api/reviews/{review_id}/run").json()
        finding_id = run["findings"][0]["findingId"]
        original_save = repository.save
        calls = 0
        def fail_once(snapshot, *, expected_version):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected projection failure")
            return original_save(snapshot, expected_version=expected_version)
        monkeypatch.setattr(repository, "save", fail_once)
        payload = {"expectedVersion": 1, "action": "REJECT_FINDING", "findingId": finding_id, "reviewerId": "pharmacist-demo"}
        assert isolated.post(f"/api/reviews/{review_id}/decisions", json=payload).status_code == 500
        before = isolated.get(f"/api/reviews/{review_id}/audit").json()
        assert isolated.post(f"/api/reviews/{review_id}/decisions", json=payload).status_code == 200
        after = isolated.get(f"/api/reviews/{review_id}/audit").json()
        assert len(after) == len(before)
        assert all(event.get("mutationId") for event in after)


def test_checkpoint_advanced_crash_recovers_after_app_restart(tmp_path: Path, monkeypatch) -> None:
    import base64
    import hashlib
    import json
    import pickle
    from langgraph.types import Command
    from medication_review_agent.workflow import state_to_snapshot

    monkeypatch.setenv("REVIEW_API_KEY", "test-secret")
    monkeypatch.setenv("REVIEW_API_REVIEWER_ID", "pharmacist-demo")
    review_path = tmp_path / "reviews.sqlite"
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    repository = ReviewRepository(review_path)
    dependencies = ReviewDependencies(
        health=FakeHealthGateway(health_context(MED1)),
        drug=FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
        repository=repository, planner=DeterministicPlanner(),
    )
    first = create_app(dependencies, checkpoint_path=checkpoint_path)
    with TestClient(first) as client:
        client.headers.update({"x-api-key": "test-secret", "x-reviewer-id": "pharmacist-demo"})
        created = client.post("/api/reviews", json={"patientId": "P001"}).json()
        review_id = created["reviewId"]
        running = client.post(f"/api/reviews/{review_id}/run").json()
        finding_id = running["findings"][0]["findingId"]
        resume = {
            "action": "COMPLETE_FINDING_REVIEW", "reviewerId": "pharmacist-demo",
            "decisions": [{"action": "REJECT_FINDING", "findingId": finding_id, "note": None}],
        }
        canonical = json.dumps(resume, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        mutation_id = hashlib.sha256(f"{review_id}:1:{fingerprint}".encode()).hexdigest()

        async def advance_without_projection_commit() -> None:
            saver = first.state.checkpointer
            await saver.setup()
            async with saver.lock:
                checkpoints = [tuple(row) for row in await (await saver.conn.execute(
                    "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ?",
                    (review_id,),
                )).fetchall()]
                writes = [tuple(row) for row in await (await saver.conn.execute(
                    "SELECT thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value FROM writes WHERE thread_id = ?",
                    (review_id,),
                )).fetchall()]
            backup = base64.b64encode(pickle.dumps({"checkpoints": checkpoints, "writes": writes}))
            repository.prepare_mutation(
                mutation_id=mutation_id, review_id=review_id, expected_version=1,
                action_fingerprint=fingerprint, checkpoint_backup=backup,
            )
            result = await first.state.graph.ainvoke(
                Command(resume=resume, update={"mutationId": mutation_id}),
                config={"configurable": {"thread_id": review_id}},
            )
            repository.mark_checkpoint_advanced(
                mutation_id, state_to_snapshot(repository.get(review_id), result)
            )

        client.portal.call(advance_without_projection_commit)
        assert repository.get(review_id).version == 1
        assert repository.get_mutation(mutation_id)["state"] == "CHECKPOINT_ADVANCED"

    restarted_repository = ReviewRepository(review_path)
    restarted_dependencies = ReviewDependencies(
        health=FakeHealthGateway(health_context(MED1)),
        drug=FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
        repository=restarted_repository, planner=DeterministicPlanner(),
    )
    restarted = create_app(restarted_dependencies, checkpoint_path=checkpoint_path)
    with TestClient(restarted) as client:
        client.headers.update({"x-api-key": "test-secret", "x-reviewer-id": "pharmacist-demo"})
        response = client.post(f"/api/reviews/{review_id}/decisions", json={
            "expectedVersion": 1, "action": "REJECT_FINDING",
            "findingId": finding_id, "reviewerId": "pharmacist-demo",
        })
        assert response.status_code == 200
        assert response.json()["version"] == 2
        assert restarted_repository.get_mutation(mutation_id)["state"] == "COMMITTED"
        audits = client.get(f"/api/reviews/{review_id}/audit").json()
        assert all(event.get("mutationId") for event in audits)


def test_cancel_recovers_checkpoint_advanced_decision_before_version_check(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_API_KEY", "test-secret")
    monkeypatch.setenv("REVIEW_API_REVIEWER_ID", "pharmacist-demo")
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    review = repository.create(patient_ref="P001", review_id="recover-before-cancel")
    review.status = ReviewStatus.AWAITING_FINDING_REVIEW
    repository.save(review, expected_version=0)
    projected = repository.get(review.reviewId)
    projected.status = ReviewStatus.READY_FOR_SIGN_OFF
    repository.prepare_mutation(
        mutation_id="pending-decision", review_id=review.reviewId, expected_version=1,
        action_fingerprint="f" * 64,
        checkpoint_backup=__import__("base64").b64encode(__import__("pickle").dumps({"checkpoints": [], "writes": []})),
    )
    repository.mark_checkpoint_advanced("pending-decision", projected)
    dependencies = ReviewDependencies(
        health=FakeHealthGateway(health_context(MED1)),
        drug=FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
        repository=repository, planner=DeterministicPlanner(),
    )
    app = create_app(dependencies, checkpoint_path=tmp_path / "checkpoints.sqlite")
    with TestClient(app) as client:
        client.headers.update({"x-api-key": "test-secret", "x-reviewer-id": "pharmacist-demo"})
        response = client.post(f"/api/reviews/{review.reviewId}/cancel", json={"expectedVersion": 1})
        assert response.status_code == 409
        durable = repository.get(review.reviewId)
        assert durable.version == 2
        assert durable.status == ReviewStatus.READY_FOR_SIGN_OFF
        assert repository.get_mutation("pending-decision")["state"] == "COMMITTED"


@pytest.mark.parametrize("journal_state", ["PREPARED", "CHECKPOINT_ADVANCED"])
def test_run_recovers_owned_journal_before_orphan_checkpoint_rejection(tmp_path: Path, monkeypatch, journal_state: str) -> None:
    monkeypatch.setenv("REVIEW_API_KEY", "test-secret")
    monkeypatch.setenv("REVIEW_API_REVIEWER_ID", "pharmacist-demo")
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    review = repository.create(patient_ref="P001", review_id=f"owned-{journal_state.lower()}")
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    payload = {"action": "RUN", "patientRef": "P001", "asOf": None}
    import hashlib, json, base64, pickle
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    mutation_id = hashlib.sha256(f"{review.reviewId}:0:{fingerprint}".encode()).hexdigest()
    repository.prepare_mutation(
        mutation_id=mutation_id, review_id=review.reviewId, expected_version=0,
        action_fingerprint=fingerprint,
        checkpoint_backup=base64.b64encode(pickle.dumps({"checkpoints": [], "writes": []})),
    )
    if journal_state == "CHECKPOINT_ADVANCED":
        projected = review.model_copy(deep=True)
        projected.status = ReviewStatus.AWAITING_FINDING_REVIEW
        repository.mark_checkpoint_advanced(mutation_id, projected)
    dependencies = ReviewDependencies(
        health=FakeHealthGateway(health_context(MED1)),
        drug=FakeDrugGateway(standard_drug_responses({"ARNICA": mapped_response()})),
        repository=repository, planner=DeterministicPlanner(),
    )
    app = create_app(dependencies, checkpoint_path=checkpoint_path)
    with TestClient(app) as client:
        client.headers.update({"x-api-key": "test-secret", "x-reviewer-id": "pharmacist-demo"})
        client.portal.call(app.state.checkpointer.setup)
        import sqlite3
        with sqlite3.connect(checkpoint_path) as connection:
            connection.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (review.reviewId, "", "owned", "msgpack", b"owned", b"owned"),
            )
        response = client.post(f"/api/reviews/{review.reviewId}/run")
        if journal_state == "CHECKPOINT_ADVANCED":
            assert response.status_code == 200
            assert response.json()["version"] == 1
            assert response.json()["status"] == "AWAITING_FINDING_REVIEW"
        else:
            assert response.status_code == 200
            assert response.json()["version"] == 1
