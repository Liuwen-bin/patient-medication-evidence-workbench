from collections.abc import Callable
from pathlib import Path
import socket
import threading
import time
from urllib.request import urlopen

import pytest
import uvicorn
from fastapi.testclient import TestClient

from medication_review_agent.api import create_app
from medication_review_agent.models import (
    EvidenceItem,
    Finding,
    GraphEvidenceProvenance,
    MedicationMapping,
    MedicationRecord,
    ReviewStatus,
)
from medication_review_agent.planner import DeterministicPlanner
from medication_review_agent.repository import ReviewRepository
from medication_review_agent.workflow import ReviewDependencies
from tests.fakes import FakeDrugGateway, FakeHealthGateway, envelope, health_context, mapped_response, standard_drug_responses
from tests.test_workflow import MED1, MED2


_CHROMIUM_BLOCKED_PORTS = frozenset({
    1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 69,
    77, 79, 87, 95, 101, 102, 103, 104, 109, 110, 111, 113, 115, 117, 119,
    123, 135, 137, 139, 143, 161, 179, 389, 427, 465, 512, 513, 514, 515,
    526, 530, 531, 532, 540, 548, 554, 556, 563, 587, 601, 636, 989, 990,
    993, 995, 1719, 1720, 1723, 2049, 3659, 4045, 4190, 5060, 5061, 6000,
    6566, 6665, 6666, 6667, 6668, 6669, 6697, 10080,
})


def _allocate_ephemeral_port() -> int:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return int(reservation.getsockname()[1])


def _select_browser_safe_port(
    allocate: Callable[[], int] = _allocate_ephemeral_port,
) -> int:
    for _ in range(32):
        port = allocate()
        if port not in _CHROMIUM_BLOCKED_PORTS:
            return port
    raise RuntimeError("could not allocate a browser-safe test server port")


def _make_app(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REVIEW_API_KEY", "test-secret")
    monkeypatch.setenv("REVIEW_API_REVIEWER_ID", "pharmacist-demo")
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    stale = repository.create(
        patient_ref="P001",
        question="默认用药证据核查",
        review_id="review-1",
        as_of="2026-08-31",
    )
    stale.status = ReviewStatus.RUNNING
    graph = GraphEvidenceProvenance.model_validate({
        "graphBackend": "neo4j",
        "graphWorkspace": "dm1000_xml_v1",
        "graphDatabase": "neo4j",
        "fallbackUsed": False,
        "consistency": {"status": "CONSISTENT"},
    })
    stale.contextSnapshot = {
        "patient": {"id": "p1", "age": 42, "evidenceRef": "FHIR:Patient/p1"},
        "activeConditions": [],
        "allergies": [],
        "recentObservations": [],
    }
    stale.contextMissingFields = ["pregnancyStatus"]
    stale.medications = [MedicationRecord.model_validate({
        "medicationId": "med-1",
        "name": "Arnica montana",
        "dosage": "每日一次",
        "route": "口服",
        "patientEvidenceRefs": ["FHIR:MedicationRequest/med-1"],
    })]
    stale.medicationMappings = [MedicationMapping.model_validate({
        "medicationId": "med-1",
        "sourceName": "Arnica montana",
        "matchClass": "EXACT_IDENTIFIER",
        "selectedProductId": "DRUG_PRODUCT::1",
        "graphProvenance": graph,
    })]
    stale.findings = [Finding.model_validate({
        "findingId": "finding-1",
        "reviewType": "LABEL_PRECAUTION",
        "ruleId": "label-precaution-v1",
        "summary": "核对标签注意事项",
        "attentionLevel": "MEDIUM",
        "confidence": 0.92,
        "medicationIds": ["med-1"],
        "patientEvidenceRefs": ["FHIR:MedicationRequest/med-1"],
        "labelEvidenceRefs": ["SPL:doc-1#warnings"],
        "requiresHumanReview": True,
        "graphProvenance": graph,
    })]
    stale.evidenceIndex = [EvidenceItem.model_validate({
        "evidenceId": "evidence-1",
        "source": "SPL",
        "evidenceRef": "SPL:doc-1#warnings",
        "medicationIds": ["med-1"],
        "productIds": ["DRUG_PRODUCT::1"],
        "topic": "warnings",
        "summary": "Label warning",
        "graphProvenance": graph,
    })]
    repository.save(stale, expected_version=0)
    health = FakeHealthGateway(health_context(MED1, MED2))
    drug = FakeDrugGateway(standard_drug_responses({
        "ARNICA": mapped_response(),
        "METFORMIN": envelope("UNMAPPED", {"matchClass": "UNMAPPED", "selectedProductId": None, "candidates": [], "unmatchedFields": []}),
    }))
    dependencies = ReviewDependencies(health=health, drug=drug, repository=repository, planner=DeterministicPlanner())
    return create_app(dependencies, checkpoint_path=tmp_path / "checkpoints.sqlite")


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as test_client:
        test_client.headers.update({"x-api-key": "test-secret", "x-reviewer-id": "pharmacist-demo"})
        yield test_client


@pytest.fixture
def live_server_url(tmp_path: Path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    port = _select_browser_safe_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            with urlopen(f"{root}/api/health", timeout=.2) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail("workbench test server did not start")
    yield root
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    return {
        **browser_context_args,
        "extra_http_headers": {
            **browser_context_args.get("extra_http_headers", {}),
            "x-api-key": "test-secret",
            "x-reviewer-id": "pharmacist-demo",
        },
    }
