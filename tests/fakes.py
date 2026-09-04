from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from medication_review_agent.gateways import TimedToolResult
from medication_review_agent.models import ToolEnvelope
from medication_review_agent.planner import DeterministicPlanner
from medication_review_agent.repository import ReviewRepository
from medication_review_agent.workflow import (
    ReviewDependencies,
    build_review_graph,
    open_sqlite_checkpointer,
)


def envelope(
    status: str = "OK", data: dict[str, Any] | None = None,
    *, refs: list[str] | None = None, provenance: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> TimedToolResult:
    return TimedToolResult(
        envelope=ToolEnvelope.model_validate({
            "schemaVersion": "1.0", "status": status, "data": data or {},
            "evidenceRefs": refs or [], "warnings": [], "errors": errors or [],
            "provenance": provenance or {}, "requestId": f"req-{status.lower()}",
        }),
        latency_ms=3,
    )


def health_context(*medications: dict[str, Any], missing: list[str] | None = None) -> TimedToolResult:
    return envelope(
        "INSUFFICIENT_EVIDENCE" if missing else "OK",
        {
            "patient": {"id": "p1", "age": 42, "evidenceRef": "FHIR:Patient/p1"},
            "asOf": "2026-08-31", "activeMedications": list(medications),
            "activeConditions": [], "allergies": [], "recentObservations": [],
            "specialPopulations": [], "missingFields": missing or [],
        },
        refs=["FHIR:Patient/p1", *[item["evidenceRef"] for item in medications]],
    )


class FakeHealthGateway:
    def __init__(self, response: TimedToolResult) -> None:
        self.response = response
        self.calls: list[tuple[str | None, str | None]] = []

    async def get_review_context(self, patient_id: str | None, as_of: str | None) -> TimedToolResult:
        self.calls.append((patient_id, as_of))
        return self.response


class FakeDrugGateway:
    def __init__(self, responses: dict[str, list[TimedToolResult] | TimedToolResult]) -> None:
        self.responses = {
            key: deque(value if isinstance(value, list) else [value])
            for key, value in responses.items()
        }
        self.calls: list[tuple[str, Any]] = []

    def _take(self, key: str) -> TimedToolResult:
        queue = self.responses[key]
        return queue[0] if len(queue) == 1 else queue.popleft()

    async def resolve_medication(self, **kwargs: Any) -> TimedToolResult:
        self.calls.append(("resolve_medication", kwargs))
        return self._take(f"resolve:{kwargs['name']}")

    async def get_product_facts(self, product_id: str) -> TimedToolResult:
        self.calls.append(("get_product_facts", product_id))
        return self._take("facts")

    async def search_label_evidence(self, product_ids: list[str], topics: list[str], question: str | None) -> TimedToolResult:
        self.calls.append(("search_label_evidence", product_ids))
        result = self._take("search")
        if result.envelope.status.value != "OK":
            return result
        data = dict(result.envelope.data)
        data["evidence"] = [
            item
            for item in data.get("evidence", [])
            if not item.get("topic") or item.get("topic") in topics
        ]
        return TimedToolResult(
            envelope=result.envelope.model_copy(update={"data": data}),
            latency_ms=result.latency_ms,
        )

    async def compare_product_ingredients(self, product_ids: list[str]) -> TimedToolResult:
        self.calls.append(("compare_product_ingredients", product_ids))
        return self._take("compare")

    async def validate_evidence(self, claims: list[dict[str, Any]]) -> TimedToolResult:
        self.calls.append(("validate_evidence", claims))
        result = self._take("validate")
        if result.envelope.status.value == "OK" and not result.envelope.data.get("claims"):
            return envelope("OK", {"claims": [{**claim, "valid": True, "errors": []} for claim in claims]})
        return result


def mapped_response(*, backend: str = "neo4j", fallback: bool = False, consistency: str = "CONSISTENT") -> TimedToolResult:
    return envelope("OK", {
        "matchClass": "EXACT_IDENTIFIER", "autoAcceptable": True,
        "selectedProductId": "DRUG_PRODUCT::1", "candidates": [], "unmatchedFields": [],
    }, provenance={
        "graphBackend": backend, "graphWorkspace": "dailymed", "graphDatabase": "neo4j",
        "fallbackUsed": fallback, "consistency": {"status": consistency},
    })


def standard_drug_responses(
    resolve: dict[str, TimedToolResult | list[TimedToolResult]],
) -> dict[str, TimedToolResult | list[TimedToolResult]]:
    document = {
        "documentId": "doc-1",
        "documentVersion": "3",
        "effectiveTime": "20260831",
        "sourcePath": "labels/doc-1.xml",
        "contentHash": "a" * 64,
    }
    topics = ["identity", "ingredients", "route", "dosage_form", "warnings"]
    product_ids: set[str] = set()
    for configured in resolve.values():
        for response in configured if isinstance(configured, list) else [configured]:
            data = response.envelope.data
            selected = data.get("selectedProductId")
            if isinstance(selected, str) and selected:
                product_ids.add(selected)
            product_ids.update(
                candidate["productId"]
                for candidate in data.get("candidates", [])
                if isinstance(candidate, dict)
                and isinstance(candidate.get("productId"), str)
                and candidate["productId"]
            )
    product_ids = product_ids or {"DRUG_PRODUCT::1"}
    fact_responses = [
        envelope("OK", {"product": {
            "productId": product_id,
            **document,
        }}, provenance={
            "graphBackend": "neo4j", "graphWorkspace": "dailymed",
            "graphDatabase": "neo4j", "fallbackUsed": False,
            "consistency": {"status": "CONSISTENT"},
        })
        for product_id in sorted(product_ids)
    ]
    search_responses = [
        envelope("OK", {"evidence": [{
            "referenceId": f"S-{topic}",
            "productId": product_id,
            **document,
            "sectionId": topic,
            "sectionCode": "34071-1",
            "topic": topic,
            "content": f"Label evidence for {topic}.",
            "evidenceRef": f"SPL:doc-1#{topic}",
        } for topic in topics]}, refs=[
            f"SPL:doc-1#{topic}" for topic in topics
        ], provenance={
            "graphBackend": "neo4j", "graphWorkspace": "dailymed",
            "graphDatabase": "neo4j", "fallbackUsed": False,
            "consistency": {"status": "CONSISTENT"},
        })
        for product_id in sorted(product_ids)
    ]
    return {
        **{f"resolve:{name}": value for name, value in resolve.items()},
        "facts": fact_responses[0] if len(fact_responses) == 1 else fact_responses,
        "search": search_responses[0] if len(search_responses) == 1 else search_responses,
        "compare": envelope("OK", {"sharedActiveIngredients": []}),
        "validate": envelope("OK", {"claims": []}),
    }


def build_test_graph(
    tmp_path: Path,
    health: FakeHealthGateway,
    drug: FakeDrugGateway,
    *,
    review_id: str = "review-1",
    question: str = "默认用药证据核查",
    planner=None,
    grader=None,
):
    repository = ReviewRepository(tmp_path / "reviews.sqlite")
    try:
        repository.get(review_id)
    except KeyError:
        repository.create(
            patient_ref="P001",
            question=question,
            review_id=review_id,
            as_of="2026-08-31",
        )
    dependencies = ReviewDependencies(
        health=health,
        drug=drug,
        repository=repository,
        planner=planner if planner is not None else DeterministicPlanner(),
        grader=grader,
    )
    checkpointer = open_sqlite_checkpointer(tmp_path / "checkpoints.sqlite")
    return build_review_graph(dependencies, checkpointer)
