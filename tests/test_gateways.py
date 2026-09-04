import pytest

from medication_review_agent.gateways import (
    DrugEvidenceGateway,
    HealthRecordGateway,
    StreamableHTTPMCPCaller,
    ToolContractError,
)


class FakeCaller:
    def __init__(self, responses: dict[str, dict]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return self.responses[name]


def envelope(status: str = "OK", data: dict | None = None, provenance: dict | None = None) -> dict:
    return {
        "schemaVersion": "1.0",
        "status": status,
        "data": data or {},
        "evidenceRefs": [],
        "warnings": [],
        "errors": [],
        "provenance": provenance or {},
        "requestId": "r1",
    }


@pytest.mark.asyncio
async def test_health_gateway_uses_structured_context_tool() -> None:
    caller = FakeCaller({"get_medication_review_context": envelope()})
    gateway = HealthRecordGateway(caller)
    await gateway.get_review_context("P001", "2026-08-31")
    assert caller.calls == [(
        "get_medication_review_context",
        {"patientId": "P001", "asOf": "2026-08-31"},
    )]


@pytest.mark.asyncio
async def test_health_gateway_calls_preview_with_payload() -> None:
    caller = FakeCaller({
        "validate_medication_review_writeback": envelope(data={"jobId": "job-1"})
    })
    gateway = HealthRecordGateway(caller)

    await gateway.validate_writeback({"schemaVersion": "1.0", "reviewId": "review-1"})

    assert caller.calls == [("validate_medication_review_writeback", {
        "payload": {"schemaVersion": "1.0", "reviewId": "review-1"}
    })]


@pytest.mark.asyncio
async def test_health_gateway_commit_preserves_confirmation_fields() -> None:
    caller = FakeCaller({"commit_medication_review_writeback": envelope()})
    gateway = HealthRecordGateway(caller)

    await gateway.commit_writeback("job-1", "a" * 64, 7, True)

    assert caller.calls == [("commit_medication_review_writeback", {
        "jobId": "job-1",
        "bundleHash": "a" * 64,
        "expectedVersion": 7,
        "confirmed": True,
    })]


@pytest.mark.asyncio
async def test_drug_gateway_preserves_unmapped_status() -> None:
    caller = FakeCaller({
        "resolve_medication": envelope("UNMAPPED", {"selectedProductId": None})
    })
    result = await DrugEvidenceGateway(caller).resolve_medication(
        name="METFORMIN", identifiers=[], strength=None, dosage_form=None, route=None
    )
    assert result.envelope.status.value == "UNMAPPED"


@pytest.mark.asyncio
async def test_drug_gateway_preserves_graph_provenance() -> None:
    caller = FakeCaller({"get_product_facts": envelope(provenance={
        "graphBackend": "snapshot", "fallbackUsed": True,
        "consistency": {"status": "UNAVAILABLE"},
    })})
    result = await DrugEvidenceGateway(caller).get_product_facts("DRUG_PRODUCT::1")
    assert result.envelope.graph_provenance.fallbackUsed is True


@pytest.mark.asyncio
async def test_gateway_rejects_wrong_schema_without_erasing_payload_status() -> None:
    payload = envelope()
    payload["schemaVersion"] = "2.0"
    caller = FakeCaller({"get_medication_review_context": payload})
    with pytest.raises(ToolContractError):
        await HealthRecordGateway(caller).get_review_context("P001", None)


def test_streamable_http_rejects_remote_plain_http_by_default() -> None:
    with pytest.raises(ValueError, match="Remote plain HTTP"):
        StreamableHTTPMCPCaller("http://records.example/mcp", allow_remote=False)


def test_streamable_http_allows_loopback_plain_http() -> None:
    caller = StreamableHTTPMCPCaller("http://127.0.0.1:8000/mcp", allow_remote=False)
    assert caller.url.endswith("/mcp")
