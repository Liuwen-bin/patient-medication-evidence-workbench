from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import ValidationError

from .models import ToolEnvelope


class ToolContractError(RuntimeError):
    """The MCP response could not be validated against the frozen contract."""


class MCPToolCaller(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TimedToolResult:
    envelope: ToolEnvelope
    latency_ms: int


class StreamableHTTPMCPCaller:
    def __init__(self, url: str, *, allow_remote: bool = False) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("MCP URL must be an absolute HTTP(S) URL")
        local_hosts = {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme == "http" and parsed.hostname not in local_hosts and not allow_remote:
            raise ValueError("Remote plain HTTP MCP requires ALLOW_REMOTE_MCP=true")
        self.url = url

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with streamablehttp_client(self.url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            return structured
        payloads: list[dict[str, Any]] = []
        for block in result.content:
            text = getattr(block, "text", None)
            if not isinstance(text, str):
                continue
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ToolContractError("MCP returned non-JSON text content") from exc
            if not isinstance(decoded, dict):
                raise ToolContractError("MCP JSON payload must be an object")
            payloads.append(decoded)
        if not payloads:
            raise ToolContractError("MCP returned no JSON payload")
        if any(item != payloads[0] for item in payloads[1:]):
            raise ToolContractError("MCP returned conflicting JSON payloads")
        return payloads[0]


class _Gateway:
    def __init__(self, caller: MCPToolCaller) -> None:
        self.caller = caller

    async def _call(self, name: str, arguments: dict[str, Any]) -> TimedToolResult:
        started = time.perf_counter()
        try:
            payload = await self.caller.call_tool(name, arguments)
            envelope = ToolEnvelope.model_validate(payload)
        except (ValidationError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ToolContractError(f"Invalid {name} response contract") from exc
        latency = max(0, round((time.perf_counter() - started) * 1000))
        return TimedToolResult(envelope=envelope, latency_ms=latency)


def _allow_remote() -> bool:
    return os.getenv("ALLOW_REMOTE_MCP", "false").lower() in {"1", "true", "yes", "on"}


class HealthRecordGateway(_Gateway):
    @classmethod
    def from_env(cls) -> "HealthRecordGateway":
        url = os.getenv("HEALTH_MCP_URL")
        if not url:
            raise RuntimeError("HEALTH_MCP_URL is required")
        return cls(StreamableHTTPMCPCaller(url, allow_remote=_allow_remote()))

    async def get_review_context(self, patient_id: str | None, as_of: str | None) -> TimedToolResult:
        return await self._call(
            "get_medication_review_context", {"patientId": patient_id, "asOf": as_of}
        )


class DrugEvidenceGateway(_Gateway):
    @classmethod
    def from_env(cls) -> "DrugEvidenceGateway":
        url = os.getenv("DRUG_MCP_URL")
        if not url:
            raise RuntimeError("DRUG_MCP_URL is required")
        return cls(StreamableHTTPMCPCaller(url, allow_remote=_allow_remote()))

    async def resolve_medication(
        self, *, name: str, identifiers: list[dict[str, str]], strength: str | None,
        dosage_form: str | None, route: str | None,
    ) -> TimedToolResult:
        return await self._call("resolve_medication", {
            "name": name, "identifiers": identifiers, "strength": strength,
            "dosageForm": dosage_form, "route": route,
        })

    async def get_product_facts(self, product_id: str) -> TimedToolResult:
        return await self._call("get_product_facts", {"productId": product_id})

    async def search_label_evidence(
        self, product_ids: list[str], topics: list[str], question: str | None,
    ) -> TimedToolResult:
        return await self._call("search_label_evidence", {
            "productIds": product_ids, "topics": topics, "question": question,
        })

    async def compare_product_ingredients(self, product_ids: list[str]) -> TimedToolResult:
        return await self._call("compare_product_ingredients", {"productIds": product_ids})

    async def validate_evidence(self, claims: list[dict[str, Any]]) -> TimedToolResult:
        return await self._call("validate_evidence", {"claims": claims})
