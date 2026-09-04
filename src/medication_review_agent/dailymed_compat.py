from __future__ import annotations

import hashlib
import re
from types import ModuleType
from typing import Any


_ENTITY_TYPE_PREDICATE = re.compile(
    r"\b(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\.entity_type\s*=\s*"
    r"'(?P<entity_type>DrugProduct|Ingredient|LabelDocument)'"
)
_DOCUMENT_ID = re.compile(r"^Document ID:\s*(\S+)\s*$", re.MULTILINE)
_DOCUMENT_VERSION = re.compile(r"^Version:\s*(\S+)\s*$", re.MULTILINE)
_EFFECTIVE_TIME = re.compile(
    r"^Effective date:\s*(\d{4}-\d{2}-\d{2}|\d{8})\s*$",
    re.MULTILINE,
)
_SECTION_TITLE = re.compile(r"^Section title:\s*(.+?)\s*$", re.MULTILINE)
_TOPIC_MARKERS = {
    "identity": ("drugproduct", "drug product", "product code", "document id"),
    "ingredients": ("ingredient", "active substance", "成分"),
    "route": (
        "administration route", "administered_via", "sublingual", "oral",
        "topical", "intramuscular", "intravenous",
    ),
    "dosage_form": (
        "dosage form", '"form"', "pellet", "tablet", "capsule", "liquid",
        "cream", "ointment", "spray", "drops",
    ),
    "warnings": ("warning", "caution", "keep out", "警告"),
    "dosage": ("dosage", "directions", "dose", "take ", "apply ", "dissolve"),
    "storage": ("storage", "store ", "储存"),
    "indications": ("indication", "purpose", "used for", "适应症"),
    "pregnancy": ("pregnan", "nursing", "breastfeed", "哺乳"),
    "stop_use": ("stop use", "discontinue", "consult", "ask a doctor"),
    "images": ("image", "media", "principal display panel"),
}


def _unique_match(pattern: re.Pattern[str], values: list[str]) -> str | None:
    matches = {
        match
        for value in values
        for match in pattern.findall(value)
        if match
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _document_metadata(rag_service: object | None) -> dict[str, dict[str, str]]:
    if rag_service is None:
        return {}
    provenance = getattr(rag_service, "provenance", {})
    semantic_graph = getattr(rag_service, "semantic_graph", None)
    chunks = getattr(semantic_graph, "chunks_by_source", {})
    narratives = getattr(rag_service, "narratives", [])
    if not isinstance(provenance, dict) or not isinstance(chunks, dict):
        return {}

    provenance_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rows in provenance.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            document_id = str(row.get("document_id") or "")
            source_id = str(row.get("source_id") or "")
            source_path = str(row.get("file_path") or "")
            if document_id and source_id and source_path:
                provenance_rows[(document_id, source_id, source_path)] = row

    sources: dict[str, list[tuple[str, str]]] = {}
    for (document_id, source_id, source_path), _row in provenance_rows.items():
        chunk = chunks.get(source_id)
        if not isinstance(chunk, dict):
            continue
        content = chunk.get("content")
        if isinstance(content, str) and content.strip():
            sources.setdefault(document_id, []).append((source_path, content))

    narrative_sources: dict[str, list[tuple[str, str]]] = {}
    for narrative in narratives if isinstance(narratives, list) else []:
        if not isinstance(narrative, dict):
            continue
        document_id = str(narrative.get("document_id") or "")
        source_path = str(narrative.get("file_path") or "")
        content = narrative.get("content")
        provenance_confirms_path = any(
            row_document == document_id and row_path == source_path
            for row_document, _source_id, row_path in provenance_rows
        )
        if (
            document_id
            and source_path
            and isinstance(content, str)
            and content.strip()
            and provenance_confirms_path
        ):
            narrative_sources.setdefault(document_id, []).append(
                (source_path, content)
            )

    result: dict[str, dict[str, str]] = {}
    for document_id, structured_sources in sources.items():
        all_sources = [*structured_sources, *narrative_sources.get(document_id, [])]
        contents = [content for _path, content in all_sources]
        parsed_document = _unique_match(_DOCUMENT_ID, contents)
        version = _unique_match(_DOCUMENT_VERSION, contents)
        effective_time = _unique_match(_EFFECTIVE_TIME, contents)
        if parsed_document != document_id or not version or not effective_time:
            continue
        preferred_sources = narrative_sources.get(document_id) or structured_sources
        source_path, content = sorted(preferred_sources, key=lambda item: item[0])[0]
        result[document_id] = {
            "documentId": document_id,
            "documentVersion": version,
            "effectiveTime": effective_time,
            "sourcePath": source_path,
            "contentHash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
    return result


def _section_metadata(
    rag_service: object | None,
    documents: dict[str, dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    if rag_service is None:
        return {}
    provenance = getattr(rag_service, "provenance", {})
    semantic_graph = getattr(rag_service, "semantic_graph", None)
    chunks = getattr(semantic_graph, "chunks_by_source", {})
    narratives = getattr(rag_service, "narratives", [])
    if not isinstance(provenance, dict) or not isinstance(chunks, dict):
        return {}

    narrative_contents: dict[tuple[str, str, str], list[str]] = {}
    for item in narratives if isinstance(narratives, list) else []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        key = (
            str(item.get("document_id") or ""),
            str(item.get("section_id") or ""),
            str(item.get("file_path") or ""),
        )
        narrative_contents.setdefault(key, []).append(content)
    result: dict[tuple[str, str], list[dict[str, str]]] = {}
    for rows in provenance.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            document_id = str(row.get("document_id") or "")
            section_id = str(row.get("section_id") or "")
            source_id = str(row.get("source_id") or "")
            source_path = str(row.get("file_path") or "")
            document = documents.get(document_id)
            if document is None or not section_id or not source_path:
                continue
            chunk = chunks.get(source_id)
            candidates = result.setdefault((document_id, section_id), [])
            contents = [
                chunk.get("content") if isinstance(chunk, dict) else None,
                *narrative_contents.get((document_id, section_id, source_path), []),
            ]
            for content in contents:
                if not isinstance(content, str) or not content.strip():
                    continue
                metadata = {
                    **document,
                    "sourcePath": source_path,
                    "contentHash": hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    "_content": content,
                }
                if metadata not in candidates:
                    candidates.append(metadata)
    return result


def _metadata_for_evidence(
    item: dict[str, Any],
    sections: dict[tuple[str, str], list[dict[str, str]]],
) -> dict[str, str] | None:
    candidates = sections.get((
        str(item.get("documentId") or ""),
        str(item.get("sectionId") or ""),
    ), [])
    source_path = str(item.get("sourcePath") or "")
    if source_path:
        candidates = [
            candidate for candidate in candidates
            if candidate["sourcePath"] == source_path
        ]
    content = item.get("content")
    if isinstance(content, str) and content:
        candidates = [
            candidate for candidate in candidates
            if candidate.get("_content") == content
        ]
    if len(candidates) != 1:
        return None
    return {
        key: value for key, value in candidates[0].items()
        if not key.startswith("_")
    }


def _matching_topics(item: dict[str, Any], requested: list[str]) -> list[str]:
    searchable = "\n".join(
        str(item.get(field) or "")
        for field in ("sectionTitle", "sectionId", "content")
    ).casefold()
    return [
        topic
        for topic in requested
        if any(marker in searchable for marker in _TOPIC_MARKERS.get(topic, ()))
    ]


def _confirmed_local_narratives(
    rag_service: object | None,
    allowed_document_ids: set[str],
    requested_topics: list[str],
) -> list[dict[str, Any]]:
    if rag_service is None:
        return []
    provenance = getattr(rag_service, "provenance", {})
    narratives = getattr(rag_service, "narratives", [])
    if not isinstance(provenance, dict) or not isinstance(narratives, list):
        return []
    confirmed_paths = {
        (str(row.get("document_id") or ""), str(row.get("file_path") or ""))
        for rows in provenance.values()
        if isinstance(rows, list)
        for row in rows
        if isinstance(row, dict)
    }
    result: list[dict[str, Any]] = []
    for narrative in narratives:
        if not isinstance(narrative, dict):
            continue
        document_id = str(narrative.get("document_id") or "")
        source_path = str(narrative.get("file_path") or "")
        section_id = str(narrative.get("section_id") or "")
        content = narrative.get("content")
        if (
            document_id not in allowed_document_ids
            or (document_id, source_path) not in confirmed_paths
            or not section_id
            or not isinstance(content, str)
            or not content.strip()
        ):
            continue
        section_title_match = _SECTION_TITLE.search(content)
        section_title = (
            section_title_match.group(1)
            if section_title_match and section_title_match.group(1) != "N/A"
            else None
        )
        item = {
            "referenceId": f"COMPAT-{section_id}",
            "documentId": document_id,
            "sectionId": section_id,
            "sectionTitle": section_title,
            "content": content,
            "evidenceRef": f"SPL:{document_id}#{section_id}",
        }
        for topic in _matching_topics(item, requested_topics):
            result.append({**item, "topic": topic})
    return result


def build_compatible_evidence_service(base_service: type) -> type:
    if getattr(base_service, "_medication_review_compatible", False):
        return base_service

    class CompatibleDrugEvidenceService(base_service):
        _medication_review_compatible = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._rag_service = getattr(self, "rag_service", None)
            self._document_metadata = _document_metadata(
                self._rag_service
            )
            self._section_metadata = _section_metadata(
                self._rag_service, self._document_metadata
            )

        async def get_product_facts(self, product_id: str) -> dict[str, Any]:
            result = await super().get_product_facts(product_id)
            product = (result.get("data") or {}).get("product")
            if not isinstance(product, dict):
                return result
            documents: list[dict[str, str]] = []
            for raw_document_id in product.get("documentIds") or []:
                document_id = str(raw_document_id)
                section_documents = [
                    metadata
                    for (candidate_document_id, _section_id), candidates
                    in self._section_metadata.items()
                    if candidate_document_id == document_id
                    for metadata in candidates
                ]
                if section_documents:
                    candidates = section_documents
                elif document_id in self._document_metadata:
                    candidates = [self._document_metadata[document_id]]
                else:
                    candidates = []
                for metadata in candidates:
                    metadata = {
                        key: value for key, value in metadata.items()
                        if not key.startswith("_")
                    }
                    if metadata not in documents:
                        documents.append(metadata)
            if not documents:
                result.setdefault("warnings", []).append(
                    "Indexed document provenance is unavailable; narrative retrieval "
                    "remains disabled for this product."
                )
                return result
            product["documents"] = documents
            if len(documents) == 1:
                product.update(documents[0])
            return result

        async def search_label_evidence(
            self,
            product_ids: list[str],
            topics: list[str],
            question: str | None = None,
        ) -> dict[str, Any]:
            result = await super().search_label_evidence(
                product_ids, topics, question
            )
            data = result.get("data") or {}
            evidence = data.get("evidence")
            if not isinstance(evidence, list):
                return result
            documents_by_product = {
                product_id: await self.graph.document_ids_for_products({product_id})
                for product_id in dict.fromkeys(product_ids)
            }
            products_by_document = {
                document_id: [
                    product_id for product_id in product_ids
                    if document_id in documents_by_product[product_id]
                ]
                for document_id in {
                    document_id
                    for document_ids in documents_by_product.values()
                    for document_id in document_ids
                }
            }
            allowed = set(products_by_document)
            enriched: list[dict[str, Any]] = []
            warnings = result.setdefault("warnings", [])
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                document_id = str(item.get("documentId") or "")
                if document_id not in allowed:
                    warnings.append(
                        "Dropped out-of-scope label reference during provenance "
                        f"enrichment: {item.get('referenceId')}"
                    )
                    continue
                metadata = _metadata_for_evidence(item, self._section_metadata)
                if metadata is None:
                    warnings.append(
                        "Dropped label reference without confirmed section provenance: "
                        f"{item.get('referenceId')}"
                    )
                    continue
                for topic in _matching_topics(item, topics):
                    enriched.append({
                        **item,
                        **metadata,
                        "productIds": products_by_document[document_id],
                        "topic": topic,
                    })
            covered_topics = {item["topic"] for item in enriched}
            missing_topics = [
                topic for topic in topics if topic not in covered_topics
            ]
            for item in _confirmed_local_narratives(
                self._rag_service, set(allowed), missing_topics
            ):
                metadata = _metadata_for_evidence(item, self._section_metadata)
                if metadata is None:
                    continue
                enriched.append({
                    **item,
                    **metadata,
                    "productIds": products_by_document[item["documentId"]],
                })
            data["evidence"] = enriched
            result["data"] = data
            result["evidenceRefs"] = list(dict.fromkeys(
                item["evidenceRef"] for item in enriched
            ))
            if enriched:
                result["status"] = "OK"
            else:
                result["status"] = "INSUFFICIENT_EVIDENCE"
                result.setdefault("errors", []).append(
                    "No in-scope label evidence with complete indexed provenance "
                    "matched the requested topics."
                )
            return result

    CompatibleDrugEvidenceService.__name__ = "CompatibleDrugEvidenceService"
    return CompatibleDrugEvidenceService


def normalize_drug_product_predicates(query: str) -> str:
    def normalize(match: re.Match[str]) -> str:
        variable = match.group("variable")
        entity_type = match.group("entity_type").casefold()
        predicate = f"toLower({variable}.entity_type) = '{entity_type}'"
        if entity_type != "drugproduct":
            return predicate
        return (
            f"({predicate} AND ("
            f"{variable}.entity_id STARTS WITH 'DRUG_PRODUCT::' OR "
            f"{variable}.entity_id STARTS WITH 'DRUG_PRODUCT_OCCURRENCE::'))"
        )

    return _ENTITY_TYPE_PREDICATE.sub(normalize, query)


def safe_graph_display(description: object, fallback: str) -> str:
    lines = str(description or fallback).splitlines()
    if not lines:
        return fallback
    return lines[0].split(":", 1)[-1].strip() or fallback


def install_dailymed_compatibility() -> ModuleType:
    from dailymed_lightrag import drug_graph, drug_mcp_server

    base_repository = drug_graph.Neo4jDrugGraphRepository

    class CompatibleNeo4jDrugGraphRepository(base_repository):
        async def _run(self, query: str, **parameters):
            return await super()._run(
                normalize_drug_product_predicates(query), **parameters
            )

    CompatibleNeo4jDrugGraphRepository.__name__ = (
        "CompatibleNeo4jDrugGraphRepository"
    )
    drug_graph._display = safe_graph_display
    drug_mcp_server.Neo4jDrugGraphRepository = CompatibleNeo4jDrugGraphRepository
    drug_mcp_server.DrugEvidenceService = build_compatible_evidence_service(
        drug_mcp_server.DrugEvidenceService
    )
    return drug_mcp_server


def main() -> None:
    install_dailymed_compatibility().main()


if __name__ == "__main__":
    main()
