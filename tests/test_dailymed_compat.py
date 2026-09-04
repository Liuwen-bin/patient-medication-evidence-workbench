import hashlib

import pytest

from medication_review_agent.dailymed_compat import (
    build_compatible_evidence_service,
    normalize_drug_product_predicates,
    safe_graph_display,
)


DOCUMENT_ID = "doc-1"
PRODUCT_ID = "DRUG_PRODUCT::10191-1246"
SOURCE_PATH = "spl-section__doc-1__section-a.txt"
SOURCE_CONTENT = "\n".join((
    "FDA SPL structured-fact provenance",
    "Document ID: doc-1",
    "Version: 3",
    "Effective date: 2026-08-31",
    "Section: WARNINGS SECTION",
    "Warnings: Stop use and ask a doctor if symptoms persist.",
))
SOURCE_HASH = "eb90d11426fc0b1338d1c538a342d2862805f230deae475a404674b3688a5b42"


class FakeSemanticGraph:
    def __init__(self, content: str = SOURCE_CONTENT) -> None:
        self.chunks_by_source = {
            "section::doc-1::section-a": {
                "source_id": "section::doc-1::section-a",
                "file_path": SOURCE_PATH,
                "content": content,
            },
        }


class FakeRagService:
    def __init__(self, content: str = SOURCE_CONTENT) -> None:
        self.semantic_graph = FakeSemanticGraph(content)
        self.narratives = [{
            "id": "spl-narrative::doc-1::section-a",
            "file_path": SOURCE_PATH,
            "document_id": DOCUMENT_ID,
            "section_id": "section-a",
            "content": content,
        }]
        self.provenance = {
            SOURCE_PATH: [{
                "source_id": "section::doc-1::section-a",
                "source_type": "section",
                "file_path": SOURCE_PATH,
                "document_id": DOCUMENT_ID,
                "section_id": "section-a",
            }],
        }


class FakeGraph:
    async def document_ids_for_products(self, product_ids: set[str]) -> set[str]:
        return {DOCUMENT_ID} if product_ids == {PRODUCT_ID} else set()


class FakeBaseEvidenceService:
    def __init__(self, graph: object, rag_service: object | None = None) -> None:
        self.graph = graph
        self.rag_service = rag_service

    async def get_product_facts(self, product_id: str) -> dict:
        return {
            "status": "OK",
            "data": {"product": {
                "productId": product_id,
                "documentIds": [DOCUMENT_ID],
            }},
            "evidenceRefs": [f"SPL:{DOCUMENT_ID}#document"],
            "warnings": [],
            "errors": [],
        }

    async def search_label_evidence(
        self,
        product_ids: list[str],
        topics: list[str],
        question: str | None = None,
    ) -> dict:
        del question
        return {
            "status": "OK",
            "data": {"evidence": [{
                "referenceId": "S1",
                "documentId": DOCUMENT_ID,
                "sectionId": "section-a",
                "sectionTitle": "WARNINGS",
                "content": SOURCE_CONTENT,
                "evidenceRef": f"SPL:{DOCUMENT_ID}#section-a",
            }]},
            "evidenceRefs": [f"SPL:{DOCUMENT_ID}#section-a"],
            "warnings": [],
            "errors": [],
        }


def compatible_service(rag_service: FakeRagService | None = None):
    service_class = build_compatible_evidence_service(FakeBaseEvidenceService)
    return service_class(FakeGraph(), rag_service or FakeRagService())


@pytest.mark.parametrize("variable", ["p", "product"])
def test_drug_product_query_accepts_normalized_neo4j_entity_type(
    variable: str,
) -> None:
    query = (
        f"MATCH ({variable}) WHERE {variable}.entity_type = 'DrugProduct' "
        "AND ingredient.entity_type = 'Ingredient' RETURN product"
    )

    normalized = normalize_drug_product_predicates(query)

    assert f"toLower({variable}.entity_type) = 'drugproduct'" in normalized
    assert f"{variable}.entity_id STARTS WITH 'DRUG_PRODUCT::'" in normalized
    assert (
        f"{variable}.entity_id STARTS WITH 'DRUG_PRODUCT_OCCURRENCE::'"
        in normalized
    )
    assert "toLower(ingredient.entity_type) = 'ingredient'" in normalized


def test_label_document_query_accepts_normalized_neo4j_entity_type() -> None:
    query = (
        "MATCH (product)-[]-(document) "
        "WHERE document.entity_type = 'LabelDocument' RETURN document"
    )

    normalized = normalize_drug_product_predicates(query)

    assert "toLower(document.entity_type) = 'labeldocument'" in normalized


def test_empty_optional_graph_relation_has_a_safe_display() -> None:
    assert safe_graph_display(None, "") == ""
    assert safe_graph_display("DrugProduct: ARNICA MONTANA\nDetails", "fallback") == (
        "ARNICA MONTANA"
    )


@pytest.mark.asyncio
async def test_product_facts_expose_complete_indexed_document_metadata() -> None:
    result = await compatible_service().get_product_facts(PRODUCT_ID)

    product = result["data"]["product"]
    assert product["documents"] == [{
        "documentId": DOCUMENT_ID,
        "documentVersion": "3",
        "effectiveTime": "2026-08-31",
        "sourcePath": SOURCE_PATH,
        "contentHash": SOURCE_HASH,
    }]
    assert product["documentId"] == DOCUMENT_ID
    assert product["documentVersion"] == "3"
    assert product["effectiveTime"] == "2026-08-31"
    assert product["sourcePath"] == SOURCE_PATH
    assert product["contentHash"] == SOURCE_HASH


@pytest.mark.asyncio
async def test_search_evidence_uses_the_confirmed_document_identity() -> None:
    service = compatible_service()
    facts = await service.get_product_facts(PRODUCT_ID)

    result = await service.search_label_evidence(
        [PRODUCT_ID], ["warnings", "stop_use"], "check warnings"
    )

    fact_document = facts["data"]["product"]["documents"][0]
    evidence = result["data"]["evidence"]
    assert {item["topic"] for item in evidence} == {"warnings", "stop_use"}
    assert all(item["productIds"] == [PRODUCT_ID] for item in evidence)
    assert all(item["documentVersion"] == fact_document["documentVersion"] for item in evidence)
    assert all(item["contentHash"] == fact_document["contentHash"] for item in evidence)
    assert all(item["effectiveTime"] == "2026-08-31" for item in evidence)
    assert all(item["sourcePath"] == SOURCE_PATH for item in evidence)


@pytest.mark.asyncio
async def test_search_evidence_uses_its_own_section_path_and_hash() -> None:
    section_b_path = "spl-section__doc-1__section-b.txt"
    section_b_content = SOURCE_CONTENT.replace(
        "Section: WARNINGS SECTION",
        "Section: DOSAGE SECTION",
    ).replace(
        "Warnings: Stop use and ask a doctor if symptoms persist.",
        "Dosage: Dissolve five pellets under the tongue.",
    )
    section_b_structured_content = section_b_content.replace(
        "Dosage: Dissolve five pellets under the tongue.",
        '"dosage form": "PELLET"',
    )
    rag_service = FakeRagService()
    rag_service.semantic_graph.chunks_by_source["section::doc-1::section-b"] = {
        "source_id": "section::doc-1::section-b",
        "file_path": section_b_path,
        "content": section_b_structured_content,
    }
    rag_service.narratives.append({
        "id": "spl-narrative::doc-1::section-b",
        "file_path": section_b_path,
        "document_id": DOCUMENT_ID,
        "section_id": "section-b",
        "content": section_b_content,
    })
    rag_service.provenance[section_b_path] = [{
        "source_id": "section::doc-1::section-b",
        "source_type": "section",
        "file_path": section_b_path,
        "document_id": DOCUMENT_ID,
        "section_id": "section-b",
    }]

    class SectionBBase(FakeBaseEvidenceService):
        async def search_label_evidence(self, product_ids, topics, question=None):
            del product_ids, topics, question
            return {
                "status": "OK",
                "data": {"evidence": [{
                    "referenceId": "S2",
                    "documentId": DOCUMENT_ID,
                    "sectionId": "section-b",
                    "sectionTitle": "DOSAGE",
                    "content": section_b_content,
                    "evidenceRef": f"SPL:{DOCUMENT_ID}#section-b",
                }]},
                "evidenceRefs": [f"SPL:{DOCUMENT_ID}#section-b"],
                "warnings": [],
                "errors": [],
            }

    service_class = build_compatible_evidence_service(SectionBBase)
    service = service_class(FakeGraph(), rag_service)

    facts = await service.get_product_facts(PRODUCT_ID)
    result = await service.search_label_evidence([PRODUCT_ID], ["dosage"])

    evidence = result["data"]["evidence"][0]
    assert evidence["sourcePath"] == section_b_path
    assert evidence["contentHash"] == hashlib.sha256(
        section_b_content.encode("utf-8")
    ).hexdigest()
    assert any(
        document == {
            key: evidence[key]
            for key in (
                "documentId",
                "documentVersion",
                "effectiveTime",
                "sourcePath",
                "contentHash",
            )
        }
        for document in facts["data"]["product"]["documents"]
    )


@pytest.mark.asyncio
async def test_multi_product_search_reports_only_products_linked_to_document() -> None:
    second_product = "DRUG_PRODUCT::second"

    class ProductScopedGraph:
        async def document_ids_for_products(self, product_ids: set[str]) -> set[str]:
            return {DOCUMENT_ID} if PRODUCT_ID in product_ids else set()

    service_class = build_compatible_evidence_service(FakeBaseEvidenceService)
    service = service_class(ProductScopedGraph(), FakeRagService())

    result = await service.search_label_evidence(
        [PRODUCT_ID, second_product], ["warnings"]
    )

    assert result["data"]["evidence"][0]["productIds"] == [PRODUCT_ID]


@pytest.mark.asyncio
async def test_search_drops_content_that_matches_no_confirmed_source() -> None:
    class UnmatchedContentBase(FakeBaseEvidenceService):
        async def search_label_evidence(self, product_ids, topics, question=None):
            result = await super().search_label_evidence(
                product_ids, topics, question
            )
            result["data"]["evidence"][0]["content"] = (
                "Unconfirmed warning text from another source."
            )
            return result

    service_class = build_compatible_evidence_service(UnmatchedContentBase)
    service = service_class(FakeGraph(), FakeRagService())

    result = await service.search_label_evidence([PRODUCT_ID], ["warnings"])

    assert all(
        item["referenceId"] != "S1" for item in result["data"]["evidence"]
    )
    assert all(
        item["content"] != "Unconfirmed warning text from another source."
        for item in result["data"]["evidence"]
    )
    assert any("section provenance" in warning for warning in result["warnings"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rag_service",
    [
        FakeRagService(SOURCE_CONTENT.replace("Version: 3", "Version: 3\nVersion: 4")),
        FakeRagService(),
    ],
    ids=["conflicting-version", "missing-provenance"],
)
async def test_unconfirmed_document_metadata_is_not_admitted(
    rag_service: FakeRagService,
) -> None:
    if rag_service.narratives[0]["content"] == SOURCE_CONTENT:
        rag_service.provenance = {}

    service = compatible_service(rag_service)
    facts = await service.get_product_facts(PRODUCT_ID)
    search = await service.search_label_evidence(
        [PRODUCT_ID], ["warnings"], "check warnings"
    )

    product = facts["data"]["product"]
    assert "documents" not in product
    assert "documentVersion" not in product
    assert search["status"] == "INSUFFICIENT_EVIDENCE"
    assert search["data"]["evidence"] == []


@pytest.mark.asyncio
async def test_compatibility_layer_does_not_restore_out_of_scope_evidence() -> None:
    class FakeFilteringBase(FakeBaseEvidenceService):
        async def search_label_evidence(
            self,
            product_ids: list[str],
            topics: list[str],
            question: str | None = None,
        ) -> dict:
            result = await super().search_label_evidence(
                product_ids, topics, question
            )
            result["warnings"].append(
                "Dropped out-of-scope label reference: S2"
            )
            return result

    service_class = build_compatible_evidence_service(FakeFilteringBase)
    service = service_class(FakeGraph(), FakeRagService())
    result = await service.search_label_evidence(
        [PRODUCT_ID], ["warnings"], "check warnings"
    )

    assert {item["documentId"] for item in result["data"]["evidence"]} == {
        DOCUMENT_ID
    }
    assert result["warnings"] == ["Dropped out-of-scope label reference: S2"]


@pytest.mark.asyncio
async def test_confirmed_local_narrative_fills_a_vector_search_gap() -> None:
    class FakeEmptySearchBase(FakeBaseEvidenceService):
        async def search_label_evidence(
            self,
            product_ids: list[str],
            topics: list[str],
            question: str | None = None,
        ) -> dict:
            del product_ids, topics, question
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "data": {"evidence": []},
                "evidenceRefs": [],
                "warnings": ["Vector retrieval unavailable."],
                "errors": ["No vector result."],
            }

    service_class = build_compatible_evidence_service(FakeEmptySearchBase)
    service = service_class(FakeGraph(), FakeRagService())

    result = await service.search_label_evidence(
        [PRODUCT_ID], ["warnings"], "check warnings"
    )

    assert result["status"] == "OK"
    assert result["data"]["evidence"] == [{
        "referenceId": "COMPAT-section-a",
        "documentId": DOCUMENT_ID,
        "sectionId": "section-a",
        "sectionTitle": None,
        "content": SOURCE_CONTENT,
        "evidenceRef": f"SPL:{DOCUMENT_ID}#section-a",
        "documentVersion": "3",
        "effectiveTime": "2026-08-31",
        "sourcePath": SOURCE_PATH,
        "contentHash": SOURCE_HASH,
        "productIds": [PRODUCT_ID],
        "topic": "warnings",
    }]
    assert "Vector retrieval unavailable." in result["warnings"]
