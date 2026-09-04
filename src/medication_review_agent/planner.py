from __future__ import annotations

from typing import Any, Protocol

from .models import MedicationMapping, ReviewPlanItem


class ReviewPlanner(Protocol):
    async def plan(
        self, patient_features: dict[str, Any], mappings: list[MedicationMapping],
        missing_fields: list[str],
    ) -> list[ReviewPlanItem]: ...


class DeterministicPlanner:
    async def plan(
        self, patient_features: dict[str, Any], mappings: list[MedicationMapping],
        missing_fields: list[str],
    ) -> list[ReviewPlanItem]:
        medication_ids = [item.medicationId for item in mappings]
        items = [
            ReviewPlanItem(planItemId="identity", reviewType="IDENTITY", medicationIds=medication_ids, topics=["identity"], rationale="Confirm the mapped label product."),
            ReviewPlanItem(planItemId="ingredients", reviewType="INGREDIENTS", medicationIds=medication_ids, topics=["active ingredients", "inactive ingredients"], rationale="Review deterministic graph ingredient facts."),
            ReviewPlanItem(planItemId="route-form", reviewType="ROUTE_FORM", medicationIds=medication_ids, topics=["route", "dosage form"], rationale="Compare recorded use with label route and form."),
            ReviewPlanItem(planItemId="evidence-gaps", reviewType="EVIDENCE_GAP", medicationIds=medication_ids, topics=["missing information"], rationale="Keep missing and unmapped information explicit.", requiresHumanReview=True),
        ]
        if patient_features.get("age") is not None:
            items.append(ReviewPlanItem(planItemId="age", reviewType="AGE", medicationIds=medication_ids, topics=["pediatric", "geriatric", "age"], rationale="Age is explicitly present in the patient record."))
        if patient_features.get("specialPopulations"):
            items.append(ReviewPlanItem(planItemId="special-population", reviewType="SPECIAL_POPULATION", medicationIds=medication_ids, topics=["pregnancy", "special populations"], rationale="A special-population fact is explicitly present."))
        if patient_features.get("allergies"):
            items.append(ReviewPlanItem(planItemId="allergy", reviewType="ALLERGY", medicationIds=medication_ids, topics=["allergy", "hypersensitivity"], rationale="Allergy information is explicitly present."))
        return items

