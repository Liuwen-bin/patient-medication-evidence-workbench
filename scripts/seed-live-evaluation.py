from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path


EXPECTED_PATIENT_NUMBERS = {
    "DEMO-LIVE-001",
    "DEMO-LIVE-AMB",
    "DEMO-LIVE-ALLERGY",
    "DEMO-LIVE-POLY",
    "DEMO-LIVE-DEG",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_resources(path: Path) -> list[dict]:
    resources = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for resource in resources:
        if not resource.get("resourceType") or not resource.get("id"):
            raise ValueError("Every seeded resource requires resourceType and id.")
        tags = (resource.get("meta") or {}).get("tag") or []
        if {
            "system": "urn:local-ehr:data-kind",
            "code": "synthetic",
        } not in tags:
            raise ValueError("Every live evaluation resource must be tagged synthetic.")
    return resources


def patient_number(resource: dict) -> str | None:
    return next(
        (
            str(item["value"])
            for item in resource.get("identifier") or []
            if item.get("system") == "urn:local-ehr:patient-number" and item.get("value")
        ),
        None,
    )


def seed(source: Path, output: Path, manifest: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    manifest = manifest.resolve()
    if source == output:
        raise ValueError("Source and output database paths must differ.")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing run database: {output}")
    resources = load_resources(manifest)
    numbers = {
        number
        for item in resources
        if item.get("resourceType") == "Patient"
        if (number := patient_number(item))
    }
    if numbers != EXPECTED_PATIENT_NUMBERS:
        raise ValueError("The live manifest does not contain the five required patient numbers.")
    ndc_present = any(
        coding.get("system") == "http://hl7.org/fhir/sid/ndc"
        and coding.get("code") == "10191-1246"
        for item in resources
        for coding in (item.get("code") or item.get("medicationCodeableConcept") or {}).get("coding") or []
    )
    if not ndc_present:
        raise ValueError("The live manifest is missing NDC 10191-1246.")
    before = file_hash(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    with sqlite3.connect(output) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(fhir_resources)")
        }
        if not {"resource_type", "resource_id", "json"} <= columns:
            raise RuntimeError("Source database has an incompatible fhir_resources schema.")
        connection.executemany(
            "INSERT OR REPLACE INTO fhir_resources(resource_type, resource_id, json) "
            "VALUES (?, ?, ?)",
            [
                (
                    item["resourceType"],
                    item["id"],
                    json.dumps(item, ensure_ascii=True, sort_keys=True),
                )
                for item in resources
            ],
        )
        connection.commit()
    if file_hash(source) != before:
        raise RuntimeError("Source database changed during isolated evaluation seeding.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed an isolated Health evaluation database")
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--output-db", required=True, type=Path)
    parser.add_argument("--resources", required=True, type=Path)
    args = parser.parse_args()
    seed(args.source_db, args.output_db, args.resources)


if __name__ == "__main__":
    main()
