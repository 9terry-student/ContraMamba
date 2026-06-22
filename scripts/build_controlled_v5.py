"""Build and validate a controlled ContraMamba-v5 intervention dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contramamba.labels import (  # noqa: E402
    FinalLabel,
    InterventionType,
    PolarityLabel,
    PrimaryFailureType,
)


REQUIRED_FIELDS = {
    "id",
    "pair_id",
    "claim",
    "evidence",
    "final_label",
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "polarity_label",
    "primary_failure_type",
    "intervention_type",
}
FINAL_LABEL_TO_ID = {label.name: int(label) for label in FinalLabel}
POLARITY_LABELS = {label.name for label in PolarityLabel}
PRIMARY_FAILURE_TYPES = {label.value for label in PrimaryFailureType}
INTERVENTION_TYPES = {label.value for label in InterventionType}
BINARY_FIELDS = {
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
}


FACT_TEMPLATES = [
    {
        "pair_id": "orion_approval",
        "title": "Dr",
        "name": "Mira Chen",
        "alternate_title": "Mr",
        "alternate_name": "Jon Bell",
        "role": "director",
        "alternate_role": "auditor",
        "predicate": "approved",
        "alternate_predicate": "reviewed",
        "object": "the Orion project",
        "alternate_object": "the Vega project",
        "time": "Monday",
        "alternate_time": "Tuesday",
        "location": "Seoul",
        "alternate_location": "Busan",
    },
    {
        "pair_id": "bridge_opening",
        "title": "Mayor",
        "name": "Elena Ruiz",
        "alternate_title": "Councillor",
        "alternate_name": "Pavel Novak",
        "role": "city leader",
        "alternate_role": "transport clerk",
        "predicate": "opened",
        "alternate_predicate": "inspected",
        "object": "the Harbor Bridge",
        "alternate_object": "the River Tunnel",
        "time": "Friday",
        "alternate_time": "Saturday",
        "location": "Lisbon",
        "alternate_location": "Porto",
    },
    {
        "pair_id": "museum_purchase",
        "title": "Ms",
        "name": "Amina Okafor",
        "alternate_title": "Dr",
        "alternate_name": "Ruth Mensah",
        "role": "curator",
        "alternate_role": "accountant",
        "predicate": "purchased",
        "alternate_predicate": "borrowed",
        "object": "the bronze sculpture",
        "alternate_object": "the marble portrait",
        "time": "June",
        "alternate_time": "July",
        "location": "Lagos",
        "alternate_location": "Abuja",
    },
    {
        "pair_id": "satellite_launch",
        "title": "Commander",
        "name": "Noah Kim",
        "alternate_title": "Captain",
        "alternate_name": "Sara Ito",
        "role": "mission chief",
        "alternate_role": "flight analyst",
        "predicate": "launched",
        "alternate_predicate": "tested",
        "object": "the Aurora satellite",
        "alternate_object": "the Nimbus probe",
        "time": "March",
        "alternate_time": "April",
        "location": "Tanegashima",
        "alternate_location": "Wenchang",
    },
    {
        "pair_id": "garden_award",
        "title": "Professor",
        "name": "Iris Wong",
        "alternate_title": "Doctor",
        "alternate_name": "Leo Park",
        "role": "botanist",
        "alternate_role": "historian",
        "predicate": "received",
        "alternate_predicate": "presented",
        "object": "the Green Garden award",
        "alternate_object": "the Urban Forest award",
        "time": "September",
        "alternate_time": "October",
        "location": "Singapore",
        "alternate_location": "Kuala Lumpur",
    },
    {
        "pair_id": "archive_release",
        "title": "Director",
        "name": "Omar Haddad",
        "alternate_title": "Deputy",
        "alternate_name": "Nadia Saleh",
        "role": "archivist",
        "alternate_role": "journalist",
        "predicate": "released",
        "alternate_predicate": "catalogued",
        "object": "the coastal records",
        "alternate_object": "the mountain records",
        "time": "January",
        "alternate_time": "February",
        "location": "Amman",
        "alternate_location": "Aqaba",
    },
    {
        "pair_id": "clinic_expansion",
        "title": "Dr",
        "name": "Sofia Marin",
        "alternate_title": "Nurse",
        "alternate_name": "Ines Costa",
        "role": "clinic chief",
        "alternate_role": "lab manager",
        "predicate": "expanded",
        "alternate_predicate": "surveyed",
        "object": "the rural clinic",
        "alternate_object": "the city laboratory",
        "time": "Wednesday",
        "alternate_time": "Thursday",
        "location": "Valencia",
        "alternate_location": "Alicante",
    },
    {
        "pair_id": "treaty_signature",
        "title": "Minister",
        "name": "Lukas Weber",
        "alternate_title": "Ambassador",
        "alternate_name": "Anna Keller",
        "role": "trade minister",
        "alternate_role": "energy adviser",
        "predicate": "signed",
        "alternate_predicate": "discussed",
        "object": "the North Sea treaty",
        "alternate_object": "the Alpine accord",
        "time": "May",
        "alternate_time": "August",
        "location": "Berlin",
        "alternate_location": "Vienna",
    },
    {
        "pair_id": "festival_selection",
        "title": "Judge",
        "name": "Priya Shah",
        "alternate_title": "Critic",
        "alternate_name": "Meera Rao",
        "role": "jury chair",
        "alternate_role": "festival editor",
        "predicate": "selected",
        "alternate_predicate": "screened",
        "object": "the film Silent River",
        "alternate_object": "the film Golden Road",
        "time": "November",
        "alternate_time": "December",
        "location": "Mumbai",
        "alternate_location": "Delhi",
    },
    {
        "pair_id": "railway_restoration",
        "title": "Engineer",
        "name": "Mateo Silva",
        "alternate_title": "Inspector",
        "alternate_name": "Diego Torres",
        "role": "project lead",
        "alternate_role": "safety officer",
        "predicate": "restored",
        "alternate_predicate": "mapped",
        "object": "the coastal railway",
        "alternate_object": "the valley highway",
        "time": "April",
        "alternate_time": "June",
        "location": "Santiago",
        "alternate_location": "Valparaiso",
    },
]


def _statement(fact: dict, *, negative: bool = False, **overrides: str) -> str:
    values = {**fact, **overrides}
    predicate = values["predicate"]
    if negative:
        predicate = f"did not {predicate}"
    return (
        f"{values['title']} {values['name']}, the {values['role']}, {predicate} "
        f"{values['object']} in {values['location']} during {values['time']}."
    )


def _paraphrase(fact: dict, *, negative: bool = False) -> str:
    polarity = "did not " if negative else ""
    return (
        f"During {fact['time']} in {fact['location']}, {fact['title']} "
        f"{fact['name']} acting as {fact['role']} {polarity}{fact['predicate']} "
        f"{fact['object']}."
    )


def _record(
    fact: dict,
    intervention: str,
    evidence: str,
    final_label: str,
    frame: int,
    predicate: int,
    sufficiency: int,
    polarity: str,
    failure: str,
    claim: str,
) -> dict:
    return {
        "id": f"{fact['pair_id']}__{intervention}",
        "pair_id": fact["pair_id"],
        "claim": claim,
        "evidence": evidence,
        "final_label": final_label,
        "frame_compatible_label": frame,
        "predicate_covered_label": predicate,
        "sufficiency_label": sufficiency,
        "polarity_label": polarity,
        "primary_failure_type": failure,
        "intervention_type": intervention,
    }


def build_seed_records() -> list[dict]:
    records: list[dict] = []
    for index, fact in enumerate(FACT_TEMPLATES):
        base_refute = index >= len(FACT_TEMPLATES) // 2
        claim = _statement(fact)
        base_final = "REFUTE" if base_refute else "SUPPORT"
        base_polarity = base_final
        flipped_final = "SUPPORT" if base_refute else "REFUTE"
        records.extend(
            [
                _record(
                    fact, "none", _statement(fact, negative=base_refute),
                    base_final, 1, 1, 1, base_polarity, "none", claim,
                ),
                _record(
                    fact, "paraphrase", _paraphrase(fact, negative=base_refute),
                    base_final, 1, 1, 1, base_polarity, "none", claim,
                ),
                _record(
                    fact, "entity_swap",
                    _statement(fact, name=fact["alternate_name"]),
                    "NOT_ENTITLED", 0, 0, 1, "NONE", "frame", claim,
                ),
                _record(
                    fact, "event_swap",
                    _statement(fact, object=fact["alternate_object"]),
                    "NOT_ENTITLED", 0, 0, 1, "NONE", "frame", claim,
                ),
                _record(
                    fact, "time_swap",
                    _statement(fact, time=fact["alternate_time"]),
                    "NOT_ENTITLED", 0, 1, 1, "NONE", "frame", claim,
                ),
                _record(
                    fact, "location_swap",
                    _statement(fact, location=fact["alternate_location"]),
                    "NOT_ENTITLED", 0, 1, 1, "NONE", "frame", claim,
                ),
                _record(
                    fact, "role_swap",
                    _statement(fact, role=fact["alternate_role"]),
                    "NOT_ENTITLED", 0, 1, 1, "NONE", "frame", claim,
                ),
                _record(
                    fact, "title_name_swap",
                    _statement(
                        fact,
                        title=fact["alternate_title"],
                        name=fact["alternate_name"],
                    ),
                    "NOT_ENTITLED", 0, 0, 1, "NONE", "frame", claim,
                ),
                _record(
                    fact, "predicate_swap",
                    _statement(fact, predicate=fact["alternate_predicate"]),
                    "NOT_ENTITLED", 1, 0, 1, "NONE", "predicate", claim,
                ),
                _record(
                    fact, "evidence_deletion",
                    "The source contains no statement about the claimed event.",
                    "NOT_ENTITLED", 1, 1, 0, "NONE", "sufficiency", claim,
                ),
                _record(
                    fact, "evidence_truncation",
                    f"{fact['title']} {fact['name']} was mentioned in the report.",
                    "NOT_ENTITLED", 1, 1, 0, "NONE", "sufficiency", claim,
                ),
                _record(
                    fact, "irrelevant_evidence",
                    "A weather bulletin reported mild winds and clear skies.",
                    "NOT_ENTITLED", 0, 0, 0, "NONE", "frame", claim,
                ),
                _record(
                    fact, "polarity_flip", _statement(fact, negative=not base_refute),
                    flipped_final, 1, 1, 1, flipped_final, "polarity", claim,
                ),
            ]
        )
    validate_records(records)
    return records


def validate_record(record: dict, row_number: int | None = None) -> None:
    prefix = f"row {row_number}: " if row_number is not None else ""
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError(f"{prefix}missing required fields: {sorted(missing)}")
    for field in ("id", "pair_id", "claim", "evidence"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{prefix}{field} must be a non-empty string")
    if record["final_label"] not in FINAL_LABEL_TO_ID:
        raise ValueError(f"{prefix}invalid final_label: {record['final_label']!r}")
    if record["polarity_label"] not in POLARITY_LABELS:
        raise ValueError(f"{prefix}invalid polarity_label: {record['polarity_label']!r}")
    if record["primary_failure_type"] not in PRIMARY_FAILURE_TYPES:
        raise ValueError(
            f"{prefix}invalid primary_failure_type: {record['primary_failure_type']!r}"
        )
    if record["intervention_type"] not in INTERVENTION_TYPES:
        raise ValueError(
            f"{prefix}invalid intervention_type: {record['intervention_type']!r}"
        )
    for field in BINARY_FIELDS:
        value = record[field]
        if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
            raise ValueError(f"{prefix}{field} must be integer 0 or 1")


def validate_records(records: Iterable[dict]) -> list[dict]:
    materialized = list(records)
    if not materialized:
        raise ValueError("dataset must contain at least one record")
    seen_ids: set[str] = set()
    for row_number, record in enumerate(materialized, start=1):
        validate_record(record, row_number)
        if record["id"] in seen_ids:
            raise ValueError(f"row {row_number}: duplicate id: {record['id']!r}")
        seen_ids.add(record["id"])
    return materialized


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return validate_records(records)


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    validated = validate_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in validated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_by_pair_id(
    records: Iterable[dict], dev_ratio: float = 0.2, seed: int = 17
) -> tuple[list[dict], list[dict]]:
    materialized = validate_records(records)
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio must be between 0 and 1")
    pair_ids = sorted({record["pair_id"] for record in materialized})
    if len(pair_ids) < 2:
        raise ValueError("at least two pair_id values are required for a split")
    random.Random(seed).shuffle(pair_ids)
    dev_count = min(len(pair_ids) - 1, max(1, round(len(pair_ids) * dev_ratio)))
    dev_pair_ids = set(pair_ids[:dev_count])
    train = [record for record in materialized if record["pair_id"] not in dev_pair_ids]
    dev = [record for record in materialized if record["pair_id"] in dev_pair_ids]
    train_pairs = {record["pair_id"] for record in train}
    dev_pairs = {record["pair_id"] for record in dev}
    if train_pairs & dev_pairs:
        raise RuntimeError("pair_id leakage detected")
    return train, dev


def dataset_statistics(records: Iterable[dict]) -> dict:
    materialized = list(records)
    return {
        "records": len(materialized),
        "pair_ids": len({record["pair_id"] for record in materialized}),
        "interventions": dict(
            sorted(Counter(record["intervention_type"] for record in materialized).items())
        ),
        "final_labels": dict(
            sorted(Counter(record["final_label"] for record in materialized).items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Validate and split an existing JSONL file")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "controlled_v5_seed.jsonl",
    )
    parser.add_argument("--train-output", type=Path)
    parser.add_argument("--dev-output", type=Path)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    records = load_jsonl(args.input) if args.input else build_seed_records()
    write_jsonl(records, args.output)
    train, dev = split_by_pair_id(records, args.dev_ratio, args.seed)
    if bool(args.train_output) != bool(args.dev_output):
        parser.error("--train-output and --dev-output must be supplied together")
    if args.train_output and args.dev_output:
        write_jsonl(train, args.train_output)
        write_jsonl(dev, args.dev_output)

    report = {
        "dataset": dataset_statistics(records),
        "train": dataset_statistics(train),
        "dev": dataset_statistics(dev),
        "pair_id_overlap": sorted(
            {record["pair_id"] for record in train}
            & {record["pair_id"] for record in dev}
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

