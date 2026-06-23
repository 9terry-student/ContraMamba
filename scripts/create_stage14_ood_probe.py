"""Create the Stage 14 controlled OOD probe suite.

This generator builds a deterministic OOD-lite probe from the controlled v5/v6
dataset schema used by Stage13. It intentionally avoids claiming full OOD
coverage; the output is a controlled probe suite for stress-testing surface,
temporal, predicate, frame, and sufficiency behavior.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "controlled_v5_v3.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "stage14_ood_probe.jsonl"
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
PROBE_TYPES = (
    "surface_distractor",
    "temporality_shift",
    "predicate_swap",
    "frame_swap",
    "sufficiency_drop",
)
DAY_ORDER = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTH_ORDER = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
FRAME_SOURCE_TYPES = (
    "entity_swap",
    "event_swap",
    "location_swap",
    "role_swap",
    "title_name_swap",
)
SUFFICIENCY_SOURCE_TYPES = ("evidence_deletion", "evidence_truncation")
SYNTHETIC_ACTORS = (
    ("Dr Mira Chen", "director"),
    ("Mayor Elena Ruiz", "city leader"),
    ("Professor Iris Wong", "botanist"),
    ("Engineer Mateo Silva", "project lead"),
    ("Director Hana Sato", "librarian"),
    ("Minister Lukas Weber", "trade minister"),
    ("Curator Nikos Pappas", "heritage curator"),
    ("Captain Maeve Kelly", "fleet captain"),
    ("Architect Camila Reyes", "lead architect"),
    ("Ranger Erik Lund", "forest ranger"),
)
SYNTHETIC_EVENTS = (
    ("approved", "the Orion project", "Seoul"),
    ("opened", "the Harbor Bridge", "Lisbon"),
    ("restored", "the coastal railway", "Santiago"),
    ("digitized", "the Edo manuscripts", "Kyoto"),
    ("protected", "the Blue Reef reserve", "Noumea"),
    ("signed", "the North Sea treaty", "Berlin"),
    ("restored", "the Apollo theater", "Athens"),
    ("launched", "the Emerald ferry", "Dublin"),
    ("renovated", "the Central Market", "Bogota"),
    ("mapped", "the Pine Valley forest", "Oslo"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"input data is empty: {path}")
    for index, record in enumerate(records, start=1):
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(f"input row {index} missing fields: {sorted(missing)}")
    return records


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def swap_temporal_value(text: str) -> str | None:
    """Swap an explicit temporal slot while preserving the surrounding frame.

    The controlled v5/v6 records use "during <time>" phrases. The first Stage14
    version only handled weekday names, which limited temporality_shift to six
    examples. This function also handles month names and keeps replacements
    within the same coarse type, avoiding ambiguous edits such as weekday <->
    month substitutions.
    """

    for values, offset in ((DAY_ORDER, 2), (MONTH_ORDER, 3)):
        for index, value in enumerate(values):
            pattern = rf"\bduring {value}\b"
            if re.search(pattern, text):
                replacement = values[(index + offset) % len(values)]
                return re.sub(pattern, f"during {replacement}", text, count=1)
    return None


def swap_day(text: str) -> str | None:
    """Backward-compatible alias for older imports/tests."""
    return swap_temporal_value(text)


def count_by_source_intervention(records: Sequence[dict[str, Any]]) -> Counter[str]:
    return Counter(record.get("source_intervention_type", "unknown") for record in records)


def select_balanced_by_source_intervention(
    records: Sequence[dict[str, Any]],
    *,
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Select records while preserving subtype coverage where possible."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record.get("source_intervention_type", "unknown")].append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for key in sorted(buckets):
            bucket = buckets[key]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    return selected


def _legacy_swap_day(text: str) -> str | None:
    for index, day in enumerate(DAY_ORDER):
        pattern = rf"\bduring {day}\b"
        if re.search(pattern, text):
            replacement = DAY_ORDER[(index + 2) % len(DAY_ORDER)]
            return re.sub(pattern, f"during {replacement}", text, count=1)
    return None


def stage14_record(
    *,
    source: dict[str, Any],
    probe_type: str,
    evidence: str,
    final_label: str,
    frame: int,
    predicate: int,
    sufficiency: int,
    polarity: str,
    failure: str,
    source_intervention_type: str,
    expected_behavior: str,
) -> dict[str, Any]:
    source_pair_id = source["pair_id"]
    source_slug = source["id"].replace("__", "_")
    pair_id = f"stage14_{probe_type}__{source_slug}"
    return {
        "id": f"{pair_id}__probe",
        "pair_id": pair_id,
        "claim": source["claim"],
        "evidence": evidence,
        "final_label": final_label,
        "frame_compatible_label": frame,
        "predicate_covered_label": predicate,
        "sufficiency_label": sufficiency,
        "polarity_label": polarity,
        "primary_failure_type": failure,
        "intervention_type": probe_type,
        "stage14_probe_type": probe_type,
        "source_intervention_type": source_intervention_type,
        "source_pair_id": source_pair_id,
        "source_id": source["id"],
        "stage14_expected_behavior": expected_behavior,
    }


def build_surface_distractor(source: dict[str, Any]) -> dict[str, Any]:
    # Keep token length unchanged for the Stage13 dummy smoke model while still
    # perturbing surface form. Textual distractors can exceed the small
    # train/dev positional range in smoke mode.
    evidence = f"{source['evidence']} --"
    return stage14_record(
        source=source,
        probe_type="surface_distractor",
        evidence=evidence,
        final_label=source["final_label"],
        frame=source["frame_compatible_label"],
        predicate=source["predicate_covered_label"],
        sufficiency=source["sufficiency_label"],
        polarity=source["polarity_label"],
        failure=source["primary_failure_type"],
        source_intervention_type=source["intervention_type"],
        expected_behavior="preserve_final_label_under_harmless_surface_distractor",
    )


def build_temporality_shift(source: dict[str, Any]) -> dict[str, Any] | None:
    evidence = swap_temporal_value(source["evidence"])
    if evidence is None or evidence == source["evidence"]:
        return None
    return stage14_record(
        source=source,
        probe_type="temporality_shift",
        evidence=evidence,
        final_label="NOT_ENTITLED",
        frame=0,
        predicate=1,
        sufficiency=1,
        polarity="NONE",
        failure="frame",
        source_intervention_type=source["intervention_type"],
        expected_behavior="reject_temporal_frame_mismatch_as_not_entitled",
    )


def synthetic_temporality_record(index: int) -> dict[str, Any]:
    actor, role = SYNTHETIC_ACTORS[index % len(SYNTHETIC_ACTORS)]
    predicate, obj, location = SYNTHETIC_EVENTS[index % len(SYNTHETIC_EVENTS)]
    if index % 2 == 0:
        source_time = DAY_ORDER[index % len(DAY_ORDER)]
        target_time = DAY_ORDER[(index + 2) % len(DAY_ORDER)]
    else:
        source_time = MONTH_ORDER[index % len(MONTH_ORDER)]
        target_time = MONTH_ORDER[(index + 4) % len(MONTH_ORDER)]

    claim = (
        f"{actor}, the {role}, {predicate} {obj} in {location} "
        f"during {source_time}."
    )
    evidence = (
        f"{actor}, the {role}, {predicate} {obj} in {location} "
        f"during {target_time}."
    )
    pair_id = f"stage14_temporality_shift__synthetic_{index:03d}"
    return {
        "id": f"{pair_id}__probe",
        "pair_id": pair_id,
        "claim": claim,
        "evidence": evidence,
        "final_label": "NOT_ENTITLED",
        "frame_compatible_label": 0,
        "predicate_covered_label": 1,
        "sufficiency_label": 1,
        "polarity_label": "NONE",
        "primary_failure_type": "frame",
        "intervention_type": "temporality_shift",
        "stage14_probe_type": "temporality_shift",
        "source_intervention_type": "synthetic_temporality",
        "source_pair_id": pair_id,
        "source_id": pair_id,
        "stage14_expected_behavior": "reject_temporal_frame_mismatch_as_not_entitled",
    }


def add_synthetic_temporality_if_needed(
    candidates: dict[str, list[dict[str, Any]]],
    *,
    target_count: int,
    enabled: bool,
) -> Counter[str]:
    notes: Counter[str] = Counter()
    current = len(candidates.get("temporality_shift", []))
    if current >= target_count:
        return notes
    missing = target_count - current
    if not enabled:
        notes["temporality_shift:synthetic_disabled_below_target"] += missing
        return notes

    existing_ids = {
        record["id"]
        for group in candidates.values()
        for record in group
        if "id" in record
    }
    synthetic: list[dict[str, Any]] = []
    index = 0
    while len(synthetic) < missing:
        record = synthetic_temporality_record(index)
        index += 1
        if record["id"] in existing_ids:
            continue
        existing_ids.add(record["id"])
        synthetic.append(record)
    candidates["temporality_shift"].extend(synthetic)
    notes["temporality_shift:synthetic_added"] += len(synthetic)
    return notes


def copy_variant(source: dict[str, Any], probe_type: str, expected_behavior: str) -> dict[str, Any]:
    return stage14_record(
        source=source,
        probe_type=probe_type,
        evidence=source["evidence"],
        final_label=source["final_label"],
        frame=source["frame_compatible_label"],
        predicate=source["predicate_covered_label"],
        sufficiency=source["sufficiency_label"],
        polarity=source["polarity_label"],
        failure=source["primary_failure_type"],
        source_intervention_type=source["intervention_type"],
        expected_behavior=expected_behavior,
    )


def candidates_by_probe_type(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        intervention = record["intervention_type"]
        if intervention == "none" and record["final_label"] == "SUPPORT":
            candidates["surface_distractor"].append(build_surface_distractor(record))
            temporal = build_temporality_shift(record)
            if temporal is not None:
                candidates["temporality_shift"].append(temporal)
        elif intervention == "predicate_swap":
            candidates["predicate_swap"].append(
                copy_variant(
                    record,
                    "predicate_swap",
                    "reject_predicate_relation_mismatch_as_not_entitled",
                )
            )
        elif intervention in FRAME_SOURCE_TYPES:
            candidates["frame_swap"].append(
                copy_variant(
                    record,
                    "frame_swap",
                    "reject_frame_slot_mismatch_as_not_entitled",
                )
            )
        elif intervention in SUFFICIENCY_SOURCE_TYPES:
            candidates["sufficiency_drop"].append(
                copy_variant(
                    record,
                    "sufficiency_drop",
                    "reject_insufficient_evidence_as_not_entitled",
                )
            )
    return candidates


def select_records(
    candidates: dict[str, list[dict[str, Any]]],
    *,
    max_per_group: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    skipped: Counter[str] = Counter()
    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    for probe_type in PROBE_TYPES:
        group = list(candidates.get(probe_type, []))
        if not group:
            skipped[f"{probe_type}:no_safe_candidates"] += 1
            selected_by_group[probe_type] = []
            continue
        if probe_type == "frame_swap":
            selected_by_group[probe_type] = select_balanced_by_source_intervention(
                group,
                limit=max_per_group,
                rng=rng,
            )
        else:
            rng.shuffle(group)
            selected_by_group[probe_type] = group[:max_per_group]
        if len(group) < max_per_group:
            skipped[f"{probe_type}:below_max_per_group"] += max_per_group - len(group)

    interleaved: list[dict[str, Any]] = []
    for index in range(max_per_group):
        for probe_type in PROBE_TYPES:
            group = selected_by_group[probe_type]
            if index < len(group):
                interleaved.append(group[index])
    return interleaved, skipped


def validate_records(records: Sequence[dict[str, Any]]) -> None:
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Stage14 probe generated duplicate ids")
    for index, record in enumerate(records, start=1):
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise RuntimeError(f"Stage14 row {index} missing fields: {sorted(missing)}")
        if record["stage14_probe_type"] not in PROBE_TYPES:
            raise RuntimeError(f"Stage14 row {index} has invalid probe type")


def print_summary(records: Sequence[dict[str, Any]], skipped: Counter[str]) -> None:
    by_probe = Counter(record["stage14_probe_type"] for record in records)
    by_label = Counter(record["final_label"] for record in records)
    by_source = count_by_source_intervention(records)
    print("STAGE14_OOD_PROBE_SUMMARY")
    print(f"total_generated\t{len(records)}")
    print("count_by_stage14_probe_type")
    for key in PROBE_TYPES:
        print(f"{key}\t{by_probe.get(key, 0)}")
    print("count_by_expected_label")
    for key in sorted(by_label):
        print(f"{key}\t{by_label[key]}")
    print("count_by_source_intervention_type")
    for key in sorted(by_source):
        print(f"{key}\t{by_source[key]}")
    print("skipped_counts_by_reason")
    if skipped:
        for key in sorted(skipped):
            print(f"{key}\t{skipped[key]}")
    else:
        print("none\t0")
    synthetic_count = skipped.get("temporality_shift:synthetic_added", 0)
    if synthetic_count:
        print(
            "WARNING\tsynthetic_temporality_used\t"
            f"{synthetic_count} deterministic temporal examples added"
        )


def build_probe(
    records: Sequence[dict[str, Any]],
    *,
    max_per_group: int,
    min_temporality: int,
    synthetic_temporality: bool,
    seed: int,
    exclude_interventions: Sequence[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if max_per_group < 1:
        raise ValueError("max_per_group must be positive")
    if min_temporality < 0:
        raise ValueError("min_temporality must be non-negative")
    excluded = set(exclude_interventions)
    filtered = [
        record for record in records if record["intervention_type"] not in excluded
    ]
    rng = random.Random(seed)
    candidates = candidates_by_probe_type(filtered)
    temporal_target = max(min_temporality, max_per_group)
    synthetic_notes = add_synthetic_temporality_if_needed(
        candidates,
        target_count=temporal_target,
        enabled=synthetic_temporality,
    )
    records_out, skipped = select_records(
        candidates,
        max_per_group=max_per_group,
        rng=rng,
    )
    skipped.update(synthetic_notes)
    validate_records(records_out)
    return records_out, skipped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-per-group", type=int, default=100)
    parser.add_argument("--min-temporality", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.set_defaults(synthetic_temporality=True)
    parser.add_argument(
        "--synthetic-temporality",
        dest="synthetic_temporality",
        action="store_true",
        help="Enable deterministic synthetic temporal examples when source data is sparse.",
    )
    parser.add_argument(
        "--no-synthetic-temporality",
        dest="synthetic_temporality",
        action="store_false",
        help="Disable deterministic synthetic temporal fallback.",
    )
    parser.add_argument(
        "--exclude-interventions",
        nargs="*",
        default=["time_swap"],
        help="Intervention types to exclude before generating probes.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_jsonl(args.input)
    generated, skipped = build_probe(
        records,
        max_per_group=args.max_per_group,
        min_temporality=args.min_temporality,
        synthetic_temporality=args.synthetic_temporality,
        seed=args.seed,
        exclude_interventions=args.exclude_interventions,
    )
    write_jsonl(args.output, generated)
    print_summary(generated, skipped)
    print(f"wrote\t{args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
