"""Create the Stage15 slot-sensitivity diagnostic probe.

The probe is derived from Stage14 OOD examples and is meant to separate
slot-level representation failures from auxiliary-head and final-aggregation
failures. It is a controlled diagnostic suite, not a benchmark-training set.
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
DEFAULT_INPUT = ROOT / "data" / "stage14_ood_probe_v2.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "stage15_slot_sensitivity_probe.jsonl"

PROBE_TYPES = (
    "temporal_mismatch",
    "temporal_erased",
    "surface_control",
    "sufficiency_control",
    "frame_location_mismatch",
    "frame_role_mismatch",
    "predicate_mismatch",
)

FINAL_LABELS = {"REFUTE", "NOT_ENTITLED", "SUPPORT"}
WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
MONTHS = (
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
TEMPORAL_VALUES = tuple(sorted(set(WEEKDAYS + MONTHS), key=len, reverse=True))
TEMPORAL_PATTERN = re.compile(
    rf"\s+(?:during|in|on)\s+(?:{'|'.join(re.escape(x) for x in TEMPORAL_VALUES)})\b"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"input file is empty: {path}")
    return records


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def clean_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    return text


def erase_temporal_phrase(text: str) -> str | None:
    erased, count = TEMPORAL_PATTERN.subn("", text, count=1)
    if count == 0:
        return None
    return clean_spacing(erased)


def source_id(row: dict[str, Any]) -> str:
    return str(row.get("stage14_source_id") or row.get("source_id") or row["id"])


def make_stage15_row(
    row: dict[str, Any],
    *,
    probe_type: str,
    expected_behavior: str,
    transformation: str,
    claim: str | None = None,
    evidence: str | None = None,
    final_label: str | None = None,
    frame: int | None = None,
    predicate: int | None = None,
    sufficiency: int | None = None,
    polarity: str | None = None,
    failure: str | None = None,
) -> dict[str, Any]:
    original_id = row["id"]
    new_id = f"stage15_{probe_type}__{original_id}"
    out = dict(row)
    out.update(
        {
            "id": new_id,
            "pair_id": f"stage15_{probe_type}__{row.get('pair_id', original_id)}",
            "claim": claim if claim is not None else row["claim"],
            "evidence": evidence if evidence is not None else row["evidence"],
            "final_label": final_label if final_label is not None else row.get("final_label"),
            "frame_compatible_label": (
                frame if frame is not None else row.get("frame_compatible_label")
            ),
            "predicate_covered_label": (
                predicate if predicate is not None else row.get("predicate_covered_label")
            ),
            "sufficiency_label": (
                sufficiency if sufficiency is not None else row.get("sufficiency_label")
            ),
            "polarity_label": polarity if polarity is not None else row.get("polarity_label"),
            "primary_failure_type": (
                failure if failure is not None else row.get("primary_failure_type")
            ),
            "intervention_type": probe_type,
            "stage15_probe_type": probe_type,
            "stage15_source_id": original_id,
            "stage15_expected_behavior": expected_behavior,
            "stage15_transformation": transformation,
            "stage15_original_claim": row["claim"],
            "stage15_original_evidence": row["evidence"],
            "stage15_original_label": row.get("final_label") or row.get("label"),
            "stage15_original_probe_type": row.get("stage14_probe_type")
            or row.get("intervention_type"),
        }
    )
    return out


def temporal_erased_variant(row: dict[str, Any]) -> dict[str, Any] | None:
    claim = erase_temporal_phrase(row["claim"])
    evidence = erase_temporal_phrase(row["evidence"])
    if claim is None or evidence is None:
        return None
    if claim == row["claim"] or evidence == row["evidence"]:
        return None
    return make_stage15_row(
        row,
        probe_type="temporal_erased",
        claim=claim,
        evidence=evidence,
        final_label="SUPPORT",
        frame=1,
        predicate=1,
        sufficiency=1,
        polarity="SUPPORT",
        failure="none",
        expected_behavior=(
            "temporal_information_removed_should_restore_support_if_only_time_mismatched"
        ),
        transformation="removed_supported_temporal_phrase_from_claim_and_evidence",
    )


def build_candidates(rows: Iterable[dict[str, Any]], skipped: Counter[str]) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stage14_type = row.get("stage14_probe_type") or row.get("intervention_type")
        source_intervention = row.get("source_intervention_type")

        if stage14_type == "temporality_shift":
            candidates["temporal_mismatch"].append(
                make_stage15_row(
                    row,
                    probe_type="temporal_mismatch",
                    expected_behavior="reject_temporal_frame_mismatch_as_not_entitled",
                    transformation="reused_stage14_temporality_shift",
                )
            )
            erased = temporal_erased_variant(row)
            if erased is None:
                skipped["temporal_erased:no_supported_temporal_phrase"] += 1
            else:
                candidates["temporal_erased"].append(erased)
        elif stage14_type == "surface_distractor":
            candidates["surface_control"].append(
                make_stage15_row(
                    row,
                    probe_type="surface_control",
                    expected_behavior="preserve_support_under_harmless_surface_control",
                    transformation="reused_stage14_surface_distractor",
                )
            )
        elif stage14_type == "sufficiency_drop":
            candidates["sufficiency_control"].append(
                make_stage15_row(
                    row,
                    probe_type="sufficiency_control",
                    expected_behavior="reject_insufficient_evidence_as_not_entitled",
                    transformation="reused_stage14_sufficiency_drop",
                )
            )
        elif stage14_type == "frame_swap" and source_intervention == "location_swap":
            candidates["frame_location_mismatch"].append(
                make_stage15_row(
                    row,
                    probe_type="frame_location_mismatch",
                    expected_behavior="reject_location_slot_mismatch_as_not_entitled",
                    transformation="reused_stage14_location_frame_swap",
                )
            )
        elif stage14_type == "frame_swap" and source_intervention == "role_swap":
            candidates["frame_role_mismatch"].append(
                make_stage15_row(
                    row,
                    probe_type="frame_role_mismatch",
                    expected_behavior="reject_role_slot_mismatch_as_not_entitled",
                    transformation="reused_stage14_role_frame_swap",
                )
            )
        elif stage14_type == "predicate_swap":
            candidates["predicate_mismatch"].append(
                make_stage15_row(
                    row,
                    probe_type="predicate_mismatch",
                    expected_behavior="reject_predicate_relation_mismatch_as_not_entitled",
                    transformation="reused_stage14_predicate_swap",
                )
            )
    return candidates


def select_rows(
    candidates: dict[str, list[dict[str, Any]]],
    *,
    max_per_type: int,
    rng: random.Random,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for probe_type in PROBE_TYPES:
        group = list(candidates.get(probe_type, []))
        if not group:
            skipped[f"{probe_type}:no_candidates"] += 1
            continue
        rng.shuffle(group)
        selected.extend(group[:max_per_type])
        if len(group) < max_per_type:
            skipped[f"{probe_type}:below_max_per_type"] += max_per_type - len(group)
    return selected


def validate_rows(rows: Sequence[dict[str, Any]]) -> None:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Stage15 probe generated duplicate ids")
    for index, row in enumerate(rows, start=1):
        if row.get("stage15_probe_type") not in PROBE_TYPES:
            raise RuntimeError(f"row {index} has invalid stage15_probe_type")
        label = row.get("final_label") or row.get("label")
        if label not in FINAL_LABELS:
            raise RuntimeError(f"row {index} has invalid label: {label!r}")


def print_summary(rows: Sequence[dict[str, Any]], skipped: Counter[str]) -> None:
    print("STAGE15_SLOT_SENSITIVITY_PROBE_SUMMARY")
    print(f"total_written\t{len(rows)}")
    print("count_by_stage15_probe_type")
    counts = Counter(row.get("stage15_probe_type") for row in rows)
    for probe_type in PROBE_TYPES:
        print(f"{probe_type}\t{counts.get(probe_type, 0)}")
    print("skip_counts")
    if skipped:
        for reason in sorted(skipped):
            print(f"{reason}\t{skipped[reason]}")
    else:
        print("none\t0")


def build_probe(rows: Sequence[dict[str, Any]], *, max_per_type: int, seed: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    if max_per_type < 1:
        raise ValueError("max_per_type must be positive")
    skipped: Counter[str] = Counter()
    candidates = build_candidates(rows, skipped)
    selected = select_rows(
        candidates,
        max_per_type=max_per_type,
        rng=random.Random(seed),
        skipped=skipped,
    )
    validate_rows(selected)
    return selected, skipped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_rows = load_jsonl(args.input)
    rows, skipped = build_probe(source_rows, max_per_type=args.max_per_type, seed=args.seed)
    write_jsonl(args.output, rows)
    print_summary(rows, skipped)
    print(f"wrote\t{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

