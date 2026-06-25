"""Build a temporal diagnostic JSONL from controlled_v5_v3.jsonl.

Selects time_swap records (temporal mismatch) and none/paraphrase records
(temporally safe controls) from the existing controlled dataset. Adds temporal
diagnostic fields to each output record.

This file is DIAGNOSTIC ONLY. It must not be mixed into the main clean
controlled train/eval classification data (controlled_v5_v3_without_time_swap.jsonl).
Stage15 is not used or read by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DEFAULT_INPUT = DATA / "controlled_v5_v3.jsonl"
DEFAULT_OUTPUT = DATA / "temporal_diagnostic_v1_from_controlled_v5_v3.jsonl"
CLEAN_CONTROLLED = DATA / "controlled_v5_v3_without_time_swap.jsonl"

DIAGNOSTIC_DATASET_NAME = "temporal_diagnostic_v1_from_controlled_v5_v3"

TEMPORAL_MISMATCH_TYPE = "time_swap"
TEMPORAL_CONTROL_TYPES = frozenset({"none", "paraphrase"})

LEAKAGE_NOTE = (
    "Constructed from controlled_v5_v3.jsonl only. "
    "Stage15 OOD records were not used in construction, selection, or labeling."
)
USAGE_NOTE = (
    "Diagnostic-only file. Must not be mixed into the main clean controlled "
    "train/eval classification data (controlled_v5_v3_without_time_swap.jsonl). "
    "Intended for temporal-specific diagnostic supervision only."
)

ALLOWED_ROLES = frozenset({"temporal_mismatch", "temporal_control"})


def _make_diagnostic_record(
    record: dict,
    *,
    label: int,
    role: str,
    source_file: str,
) -> dict:
    out = dict(record)
    out["temporal_diagnostic_label"] = label
    out["temporal_diagnostic_role"] = role
    out["source_intervention_type"] = record.get("intervention_type", "unknown")
    out["source_file"] = source_file
    out["diagnostic_dataset"] = DIAGNOSTIC_DATASET_NAME
    out["leakage_note"] = LEAKAGE_NOTE
    out["usage_note"] = USAGE_NOTE
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a temporal diagnostic dataset from controlled_v5_v3.jsonl. "
            "Selects time_swap records (label=1, role=temporal_mismatch) and "
            "none/paraphrase records (label=0, role=temporal_control). "
            "Output is diagnostic-only and must not be mixed into the main clean "
            "controlled train/eval data. Stage15 is not used."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input JSONL file (default: data/controlled_v5_v3.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output JSONL file "
            "(default: data/temporal_diagnostic_v1_from_controlled_v5_v3.jsonl)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite output if it already exists (default: fail if output exists)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print counts but do not write output",
    )
    parser.add_argument(
        "--include-paraphrase-controls",
        action="store_true",
        default=True,
        help=(
            "Include paraphrase records as temporal controls alongside none records "
            "(default: True; use --no-include-paraphrase-controls to exclude)"
        ),
    )
    parser.add_argument(
        "--no-include-paraphrase-controls",
        dest="include_paraphrase_controls",
        action="store_false",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.output.exists() and not args.overwrite and not args.dry_run:
        print(
            f"ERROR: output file already exists: {args.output}\n"
            "Use --overwrite to replace it or --dry-run to preview counts.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine which control types to include based on flag
    control_types = frozenset({"none", "paraphrase"}) if args.include_paraphrase_controls else frozenset({"none"})

    source_file_str = str(args.input)
    records_out: list[dict] = []
    input_counts: Counter[str] = Counter()
    output_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    role_counts: Counter[str] = Counter()

    with open(args.input, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"ERROR: malformed JSON on line {lineno}: {exc}", file=sys.stderr)
                sys.exit(1)
            itype = record.get("intervention_type", "unknown")
            input_counts[itype] += 1

            if itype == TEMPORAL_MISMATCH_TYPE:
                out = _make_diagnostic_record(
                    record, label=1, role="temporal_mismatch", source_file=source_file_str
                )
                records_out.append(out)
                output_counts[itype] += 1
                label_counts[1] += 1
                role_counts["temporal_mismatch"] += 1
            elif itype in control_types:
                out = _make_diagnostic_record(
                    record, label=0, role="temporal_control", source_file=source_file_str
                )
                records_out.append(out)
                output_counts[itype] += 1
                label_counts[0] += 1
                role_counts["temporal_control"] += 1
            # All other intervention types are silently excluded

    total_in = sum(input_counts.values())
    total_out = sum(output_counts.values())
    excluded_count = total_in - total_out

    # --- Safety checks ---
    # 1. Stage15 was not used (structural: path is never referenced)
    assert "stage15" not in str(args.input).lower(), (
        "BUG: input path contains 'stage15' — Stage15 must not be used for dataset construction"
    )

    # 2. All output records have required fields and valid roles
    for i, rec in enumerate(records_out):
        assert "temporal_diagnostic_label" in rec, (
            f"BUG: record {i} is missing temporal_diagnostic_label"
        )
        assert rec["temporal_diagnostic_role"] in ALLOWED_ROLES, (
            f"BUG: record {i} has unexpected role: {rec['temporal_diagnostic_role']!r}"
        )

    # 3. time_swap is present in this file as temporal_mismatch only
    assert role_counts.get("temporal_mismatch", 0) == output_counts.get(TEMPORAL_MISMATCH_TYPE, 0), (
        "BUG: temporal_mismatch role count does not match time_swap record count"
    )

    # --- Print report ---
    print(f"Input:  {total_in} records from {args.input}")
    print("  Counts by intervention_type:")
    for itype in sorted(input_counts):
        if itype == TEMPORAL_MISMATCH_TYPE:
            marker = "  [temporal_mismatch → label=1]"
        elif itype in control_types:
            marker = "  [temporal_control → label=0]"
        else:
            marker = "  [EXCLUDED]"
        print(f"    {itype}: {input_counts[itype]}{marker}")

    print(f"\nOutput: {total_out} records ({excluded_count} excluded)")
    print("  Counts by intervention_type in output:")
    for itype in sorted(output_counts):
        print(f"    {itype}: {output_counts[itype]}")
    print("  Diagnostic label distribution:")
    print(f"    label=1 (temporal_mismatch): {label_counts[1]}")
    print(f"    label=0 (temporal_control):  {label_counts[0]}")
    print("  Diagnostic role distribution:")
    for role in sorted(role_counts):
        print(f"    {role}: {role_counts[role]}")

    # 4. Optionally verify clean controlled file has zero time_swap
    if CLEAN_CONTROLLED.exists():
        clean_ts = sum(
            1
            for ln in open(CLEAN_CONTROLLED, encoding="utf-8")
            if ln.strip() and json.loads(ln).get("intervention_type") == TEMPORAL_MISMATCH_TYPE
        )
        status = "OK (zero time_swap)" if clean_ts == 0 else f"WARNING: {clean_ts} time_swap found"
        print(f"\nClean controlled file check ({CLEAN_CONTROLLED.name}): {status}")
    else:
        print(f"\nClean controlled file not found at {CLEAN_CONTROLLED} (skipping check)")

    print(
        f"\nLeakage note: {LEAKAGE_NOTE}\n"
        f"Usage note:   {USAGE_NOTE}"
    )

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    with open(args.output, "w", encoding="utf-8") as fh:
        for rec in records_out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {total_out} records to {args.output}")


if __name__ == "__main__":
    main()
