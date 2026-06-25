"""Filter controlled_v5_v3.jsonl to remove time_swap records.

Produces a clean training/eval file with all other intervention types intact.
Does not modify input records beyond JSON re-serialization (whitespace normalization).
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
DEFAULT_OUTPUT = DATA / "controlled_v5_v3_without_time_swap.jsonl"

EXCLUDED_INTERVENTION = "time_swap"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove time_swap records from controlled_v5_v3.jsonl. "
            "All other records are preserved exactly (JSON whitespace may normalize). "
            "Output has zero time_swap records."
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
        help="Output JSONL file (default: data/controlled_v5_v3_without_time_swap.jsonl)",
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

    kept: list[dict] = []
    input_counts: Counter[str] = Counter()
    output_counts: Counter[str] = Counter()

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
            if itype != EXCLUDED_INTERVENTION:
                kept.append(record)
                output_counts[itype] += 1

    removed = input_counts[EXCLUDED_INTERVENTION]
    total_in = sum(input_counts.values())
    total_out = sum(output_counts.values())

    print(f"Input:  {total_in} records from {args.input}")
    print("  Counts by intervention_type:")
    for itype in sorted(input_counts):
        marker = "  [EXCLUDED]" if itype == EXCLUDED_INTERVENTION else ""
        print(f"    {itype}: {input_counts[itype]}{marker}")

    print(f"\nOutput: {total_out} records ({removed} time_swap removed)")
    print("  Counts by intervention_type:")
    for itype in sorted(output_counts):
        print(f"    {itype}: {output_counts[itype]}")

    assert output_counts.get(EXCLUDED_INTERVENTION, 0) == 0, (
        "BUG: time_swap records found in output — filtering logic is broken"
    )

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    with open(args.output, "w", encoding="utf-8") as fh:
        for record in kept:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {total_out} records to {args.output}")


if __name__ == "__main__":
    main()
