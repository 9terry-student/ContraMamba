"""Stage22-A4b: pair-group contrastive frame dataset generator.

Builds a diagnostic contrastive dataset from controlled training data only.
Each output record pairs a preservation-side sibling (none / paraphrase) with
a frame-side sibling (entity_swap / event_swap / location_swap / role_swap /
title_name_swap) that share the same pair_id.

Data-leakage contract
---------------------
- Stage15 OOD records are NOT read, referenced, or used.
- All output records are derived exclusively from controlled training data.
- Output records carry leakage_note = "constructed_from_controlled_data_only".
- This output is a DIAGNOSTIC CONTRASTIVE DATASET, not a final training set.
  Stage22-B logit gate remains rejected until OOD ranking validation passes.

Usage
-----
    python scripts/build_stage22a4_pair_contrastive_frame_data.py \\
        --controlled-data  data/controlled_v5_v3_without_time_swap.jsonl \\
        --output-jsonl     data/stage22a4_pair_contrastive_frame.jsonl \\
        --output-summary-json results/stage22a4_pair_contrastive_summary.json \\
        --output-summary-md   results/stage22a4_pair_contrastive_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Vocabulary constants — match train_controlled_v6b_minimal.py exactly
# ---------------------------------------------------------------------------

PRESERVATION_TYPES: frozenset[str] = frozenset({"none", "paraphrase"})
FRAME_TYPES: frozenset[str] = frozenset({
    "entity_swap", "event_swap", "location_swap", "role_swap", "title_name_swap",
})
IGNORED_TYPES: frozenset[str] = frozenset({
    "time_swap", "predicate_swap",
    "evidence_deletion", "evidence_truncation", "irrelevant_evidence",
    "polarity_flip",
})

CONTRASTIVE_TARGET = "frame_more_violating_than_preservation"
SOURCE_TAG = "controlled_pair_group"
LEAKAGE_NOTE = "constructed_from_controlled_data_only"

# Optional label fields copied from source records when present
_OPTIONAL_LABEL_FIELDS = (
    "frame_compatible_label",
    "sufficiency_label",
    "predicate_covered_label",
    "polarity_label",
    "primary_failure_type",
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Core pair construction
# ---------------------------------------------------------------------------

def _copy_optional_labels(record: dict[str, Any]) -> dict[str, Any]:
    return {f: record[f] for f in _OPTIONAL_LABEL_FIELDS if f in record}


def build_contrastive_record(
    pair_id: str,
    pres: dict[str, Any],
    frame: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Construct one contrastive record from a (preservation, frame) sibling pair."""
    contrastive_id = (
        f"stage22a4__{pair_id}"
        f"__pres_{pres['intervention_type']}"
        f"__frame_{frame['intervention_type']}"
        f"__{index:04d}"
    )
    rec: dict[str, Any] = {
        "contrastive_id": contrastive_id,
        "pair_id": pair_id,
        # Shared claim text (same pair_id — use preservation side as canonical)
        "claim": pres["claim"],
        # Evidence texts from each sibling
        "preservation_evidence": pres["evidence"],
        "frame_evidence": frame["evidence"],
        # Intervention metadata
        "preservation_intervention_type": pres["intervention_type"],
        "frame_intervention_type": frame["intervention_type"],
        # Labels
        "preservation_final_label": pres.get("final_label", ""),
        "frame_final_label": frame.get("final_label", ""),
        # Contrastive semantics
        "target": CONTRASTIVE_TARGET,
        "preservation_should_be_safe": True,
        "frame_should_be_blocked": True,
        # Provenance
        "source": SOURCE_TAG,
        "leakage_note": LEAKAGE_NOTE,
        # Source record IDs for traceability
        "preservation_source_id": pres.get("id", ""),
        "frame_source_id": frame.get("id", ""),
    }
    # Copy optional label fields from preservation side (canonical)
    rec.update(_copy_optional_labels(pres))
    # Copy frame-side labels with prefix where they differ
    for field in _OPTIONAL_LABEL_FIELDS:
        if field in frame:
            rec[f"frame_{field}"] = frame[field]
    return rec


def generate_contrastive_pairs(
    records: list[dict[str, Any]],
    *,
    exclude_time_swap: bool = True,
    max_pairs_per_pair_id: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Group controlled records by pair_id and construct contrastive pairs.

    Returns (output_records, stats_dict).
    """
    # Group by pair_id
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        pid = rec.get("pair_id")
        if pid is None:
            continue
        it = rec.get("intervention_type", "")
        if exclude_time_swap and it == "time_swap":
            continue
        by_pair[pid].append(rec)

    stats: dict[str, Any] = {
        "input_records": len(records),
        "pair_ids_total": len(by_pair),
        "pair_ids_with_preservation": 0,
        "pair_ids_with_frame": 0,
        "pair_ids_used": 0,
        "skipped_pair_ids": [],
        "output_pair_count": 0,
        "pairs_by_frame_intervention_type": {},
        "pairs_by_preservation_intervention_type": {},
    }

    frame_it_counter: Counter[str] = Counter()
    pres_it_counter: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    global_index = 0

    for pair_id in sorted(by_pair):
        group = by_pair[pair_id]
        pres_members = [r for r in group if r.get("intervention_type") in PRESERVATION_TYPES]
        frame_members = [r for r in group if r.get("intervention_type") in FRAME_TYPES]

        has_pres = bool(pres_members)
        has_frame = bool(frame_members)

        if has_pres:
            stats["pair_ids_with_preservation"] += 1
        if has_frame:
            stats["pair_ids_with_frame"] += 1

        if not has_pres or not has_frame:
            reason = (
                "no_preservation_candidates" if not has_pres
                else "no_frame_candidates"
            )
            stats["skipped_pair_ids"].append({"pair_id": pair_id, "reason": reason})
            continue

        stats["pair_ids_used"] += 1

        # Cartesian product of (preservation, frame) siblings
        pairs = list(product(pres_members, frame_members))
        if max_pairs_per_pair_id and max_pairs_per_pair_id > 0:
            pairs = pairs[:max_pairs_per_pair_id]

        for pres, frame in pairs:
            rec = build_contrastive_record(pair_id, pres, frame, global_index)
            output.append(rec)
            frame_it_counter[frame["intervention_type"]] += 1
            pres_it_counter[pres["intervention_type"]] += 1
            global_index += 1

    stats["output_pair_count"] = len(output)
    stats["pairs_by_frame_intervention_type"] = dict(
        sorted(frame_it_counter.items(), key=lambda x: -x[1])
    )
    stats["pairs_by_preservation_intervention_type"] = dict(
        sorted(pres_it_counter.items(), key=lambda x: -x[1])
    )
    return output, stats


# ---------------------------------------------------------------------------
# Summary renderers
# ---------------------------------------------------------------------------

def build_summary(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        **stats,
        "leakage_statement": (
            "All output records are constructed from controlled training data only. "
            "Stage15 OOD records (data/stage15_slot_sensitivity_probe.jsonl) are NOT "
            "used, referenced, or copied. No OOD group labels are applied to output records."
        ),
        "recommended_downstream_use": (
            "Diagnostic pairwise ranking loss or auxiliary contrastive evaluator. "
            "Not a direct Stage22-B gate. Stage22-B gate remains rejected until "
            "frame_violation_prob OOD ranking criterion passes (frame mean > surface mean "
            "by >= 0.10 on Stage15 probe)."
        ),
    }


def render_summary_md(stats: dict[str, Any], args: argparse.Namespace) -> str:
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"{'#' * level} {text}\n")

    def p(text: str) -> None:
        lines.append(text + "\n")

    def table(headers: list[str], rows: list[list[str]]) -> None:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    h(1, "Stage22-A4b pair-group contrastive frame dataset — generation summary")
    p(f"**Input:** `{args.controlled_data}`")
    p(f"**Output:** `{args.output_jsonl}`")
    p(
        "**Data-leakage constraint:** Stage15 OOD records are NOT used. "
        "All output records are constructed from controlled data exclusively."
    )
    lines.append("")

    h(2, "Record counts")
    table(
        ["Metric", "Count"],
        [
            ["Input controlled records", str(stats["input_records"])],
            ["Unique pair_ids (after time_swap filter)", str(stats["pair_ids_total"])],
            ["pair_ids with preservation candidates", str(stats["pair_ids_with_preservation"])],
            ["pair_ids with frame candidates", str(stats["pair_ids_with_frame"])],
            ["pair_ids used (both sides present)", str(stats["pair_ids_used"])],
            ["Output contrastive pair records", str(stats["output_pair_count"])],
        ],
    )

    h(2, "Output pairs by frame_intervention_type")
    fi = stats.get("pairs_by_frame_intervention_type", {})
    if fi:
        table(
            ["frame_intervention_type", "pair count"],
            [[k, str(v)] for k, v in fi.items()],
        )
    else:
        p("_No pairs generated._")

    h(2, "Output pairs by preservation_intervention_type")
    pi = stats.get("pairs_by_preservation_intervention_type", {})
    if pi:
        table(
            ["preservation_intervention_type", "pair count"],
            [[k, str(v)] for k, v in pi.items()],
        )

    h(2, "Skipped pair_ids")
    skipped = stats.get("skipped_pair_ids", [])
    if skipped:
        table(
            ["pair_id", "reason"],
            [[s["pair_id"], s["reason"]] for s in skipped],
        )
    else:
        p("_No pair_ids skipped — all had both preservation and frame candidates._")

    h(2, "Leakage statement")
    p(
        "All output records are constructed from controlled training data only. "
        "Stage15 OOD records (`data/stage15_slot_sensitivity_probe.jsonl`) are NOT "
        "used, referenced, or copied. No OOD group labels are applied to output records. "
        "Each output record carries `leakage_note = \"constructed_from_controlled_data_only\"`."
    )

    h(2, "Recommended downstream use")
    p(
        "This dataset is a **diagnostic contrastive dataset**, not a final training set. "
        "Recommended use: pairwise ranking loss or auxiliary contrastive evaluator that "
        "trains a head to score `frame_evidence` higher on frame_violation_prob than "
        "`preservation_evidence` for the same `pair_id`."
    )
    p(
        "**Stage22-B positive recovery gate remains rejected** until frame_violation_prob "
        "OOD ranking passes: frame group mean > surface_control / temporal_erased mean "
        "by ≥ 0.10 on the Stage15 probe (evaluation only, not training)."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stage22-A4b: build pair-group contrastive frame dataset from controlled data."
        ),
    )
    p.add_argument(
        "--controlled-data",
        required=True,
        help="Path to controlled training JSONL.",
    )
    p.add_argument(
        "--output-jsonl",
        required=True,
        help="Path to write the output contrastive JSONL.",
    )
    p.add_argument(
        "--output-summary-json",
        required=True,
        help="Path to write the JSON generation summary.",
    )
    p.add_argument(
        "--output-summary-md",
        required=True,
        help="Path to write the Markdown generation summary.",
    )
    p.add_argument(
        "--exclude-time-swap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude records with intervention_type=time_swap (default: true).",
    )
    p.add_argument(
        "--max-pairs-per-pair-id",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Cap the number of contrastive pairs per pair_id. "
            "0 means no limit (default: 0)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    controlled_path = Path(args.controlled_data)
    if not controlled_path.exists():
        print(f"ERROR: controlled data not found: {controlled_path}", file=sys.stderr)
        return 1

    print(f"Loading controlled data: {controlled_path}")
    records = load_jsonl(controlled_path)
    print(f"  {len(records)} records loaded")

    print("Generating contrastive pairs...")
    output, stats = generate_contrastive_pairs(
        records,
        exclude_time_swap=args.exclude_time_swap,
        max_pairs_per_pair_id=args.max_pairs_per_pair_id,
    )
    print(f"  {stats['output_pair_count']} contrastive pairs from "
          f"{stats['pair_ids_used']} pair_ids")

    out_jsonl = Path(args.output_jsonl)
    write_jsonl(out_jsonl, output)
    print(f"Output JSONL written: {out_jsonl}")

    summary = build_summary(stats)
    out_json = Path(args.output_summary_json)
    write_json(out_json, summary)
    print(f"Summary JSON written: {out_json}")

    md_text = render_summary_md(stats, args)
    out_md = Path(args.output_summary_md)
    write_md(out_md, md_text)
    print(f"Summary Markdown written: {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
