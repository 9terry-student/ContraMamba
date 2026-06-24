"""Stage22-A4d: OOD-matched frame-vs-preservation pair dataset builder.

Constructs a controlled-data-only pair dataset whose preservation anchors mirror
the Stage15 OOD failure structure (surface_like / temporal_erased_like) instead of
the generic A4b/A4c none/paraphrase anchors.

Why this is needed
------------------
Stage22-A4c pair contrastive loss (frame_swap > none/paraphrase) failed the Stage15
OOD ranking criterion: `frame_violation_prob` for surface_control / temporal_erased
remained >= frame_location / frame_role mismatch. The A4c pairs are not sufficiently
aligned with the OOD structure — the model can distinguish A4c pairs via base-pair
lexical shortcuts without learning the within-OOD discrimination the probe tests.

A4d constructs pairs whose preservation side more closely mirrors OOD surface_control
and temporal_erased groups:
  - surface_like_preservation: paraphrase (or none) support-safe evidence
  - temporal_erased_like_preservation: support-safe evidence with temporal phrase removed

Frame side mirrors OOD frame_location / frame_role groups:
  - frame_location_like: location_swap record
  - frame_role_like: role_swap record

Data-leakage contract
---------------------
- Stage15 OOD records (data/stage15_slot_sensitivity_probe.jsonl) are NOT read,
  referenced, or used anywhere in this script.
- All output records are derived exclusively from controlled training data.
- Generic construction-type labels (surface_like, temporal_erased_like,
  frame_location_like, frame_role_like) describe construction method, NOT Stage15
  evaluation group membership.
- Output carries leakage_note = "constructed_from_controlled_data_only_no_stage15_records".
- This output is a DIAGNOSTIC PAIR DATASET, not a final training set.
  Stage22-B gate remains rejected until OOD ranking validation passes.

Usage
-----
    python scripts/build_stage22a4d_ood_matched_frame_pairs.py \\
        --controlled-data  data/controlled_v5_seed.jsonl \\
        --output-jsonl     data/stage22a4d_ood_matched_frame_pairs.jsonl \\
        --output-summary-json results/stage22a4d_ood_matched_summary.json \\
        --output-summary-md   results/stage22a4d_ood_matched_summary.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------

_SUPPORT_SAFE_PRES_TYPES: frozenset[str] = frozenset({"none", "paraphrase"})
_FRAME_LOCATION_TYPES: frozenset[str] = frozenset({"location_swap"})
_FRAME_ROLE_TYPES: frozenset[str] = frozenset({"role_swap"})
_FRAME_CANDIDATE_TYPES: frozenset[str] = _FRAME_LOCATION_TYPES | _FRAME_ROLE_TYPES

TARGET = "frame_more_violating_than_ood_matched_preservation"
SOURCE_TAG = "controlled_ood_matched_pair_builder"
LEAKAGE_NOTE = "constructed_from_controlled_data_only_no_stage15_records"

# Optional auxiliary label fields copied from source records when present
_AUX_LABEL_FIELDS = (
    "frame_compatible_label",
    "sufficiency_label",
    "predicate_covered_label",
    "polarity_label",
    "primary_failure_type",
)

# ---------------------------------------------------------------------------
# Temporal phrase erasure (mirrors Stage15 erased variant construction logic;
# uses only a conservative "during <phrase>" pattern on EVIDENCE text only)
# ---------------------------------------------------------------------------

# Weekdays / months — same ordering logic as create_stage15_slot_sensitivity_probe.py
_WEEKDAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
)
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_TEMPORAL_VALUES = tuple(
    sorted(set(_WEEKDAYS + _MONTHS), key=len, reverse=True)
)
# Match " during <weekday|month>" or " in/on <weekday|month>"
_TEMPORAL_PATTERN = re.compile(
    r"\s+(?:during|in|on)\s+(?:" + "|".join(re.escape(v) for v in _TEMPORAL_VALUES) + r")\b"
)


def _clean_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    return text


def _erase_temporal_phrase(text: str) -> str | None:
    """Remove the first temporal phrase from text. Return None if none found."""
    erased, count = _TEMPORAL_PATTERN.subn("", text, count=1)
    if count == 0:
        return None
    return _clean_spacing(erased)


# ---------------------------------------------------------------------------
# Support-safe anchor check
# ---------------------------------------------------------------------------

def _is_support_safe(rec: dict[str, Any]) -> bool:
    """True when the record is a conservative SUPPORT-safe preservation anchor."""
    if rec.get("intervention_type") not in _SUPPORT_SAFE_PRES_TYPES:
        return False
    if rec.get("final_label") != "SUPPORT":
        return False
    fc = rec.get("frame_compatible_label")
    if fc is not None and int(fc) != 1:
        return False
    suf = rec.get("sufficiency_label")
    if suf is not None and int(suf) != 1:
        return False
    pred = rec.get("predicate_covered_label")
    if pred is not None and int(pred) != 1:
        return False
    pol = rec.get("polarity_label")
    if pol is not None and pol != "SUPPORT":
        return False
    return True


def _is_frame_candidate(rec: dict[str, Any]) -> bool:
    """True when the record is a valid frame-side candidate."""
    if rec.get("intervention_type") not in _FRAME_CANDIDATE_TYPES:
        return False
    fl = rec.get("final_label")
    if fl is not None and fl != "NOT_ENTITLED":
        return False
    pft = rec.get("primary_failure_type")
    if pft is not None and pft != "frame":
        return False
    # Require sufficiency_label == 1 when present (evidence still sufficient for base claim)
    suf = rec.get("sufficiency_label")
    if suf is not None and int(suf) != 1:
        return False
    return True


# ---------------------------------------------------------------------------
# Label copying helpers
# ---------------------------------------------------------------------------

def _copy_pres_labels(rec: dict[str, Any]) -> dict[str, Any]:
    return {f: rec[f] for f in _AUX_LABEL_FIELDS if f in rec}


def _copy_frame_labels(rec: dict[str, Any]) -> dict[str, Any]:
    return {f"frame_{f}": rec[f] for f in _AUX_LABEL_FIELDS if f in rec}


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def _frame_construction_type(intervention_type: str) -> str:
    if intervention_type in _FRAME_LOCATION_TYPES:
        return "frame_location_like"
    if intervention_type in _FRAME_ROLE_TYPES:
        return "frame_role_like"
    return "frame_other"


def _build_pair(
    pair_id: str,
    pres_rec: dict[str, Any],
    frame_rec: dict[str, Any],
    pres_evidence_override: str | None,
    pres_construction_type: str,
    index: int,
) -> dict[str, Any]:
    pres_it = pres_rec.get("intervention_type", "")
    frame_it = frame_rec.get("intervention_type", "")
    pres_evidence = pres_evidence_override if pres_evidence_override is not None else pres_rec["evidence"]
    cid = (
        f"stage22a4d__{pair_id}"
        f"__pres_{pres_construction_type}"
        f"__frame_{_frame_construction_type(frame_it)}"
        f"__{index:04d}"
    )
    rec: dict[str, Any] = {
        "contrastive_id": cid,
        "pair_id": pair_id,
        "claim": pres_rec["claim"],
        "preservation_evidence": pres_evidence,
        "frame_evidence": frame_rec["evidence"],
        "preservation_construction_type": pres_construction_type,
        "frame_construction_type": _frame_construction_type(frame_it),
        "preservation_source_intervention_type": pres_it,
        "frame_source_intervention_type": frame_it,
        "preservation_final_label": pres_rec.get("final_label", ""),
        "frame_final_label": frame_rec.get("final_label", ""),
        # Contrastive semantics
        "preservation_should_score_low_frame_violation": True,
        "frame_should_score_high_frame_violation": True,
        "target": TARGET,
        # Provenance / leakage
        "source": SOURCE_TAG,
        "leakage_note": LEAKAGE_NOTE,
        "preservation_source_id": pres_rec.get("id", ""),
        "frame_source_id": frame_rec.get("id", ""),
    }
    # Auxiliary labels for v5.encode_records compatibility
    rec.update(_copy_pres_labels(pres_rec))
    rec.update(_copy_frame_labels(frame_rec))
    return rec


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate_ood_matched_pairs(
    records: list[dict[str, Any]],
    *,
    exclude_time_swap: bool = True,
    include_temporal_erased_like: bool = True,
    max_pairs_per_pair_id: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build OOD-matched frame-vs-preservation pairs from controlled records.

    Returns (output_records, stats).
    """
    # Group by pair_id, optionally filtering time_swap
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        pid = rec.get("pair_id")
        if pid is None:
            continue
        if exclude_time_swap and rec.get("intervention_type") == "time_swap":
            continue
        by_pair[pid].append(rec)

    stats: dict[str, Any] = {
        "input_records": len(records),
        "pair_ids_total": len(by_pair),
        "pair_ids_with_support_safe_anchor": 0,
        "pair_ids_with_location_swap": 0,
        "pair_ids_with_role_swap": 0,
        "pair_ids_used": 0,
        "skipped_pair_ids": [],
        "output_pair_count": 0,
        "count_by_preservation_construction_type": {},
        "count_by_frame_construction_type": {},
        "temporal_erased_like_success": 0,
        "temporal_erased_like_skipped_no_temporal_phrase": 0,
    }

    pres_type_counter: Counter[str] = Counter()
    frame_type_counter: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    global_index = 0

    for pair_id in sorted(by_pair):
        group = by_pair[pair_id]

        # Classify group members
        safe_pres = [r for r in group if _is_support_safe(r)]
        frame_cands = [r for r in group if _is_frame_candidate(r)]

        has_safe_pres = bool(safe_pres)
        has_loc = any(r.get("intervention_type") in _FRAME_LOCATION_TYPES for r in frame_cands)
        has_role = any(r.get("intervention_type") in _FRAME_ROLE_TYPES for r in frame_cands)

        if has_safe_pres:
            stats["pair_ids_with_support_safe_anchor"] += 1
        if has_loc:
            stats["pair_ids_with_location_swap"] += 1
        if has_role:
            stats["pair_ids_with_role_swap"] += 1

        if not has_safe_pres or not frame_cands:
            reason = (
                "no_support_safe_preservation_anchor" if not has_safe_pres
                else "no_frame_candidates"
            )
            stats["skipped_pair_ids"].append({"pair_id": pair_id, "reason": reason})
            continue

        stats["pair_ids_used"] += 1

        # Choose the best surface-like preservation anchor:
        # prefer paraphrase > none
        paraphrase_pres = [r for r in safe_pres if r.get("intervention_type") == "paraphrase"]
        none_pres = [r for r in safe_pres if r.get("intervention_type") == "none"]
        surface_pres_candidates = paraphrase_pres if paraphrase_pres else none_pres
        if not surface_pres_candidates:
            stats["skipped_pair_ids"].append({"pair_id": pair_id, "reason": "no_surface_pres_after_filter"})
            stats["pair_ids_used"] -= 1
            continue

        # Surface-like pairs: one surface-like pres × each frame candidate
        surface_pres = surface_pres_candidates[0]
        pair_count = 0
        for frame_rec in frame_cands:
            if max_pairs_per_pair_id and pair_count >= max_pairs_per_pair_id:
                break
            rec = _build_pair(
                pair_id, surface_pres, frame_rec,
                pres_evidence_override=None,
                pres_construction_type="surface_like_preservation",
                index=global_index,
            )
            output.append(rec)
            pres_type_counter["surface_like_preservation"] += 1
            frame_type_counter[_frame_construction_type(frame_rec.get("intervention_type", ""))] += 1
            global_index += 1
            pair_count += 1

        # Temporal-erased-like pairs (if requested)
        if include_temporal_erased_like:
            # Try to find a none/paraphrase pres with a temporal phrase in evidence
            te_anchor: dict[str, Any] | None = None
            te_erased_evidence: str | None = None
            for candidate in (none_pres + paraphrase_pres):
                erased = _erase_temporal_phrase(candidate["evidence"])
                if erased is not None and erased != candidate["evidence"]:
                    te_anchor = candidate
                    te_erased_evidence = erased
                    break

            if te_anchor is None or te_erased_evidence is None:
                stats["temporal_erased_like_skipped_no_temporal_phrase"] += 1
            else:
                stats["temporal_erased_like_success"] += 1
                te_pair_count = 0
                for frame_rec in frame_cands:
                    if max_pairs_per_pair_id and te_pair_count >= max_pairs_per_pair_id:
                        break
                    rec = _build_pair(
                        pair_id, te_anchor, frame_rec,
                        pres_evidence_override=te_erased_evidence,
                        pres_construction_type="temporal_erased_like_preservation",
                        index=global_index,
                    )
                    output.append(rec)
                    pres_type_counter["temporal_erased_like_preservation"] += 1
                    frame_type_counter[_frame_construction_type(frame_rec.get("intervention_type", ""))] += 1
                    global_index += 1
                    te_pair_count += 1

    stats["output_pair_count"] = len(output)
    stats["count_by_preservation_construction_type"] = dict(
        sorted(pres_type_counter.items(), key=lambda x: -x[1])
    )
    stats["count_by_frame_construction_type"] = dict(
        sorted(frame_type_counter.items(), key=lambda x: -x[1])
    )
    return output, stats


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"input file is empty: {path}")
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
# Summary renderers
# ---------------------------------------------------------------------------

def build_summary(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        **stats,
        "leakage_statement": (
            "All output records are constructed from controlled training data only. "
            "Stage15 OOD records (data/stage15_slot_sensitivity_probe.jsonl) are NOT "
            "read, referenced, or copied. No Stage15 evaluation group labels are applied "
            "to output records. Construction-type labels (surface_like, temporal_erased_like, "
            "frame_location_like, frame_role_like) describe the construction method only."
        ),
        "recommended_downstream_use": (
            "Use with the A4c pair contrastive loss (--pair-contrastive-frame-data) "
            "by filtering on preservation_construction_type. "
            "Combine surface_like and temporal_erased_like pairs to expose the model to "
            "the within-pair discrimination the Stage15 OOD probe tests. "
            "Not a Stage22-B gate. Stage22-B gate remains rejected until "
            "frame_violation_prob OOD ranking passes (frame_location/frame_role mean "
            "> surface_control/temporal_erased mean by >= 0.10 on Stage15 probe)."
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

    h(1, "Stage22-A4d OOD-matched frame-vs-preservation pairs — generation summary")
    p(f"**Input:** `{args.controlled_data}`")
    p(f"**Output:** `{args.output_jsonl}`")
    p(
        "**Data-leakage constraint:** Stage15 OOD records are NOT used. "
        "All output records are constructed from controlled data exclusively. "
        "Construction-type labels are not Stage15 evaluation group labels."
    )
    lines.append("")

    h(2, "Record counts")
    table(
        ["Metric", "Count"],
        [
            ["Input controlled records", str(stats["input_records"])],
            ["Unique pair_ids (after time_swap filter)", str(stats["pair_ids_total"])],
            ["pair_ids with support-safe preservation anchor", str(stats["pair_ids_with_support_safe_anchor"])],
            ["pair_ids with location_swap frame candidate", str(stats["pair_ids_with_location_swap"])],
            ["pair_ids with role_swap frame candidate", str(stats["pair_ids_with_role_swap"])],
            ["pair_ids used (both sides present)", str(stats["pair_ids_used"])],
            ["Output pair records", str(stats["output_pair_count"])],
        ],
    )

    h(2, "Output by preservation_construction_type")
    pct = stats.get("count_by_preservation_construction_type", {})
    if pct:
        table(
            ["preservation_construction_type", "count"],
            [[k, str(v)] for k, v in pct.items()],
        )
    else:
        p("_No pairs generated._")

    h(2, "Output by frame_construction_type")
    fct = stats.get("count_by_frame_construction_type", {})
    if fct:
        table(
            ["frame_construction_type", "count"],
            [[k, str(v)] for k, v in fct.items()],
        )

    h(2, "Temporal-erased-like construction")
    table(
        ["Outcome", "Count"],
        [
            ["Temporal phrase found and erased", str(stats["temporal_erased_like_success"])],
            ["Skipped (no temporal phrase in evidence)", str(stats["temporal_erased_like_skipped_no_temporal_phrase"])],
        ],
    )

    h(2, "Skipped pair_ids")
    skipped = stats.get("skipped_pair_ids", [])
    if skipped:
        table(
            ["pair_id", "reason"],
            [[s["pair_id"], s["reason"]] for s in skipped],
        )
    else:
        p("_No pair_ids skipped._")

    h(2, "Leakage statement")
    p(
        "All output records are constructed from controlled training data only. "
        "Stage15 OOD records (`data/stage15_slot_sensitivity_probe.jsonl`) are NOT "
        "read, referenced, or copied. No Stage15 evaluation group labels are applied "
        "to output records. Each output record carries "
        "`leakage_note = \"constructed_from_controlled_data_only_no_stage15_records\"`."
    )

    h(2, "Recommended downstream use")
    p(
        "Use with the existing A4c pair contrastive loss (`--pair-contrastive-frame-data`). "
        "The `surface_like_preservation` pairs mirror Stage15 `surface_control` structure. "
        "The `temporal_erased_like_preservation` pairs mirror Stage15 `temporal_erased` structure. "
        "The `frame_location_like` / `frame_role_like` frame sides mirror the Stage15 frame mismatch groups."
    )
    p(
        "**Stage22-B positive recovery gate remains rejected** until `frame_violation_prob` "
        "OOD ranking passes on the Stage15 probe: frame_location/frame_role mean "
        "> surface_control/temporal_erased mean by ≥ 0.10 (evaluation only, not training)."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stage22-A4d: build OOD-matched frame-vs-preservation pair dataset "
            "from controlled data only. Stage15 OOD records are NOT used."
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
        help="Path to write the output OOD-matched pair JSONL.",
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
        "--include-temporal-erased-like",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate temporal_erased_like_preservation pairs by removing a temporal phrase "
            "from support-safe evidence text (default: true). "
            "Pairs are skipped when no temporal phrase is found."
        ),
    )
    p.add_argument(
        "--max-pairs-per-pair-id",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Cap the number of contrastive pairs per pair_id per construction type. "
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

    print("Generating OOD-matched pairs...")
    output, stats = generate_ood_matched_pairs(
        records,
        exclude_time_swap=args.exclude_time_swap,
        include_temporal_erased_like=args.include_temporal_erased_like,
        max_pairs_per_pair_id=args.max_pairs_per_pair_id,
    )
    print(
        f"  {stats['output_pair_count']} pairs from {stats['pair_ids_used']} pair_ids"
        f" (surface_like={stats['count_by_preservation_construction_type'].get('surface_like_preservation', 0)}"
        f" temporal_erased_like={stats['count_by_preservation_construction_type'].get('temporal_erased_like_preservation', 0)})"
    )

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
