"""Stage28-C: Location Manual Audit Sheet builder.

Reads the Stage28-B v2 location validity audit JSONL and produces a compact
manual audit sheet (CSV + JSONL) for classifying location_swap cases as:
  clean_invalid_entailment / ambiguous / invalid_artifact / unknown

This is NOT a model-training stage. product_power=0.90 remains the Stage27 baseline.

Usage:
    python scripts/build_stage28c_location_manual_audit_sheet.py \\
        --input-jsonl   /kaggle/working/stage28b_location_validity_audit_v2.jsonl \\
        --output-csv    /kaggle/working/stage28c_location_manual_audit_sheet.csv \\
        --output-jsonl  /kaggle/working/stage28c_location_manual_audit_sheet.jsonl \\
        --output-md     /kaggle/working/stage28c_location_manual_audit_sheet_summary.md \\
        --output-json   /kaggle/working/stage28c_location_manual_audit_sheet_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AUDIT_QUESTION = (
    "Does the location substitution genuinely break entitlement, or is the case "
    "ambiguous because of aliasing, containment, hierarchy, background context, "
    "or partial support?"
)

_AUDIT_FLAG_NAMES = (
    "possible_alias",
    "possible_hierarchical_location",
    "possible_containment_relation",
    "multiple_locations_in_evidence",
    "location_is_background_context",
    "claim_location_not_predicate_argument",
    "partial_support_possible",
    "surface_only_swap",
    "needs_external_knowledge",
)

# Column order for CSV / JSONL output
_COLUMNS = (
    "audit_row_id",
    "split",
    "source_id",
    "stage28b_id",
    "claim",
    "evidence",
    "normalized_label",
    "normalized_intervention",
    "diagnostic_bucket",
    "expected_failure_mode",
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "primary_failure_type",
    "audit_validity",
    "audit_reason",
    "audit_flags_possible_alias",
    "audit_flags_possible_hierarchical_location",
    "audit_flags_possible_containment_relation",
    "audit_flags_multiple_locations_in_evidence",
    "audit_flags_location_is_background_context",
    "audit_flags_claim_location_not_predicate_argument",
    "audit_flags_partial_support_possible",
    "audit_flags_surface_only_swap",
    "audit_flags_needs_external_knowledge",
    "manual_notes",
    "audit_question",
)

_AUDIT_EDITABLE_COLS = frozenset((
    "audit_validity", "audit_reason",
    "audit_flags_possible_alias",
    "audit_flags_possible_hierarchical_location",
    "audit_flags_possible_containment_relation",
    "audit_flags_multiple_locations_in_evidence",
    "audit_flags_location_is_background_context",
    "audit_flags_claim_location_not_predicate_argument",
    "audit_flags_partial_support_possible",
    "audit_flags_surface_only_swap",
    "audit_flags_needs_external_knowledge",
    "manual_notes",
))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _str_or_none(v: Any) -> "str | None":
    return str(v) if v is not None else None


def _str_for_csv(v: Any) -> str:
    """Return empty string for None/null values in CSV."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    return str(v)


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    errors = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors += 1
                if errors <= 5:
                    print(
                        f"[S28C] WARNING: parse error at line {lineno}: {exc}",
                        file=sys.stderr,
                    )
    print(
        f"[S28C] Read {len(records)} records ({errors} parse errors) from {path}.",
        file=sys.stderr,
    )
    return records


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _extract_from_record(rec: dict) -> dict[str, Any]:
    """Extract all fields needed for an audit row from one Stage28-B record."""
    raw = rec.get("raw_record") or {}
    audit_flags = rec.get("audit_flags") or {}

    # Sub-label fields come from raw_record
    frame_compat   = _safe_get(raw, "frame_compatible_label")
    pred_covered   = _safe_get(raw, "predicate_covered_label")
    sufficiency    = _safe_get(raw, "sufficiency_label")
    primary_fail   = _safe_get(raw, "primary_failure_type")

    # Propagate heuristic flags from Stage28-B if present, otherwise null
    flag_vals: dict[str, Any] = {}
    for flag in _AUDIT_FLAG_NAMES:
        v = audit_flags.get(flag)
        flag_vals[f"audit_flags_{flag}"] = v

    return {
        "source_id":            rec.get("source_id"),
        "stage28b_id":          rec.get("stage28b_id"),
        "claim":                rec.get("claim"),
        "evidence":             rec.get("evidence"),
        "normalized_label":     rec.get("normalized_label"),
        "normalized_intervention": rec.get("normalized_intervention"),
        "diagnostic_bucket":    rec.get("diagnostic_bucket"),
        "expected_failure_mode": rec.get("expected_failure_mode"),
        "frame_compatible_label": _str_or_none(frame_compat),
        "predicate_covered_label": _str_or_none(pred_covered),
        "sufficiency_label":    _str_or_none(sufficiency),
        "primary_failure_type": _str_or_none(primary_fail),
        "audit_validity":       None,  # to be filled by manual audit
        "audit_reason":         None,
        "manual_notes":         None,
        "audit_question":       _AUDIT_QUESTION,
        **flag_vals,
    }


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _stratified_location_sample(
    location_recs: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Sample n records from location_swap_audit, stratifying by source_id prefix."""
    gen_fact = [r for r in location_recs
                if str(r.get("source_id") or "").startswith("generated_fact_")]
    named    = [r for r in location_recs
                if not str(r.get("source_id") or "").startswith("generated_fact_")]

    rng.shuffle(gen_fact)
    rng.shuffle(named)

    if not gen_fact:
        return named[:n]
    if not named:
        return gen_fact[:n]

    half = n // 2
    gen_take   = min(half, len(gen_fact))
    named_take = min(half, len(named))

    # Fill any shortfall from the other stratum
    gen_take   = min(gen_take   + max(0, half - named_take), len(gen_fact))
    named_take = min(named_take + max(0, half - gen_take),   len(named))

    return (gen_fact[:gen_take] + named[:named_take])[:n]


def _control_sample(
    control_recs: list[dict],
    preferred_source_ids: frozenset,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Sample n controls, preferring those whose source_id is in preferred_source_ids."""
    preferred = [r for r in control_recs
                 if r.get("source_id") in preferred_source_ids]
    others    = [r for r in control_recs
                 if r.get("source_id") not in preferred_source_ids]
    rng.shuffle(preferred)
    rng.shuffle(others)
    combined = preferred + others
    return combined[:n]


# ---------------------------------------------------------------------------
# Output row builder
# ---------------------------------------------------------------------------

def _build_audit_row(
    seq: int,
    split: str,
    rec: dict,
) -> dict[str, Any]:
    fields = _extract_from_record(rec)
    row: dict[str, Any] = {"audit_row_id": f"s28c_{seq:04d}", "split": split}
    for col in _COLUMNS:
        if col in ("audit_row_id", "split"):
            continue
        row[col] = fields.get(col)
    return row


# ---------------------------------------------------------------------------
# CSV / JSONL writers
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(_COLUMNS),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            csv_row = {col: _str_for_csv(row.get(col)) for col in _COLUMNS}
            writer.writerow(csv_row)
    print(f"[S28C] Wrote {len(rows)} rows to {path}.", file=sys.stderr)


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            jsonl_row = {col: row.get(col) for col in _COLUMNS}
            f.write(json.dumps(jsonl_row, ensure_ascii=False) + "\n")
    print(f"[S28C] Wrote {len(rows)} records to {path}.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(
    args: argparse.Namespace,
    rows: list[dict],
    loc_sample: list[dict],
    ctrl_sample: list[dict],
    selected_loc_source_ids: frozenset,
) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    for r in rows:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    gen_fact_count = sum(
        1 for r in loc_sample
        if str(r.get("source_id") or "").startswith("generated_fact_")
    )
    named_count = len(loc_sample) - gen_fact_count

    ctrl_overlap = sum(
        1 for r in ctrl_sample
        if r.get("source_id") in selected_loc_source_ids
    )
    ctrl_cov = (
        round(ctrl_overlap / len(selected_loc_source_ids), 4)
        if selected_loc_source_ids else 0.0
    )

    return {
        "stage": "Stage28-C",
        "objective": (
            "Build a compact manual audit sheet from Stage28-B location validity audit "
            "JSONL to classify location_swap cases as: clean_invalid_entailment / "
            "ambiguous / invalid_artifact / unknown. "
            "product_power=0.90 remains the Stage27 controlled-setting baseline."
        ),
        "input_jsonl":  str(args.input_jsonl),
        "output_csv":   str(args.output_csv),
        "output_jsonl": str(args.output_jsonl),
        "output_md":    str(args.output_md),
        "output_json":  str(args.output_json),
        "seed":         args.seed,
        "location_sample_size_requested":  args.location_sample_size,
        "role_reference_size_requested":   args.role_reference_size,
        "control_reference_size_requested": args.control_reference_size,
        "total_audit_rows": len(rows),
        "split_counts": split_counts,
        "selected_location_source_count":         len(selected_loc_source_ids),
        "selected_location_generated_fact_count": gen_fact_count,
        "selected_location_named_scenario_count": named_count,
        "selected_location_control_overlap_count": ctrl_overlap,
        "selected_location_control_coverage": ctrl_cov,
        "recommended_manual_label_schema": {
            "clean_invalid_entailment": (
                "The location substitution genuinely breaks entailment; "
                "NOT_ENTITLED is a valid label."
            ),
            "ambiguous": (
                "The case may be ambiguous due to aliasing, containment, hierarchy, "
                "background context, partial support, or unclear predicate argument structure."
            ),
            "invalid_artifact": (
                "The generated example is malformed or the location substitution does "
                "not actually test the intended boundary."
            ),
            "unknown": "Insufficient information to judge.",
        },
        "decision_thresholds": {
            "clean_rate_ge_0.80": (
                "Treat location_swap as a valid hard diagnostic axis and proceed to "
                "Stage28-D specialist design."
            ),
            "clean_rate_0.50_to_0.79": (
                "Split clean_location_swap from ambiguous/invalid cases before using "
                "this axis for evaluation."
            ),
            "clean_rate_lt_0.50": (
                "Treat location_swap as construction-artifact-prone and consider "
                "separating it from main validation."
            ),
        },
        "limitations": [
            "Audit sheet contains a sample of up to "
            f"{args.location_sample_size} location_swap records. "
            "Results may not generalize to the full set.",
            "Stratification by generated_fact_ prefix is a heuristic. "
            "Records without this prefix may still be synthetic.",
            "Heuristic audit_flags from Stage28-B are carried forward but must not "
            "be treated as ground-truth validity labels.",
            "Control reference rows are for context only; they do not require labeling.",
            "Role reference rows are for axis comparison only; they do not require labeling.",
            "product_power=0.90 remains the Stage27 controlled-setting baseline. "
            "Nothing in Stage28-C changes any model configuration.",
        ],
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt_f(v: Any, fmt: str = ".4f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def _build_markdown(s: dict[str, Any]) -> str:
    sc = s.get("split_counts", {})
    schema = s.get("recommended_manual_label_schema", {})
    thresholds = s.get("decision_thresholds", {})

    schema_rows = "\n".join(
        f"| `{k}` | {v} |" for k, v in schema.items()
    )
    threshold_rows = "\n".join(
        f"| {k} | {v} |" for k, v in thresholds.items()
    )
    limitation_lines = "\n".join(
        f"- {lim}" for lim in s.get("limitations", [])
    )

    return f"""\
# Stage28-C Location Manual Audit Sheet Summary

## Objective

Build a compact manual audit sheet to classify location_swap cases from Stage28-B.
The goal is to estimate how many location_swap cases are genuinely clean
(NOT_ENTITLED is a valid label) versus ambiguous, invalid artifacts, or unknown.

This is a manual audit preparation stage, not a model-training stage.
`product_power=0.90` remains the Stage27 controlled-setting baseline.

## Input

| Parameter | Value |
|---|---|
| input_jsonl | {s.get('input_jsonl', 'N/A')} |
| seed | {s.get('seed', 'N/A')} |
| location_sample_size_requested | {s.get('location_sample_size_requested', 'N/A')} |
| role_reference_size_requested | {s.get('role_reference_size_requested', 'N/A')} |
| control_reference_size_requested | {s.get('control_reference_size_requested', 'N/A')} |

## Sampling Design

**Location swap audit sample** (`split=location_audit`):
- Stratified by source_id prefix:
  - `generated_fact_*` records: {s.get('selected_location_generated_fact_count', 0)}
  - Named scenario records: {s.get('selected_location_named_scenario_count', 0)}
- Total location_audit rows: {sc.get('location_audit', 0)}

**Role swap reference** (`split=role_reference`):
- Sampled from `role_swap_reference` bucket for axis comparison.
- Total role_reference rows: {sc.get('role_reference', 0)}

**Location control reference** (`split=location_control`):
- Sampled from `location_controls` bucket, preferring matched source_ids.
- Control source overlap with location sample: {s.get('selected_location_control_overlap_count', 0)} /
  {s.get('selected_location_source_count', 0)}
  (coverage = {_fmt_f(s.get('selected_location_control_coverage'))})
- Total location_control rows: {sc.get('location_control', 0)}

## Output Files

| File | Purpose |
|---|---|
| {s.get('output_csv', 'N/A')} | Manual audit sheet (CSV, open in Excel/Sheets) |
| {s.get('output_jsonl', 'N/A')} | Same data as JSONL for programmatic access |
| {s.get('output_md', 'N/A')} | This summary |
| {s.get('output_json', 'N/A')} | Summary metadata JSON |

## Split Counts

| split | rows |
|---|---:|
| location_audit | {sc.get('location_audit', 0)} |
| role_reference | {sc.get('role_reference', 0)} |
| location_control | {sc.get('location_control', 0)} |
| **total** | **{s.get('total_audit_rows', 0)}** |

## Manual Label Schema

The auditor should fill in `audit_validity` for each `location_audit` row.
Role reference and control rows are for context and do not require labeling.

| label | description |
|---|---|
{schema_rows}

The `audit_question` column in each row contains the canonical question to ask.

## Decision Thresholds

After audit is complete, count the `clean_invalid_entailment` rate among
`location_audit` rows and apply:

| threshold | decision |
|---|---|
{threshold_rows}

## How to Use the Audit Sheet

1. Open `{s.get('output_csv', 'the CSV file')}` in a spreadsheet tool.
2. For each row where `split=location_audit`:
   a. Read `claim` and `evidence`.
   b. Consider the `audit_question`.
   c. Fill in `audit_validity` with one of:
      `clean_invalid_entailment`, `ambiguous`, `invalid_artifact`, `unknown`.
   d. Optionally fill `audit_reason` and any `audit_flags_*` columns.
   e. Add free-text notes in `manual_notes`.
3. Rows where `split=role_reference` or `split=location_control` are reference
   only -- they show the same claim/evidence under a different or no intervention.
   They do not need to be labeled.
4. Save the CSV and re-import to compute clean_invalid_entailment_rate.

## Limitations

{limitation_lines}
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage28-C: Location Manual Audit Sheet builder. "
            "Reads Stage28-B location validity audit JSONL and produces a compact "
            "manual audit sheet (CSV + JSONL) for classifying location_swap cases. "
            "product_power=0.90 remains the stable Stage27 baseline."
        )
    )
    p.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("/kaggle/working/stage28b_location_validity_audit_v2.jsonl"),
        help="Stage28-B v2 audit JSONL (default: /kaggle/working/stage28b_location_validity_audit_v2.jsonl).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/kaggle/working/stage28c_location_manual_audit_sheet.csv"),
        help="Output CSV audit sheet path.",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("/kaggle/working/stage28c_location_manual_audit_sheet.jsonl"),
        help="Output JSONL audit sheet path.",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=Path("/kaggle/working/stage28c_location_manual_audit_sheet_summary.md"),
        help="Output markdown summary path.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/kaggle/working/stage28c_location_manual_audit_sheet_summary.json"),
        help="Output JSON summary path.",
    )
    p.add_argument(
        "--location-sample-size",
        type=int,
        default=80,
        dest="location_sample_size",
        help="Number of location_swap records to include in audit (default: 80).",
    )
    p.add_argument(
        "--role-reference-size",
        type=int,
        default=20,
        dest="role_reference_size",
        help="Number of role_swap records to include as reference (default: 20).",
    )
    p.add_argument(
        "--control-reference-size",
        type=int,
        default=20,
        dest="control_reference_size",
        help="Number of location_controls records to include as reference (default: 20).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=28,
        help="Random seed for deterministic sampling (default: 28).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    args = parse_args(argv)

    if not args.input_jsonl.exists():
        print(
            f"[S28C] ERROR: input JSONL not found: {args.input_jsonl}",
            file=sys.stderr,
        )
        return 1

    all_records = _read_jsonl(args.input_jsonl)
    rng = random.Random(args.seed)

    # Partition by bucket
    loc_recs  = [r for r in all_records if r.get("diagnostic_bucket") == "location_swap_audit"]
    role_recs = [r for r in all_records if r.get("diagnostic_bucket") == "role_swap_reference"]
    ctrl_recs = [r for r in all_records if r.get("diagnostic_bucket") == "location_controls"]

    print(
        f"[S28C] bucket sizes: location={len(loc_recs)}, "
        f"role={len(role_recs)}, controls={len(ctrl_recs)}.",
        file=sys.stderr,
    )

    # Sample location audit records (stratified)
    loc_sample = _stratified_location_sample(loc_recs, args.location_sample_size, rng)
    selected_loc_source_ids = frozenset(r.get("source_id") for r in loc_sample)

    # Sample role reference records
    rng.shuffle(role_recs)
    role_sample = role_recs[: args.role_reference_size]

    # Sample control reference records (prefer matched source_ids)
    ctrl_sample = _control_sample(
        ctrl_recs, selected_loc_source_ids, args.control_reference_size, rng
    )

    print(
        f"[S28C] samples: location={len(loc_sample)}, "
        f"role={len(role_sample)}, controls={len(ctrl_sample)}.",
        file=sys.stderr,
    )

    # Build output rows
    rows: list[dict] = []
    seq = 1
    for rec in loc_sample:
        rows.append(_build_audit_row(seq, "location_audit", rec))
        seq += 1
    for rec in role_sample:
        rows.append(_build_audit_row(seq, "role_reference", rec))
        seq += 1
    for rec in ctrl_sample:
        rows.append(_build_audit_row(seq, "location_control", rec))
        seq += 1

    # Build summary
    summary = _build_summary(args, rows, loc_sample, ctrl_sample, selected_loc_source_ids)

    # Write outputs
    for out_path in (
        args.output_csv, args.output_jsonl, args.output_md, args.output_json
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(rows, args.output_csv)
    _write_jsonl(rows, args.output_jsonl)

    md = _build_markdown(summary)
    args.output_md.write_text(md, encoding="utf-8")
    print(f"[S28C] Wrote: {args.output_md}", file=sys.stderr)

    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[S28C] Wrote: {args.output_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
