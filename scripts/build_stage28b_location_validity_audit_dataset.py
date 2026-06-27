"""Stage28-B: Location Validity Audit Dataset builder.

Constructs a diagnostic/audit dataset from controlled_v5_v3_without_time_swap.jsonl
to separate clean, ambiguous, and potentially invalid location_swap cases.

This is NOT a training stage. product_power=0.90 remains the stable Stage27 baseline.
The goal is to audit whether location_swap is a valid hard frame-boundary axis before
any specialist gate is designed.

Usage:
    python scripts/build_stage28b_location_validity_audit_dataset.py \\
        --input-jsonl  data/controlled_v5_v3_without_time_swap.jsonl \\
        --output-jsonl data/stage28b_location_validity_audit.jsonl \\
        --output-md    reports/stage28b_location_validity_audit_summary.md \\
        --output-json  reports/stage28b_location_validity_audit_summary.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ID_KEYS      = ("id", "example_id", "uid", "original_id", "source_id")
_SRC_ID_KEYS  = ("source_id", "original_id")
_CLAIM_KEYS   = ("claim", "hypothesis", "sentence2", "text_b")
_EVID_KEYS    = ("evidence", "premise", "sentence1", "text_a")
_LABEL_KEYS   = ("label", "gold_label", "target", "y_true")
_INTERV_KEYS  = ("intervention", "intervention_type", "perturbation", "probe_type")

_LABEL_NORM: dict[str, str] = {
    "support": "SUPPORT", "supports": "SUPPORT", "2": "SUPPORT",
    "not_entitled": "NOT_ENTITLED", "ne": "NOT_ENTITLED",
    "not_enough_info": "NOT_ENTITLED", "1": "NOT_ENTITLED",
    "refute": "REFUTE", "refutes": "REFUTE", "0": "REFUTE",
}

_INTERV_NORM: dict[str, str] = {
    "location_swap": "location_swap",
    "role_swap": "role_swap",
    "predicate_swap": "predicate_swap",
    "entity_swap": "entity_swap",
    "event_swap": "event_swap",
    "title_name_swap": "title_name_swap",
    "evidence_deletion": "evidence_deletion",
    "evidence_truncation": "evidence_truncation",
    "irrelevant_evidence": "irrelevant_evidence",
    "none": "none",
    "paraphrase": "paraphrase",
    "polarity_flip": "polarity_flip",
}

CONTROL_INTERVENTIONS = frozenset(("none", "paraphrase", "polarity_flip"))
MISSING_EVIDENCE_INTERVENTIONS = frozenset(
    ("evidence_deletion", "evidence_truncation", "irrelevant_evidence")
)
OTHER_FRAME_INTERVENTIONS = frozenset(("entity_swap", "event_swap", "title_name_swap"))

BUCKET_ORDER = (
    "location_swap_audit",
    "role_swap_reference",
    "location_controls",
    "role_controls",
    "predicate_contrast",
    "missing_evidence_contrast",
    "other_frame_contrast",
)

BUCKET_META: dict[str, dict[str, str]] = {
    "location_swap_audit": {
        "diagnostic_focus": "location_boundary_audit",
        "expected_failure_mode": "location_false_support",
    },
    "role_swap_reference": {
        "diagnostic_focus": "role_boundary_reference",
        "expected_failure_mode": "role_false_support",
    },
    "location_controls": {
        "diagnostic_focus": "location_control",
        "expected_failure_mode": "control_preservation",
    },
    "role_controls": {
        "diagnostic_focus": "role_control",
        "expected_failure_mode": "control_preservation",
    },
    "predicate_contrast": {
        "diagnostic_focus": "predicate_contrast",
        "expected_failure_mode": "predicate_false_support",
    },
    "missing_evidence_contrast": {
        "diagnostic_focus": "missing_evidence_contrast",
        "expected_failure_mode": "missing_evidence_rejection",
    },
    "other_frame_contrast": {
        "diagnostic_focus": "other_frame_contrast",
        "expected_failure_mode": "other_frame_false_support",
    },
}

AUDIT_FLAG_NAMES = (
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

_AUDIT_QUESTION = (
    "Does the location substitution genuinely break entailment, or is the case "
    "ambiguous because of aliasing, containment, hierarchy, background context, "
    "or partial support?"
)

# Raw record fields to exclude when building the shallow copy (too large / redundant)
_LARGE_FIELD_NAMES = frozenset(
    ("tokens", "token_ids", "input_ids", "attention_mask", "embeddings",
     "features", "context_window")
)

_MAX_RAW_JSON_CHARS = 4_000

# Simple patterns for heuristics (conservative - prefer null over false positive)
_RE_CAP_WORD = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_RE_CITY_STATE = re.compile(r"\b[A-Z][a-zA-Z]{2,},\s*[A-Z][a-zA-Z]{2,}\b")
_RE_COMMA_CAP_LIST = re.compile(
    r"(?:[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)(?:\s*,\s*[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*){2,}"
)


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _safe_get_meta(record: dict, *keys: str) -> Any:
    meta = record.get("metadata")
    if isinstance(meta, dict):
        for k in keys:
            v = meta.get(k)
            if v is not None:
                return v
    return None


def _extract_stable_id(record: dict, fallback_idx: int) -> str:
    for k in _ID_KEYS:
        v = record.get(k)
        if v is not None:
            return str(v)
    v = _safe_get_meta(record, "id", "example_id", "uid")
    if v is not None:
        return str(v)
    return f"__seq_{fallback_idx}__"


def _extract_source_id(record: dict, stable_id: str) -> str:
    for k in _SRC_ID_KEYS:
        v = record.get(k)
        if v is not None:
            return str(v)
    v = _safe_get_meta(record, "original_id", "source_id")
    if v is not None:
        return str(v)
    return stable_id


def _extract_field(record: dict, keys: tuple) -> Any:
    for k in keys:
        v = record.get(k)
        if v is not None:
            return v
    return None


def _normalize_label(raw: Any) -> "str | None":
    if raw is None:
        return None
    return _LABEL_NORM.get(str(raw).strip().lower())


def _normalize_intervention(raw: Any) -> "str | None":
    if raw is None:
        return None
    s = str(raw).strip().lower()
    return _INTERV_NORM.get(s, s)  # keep unknown interventions as-is


def _extract_intervention(record: dict) -> Any:
    for k in _INTERV_KEYS:
        v = record.get(k)
        if v is not None:
            return v
    return _safe_get_meta(record, "intervention", "intervention_type")


def _shallow_copy(record: dict) -> dict:
    """Build a shallow copy, excluding very large fields and truncating strings."""
    out: dict[str, Any] = {}
    for k, v in record.items():
        if k in _LARGE_FIELD_NAMES:
            continue
        if isinstance(v, str) and len(v) > 500:
            out[k] = v[:500] + " ...[truncated]"
        else:
            out[k] = v
    return out


def _raw_record_field(record: dict) -> Any:
    try:
        raw_json = json.dumps(record, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
    if len(raw_json) <= _MAX_RAW_JSON_CHARS:
        return record
    return _shallow_copy(record)


# ---------------------------------------------------------------------------
# Heuristic flags (text-based only, no external knowledge)
# ---------------------------------------------------------------------------

def _heuristic_flags(claim: Any, evidence: Any) -> dict[str, "bool | None"]:
    """Compute heuristic audit flags from claim/evidence text. Conservative: prefer null."""
    flags: dict[str, "bool | None"] = {k: None for k in AUDIT_FLAG_NAMES}

    cl_text = str(claim) if claim is not None else ""
    ev_text = str(evidence) if evidence is not None else ""
    combined = cl_text + " " + ev_text

    # multiple_locations_in_evidence:
    # True if evidence contains 3+ capitalized words separated by commas
    if ev_text:
        if _RE_COMMA_CAP_LIST.search(ev_text):
            flags["multiple_locations_in_evidence"] = True
        else:
            # Count capitalized words; if only 0-1, mark False, else keep null
            cap_words = _RE_CAP_WORD.findall(ev_text)
            if len(cap_words) < 2:
                flags["multiple_locations_in_evidence"] = False

    # possible_hierarchical_location:
    # True if claim or evidence contains "Word, Word" geographic-style patterns
    if cl_text or ev_text:
        if _RE_CITY_STATE.search(combined):
            flags["possible_hierarchical_location"] = True

    # partial_support_possible:
    # True if BOTH claim and evidence each contain >= 2 distinct capitalized spans
    if cl_text and ev_text:
        cl_caps = set(_RE_CAP_WORD.findall(cl_text))
        ev_caps = set(_RE_CAP_WORD.findall(ev_text))
        if len(cl_caps) >= 2 and len(ev_caps) >= 2:
            flags["partial_support_possible"] = True
        elif len(cl_caps) == 0 or len(ev_caps) == 0:
            flags["partial_support_possible"] = False

    # All other flags remain null (require external knowledge or deeper analysis)
    return flags


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------

def _parse_record(raw: dict, idx: int) -> dict[str, Any]:
    stable_id   = _extract_stable_id(raw, idx)
    source_id   = _extract_source_id(raw, stable_id)
    original_id = raw.get("original_id") or _safe_get_meta(raw, "original_id")

    raw_intervention = _extract_intervention(raw)
    norm_intervention = _normalize_intervention(raw_intervention)

    raw_label   = _extract_field(raw, _LABEL_KEYS)
    norm_label  = _normalize_label(raw_label)

    claim    = _extract_field(raw, _CLAIM_KEYS)
    evidence = _extract_field(raw, _EVID_KEYS)

    return {
        "stable_id":            stable_id,
        "source_id":            source_id,
        "original_id":          original_id,
        "claim":                claim,
        "evidence":             evidence,
        "raw_label":            raw_label,
        "normalized_label":     norm_label,
        "raw_intervention":     raw_intervention,
        "normalized_intervention": norm_intervention,
        "metadata":             raw.get("metadata"),
        "_raw":                 raw,
    }


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
                        f"[S28B] WARNING: JSON parse error at line {lineno}: {exc}",
                        file=sys.stderr,
                    )
    print(
        f"[S28B] Read {len(records)} records ({errors} parse errors) from {path}.",
        file=sys.stderr,
    )
    return records


# ---------------------------------------------------------------------------
# Bucket assignment
# ---------------------------------------------------------------------------

def _assign_buckets(
    parsed: list[dict],
    location_source_ids: frozenset,
    role_source_ids: frozenset,
    rng: random.Random,
    max_per_bucket: int,
) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {b: [] for b in BUCKET_ORDER}

    # First-pass: assign each record to at most one bucket
    control_candidates_loc:  list[dict] = []
    control_candidates_role: list[dict] = []
    control_seen: set[str] = set()

    for p in parsed:
        ni = p["normalized_intervention"]

        if ni == "location_swap":
            buckets["location_swap_audit"].append(p)
        elif ni == "role_swap":
            buckets["role_swap_reference"].append(p)
        elif ni == "predicate_swap":
            buckets["predicate_contrast"].append(p)
        elif ni in MISSING_EVIDENCE_INTERVENTIONS:
            buckets["missing_evidence_contrast"].append(p)
        elif ni in OTHER_FRAME_INTERVENTIONS:
            buckets["other_frame_contrast"].append(p)
        elif ni in CONTROL_INTERVENTIONS:
            sid = p["source_id"]
            if sid in location_source_ids:
                control_candidates_loc.append(p)
            if sid in role_source_ids:
                control_candidates_role.append(p)

    # Shuffle and assign controls (may overlap source sets, deduplicate by stable_id)
    rng.shuffle(control_candidates_loc)
    rng.shuffle(control_candidates_role)

    for p in control_candidates_loc[:max_per_bucket]:
        buckets["location_controls"].append(p)
        control_seen.add(p["stable_id"])

    for p in control_candidates_role[:max_per_bucket]:
        if p["stable_id"] not in control_seen:
            buckets["role_controls"].append(p)

    # Shuffle and cap non-priority buckets
    for bucket in BUCKET_ORDER:
        items = buckets[bucket]
        rng.shuffle(items)
        if len(items) > max_per_bucket:
            buckets[bucket] = items[:max_per_bucket]

    return buckets


# ---------------------------------------------------------------------------
# Output record builder
# ---------------------------------------------------------------------------

def _build_output_record(
    parsed: dict,
    bucket: str,
) -> dict[str, Any]:
    meta  = BUCKET_META[bucket]
    is_loc = bucket == "location_swap_audit"

    stable_id = parsed["stable_id"]
    stage28b_id = f"stage28b::{bucket}::{stable_id}"

    if is_loc:
        flags = _heuristic_flags(parsed["claim"], parsed["evidence"])
        audit_status   = "needs_manual_audit"
        audit_validity = None
        audit_reason   = None
        audit_question = _AUDIT_QUESTION
    else:
        flags          = None
        audit_status   = "not_location_audit"
        audit_validity = None
        audit_reason   = None
        audit_question = None

    return {
        "stage28b_id":          stage28b_id,
        "source_id":            parsed["source_id"],
        "original_id":          parsed["original_id"],
        "claim":                parsed["claim"],
        "evidence":             parsed["evidence"],
        "label":                parsed["raw_label"],
        "normalized_label":     parsed["normalized_label"],
        "intervention":         parsed["raw_intervention"],
        "normalized_intervention": parsed["normalized_intervention"],
        "diagnostic_bucket":    bucket,
        "diagnostic_focus":     meta["diagnostic_focus"],
        "expected_failure_mode": meta["expected_failure_mode"],
        "audit_status":         audit_status,
        "audit_validity":       audit_validity,
        "audit_reason":         audit_reason,
        "audit_flags":          flags,
        "audit_question":       audit_question,
        "metadata":             parsed["metadata"],
        "raw_record":           _raw_record_field(parsed["_raw"]),
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _count_dict(items: list, key_fn) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        k = key_fn(item)
        counts[str(k) if k is not None else "null"] += 1
    return dict(sorted(counts.items()))


def _flag_counts(location_records: list[dict]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for flag in AUDIT_FLAG_NAMES:
        t = f = n = 0
        for rec in location_records:
            flags = rec.get("audit_flags") or {}
            v = flags.get(flag)
            if v is True:
                t += 1
            elif v is False:
                f += 1
            else:
                n += 1
        result[flag] = {"true": t, "false": f, "null": n}
    return result


def _build_summary(
    input_path: Path,
    output_path: Path,
    seed: int,
    max_per_bucket: int,
    total_input: int,
    buckets: dict[str, list[dict]],
    output_records: list[dict],
    parsed: list[dict],
    location_source_ids: frozenset,
    role_source_ids: frozenset,
) -> dict[str, Any]:
    # Intervention counts from input
    interv_counts: dict[str, int] = defaultdict(int)
    for p in parsed:
        ni = p["normalized_intervention"] or "null"
        interv_counts[ni] += 1

    # Label counts from output
    label_counts = _count_dict(output_records, lambda r: r.get("normalized_label"))

    # Per-bucket counts
    bucket_counts = {b: len(buckets[b]) for b in BUCKET_ORDER}

    # Per-bucket label distribution
    bucket_label_counts: dict[str, Any] = {}
    for b, recs in buckets.items():
        bucket_label_counts[b] = _count_dict(
            recs, lambda r: _normalize_label(r["_raw"].get("label") or r["_raw"].get("gold_label"))
        )

    # Location audit counts
    loc_recs = output_records  # all output records with bucket=location_swap_audit
    loc_out  = [r for r in output_records if r["diagnostic_bucket"] == "location_swap_audit"]
    n_loc_swap = len(buckets["location_swap_audit"])
    n_audit    = len(loc_out)
    n_needs    = sum(1 for r in loc_out if r["audit_status"] == "needs_manual_audit")

    # Source-ID coverage
    loc_ctrl_parsed = buckets.get("location_controls", [])
    role_ctrl_parsed = buckets.get("role_controls", [])
    n_loc_ctrl_match  = len({p["source_id"] for p in loc_ctrl_parsed} & location_source_ids)
    n_role_ctrl_match = len({p["source_id"] for p in role_ctrl_parsed} & role_source_ids)
    n_loc_src  = len(location_source_ids)
    n_role_src = len(role_source_ids)
    loc_cov  = round(n_loc_ctrl_match / n_loc_src, 4) if n_loc_src > 0 else 0.0
    role_cov = round(n_role_ctrl_match / n_role_src, 4) if n_role_src > 0 else 0.0

    # Recommended next stage
    if n_audit >= 50 and loc_cov >= 0.5:
        next_stage = (
            "Stage28-C manual audit of location validity "
            "followed by clean/ambiguous split"
        )
    elif n_audit >= 50:
        next_stage = (
            "Stage28-C paired location-control expansion "
            "before specialist modeling"
        )
    else:
        next_stage = "Stage28-C location diagnostic data expansion"

    return {
        "stage": "Stage28-B",
        "objective": (
            "Construct a diagnostic/audit dataset separating clean, ambiguous, "
            "and potentially invalid location_swap cases. "
            "product_power=0.90 remains the Stage27 controlled-setting baseline."
        ),
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "seed": seed,
        "max_per_bucket": max_per_bucket,
        "total_input_records": total_input,
        "total_output_records": len(output_records),
        "bucket_counts": bucket_counts,
        "normalized_intervention_counts_input": dict(
            sorted(interv_counts.items(), key=lambda x: -x[1])
        ),
        "normalized_label_counts_output": label_counts,
        "bucket_label_counts": bucket_label_counts,
        "location_audit_counts": {
            "n_location_swap_records": n_loc_swap,
            "n_location_audit_records": n_audit,
            "n_needs_manual_audit": n_needs,
            "n_clean_invalid_entailment": sum(
                1 for r in loc_out if r.get("audit_validity") == "clean_invalid_entailment"
            ),
            "n_ambiguous": sum(
                1 for r in loc_out if r.get("audit_validity") == "ambiguous"
            ),
            "n_invalid_artifact": sum(
                1 for r in loc_out if r.get("audit_validity") == "invalid_artifact"
            ),
            "n_unknown": sum(
                1 for r in loc_out if r.get("audit_validity") == "unknown"
            ),
        },
        "source_id_overlap_stats": {
            "n_location_sources": n_loc_src,
            "n_role_sources": n_role_src,
            "n_location_control_matches": n_loc_ctrl_match,
            "n_role_control_matches": n_role_ctrl_match,
            "location_control_coverage": loc_cov,
            "role_control_coverage": role_cov,
        },
        "heuristic_flag_counts": _flag_counts(loc_out),
        "recommended_next_stage": next_stage,
        "limitations": [
            "Heuristic flags (multiple_locations_in_evidence, possible_hierarchical_location, "
            "partial_support_possible) are text-pattern-based only. They may have false "
            "positives and false negatives. Do not treat as final validity labels.",
            "audit_validity is null for all records at construction time. "
            "Manual review is required to populate clean_invalid_entailment / ambiguous / "
            "invalid_artifact / unknown.",
            "source_id overlap depends on source_id being consistent across intervention "
            "variants. If source_id is absent or equals stable_id, control coverage will "
            "appear lower than the true overlap.",
            "All records come from controlled_v5_v3_without_time_swap.jsonl. "
            "No OOD or time_swap records are included.",
            "location_swap is treated as a plausible valid diagnostic axis. "
            "It has not been removed like time_swap. The validity audit is the mechanism "
            "for deciding whether to clean, retain, or split this axis.",
            "Capping by --max-per-bucket may discard minority-label records from "
            "larger buckets. The shuffle seed is fixed for reproducibility.",
        ],
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt_int(v: Any) -> str:
    return str(v) if v is not None else "N/A"


def _fmt_f(v: Any, fmt: str = ".4f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def _bucket_table_md(summary: dict[str, Any]) -> str:
    bc = summary.get("bucket_counts", {})
    headers = ["bucket", "count", "diagnostic_focus", "expected_failure_mode"]
    sep  = "|" + "|".join(["---"] * len(headers)) + "|"
    hrow = "| " + " | ".join(headers) + " |"
    rows = [
        f"| {b} | {bc.get(b, 0)} "
        f"| {BUCKET_META[b]['diagnostic_focus']} "
        f"| {BUCKET_META[b]['expected_failure_mode']} |"
        for b in BUCKET_ORDER
    ]
    return "\n".join([hrow, sep] + rows) + "\n"


def _label_dist_md(summary: dict[str, Any]) -> str:
    blc = summary.get("bucket_label_counts", {})
    all_labels = set()
    for counts in blc.values():
        all_labels.update(counts.keys())
    all_labels_sorted = sorted(all_labels)
    if not all_labels_sorted:
        return "_No label data available._\n"
    headers = ["bucket"] + all_labels_sorted
    sep  = "|" + "|".join(["---"] * len(headers)) + "|"
    hrow = "| " + " | ".join(headers) + " |"
    rows = []
    for b in BUCKET_ORDER:
        counts = blc.get(b, {})
        cells = [b] + [str(counts.get(lbl, 0)) for lbl in all_labels_sorted]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([hrow, sep] + rows) + "\n"


def _flag_table_md(summary: dict[str, Any]) -> str:
    fc = summary.get("heuristic_flag_counts", {})
    if not fc:
        return "_No location_swap records in output._\n"
    headers = ["flag", "true", "false", "null"]
    sep  = "|" + "|".join(["---"] * len(headers)) + "|"
    hrow = "| " + " | ".join(headers) + " |"
    rows = [
        f"| {flag} | {counts.get('true', 0)} | {counts.get('false', 0)} | {counts.get('null', 0)} |"
        for flag, counts in fc.items()
    ]
    return "\n".join([hrow, sep] + rows) + "\n"


def _build_markdown(summary: dict[str, Any]) -> str:
    s = summary
    lac = s.get("location_audit_counts", {})
    sov = s.get("source_id_overlap_stats", {})

    return f"""\
# Stage28-B Location Validity Audit Dataset Summary

## Objective

Construct a diagnostic/audit dataset that separates clean, ambiguous, and potentially
invalid location_swap cases from Stage27 controlled-setting predictions. The purpose
is NOT to remove location_swap like time_swap was removed. The purpose is to audit
whether location_swap is a valid hard frame-boundary axis and to enable a
clean/ambiguous/invalid separation before any specialist gate is designed.

`product_power=0.90` remains the stable Stage27 controlled-setting baseline throughout.

## Why Location Swap Is Not Removed by Default

`time_swap` was excluded from Stage27 evaluation because Stage12 analysis identified
it as corrupted or problematic - an artifact of dataset construction rather than a
genuine hard diagnostic axis.

`location_swap` is different. Stage27-H2F shows it is the dominant false-SUPPORT axis
in product_power=0.90 runs (location_role_balance=4.50 from H2F data;
location_SUPPORT=27 vs role_SUPPORT=6 in product 3-seed totals). This
may reflect a genuine hard frame-boundary problem where the model does not sufficiently
discriminate spatially invalid claims. Until manual validity audit proves otherwise,
location_swap should be treated as a plausible valid diagnostic axis.

The correct next step is structured validity audit, not removal.

## Input

| Parameter | Value |
|---|---|
| input_jsonl | {s.get('input_jsonl', 'N/A')} |
| seed | {s.get('seed', 'N/A')} |
| max_per_bucket | {s.get('max_per_bucket', 'N/A')} |
| total_input_records | {s.get('total_input_records', 'N/A')} |

**Input normalized intervention distribution:**

| intervention | count |
|---|---:|
{"".join(f'| {k} | {v} |{chr(10)}' for k, v in s.get('normalized_intervention_counts_input', {}).items())}

## Output

| Parameter | Value |
|---|---|
| output_jsonl | {s.get('output_jsonl', 'N/A')} |
| total_output_records | {s.get('total_output_records', 'N/A')} |

## Bucket Design

Each output record belongs to exactly one diagnostic bucket:

{_bucket_table_md(s)}

**Control selection:** `location_controls` contains none/paraphrase/polarity_flip
records that share a `source_id` with a location_swap record. Similarly for
`role_controls`. This enables paired analysis of how the model handles a claim
before and after the location swap.

## Bucket Counts

| bucket | count |
|---|---:|
{"".join(f'| {b} | {s.get("bucket_counts", {}).get(b, 0)} |{chr(10)}' for b in BUCKET_ORDER)}

## Label Distribution

Label distribution per bucket (normalized_label from source records):

{_label_dist_md(s)}

## Location Validity Audit Fields

All records in `location_swap_audit` bucket carry:

| field | value at construction |
|---|---|
| audit_status | `needs_manual_audit` |
| audit_validity | `null` (to be filled after manual review) |
| audit_reason | `null` |
| audit_flags | dictionary of heuristic booleans (see below) |
| audit_question | "{_AUDIT_QUESTION}" |

Allowed future values for `audit_validity`:
- `clean_invalid_entailment` - location swap genuinely breaks entailment
- `ambiguous` - validity unclear (aliasing, containment, hierarchy, etc.)
- `invalid_artifact` - dataset artifact, location swap is trivially invalid
- `unknown` - cannot determine

**Location audit summary:**

| metric | count |
|---|---:|
| n_location_swap_records | {lac.get('n_location_swap_records', 0)} |
| n_location_audit_records (in output) | {lac.get('n_location_audit_records', 0)} |
| n_needs_manual_audit | {lac.get('n_needs_manual_audit', 0)} |
| n_clean_invalid_entailment | {lac.get('n_clean_invalid_entailment', 0)} |
| n_ambiguous | {lac.get('n_ambiguous', 0)} |
| n_invalid_artifact | {lac.get('n_invalid_artifact', 0)} |
| n_unknown | {lac.get('n_unknown', 0)} |

## Source-ID Control Coverage

Control records are matched by `source_id` to enable paired analysis:

| metric | value |
|---|---:|
| n_location_sources | {sov.get('n_location_sources', 0)} |
| n_role_sources | {sov.get('n_role_sources', 0)} |
| n_location_control_matches | {sov.get('n_location_control_matches', 0)} |
| n_role_control_matches | {sov.get('n_role_control_matches', 0)} |
| location_control_coverage | {_fmt_f(sov.get('location_control_coverage'))} |
| role_control_coverage | {_fmt_f(sov.get('role_control_coverage'))} |

If coverage is low, future dataset construction should add paired controls.

## Heuristic Flag Summary

Heuristic flags for `location_swap_audit` records. All flags are text-pattern-based
only and must NOT be treated as final validity labels. They are starting points for
manual audit.

{_flag_table_md(s)}

Flags marked `null` require manual review or external knowledge to determine.

## Interpretation

- Stage28-B constructs a diagnostic/audit dataset. This is not a training result and
  does not change any model prediction.
- location_swap should not be removed by default. time_swap was removed because it
  appeared corrupted/problematic. location_swap is treated as a plausible hard
  frame-boundary diagnostic axis pending manual audit.
- The correct next step is manual validity audit followed by clean/ambiguous/invalid
  separation, not immediate deletion or model change.
- product_power=0.90 remains the Stage27 controlled-setting baseline. Nothing in
  Stage28-B changes the final configuration.
- If many location_swap records are labeled ambiguous or invalid after audit, future
  clean validation should separate: clean_location_swap, ambiguous_location_swap,
  invalid_location_swap.
- If location_controls have low coverage, future dataset construction should add
  paired controls for better before/after analysis.

## Recommended Next Stage

**{s.get('recommended_next_stage', 'N/A')}**

## Limitations

{"".join(f'- {lim}{chr(10)}' for lim in s.get('limitations', []))}
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage28-B: Location Validity Audit Dataset builder. "
            "Constructs a diagnostic/audit dataset from controlled_v5_v3_without_time_swap.jsonl "
            "to separate clean, ambiguous, and potentially invalid location_swap cases. "
            "Does not modify model code. Does not perform training. "
            "product_power=0.90 remains the stable Stage27 baseline."
        )
    )
    p.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data/controlled_v5_v3_without_time_swap.jsonl"),
        help="Input JSONL file (default: data/controlled_v5_v3_without_time_swap.jsonl).",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/stage28b_location_validity_audit.jsonl"),
        help="Output audit JSONL path (default: data/stage28b_location_validity_audit.jsonl).",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/stage28b_location_validity_audit_summary.md"),
        help="Output markdown summary path.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/stage28b_location_validity_audit_summary.json"),
        help="Output JSON summary path.",
    )
    p.add_argument(
        "--max-per-bucket",
        type=int,
        default=300,
        dest="max_per_bucket",
        help="Maximum records per bucket; all location/role records kept if <= this (default: 300).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=28,
        help="Random seed for deterministic shuffling (default: 28).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    args = parse_args(argv)

    if not args.input_jsonl.exists():
        print(
            f"[S28B] ERROR: input JSONL not found: {args.input_jsonl}",
            file=sys.stderr,
        )
        return 1

    # Read and parse all records
    raw_records = _read_jsonl(args.input_jsonl)
    parsed: list[dict] = [_parse_record(r, i) for i, r in enumerate(raw_records)]
    print(f"[S28B] Parsed {len(parsed)} records.", file=sys.stderr)

    # Collect source-ID sets for location/role controls
    location_source_ids: frozenset = frozenset(
        p["source_id"] for p in parsed
        if p["normalized_intervention"] == "location_swap"
    )
    role_source_ids: frozenset = frozenset(
        p["source_id"] for p in parsed
        if p["normalized_intervention"] == "role_swap"
    )
    print(
        f"[S28B] location sources: {len(location_source_ids)}, "
        f"role sources: {len(role_source_ids)}.",
        file=sys.stderr,
    )

    # Assign buckets
    rng = random.Random(args.seed)
    buckets = _assign_buckets(
        parsed, location_source_ids, role_source_ids, rng, args.max_per_bucket
    )
    for b, items in buckets.items():
        print(f"[S28B] bucket {b}: {len(items)} records.", file=sys.stderr)

    # Build output records
    output_records: list[dict] = []
    for bucket in BUCKET_ORDER:
        for p in buckets[bucket]:
            output_records.append(_build_output_record(p, bucket))
    print(f"[S28B] Total output records: {len(output_records)}.", file=sys.stderr)

    # Build summary
    summary = _build_summary(
        input_path=args.input_jsonl,
        output_path=args.output_jsonl,
        seed=args.seed,
        max_per_bucket=args.max_per_bucket,
        total_input=len(raw_records),
        buckets=buckets,
        output_records=output_records,
        parsed=parsed,
        location_source_ids=location_source_ids,
        role_source_ids=role_source_ids,
    )

    # Write outputs
    for out_dir in (
        args.output_jsonl.parent,
        args.output_md.parent,
        args.output_json.parent,
    ):
        out_dir.mkdir(parents=True, exist_ok=True)

    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[S28B] Wrote {len(output_records)} records to {args.output_jsonl}.", file=sys.stderr)

    md = _build_markdown(summary)
    args.output_md.write_text(md, encoding="utf-8")
    print(f"[S28B] Wrote: {args.output_md}", file=sys.stderr)

    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[S28B] Wrote: {args.output_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
