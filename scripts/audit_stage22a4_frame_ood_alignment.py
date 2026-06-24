"""Stage22-A4a audit: OOD-aligned frame supervision alignment.

Checks whether the controlled training data has enough schema structure
to generate train-side OOD-aligned frame mismatch supervision, without
training on Stage15 OOD records.

Usage:
    python scripts/audit_stage22a4_frame_ood_alignment.py \
        --controlled-data data/controlled_v5_v3_without_time_swap.jsonl \
        --ood-data        data/stage15_slot_sensitivity_probe.jsonl \
        --output-json     results/stage22a4_frame_ood_alignment_audit.json \
        --output-md       results/stage22a4_frame_ood_alignment_audit.md

Data-leakage contract:
    Stage15 OOD is read for SCHEMA INSPECTION ONLY.
    No OOD records are proposed for use in training.
    All candidate generation routes draw exclusively from controlled data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Schema / vocabulary helpers
# ---------------------------------------------------------------------------

def collect_schema(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect per-field presence rate and example values."""
    if not records:
        return {"n_records": 0, "fields": {}}
    field_counts: Counter[str] = Counter()
    field_examples: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        for k, v in rec.items():
            field_counts[k] += 1
            if v is not None and len(field_examples[k]) < 5:
                field_examples[k].add(str(v)[:80])
    n = len(records)
    fields = {
        k: {
            "presence_rate": round(field_counts[k] / n, 4),
            "examples": sorted(field_examples[k])[:5],
        }
        for k in sorted(field_counts)
    }
    return {"n_records": n, "fields": fields}


def collect_vocab(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    c: Counter[str] = Counter()
    for rec in records:
        v = rec.get(field)
        if v is not None:
            c[str(v)] += 1
    return dict(c.most_common())


def field_present(records: list[dict[str, Any]], field: str) -> bool:
    return any(rec.get(field) is not None for rec in records)


def presence_rate(records: list[dict[str, Any]], field: str) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.get(field) is not None) / len(records)


# ---------------------------------------------------------------------------
# OOD-aligned generation capability checks
# ---------------------------------------------------------------------------

_FRAME_INTERVENTION_TYPES = frozenset({
    "location_swap", "role_swap", "entity_swap", "event_swap", "title_name_swap",
})
_PRESERVATION_INTERVENTION_TYPES = frozenset({
    "none", "paraphrase",
})
_SURFACE_LIKE = frozenset({
    "none", "paraphrase", "evidence_deletion", "evidence_truncation",
    "irrelevant_evidence", "polarity_flip",
})
_OOD_FRAME_GROUPS = frozenset({"frame_location_mismatch", "frame_role_mismatch"})
_OOD_PRESERVATION_GROUPS = frozenset({"surface_control", "temporal_erased"})


def check_generation_capabilities(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check which OOD-aligned generation routes are feasible from controlled data."""
    caps: dict[str, Any] = {}

    # Stable IDs
    caps["has_pair_id"] = field_present(records, "pair_id")
    caps["pair_id_presence_rate"] = presence_rate(records, "pair_id")
    caps["has_id"] = field_present(records, "id")

    # Core text fields
    caps["has_claim"] = field_present(records, "claim")
    caps["has_evidence"] = field_present(records, "evidence")

    # Intervention metadata
    caps["has_intervention_type"] = field_present(records, "intervention_type")
    it_vocab = collect_vocab(records, "intervention_type")
    caps["intervention_type_vocab"] = it_vocab
    frame_types_present = sorted(_FRAME_INTERVENTION_TYPES & set(it_vocab))
    preservation_types_present = sorted(_PRESERVATION_INTERVENTION_TYPES & set(it_vocab))
    caps["frame_intervention_types_present"] = frame_types_present
    caps["preservation_intervention_types_present"] = preservation_types_present

    # Explicit slot fields (direct slot-based generation)
    slot_fields = [
        "location", "role", "entity", "event", "title",
        "original_location", "original_role", "original_entity",
        "original_event", "original_title",
        "perturbed_location", "perturbed_role", "perturbed_entity",
        "perturbed_event", "perturbed_title",
        "slot_type", "slot_original", "slot_perturbed",
    ]
    slot_presence = {f: presence_rate(records, f) for f in slot_fields}
    caps["slot_fields_presence"] = {
        k: round(v, 4) for k, v in slot_presence.items() if v > 0.0
    }
    explicit_slot_fields = [k for k, v in slot_presence.items() if v > 0.05]
    caps["explicit_slot_fields_available"] = explicit_slot_fields

    # Pair / group structure (pair-based generation)
    pair_ids = [r.get("pair_id") for r in records if r.get("pair_id") is not None]
    pid_counts: Counter = Counter(pair_ids)
    multi_member_pairs = sum(1 for c in pid_counts.values() if c > 1)
    caps["unique_pair_ids"] = len(pid_counts)
    caps["pair_ids_with_multiple_records"] = multi_member_pairs
    if pid_counts:
        caps["mean_records_per_pair_id"] = round(
            sum(pid_counts.values()) / len(pid_counts), 2
        )

    # Check whether we can find frame vs preservation siblings under same pair_id
    pair_interventions: dict[str, set[str]] = defaultdict(set)
    for r in records:
        pid = r.get("pair_id")
        it = r.get("intervention_type")
        if pid and it:
            pair_interventions[pid].add(it)

    pairs_with_frame_and_preservation = sum(
        1 for its in pair_interventions.values()
        if its & _FRAME_INTERVENTION_TYPES and its & _PRESERVATION_INTERVENTION_TYPES
    )
    caps["pairs_with_frame_and_preservation_sibling"] = pairs_with_frame_and_preservation
    caps["pair_based_generation_feasible"] = pairs_with_frame_and_preservation > 0

    # Text-template fallback (minimum requirement: claim + evidence + intervention_type)
    caps["text_template_fallback_feasible"] = (
        caps["has_claim"] and caps["has_evidence"] and caps["has_intervention_type"]
    )

    # Route summary
    caps["candidate_routes"] = _summarize_routes(caps)
    return caps


def _summarize_routes(caps: dict[str, Any]) -> list[dict[str, str]]:
    routes = []
    if caps.get("explicit_slot_fields_available"):
        routes.append({
            "route": "direct_slot_based",
            "feasibility": "high",
            "description": (
                "Explicit slot fields present — can generate frame-aligned negatives "
                "by swapping slot fillers between records with the same pair_id."
            ),
            "required_fields": ", ".join(caps["explicit_slot_fields_available"]),
        })
    if caps.get("pair_based_generation_feasible"):
        routes.append({
            "route": "pair_group_based",
            "feasibility": "medium" if caps["explicit_slot_fields_available"] else "medium_high",
            "description": (
                f"{caps['pairs_with_frame_and_preservation_sibling']} pair_ids contain both "
                "a frame intervention record and a preservation record — can generate "
                "OOD-aligned contrastive pairs by grouping siblings."
            ),
            "required_fields": "pair_id, intervention_type, claim, evidence",
        })
    if caps.get("text_template_fallback_feasible"):
        routes.append({
            "route": "text_template_fallback",
            "feasibility": "low" if not caps.get("pair_based_generation_feasible") else "supplemental",
            "description": (
                "Minimum viable route: claim + evidence + intervention_type available. "
                "Can apply rule-based slot-filling templates to generate frame mismatch "
                "variants without explicit slot fields, using NER or regex heuristics."
            ),
            "required_fields": "claim, evidence, intervention_type",
        })
    if not routes:
        routes.append({
            "route": "none_feasible",
            "feasibility": "blocked",
            "description": (
                "Insufficient schema: no pair_id, no slot fields, or no intervention_type."
            ),
            "required_fields": "",
        })
    return routes


# ---------------------------------------------------------------------------
# OOD schema inspection (read-only — no training use)
# ---------------------------------------------------------------------------

def inspect_ood_schema(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Inspect OOD data for evaluation schema only. Never propose training use."""
    out: dict[str, Any] = {"leakage_warning": (
        "Stage15 OOD records are read for schema inspection only. "
        "No OOD records are proposed for training. "
        "All generation routes draw exclusively from controlled data."
    )}
    out["n_records"] = len(records)
    for group_field in ("stage15_probe_type", "probe_type", "ood_group"):
        if field_present(records, group_field):
            out["group_field_used"] = group_field
            out["group_vocab"] = collect_vocab(records, group_field)
            break
    else:
        out["group_field_used"] = None
        out["group_vocab"] = {}

    # Which slot / structural fields are present in OOD?
    ood_slot_fields = [
        "slot_type", "slot_original", "slot_perturbed",
        "location", "role", "entity", "event", "title",
        "frame", "predicate", "sufficiency", "polarity",
        "transformation", "expected_behavior",
    ]
    out["ood_structural_fields"] = {
        f: round(presence_rate(records, f), 4)
        for f in ood_slot_fields
        if presence_rate(records, f) > 0.0
    }

    # Alignment gap: what OOD frame groups look like structurally
    group_field = out.get("group_field_used")
    if group_field:
        frame_ood = [r for r in records if r.get(group_field) in _OOD_FRAME_GROUPS]
        pres_ood = [r for r in records if r.get(group_field) in _OOD_PRESERVATION_GROUPS]
        out["frame_ood_n"] = len(frame_ood)
        out["preservation_ood_n"] = len(pres_ood)
        frame_slot_types = collect_vocab(frame_ood, "slot_type")
        pres_slot_types = collect_vocab(pres_ood, "slot_type")
        out["frame_ood_slot_type_vocab"] = frame_slot_types
        out["preservation_ood_slot_type_vocab"] = pres_slot_types
    return out


# ---------------------------------------------------------------------------
# Leakage risk report
# ---------------------------------------------------------------------------

def leakage_risk_report(
    caps: dict[str, Any],
    ood_info: dict[str, Any],
) -> list[dict[str, str]]:
    risks = []
    risks.append({
        "risk": "direct_training_on_ood",
        "severity": "critical",
        "status": "blocked_by_policy",
        "description": (
            "Training directly on Stage15 OOD records (data/stage15_slot_sensitivity_probe.jsonl) "
            "is prohibited. Any new frame supervision must be derived from controlled data or "
            "a separate synthetic generator."
        ),
    })
    if caps.get("pair_based_generation_feasible"):
        risks.append({
            "risk": "ood_schema_leakage_via_templates",
            "severity": "medium",
            "status": "requires_design_review",
            "description": (
                "If OOD slot structures (slot_type, transformation patterns) are used to "
                "design template rules applied to controlled data, the resulting synthetic "
                "records may indirectly mirror OOD distribution. Mitigation: template rules "
                "must be derived from the controlled intervention_type taxonomy only, not "
                "from Stage15 record inspection."
            ),
        })
    risks.append({
        "risk": "eval_group_label_leakage",
        "severity": "low",
        "status": "already_mitigated",
        "description": (
            "The existing training script does not pass stage15_probe_type or ood_group "
            "to model forward/loss computation. OOD group names are used only in post-hoc "
            "metric grouping, not in gate or loss logic."
        ),
    })
    return risks


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_markdown(
    controlled_schema: dict[str, Any],
    ood_info: dict[str, Any],
    caps: dict[str, Any],
    risks: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
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

    h(1, "Stage22-A4a: OOD-aligned frame supervision audit")
    p(f"**Controlled data:** `{args.controlled_data}`")
    p(f"**OOD data (schema reference only):** `{args.ood_data}`")
    p(
        "**Data-leakage constraint:** Stage15 OOD is read for schema inspection only. "
        "No OOD records are proposed for training. "
        "All candidate generation routes use controlled data exclusively."
    )
    lines.append("")

    h(2, "1. Controlled data schema summary")
    p(f"Total records: **{controlled_schema['n_records']}**")
    schema_rows = []
    for field, info in controlled_schema["fields"].items():
        ex = ", ".join(info["examples"][:2])
        schema_rows.append([field, f"{info['presence_rate']:.3f}", ex])
    table(["Field", "Presence rate", "Examples (≤2)"], schema_rows)

    h(2, "2. Controlled intervention_type vocabulary")
    it_vocab = caps.get("intervention_type_vocab", {})
    if it_vocab:
        table(
            ["intervention_type", "count"],
            [[k, str(v)] for k, v in sorted(it_vocab.items(), key=lambda x: -x[1])],
        )
    else:
        p("_intervention_type field absent or empty._")

    p("Frame intervention types present (map to OOD frame mismatch groups):")
    for t in caps.get("frame_intervention_types_present", []):
        lines.append(f"- `{t}`")
    lines.append("")
    p("Preservation intervention types present (map to OOD preservation groups):")
    for t in caps.get("preservation_intervention_types_present", []):
        lines.append(f"- `{t}`")
    lines.append("")

    h(2, "3. OOD data schema (Stage15 — evaluation reference only)")
    p(f"Total OOD records: **{ood_info.get('n_records', 'N/A')}**")
    p(f"Group field: `{ood_info.get('group_field_used', 'not found')}`")
    gv = ood_info.get("group_vocab", {})
    if gv:
        table(
            ["OOD group", "count"],
            [[k, str(v)] for k, v in sorted(gv.items(), key=lambda x: -x[1])],
        )
    sf = ood_info.get("ood_structural_fields", {})
    if sf:
        p("OOD structural / slot fields present:")
        table(
            ["Field", "Presence rate"],
            [[k, f"{v:.3f}"] for k, v in sf.items()],
        )
    fst = ood_info.get("frame_ood_slot_type_vocab", {})
    if fst:
        p("OOD frame-mismatch group slot_type vocabulary:")
        for k, v in fst.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    h(2, "4. OOD-aligned generation capability check")
    p(f"pair_id available: **{caps.get('has_pair_id')}** "
      f"(presence rate {caps.get('pair_id_presence_rate', 0.0):.3f})")
    p(f"Unique pair_ids: **{caps.get('unique_pair_ids', 0)}**")
    p(f"pair_ids with ≥2 records: **{caps.get('pair_ids_with_multiple_records', 0)}**")
    p(f"pair_ids with frame + preservation sibling: "
      f"**{caps.get('pairs_with_frame_and_preservation_sibling', 0)}**")
    p(f"Explicit slot fields available: {caps.get('explicit_slot_fields_available', [])}")
    lines.append("")

    h(2, "5. Candidate generation routes")
    for route in caps.get("candidate_routes", []):
        lines.append(f"### {route['route']}")
        lines.append(f"**Feasibility:** {route['feasibility']}")
        lines.append(f"**Description:** {route['description']}")
        if route.get("required_fields"):
            lines.append(f"**Required fields:** `{route['required_fields']}`")
        lines.append("")

    h(2, "6. Leakage risks")
    for r in risks:
        lines.append(f"### {r['risk']}")
        lines.append(f"**Severity:** {r['severity']} | **Status:** {r['status']}")
        lines.append(r['description'])
        lines.append("")

    h(2, "7. Summary and recommended next step")
    frame_route = any(
        rt["route"] in ("direct_slot_based", "pair_group_based")
        for rt in caps.get("candidate_routes", [])
    )
    if frame_route:
        p(
            "The controlled data contains sufficient schema structure "
            "to generate train-side OOD-aligned frame mismatch supervision "
            "without using Stage15 OOD records directly."
        )
        p(
            "Recommended first path: **pair_group_based contrastive supervision** — "
            "group records by pair_id, construct (frame-intervention, preservation-control) "
            "sibling pairs, and train a contrastive or classification head on those pairs "
            "using the OOD frame mismatch structure as the alignment target."
        )
    else:
        p(
            "Controlled data schema is insufficient for direct OOD-aligned generation. "
            "Fallback: generate a synthetic controlled extension that adds explicit slot fields."
        )
    p(
        "**Stage22-B positive recovery gate remains rejected** until "
        "frame_violation_prob ranks OOD frame mismatch groups higher than "
        "surface_control and temporal_erased consistently."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage22-A4a: audit OOD-aligned frame supervision feasibility.",
    )
    p.add_argument(
        "--controlled-data",
        required=True,
        help="Path to controlled training JSONL (e.g. controlled_v5_v3_without_time_swap.jsonl).",
    )
    p.add_argument(
        "--ood-data",
        required=True,
        help=(
            "Path to Stage15 OOD JSONL — schema inspection only. "
            "No OOD records will be proposed for training."
        ),
    )
    p.add_argument(
        "--output-json",
        required=True,
        help="Path to write the JSON audit summary.",
    )
    p.add_argument(
        "--output-md",
        required=True,
        help="Path to write the Markdown audit report.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    controlled_path = Path(args.controlled_data)
    ood_path = Path(args.ood_data)

    if not controlled_path.exists():
        print(f"ERROR: controlled data not found: {controlled_path}", file=sys.stderr)
        return 1
    if not ood_path.exists():
        print(f"ERROR: OOD data not found: {ood_path}", file=sys.stderr)
        return 1

    print(f"Loading controlled data: {controlled_path}")
    controlled = load_jsonl(controlled_path)
    print(f"  {len(controlled)} records")

    print(f"Loading OOD data (schema inspection only): {ood_path}")
    ood = load_jsonl(ood_path)
    print(f"  {len(ood)} records")

    controlled_schema = collect_schema(controlled)
    caps = check_generation_capabilities(controlled)
    ood_info = inspect_ood_schema(ood)
    risks = leakage_risk_report(caps, ood_info)

    summary = {
        "controlled_schema": controlled_schema,
        "generation_capabilities": caps,
        "ood_schema_reference": ood_info,
        "leakage_risks": risks,
        "leakage_contract": (
            "Stage15 OOD data is read for schema inspection only. "
            "No OOD records are proposed for training. "
            "All generation routes draw exclusively from controlled data."
        ),
    }

    out_json = Path(args.output_json)
    write_json(out_json, summary)
    print(f"JSON summary written: {out_json}")

    md_text = render_markdown(controlled_schema, ood_info, caps, risks, args)
    out_md = Path(args.output_md)
    write_md(out_md, md_text)
    print(f"Markdown report written: {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
