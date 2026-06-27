"""Stage27-H2C: Learned entitlement gate diagnostic analysis.

Reads per-seed summary JSON files from H2A and H2B run directories,
compares learned/product/product-power configurations, and generates a
markdown + JSON diagnostic report.

Usage:
    python scripts/analyze_stage27_h2c_learned_gate.py \\
        --h2a-dir results/h2a \\
        --h2b-dir results/h2b \\
        --output-md  reports/stage27_h2c_learned_gate_diagnostic.md \\
        --output-json reports/stage27_h2c_learned_gate_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTERVENTION_KEYS: tuple[str, ...] = (
    "location_swap",
    "role_swap",
    "predicate_swap",
    "entity_swap",
    "event_swap",
    "title_name_swap",
    "evidence_deletion",
    "evidence_truncation",
    "irrelevant_evidence",
)

DIAG_PROB_KEYS: tuple[str, ...] = (
    "entitlement_prob",
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "v7_h1_entitlement_for_decision",
)

TOP_LEVEL_METRIC_KEYS: tuple[str, ...] = (
    "best_dev_acc",
    "best_dev_macro_f1",
    "v7_support_recall",
    "v7_ne_recall",
    "v7_refute_recall",
    "v7_final_logit_composition",
    "v7_h1_entitlement_decision_signal",
    "v7_h1_entitlement_product_power",
)

# H2A aggregate reference (3-seed, no-time) — used when file data is absent
_H2A_REFERENCE: dict[str, dict[str, float]] = {
    "product": {"macro_mean": 0.940351, "macro_std": 0.025612, "acc_mean": 0.957870,
                "acc_std": 0.018651, "SUP_r_mean": 0.944444, "NE_r_mean": 0.953086,
                "bad_SUP_total": 76, "location_role_SUP_total": 60, "missing_SUP_total": 0},
    "min":     {"macro_mean": 0.928270, "macro_std": 0.020529, "acc_mean": 0.945833,
                "acc_std": 0.017067, "SUP_r_mean": 0.988889, "NE_r_mean": 0.929630,
                "bad_SUP_total": 114, "location_role_SUP_total": 78, "missing_SUP_total": 0},
    "learned": {"macro_mean": 0.900660, "macro_std": 0.046553, "acc_mean": 0.922685,
                "acc_std": 0.045354, "SUP_r_mean": 0.966952, "NE_r_mean": 0.902469,
                "bad_SUP_total": 136, "location_role_SUP_total": 104, "missing_SUP_total": 0},
}

_H2B_REFERENCE: dict[str, Any] = {
    "power": 0.90, "macro_mean": 0.951136, "macro_std": 0.021341,
    "acc_mean": 0.968056, "acc_std": 0.013679,
    "SUP_r_mean": 0.907407, "NE_r_mean": 0.972840,
    "bad_SUP_total": 44, "location_role_SUP_total": 33, "missing_SUP_total": 0,
}

# Filename inference patterns
_RE_POWER = re.compile(r"product_power[_-](\d+p\d+|\d+\.\d+)", re.IGNORECASE)
_RE_SEED  = re.compile(r"seed(\d+)", re.IGNORECASE)
_RE_MODE  = re.compile(
    r"^(frame_predicate_(?:product|min)|product|min|learned)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_filename(path: Path) -> dict[str, Any]:
    """Infer mode, seed, product_power from a summary JSON filename."""
    stem = path.stem
    info: dict[str, Any] = {
        "filename": path.name,
        "mode": None,
        "seed": None,
        "product_power": None,
    }

    seed_m = _RE_SEED.search(stem)
    if seed_m:
        info["seed"] = int(seed_m.group(1))

    power_m = _RE_POWER.search(stem)
    if power_m:
        raw = power_m.group(1).replace("p", ".")
        try:
            info["product_power"] = float(raw)
        except ValueError:
            pass
        info["mode"] = "product"
    else:
        mode_m = _RE_MODE.match(stem)
        if mode_m:
            info["mode"] = mode_m.group(1).lower()

    return info


# ---------------------------------------------------------------------------
# Safe nested lookup
# ---------------------------------------------------------------------------

def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Drill through nested dicts with a fallback."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _extract_intervention_support_counts(run: dict) -> dict[str, Any]:
    """Extract per-intervention SUPPORT prediction counts."""
    interventions: dict = run.get("best_dev_interventions") or {}
    result: dict[str, Any] = {}
    for key in INTERVENTION_KEYS:
        block = interventions.get(key) or {}
        count: "int | None" = None
        for path in (
            ("support_predictions",),
            ("SUPPORT_predictions",),
            ("predictions", "SUPPORT"),
            ("SUPPORT_count",),
            ("support_count",),
        ):
            val = _safe_get(block, *path)
            if val is not None:
                try:
                    count = int(val)
                except (TypeError, ValueError):
                    pass
                break
        result[key] = count
    return result


def _extract_diag_prob_means(run: dict) -> dict[str, "float | None"]:
    """Extract diagnostic probability means from multiple possible locations."""
    result: dict[str, "float | None"] = {k: None for k in DIAG_PROB_KEYS}

    # 1. Top-level scalars (some summaries lift these)
    for k in DIAG_PROB_KEYS:
        v = run.get(k)
        if isinstance(v, (int, float)):
            result[k] = float(v)

    # 2. v7_best_dev_logit_summary → per-tensor {mean, std, min, max}
    logit_summary: dict = run.get("v7_best_dev_logit_summary") or {}
    for k in DIAG_PROB_KEYS:
        if result[k] is None:
            sub = logit_summary.get(k)
            if isinstance(sub, dict):
                v = sub.get("mean")
                if v is not None:
                    result[k] = float(v)

    # 3. best_dev_interventions → control → mean_<key> or <key>.mean
    interventions: dict = run.get("best_dev_interventions") or {}
    control: dict = interventions.get("control") or {}
    for k in DIAG_PROB_KEYS:
        if result[k] is None:
            for path in ((f"mean_{k}",), (k, "mean"), (k,)):
                v = _safe_get(control, *path)
                if isinstance(v, (int, float)):
                    result[k] = float(v)
                    break

    # 4. *_mean suffixed top-level keys (e.g. frame_prob_mean)
    for k in DIAG_PROB_KEYS:
        if result[k] is None:
            v = run.get(f"{k}_mean")
            if isinstance(v, (int, float)):
                result[k] = float(v)

    return result


def _extract_run_metrics(run: dict, path_info: dict) -> dict[str, Any]:
    """Build a flat metric record for one run JSON."""
    rec: dict[str, Any] = dict(path_info)

    for k in TOP_LEVEL_METRIC_KEYS:
        rec[k] = run.get(k)

    # Override mode/power from JSON content when filename parse failed
    config: dict = run.get("configuration") or {}
    for src in (run, config):
        if rec["mode"] is None:
            rec["mode"] = src.get("v7_h1_entitlement_decision_signal")
        if rec["product_power"] is None and src.get("v7_h1_entitlement_product_power") is not None:
            rec["product_power"] = src.get("v7_h1_entitlement_product_power")

    # Intervention SUPPORT counts
    sup_counts = _extract_intervention_support_counts(run)
    rec["intervention_support_counts"] = sup_counts

    def _n(v: Any) -> "int | None":
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    loc  = _n(sup_counts.get("location_swap"))
    role = _n(sup_counts.get("role_swap"))
    evd  = _n(sup_counts.get("evidence_deletion"))
    evt  = _n(sup_counts.get("evidence_truncation"))
    irr  = _n(sup_counts.get("irrelevant_evidence"))

    rec["location_role_SUP"] = (
        (loc or 0) + (role or 0)
        if loc is not None or role is not None else None
    )
    rec["missing_SUP"] = (
        (evd or 0) + (evt or 0) + (irr or 0)
        if any(v is not None for v in (evd, evt, irr)) else None
    )

    all_vals = [_n(sup_counts.get(k)) for k in INTERVENTION_KEYS]
    rec["bad_SUP"] = (
        sum(v for v in all_vals if v is not None)
        if any(v is not None for v in all_vals) else None
    )

    rec["diag_prob_means"] = _extract_diag_prob_means(run)
    return rec


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def _scan_dir(directory: Path, label: str) -> list[dict[str, Any]]:
    """Scan a directory for *summary*.json files and extract metrics."""
    if not directory.exists():
        print(f"[H2C] WARNING: {label} directory not found: {directory}", file=sys.stderr)
        return []
    results: list[dict[str, Any]] = []
    for p in sorted(directory.glob("*summary*.json")):
        try:
            with p.open(encoding="utf-8") as f:
                run = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[H2C] WARNING: could not read {p}: {exc}", file=sys.stderr)
            continue
        info = _parse_filename(p)
        info["source_dir"] = label
        info["source_file"] = str(p)
        results.append(_extract_run_metrics(run, info))
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _mean_or_none(vals: list) -> "float | None":
    valid = [float(v) for v in vals if v is not None]
    return round(sum(valid) / len(valid), 6) if valid else None


def _rank_learned(runs: list[dict]) -> list[dict]:
    """Sort learned runs: macro_f1 desc, bad_SUP asc, location_role_SUP asc."""
    learned = [r for r in runs if (r.get("mode") or "").lower() == "learned"]

    def _key(r: dict) -> tuple:
        macro = r.get("best_dev_macro_f1")
        bad   = r.get("bad_SUP")
        lr    = r.get("location_role_SUP")
        return (
            -(macro if macro is not None else -1.0),
             bad if bad is not None else 99_999,
             lr  if lr  is not None else 99_999,
        )

    return sorted(learned, key=_key)


def _compare_runs(a: "dict | None", b: "dict | None") -> dict[str, Any]:
    """Compare two run records on key metrics."""
    if a is None or b is None:
        return {}
    keys = [
        "best_dev_macro_f1", "best_dev_acc",
        "v7_support_recall", "v7_ne_recall",
        "bad_SUP", "location_role_SUP", "missing_SUP",
    ]
    comp: dict[str, Any] = {}
    for k in keys:
        va, vb = a.get(k), b.get(k)
        comp[k] = {"a": va, "b": vb}
        if va is not None and vb is not None:
            try:
                comp[k]["delta_a_minus_b"] = round(float(va) - float(vb), 6)
            except (TypeError, ValueError):
                pass
    da = a.get("diag_prob_means") or {}
    db = b.get("diag_prob_means") or {}
    comp["diag_prob_means"] = {k: {"a": da.get(k), "b": db.get(k)} for k in DIAG_PROB_KEYS}
    return comp


def _interpret(
    learned_ranked: list[dict],
    product_runs: list[dict],
    product_power_runs: list[dict],
) -> list[str]:
    lines: list[str] = []

    if not learned_ranked:
        lines.append(
            "No learned runs found in the provided directories. "
            "Interpretation is based on H2A aggregate references only."
        )
        lines.append(
            "H2A aggregate: learned macro_mean=0.9007 (std=0.0466) vs "
            "product macro_mean=0.9404 (std=0.0256). "
            "High std in learned suggests instability rather than uniform weakness."
        )
        lines.append(
            "product_power=0.90 remains the current final configuration. "
            "Learned gate is treated as an unstable diagnostic branch pending per-seed data."
        )
        return lines

    best  = learned_ranked[0]
    worst = learned_ranked[-1] if len(learned_ranked) > 1 else None

    best_macro  = best.get("best_dev_macro_f1")
    prod_macros = [r.get("best_dev_macro_f1") for r in product_runs if r.get("best_dev_macro_f1") is not None]
    prod_mean   = _mean_or_none(prod_macros) or _H2A_REFERENCE["product"]["macro_mean"]
    pp_macros   = [r.get("best_dev_macro_f1") for r in product_power_runs if r.get("best_dev_macro_f1") is not None]
    pp_mean     = _mean_or_none(pp_macros) or _H2B_REFERENCE["macro_mean"]

    learned_macros = [r.get("best_dev_macro_f1") for r in learned_ranked if r.get("best_dev_macro_f1") is not None]
    if len(learned_macros) >= 2:
        spread = max(learned_macros) - min(learned_macros)
        tag = "high" if spread > 0.03 else "moderate"
        lines.append(
            f"Learned gate shows {tag} seed variance (macro spread {spread:.4f} across "
            f"{len(learned_ranked)} seeds), "
            + ("confirming instability observed in H2A aggregates (std=0.0466)."
               if spread > 0.03 else
               "lower than H2A aggregate std=0.0466 — stability may improve with more seeds.")
        )

    if best_macro is not None:
        if best_macro >= prod_mean - 0.005:
            lines.append(
                f"Learned best seed (macro={best_macro:.4f}) matches or exceeds "
                f"product H2A mean ({prod_mean:.4f}), "
                "suggesting the instability is seed-dependent rather than structural."
            )
        else:
            lines.append(
                f"Learned best seed (macro={best_macro:.4f}) is below "
                f"product H2A mean ({prod_mean:.4f}), "
                "indicating the learned gate is weaker even on its best seed."
            )

        if best_macro >= pp_mean - 0.005:
            lines.append(
                f"Learned best seed is competitive with product_power=0.90 mean "
                f"({pp_mean:.4f}), but this is a single-seed result and does not "
                "establish reliability across seeds."
            )
        else:
            lines.append(
                f"Learned best seed does not match product_power=0.90 mean ({pp_mean:.4f}). "
                "product_power=0.90 remains the stronger consistent configuration."
            )

    best_lr   = best.get("location_role_SUP")
    pp_lr_ref = _H2B_REFERENCE["location_role_SUP_total"]
    pp_lr_vals = [r.get("location_role_SUP") for r in product_power_runs if r.get("location_role_SUP") is not None]
    pp_lr_mean = _mean_or_none(pp_lr_vals) or float(pp_lr_ref)
    if best_lr is not None:
        if best_lr > pp_lr_mean * 1.2:
            lines.append(
                f"Learned best seed location/role false SUPPORT ({best_lr}) "
                f"exceeds product_power=0.90 mean ({pp_lr_mean:.1f}) by >20%%. "
                "Product gate better suppresses location/role false SUPPORT."
            )
        else:
            lines.append(
                f"Learned best seed location/role false SUPPORT ({best_lr}) "
                f"is close to product_power=0.90 mean ({pp_lr_mean:.1f}) — "
                "learned may match product gate on this metric in its best seed."
            )

    diag = best.get("diag_prob_means") or {}
    ent  = diag.get("entitlement_prob")
    if ent is not None:
        if 0.45 < ent < 0.55:
            lines.append(
                f"Learned best seed entitlement_prob mean ({ent:.3f}) is near 0.5, "
                "suggesting the learned gate collapses to a weak near-constant scalar "
                "with little discriminative separation across examples."
            )
        else:
            lines.append(
                f"Learned best seed entitlement_prob mean ({ent:.3f}) deviates meaningfully "
                "from 0.5, indicating some learned separation — this seed may be genuinely "
                "informative but instability prevents reliable use."
            )
    else:
        lines.append(
            "Entitlement prob diagnostics not available in run summaries; "
            "cannot directly assess learned gate collapse or separation from this data. "
            "Run with Stage26-F extended diagnostics to capture these values."
        )

    if worst is not None and best_macro is not None:
        worst_macro = worst.get("best_dev_macro_f1")
        if worst_macro is not None:
            gap = best_macro - worst_macro
            lines.append(
                f"Learned worst seed macro ({worst_macro:.4f}) is {gap:.4f} below "
                f"best seed ({best_macro:.4f}). This gap drives the H2A aggregate std "
                "and makes learned unreliable as a production configuration without "
                "seed-selection or regularization."
            )

    lines.append(
        "product_power=0.90 remains the current final configuration. "
        "Learned gate is treated as an unstable diagnostic branch; H2D/H3 should "
        "investigate whether entitlement BCE loss or initialization changes can "
        "reduce the observed seed variance."
    )

    return lines


def _recommend(learned_ranked: list[dict], product_power_runs: list[dict]) -> str:
    if not learned_ranked:
        return (
            "Insufficient per-seed learned data. "
            "Keep product_power=0.90 as the current final configuration. "
            "Re-run this script after collecting individual seed summaries."
        )

    best       = learned_ranked[0]
    best_macro = best.get("best_dev_macro_f1")
    best_lr    = best.get("location_role_SUP")

    pp_macros  = [r.get("best_dev_macro_f1") for r in product_power_runs if r.get("best_dev_macro_f1") is not None]
    pp_mean    = _mean_or_none(pp_macros) or _H2B_REFERENCE["macro_mean"]
    pp_lr_vals = [r.get("location_role_SUP") for r in product_power_runs if r.get("location_role_SUP") is not None]
    pp_lr_mean = _mean_or_none(pp_lr_vals) or float(_H2B_REFERENCE["location_role_SUP_total"])

    if best_macro is not None and best_macro < pp_mean - 0.010:
        return (
            "Keep product_power=0.90 as the final v7-H1 configuration. "
            "Even the learned best seed does not match product_power=0.90 mean macro. "
            "Schedule H2D to address learned gate instability (e.g., entitlement BCE loss, "
            "ne_bias initialization, or seed-robust regularization)."
        )

    if best_lr is not None and best_lr > pp_lr_mean * 1.3:
        return (
            "Keep product_power=0.90 as the final v7-H1 configuration. "
            "Learned best seed produces more location/role false SUPPORT than product_power=0.90. "
            "Treat learned as a diagnostic branch; explore H2D for stability improvements."
        )

    return (
        "Keep product_power=0.90 as the final v7-H1 configuration. "
        "A single competitive learned seed is insufficient to prefer learned over the "
        "more reliable product gate (H2A learned std=0.0466 vs product std=0.0256). "
        "Learned gate may be revisited in H2D/H3 if seed variance can be reduced."
    )


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------

def _safe_run_slice(r: "dict | None") -> "dict | None":
    """Return a JSON-safe subset of a run record."""
    if r is None:
        return None
    return {k: r.get(k) for k in (
        "filename", "mode", "seed", "product_power", "source_dir",
        "best_dev_macro_f1", "best_dev_acc",
        "v7_support_recall", "v7_ne_recall", "v7_refute_recall",
        "bad_SUP", "location_role_SUP", "missing_SUP",
        "intervention_support_counts", "diag_prob_means",
    )}


def _per_run_table_md(runs: list[dict]) -> str:
    if not runs:
        return "_No runs found in the provided directories._\n"
    header = "| source | mode | seed | power | macro_f1 | acc | SUP_r | NE_r | bad_SUP | lr_SUP | miss_SUP |"
    sep    = "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    rows   = []
    for r in sorted(runs, key=lambda x: (
        x.get("mode") or "",
        x.get("product_power") or 1.0,
        x.get("seed") or 0,
    )):
        def _f(key: str, fmt: str = ".4f") -> str:
            v = r.get(key)
            if v is None:
                return "N/A"
            try:
                return format(float(v), fmt) if fmt != "s" else str(v)
            except (TypeError, ValueError):
                return str(v)

        rows.append(
            f"| {r.get('source_dir','?')} | {r.get('mode') or '?'} "
            f"| {r.get('seed') or '?'} "
            f"| {r.get('product_power') if r.get('product_power') is not None else '-'} "
            f"| {_f('best_dev_macro_f1')} | {_f('best_dev_acc')} "
            f"| {_f('v7_support_recall')} | {_f('v7_ne_recall')} "
            f"| {r.get('bad_SUP') if r.get('bad_SUP') is not None else 'N/A'} "
            f"| {r.get('location_role_SUP') if r.get('location_role_SUP') is not None else 'N/A'} "
            f"| {r.get('missing_SUP') if r.get('missing_SUP') is not None else 'N/A'} |"
        )
    return "\n".join([header, sep] + rows) + "\n"


def _learned_ranking_md(learned_ranked: list[dict]) -> str:
    if not learned_ranked:
        return "_No learned runs found._\n"
    lines = []
    for i, r in enumerate(learned_ranked, 1):
        macro = r.get("best_dev_macro_f1")
        bad   = r.get("bad_SUP")
        lr    = r.get("location_role_SUP")
        lines.append(
            f"{i}. **Seed {r.get('seed', '?')}** — "
            f"macro={f'{macro:.4f}' if macro is not None else 'N/A'}, "
            f"bad_SUP={bad if bad is not None else 'N/A'}, "
            f"location_role_SUP={lr if lr is not None else 'N/A'}"
        )
    return "\n".join(lines) + "\n"


def _bvw_table_md(best: "dict | None", worst: "dict | None") -> str:
    if best is None:
        return "_No learned runs available._\n"
    if worst is None or worst is best:
        return "_Only one learned seed available; best-vs-worst comparison requires >= 2 seeds._\n"
    rows = ""
    for key, label in [
        ("best_dev_macro_f1",  "macro_f1"),
        ("best_dev_acc",       "acc"),
        ("v7_support_recall",  "SUP_r"),
        ("v7_ne_recall",       "NE_r"),
        ("bad_SUP",            "bad_SUP"),
        ("location_role_SUP",  "location_role_SUP"),
        ("missing_SUP",        "missing_SUP"),
    ]:
        vb = best.get(key)
        vw = worst.get(key)
        def _fmt(v: Any) -> str:
            if v is None:
                return "N/A"
            try:
                return f"{float(v):.4f}" if isinstance(v, float) else str(v)
            except (TypeError, ValueError):
                return str(v)
        rows += f"| {label} | {_fmt(vb)} | {_fmt(vw)} |\n"
    header = (
        f"| metric | best (seed {best.get('seed','?')}) "
        f"| worst (seed {worst.get('seed','?')}) |\n"
        "|---|---|---|\n"
    )
    return header + rows


def _diag_table_md(best: "dict | None") -> str:
    if best is None:
        return "_No learned runs available._\n"
    diag = best.get("diag_prob_means") or {}
    rows = ""
    for k in DIAG_PROB_KEYS:
        v = diag.get(k)
        rows += f"| {k} | {f'{v:.4f}' if v is not None else 'N/A (missing in summary)'} |\n"
    return "| signal | mean |\n|---|---|\n" + rows


def _build_markdown(
    all_runs: list[dict],
    learned_ranked: list[dict],
    interpretation: list[str],
    recommendation: str,
    inputs: dict,
) -> str:
    best  = learned_ranked[0] if learned_ranked else None
    worst = learned_ranked[-1] if len(learned_ranked) > 1 else None

    interp_md = "\n".join(f"- {ln}" for ln in interpretation)

    return f"""\
# Stage27-H2C Learned Gate Diagnostic

## Objective

Investigate whether the high seed variance in the `learned` v7-H1 entitlement decision
signal (H2A: macro_std=0.0466, acc_std=0.0454) reflects fundamental gate instability or
a recoverable training issue. Compare per-seed learned runs against the H2A product
baseline and the H2B selected configuration (product_power=0.90).

## Inputs

| Parameter | Value |
|---|---|
| h2a_dir | {inputs.get("h2a_dir", "N/A")} |
| h2b_dir | {inputs.get("h2b_dir", "N/A")} |
| output_md | {inputs.get("output_md", "N/A")} |
| output_json | {inputs.get("output_json", "N/A")} |

### H2A Reference Aggregates (3-seed, no-time)

| mode | macro_mean | macro_std | acc_mean | acc_std | SUP_r_mean | NE_r_mean | REF_r_mean | bad_SUP_total | location_role_SUP_total | predicate_SUP_total | missing_SUP_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| product | 0.940351 | 0.025612 | 0.957870 | 0.018651 | 0.944444 | 0.953086 | 1.0 | 76 | 60 | 7 | 0 |
| min | 0.928270 | 0.020529 | 0.945833 | 0.017067 | 0.988889 | 0.929630 | 1.0 | 114 | 78 | 19 | 0 |
| learned | 0.900660 | 0.046553 | 0.922685 | 0.045354 | 0.966952 | 0.902469 | 1.0 | 136 | 104 | 8 | 0 |

### H2B Selected Result (product_power=0.90)

| power | macro_mean | macro_std | acc_mean | acc_std | SUP_r_mean | NE_r_mean | REF_r_mean | bad_SUP_total | location_role_SUP_total | predicate_SUP_total | missing_SUP_total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.951136 | 0.021341 | 0.968056 | 0.013679 | 0.907407 | 0.972840 | 1.0 | 44 | 33 | 3 | 0 |

## Per-Run Table

{_per_run_table_md(all_runs)}

## Learned Seed Ranking

Ranked by macro_f1 descending, then bad_SUP ascending, then location_role_SUP ascending.

{_learned_ranking_md(learned_ranked)}

## Learned Best-vs-Worst Analysis

{_bvw_table_md(best, worst)}
### Diagnostic Probability Means (Best Seed)

{_diag_table_md(best)}

## Comparison to Product Gates

See Per-Run Table for individual comparisons.

- H2A product mean macro: **0.9404** (std 0.0256), bad_SUP_total=76, location_role_SUP_total=60
- H2B product_power=0.90 mean macro: **0.9511** (std 0.0213), bad_SUP_total=44, location_role_SUP_total=33

## Interpretation

{interp_md}

## Recommendation

{recommendation}

## Remaining Risks

- Results are based on the controlled no-time validation setting
  (`controlled_v5_v3_without_time_swap.jsonl`). Generalization beyond this setting is not
  established.
- time_swap was excluded because earlier Stage12 analysis identified it as
  corrupted/problematic. Results do not cover the time_swap evaluation distribution.
- Per-seed learned analysis carries high uncertainty with only 3 seeds. A single best seed
  cannot establish that learned is reliably competitive.
- Diagnostic probability means (entitlement_prob, frame_prob, etc.) may be absent from
  older summary JSONs that predate Stage26-F extended diagnostics. Missing fields are
  reported as N/A and do not invalidate macro/false-SUPPORT comparisons.
- T4-safe frozen-encoder setting used max_length=64. Claims should be framed as
  controlled-setting evidence unless confirmed by full-encoder runs.

## Conclusion

Stage27-H2C treats learned as an unstable but potentially informative diagnostic branch, while keeping product_power=0.90 as the current final v7-H1 entitlement decision configuration.
"""


def _build_json_report(
    all_runs: list[dict],
    learned_ranked: list[dict],
    interpretation: list[str],
    recommendation: str,
    inputs: dict,
    product_runs: list[dict],
    product_power_runs: list[dict],
) -> dict[str, Any]:
    best  = learned_ranked[0] if learned_ranked else None
    worst = learned_ranked[-1] if len(learned_ranked) > 1 else None

    return {
        "stage": "Stage27-H2C",
        "objective": (
            "Diagnose whether learned v7-H1 entitlement gate instability "
            "(H2A macro_std=0.0466) is recoverable or fundamental, and compare "
            "per-seed learned runs against product_power=0.90 (H2B winner)."
        ),
        "inputs": inputs,
        "h2a_reference": _H2A_REFERENCE,
        "h2b_reference": _H2B_REFERENCE,
        "per_run_results": [_safe_run_slice(r) for r in all_runs],
        "learned_seed_ranking": [_safe_run_slice(r) for r in learned_ranked],
        "learned_best_seed": _safe_run_slice(best),
        "learned_worst_seed": _safe_run_slice(worst),
        "comparison_to_product": {
            "h2a_product_from_files": {
                "n_runs": len(product_runs),
                "macro_mean": _mean_or_none(
                    [r.get("best_dev_macro_f1") for r in product_runs]
                ),
                "bad_SUP_mean": _mean_or_none(
                    [r.get("bad_SUP") for r in product_runs]
                ),
                "location_role_SUP_mean": _mean_or_none(
                    [r.get("location_role_SUP") for r in product_runs]
                ),
            },
            "h2b_product_power_from_files": {
                "n_runs": len(product_power_runs),
                "selected_power": 0.90,
                "macro_mean": _mean_or_none(
                    [r.get("best_dev_macro_f1") for r in product_power_runs]
                ),
                "bad_SUP_mean": _mean_or_none(
                    [r.get("bad_SUP") for r in product_power_runs]
                ),
                "location_role_SUP_mean": _mean_or_none(
                    [r.get("location_role_SUP") for r in product_power_runs]
                ),
            },
            "learned_best_vs_product_power_first_run": _compare_runs(
                best,
                product_power_runs[0] if product_power_runs else None,
            ),
        },
        "interpretation": interpretation,
        "recommendation": recommendation,
        "conclusion": (
            "Stage27-H2C treats learned as an unstable but potentially informative "
            "diagnostic branch, while keeping product_power=0.90 as the current final "
            "v7-H1 entitlement decision configuration."
        ),
        "remaining_risks": [
            "Results are based on the controlled no-time validation setting. "
            "Generalization beyond this setting is not established.",
            "time_swap was excluded because earlier Stage12 analysis identified it as "
            "corrupted/problematic. Results do not cover the time_swap distribution.",
            "Per-seed learned analysis carries high uncertainty with only 3 seeds. "
            "A single best seed cannot establish reliable competitiveness.",
            "Diagnostic probability means may be absent from older summary JSONs; "
            "missing fields are reported as null.",
            "T4-safe frozen-encoder setting (max_length=64). Claims should be framed "
            "as controlled-setting evidence unless confirmed by full-encoder runs.",
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage27-H2C: Analyze per-seed learned v7-H1 entitlement gate runs "
            "and compare against H2A product and H2B product_power=0.90."
        )
    )
    p.add_argument(
        "--h2a-dir",
        type=Path,
        default=Path("results/h2a"),
        help="Directory containing H2A per-seed *summary*.json files (default: results/h2a).",
    )
    p.add_argument(
        "--h2b-dir",
        type=Path,
        default=Path("results/h2b"),
        help="Directory containing H2B per-seed *summary*.json files (default: results/h2b).",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/stage27_h2c_learned_gate_diagnostic.md"),
        help="Output markdown report path.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/stage27_h2c_learned_gate_diagnostic.json"),
        help="Output JSON report path.",
    )
    return p.parse_args(argv)


def main(argv: "list[str] | None" = None) -> int:
    args = parse_args(argv)

    inputs = {
        "h2a_dir": str(args.h2a_dir),
        "h2b_dir": str(args.h2b_dir),
        "output_md": str(args.output_md),
        "output_json": str(args.output_json),
    }

    print(f"[H2C] Scanning H2A: {args.h2a_dir}", file=sys.stderr)
    h2a_runs = _scan_dir(args.h2a_dir, "h2a")
    print(f"[H2C] {len(h2a_runs)} H2A run(s) found.", file=sys.stderr)

    print(f"[H2C] Scanning H2B: {args.h2b_dir}", file=sys.stderr)
    h2b_runs = _scan_dir(args.h2b_dir, "h2b")
    print(f"[H2C] {len(h2b_runs)} H2B run(s) found.", file=sys.stderr)

    all_runs = h2a_runs + h2b_runs

    product_runs = [
        r for r in h2a_runs
        if (r.get("mode") or "").lower() == "product"
        and r.get("product_power") is None
    ]
    product_power_runs = [
        r for r in h2b_runs
        if (r.get("mode") or "").lower() == "product"
        and r.get("product_power") is not None
        and abs((r.get("product_power") or 0.0) - 0.90) < 0.01
    ]

    learned_ranked = _rank_learned(all_runs)
    print(f"[H2C] {len(learned_ranked)} learned run(s) found.", file=sys.stderr)

    interpretation = _interpret(learned_ranked, product_runs, product_power_runs)
    recommendation = _recommend(learned_ranked, product_power_runs)

    md      = _build_markdown(all_runs, learned_ranked, interpretation, recommendation, inputs)
    report  = _build_json_report(
        all_runs, learned_ranked, interpretation, recommendation,
        inputs, product_runs, product_power_runs,
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    args.output_md.write_text(md, encoding="utf-8")
    print(f"[H2C] Wrote: {args.output_md}", file=sys.stderr)

    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[H2C] Wrote: {args.output_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
