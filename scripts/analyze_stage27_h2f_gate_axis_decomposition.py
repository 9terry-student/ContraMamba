"""Stage27-H2F: Gate-axis decomposition analyzer.

Compares learned / product_power=0.90 / product_learned_residual (hybrid) v7-H1
entitlement gates by intervention axis rather than aggregate macro-F1 alone.

Usage:
    python scripts/analyze_stage27_h2f_gate_axis_decomposition.py \\
        --learned-dir  results/h2a_learned \\
        --product-dir  results/h2b_product \\
        --hybrid-dir   results/h2e_hybrid \\
        --output-md    reports/stage27_h2f_gate_axis_decomposition.md \\
        --output-json  reports/stage27_h2f_gate_axis_decomposition.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_INTERVENTIONS: tuple[str, ...] = (
    "none",
    "paraphrase",
    "polarity_flip",
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

FRAME_AXIS: tuple[str, ...] = (
    "location_swap", "role_swap", "entity_swap", "event_swap", "title_name_swap",
)
LOCATION_ROLE_AXIS: tuple[str, ...] = ("location_swap", "role_swap")
PREDICATE_AXIS: tuple[str, ...] = ("predicate_swap",)
EVIDENCE_MISSING_AXIS: tuple[str, ...] = (
    "evidence_deletion", "evidence_truncation", "irrelevant_evidence",
)
CONTROL_AXIS: tuple[str, ...] = ("none", "paraphrase", "polarity_flip")

# bad_SUP covers non-control failure axes
BAD_SUP_AXES: tuple[str, ...] = FRAME_AXIS + PREDICATE_AXIS + EVIDENCE_MISSING_AXIS

DIAG_PROB_KEYS: tuple[str, ...] = (
    "entitlement_prob",
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "v7_h1_entitlement_for_decision",
    "product_base",
    "learned_residual",
)

PREDICTION_LABELS: tuple[str, ...] = ("REFUTE", "NOT_ENTITLED", "SUPPORT")

_RE_SEED    = re.compile(r"seed(\d+)", re.IGNORECASE)
_RE_LEARNED = re.compile(r"learned", re.IGNORECASE)
_RE_HYBRID  = re.compile(r"hybrid|residual", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Utility
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


def _mean(vals: list) -> "float | None":
    valid = [float(v) for v in vals if v is not None]
    return round(sum(valid) / len(valid), 6) if valid else None


def _std(vals: list) -> "float | None":
    valid = [float(v) for v in vals if v is not None]
    n = len(valid)
    if n < 2:
        return None
    mu = sum(valid) / n
    return round(math.sqrt(sum((x - mu) ** 2 for x in valid) / (n - 1)), 6)


def _int_or_none(v: Any) -> "int | None":
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# JSON resolution helpers
# ---------------------------------------------------------------------------

def _resolve_best_dev_interventions(data: dict) -> dict:
    """Find best_dev_interventions from several possible nesting locations."""
    if isinstance(data.get("best_dev_interventions"), dict):
        return data["best_dev_interventions"]
    runs = data.get("runs")
    if isinstance(runs, dict):
        single = runs.get("single")
        if isinstance(single, dict) and isinstance(
            single.get("best_dev_interventions"), dict
        ):
            return single["best_dev_interventions"]
        for v in runs.values():
            if isinstance(v, dict) and isinstance(
                v.get("best_dev_interventions"), dict
            ):
                return v["best_dev_interventions"]
    return {}


def _extract_prediction_dist(block: "dict | None") -> "dict | None":
    """Return {REFUTE, NOT_ENTITLED, SUPPORT} counts or None if block absent."""
    if block is None:
        return None
    pred_dist = block.get("prediction_distribution")
    if isinstance(pred_dist, dict):
        return {label: int(pred_dist.get(label, 0)) for label in PREDICTION_LABELS}
    # Legacy flat keys
    result: dict[str, int] = {}
    for label in PREDICTION_LABELS:
        for key in (label, label.lower(), f"{label.lower()}_predictions",
                    f"{label}_predictions", f"{label}_count"):
            val = block.get(key)
            if val is not None:
                try:
                    result[label] = int(val)
                    break
                except (TypeError, ValueError):
                    pass
        if label not in result:
            result[label] = 0
    return result if any(v > 0 for v in result.values()) else None


def _diag_probs_from_block(block: "dict | None") -> "dict[str, float | None]":
    """Extract per-intervention diagnostic probability means from one block."""
    result: dict[str, "float | None"] = {k: None for k in DIAG_PROB_KEYS}
    if not isinstance(block, dict):
        return result
    for k in DIAG_PROB_KEYS:
        for path in (
            (k,),
            (f"mean_{k}",),
            (k, "mean"),
            ("prob_means", k),
            ("diag_prob_means", k),
        ):
            v = _safe_get(block, *path)
            if isinstance(v, (int, float)):
                result[k] = float(v)
                break
    return result


# ---------------------------------------------------------------------------
# Run-level extraction
# ---------------------------------------------------------------------------

def _infer_seed(data: dict, path: Path) -> "int | None":
    for src in (data, data.get("configuration") or {}):
        v = src.get("seed")
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    m = _RE_SEED.search(path.stem)
    return int(m.group(1)) if m else None


def _infer_mode(data: dict, path: Path, forced_label: str) -> str:
    """Return 'learned', 'product', or 'hybrid' for this run."""
    sig = (
        data.get("v7_h1_entitlement_decision_signal")
        or _safe_get(data, "configuration", "v7_h1_entitlement_decision_signal")
    )
    if sig == "product_learned_residual" or _RE_HYBRID.search(path.stem):
        return "hybrid"
    if sig == "learned" or _RE_LEARNED.search(path.stem):
        return "learned"
    if sig == "product":
        return "product"
    # Fall back to directory label
    return forced_label


def _extract_run(data: dict, path: Path, dir_label: str) -> dict[str, Any]:
    """Build a flat metric record from one run JSON."""
    rec: dict[str, Any] = {
        "filename": path.name,
        "source_file": str(path),
        "dir_label": dir_label,
        "seed": _infer_seed(data, path),
        "mode": _infer_mode(data, path, dir_label),
    }

    # Core metrics
    config: dict = data.get("configuration") or {}
    for key in (
        "best_dev_macro_f1", "best_dev_acc",
        "v7_support_recall", "v7_ne_recall", "v7_refute_recall",
        "v7_h1_entitlement_decision_signal", "v7_h1_entitlement_product_power",
        "v7_h1_hybrid_residual_beta", "v7_h1_entitlement_for_decision_source",
    ):
        val = data.get(key)
        if val is None:
            val = config.get(key)
        rec[key] = val

    # Intervention extraction
    interventions = _resolve_best_dev_interventions(data)
    pred_dists: dict[str, "dict | None"] = {}
    diag_probs: dict[str, "dict[str, float | None]"] = {}
    for iv in ALL_INTERVENTIONS:
        block = interventions.get(iv) if interventions else None
        pred_dists[iv] = _extract_prediction_dist(block)
        diag_probs[iv] = _diag_probs_from_block(block)

    rec["prediction_dists"] = pred_dists
    rec["diag_probs"] = diag_probs

    # Axis SUPPORT counts
    def _sup(iv: str) -> "int | None":
        d = pred_dists.get(iv)
        return d["SUPPORT"] if d is not None else None

    # Per-intervention named counts
    rec["location_SUP"]   = _sup("location_swap")
    rec["role_SUP"]       = _sup("role_swap")
    rec["entity_SUP"]     = _sup("entity_swap")
    rec["event_SUP"]      = _sup("event_swap")
    rec["title_SUP"]      = _sup("title_name_swap")
    rec["predicate_SUP"]  = _sup("predicate_swap")
    rec["deletion_SUP"]   = _sup("evidence_deletion")
    rec["truncation_SUP"] = _sup("evidence_truncation")
    rec["irrelevant_SUP"] = _sup("irrelevant_evidence")

    # Axis aggregates - None if any constituent is None
    def _sum_axis(keys: tuple) -> "int | None":
        vals = [_sup(k) for k in keys]
        return sum(v for v in vals if v is not None) if any(
            v is not None for v in vals
        ) else None

    rec["frame_SUP"]         = _sum_axis(FRAME_AXIS)
    rec["location_role_SUP"] = _sum_axis(LOCATION_ROLE_AXIS)
    rec["predicate_axis_SUP"] = _sum_axis(PREDICATE_AXIS)
    rec["missing_SUP"]       = _sum_axis(EVIDENCE_MISSING_AXIS)
    rec["bad_SUP"]           = _sum_axis(BAD_SUP_AXES)

    # Control axis
    ctrl_vals = [pred_dists.get(iv) for iv in CONTROL_AXIS]
    ctrl_non_ne = None
    ctrl_sup = None
    ctrl_ref = None
    for d in ctrl_vals:
        if d is not None:
            r = d.get("REFUTE", 0)
            s = d.get("SUPPORT", 0)
            ctrl_non_ne = (ctrl_non_ne or 0) + r + s
            ctrl_sup    = (ctrl_sup or 0) + s
            ctrl_ref    = (ctrl_ref or 0) + r
    rec["control_non_ne"] = ctrl_non_ne
    rec["control_support"] = ctrl_sup
    rec["control_refute"]  = ctrl_ref

    # Mean diag probs across all interventions that have data
    agg_diag: dict[str, "float | None"] = {}
    for k in DIAG_PROB_KEYS:
        vals = [
            diag_probs[iv][k]
            for iv in ALL_INTERVENTIONS
            if diag_probs[iv][k] is not None
        ]
        agg_diag[k] = _mean(vals) if vals else None
    rec["mean_diag_probs"] = agg_diag

    return rec


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def _scan_dir(directory: Path, label: str) -> list[dict[str, Any]]:
    if not directory.exists():
        print(f"[H2F] WARNING: {label} dir not found: {directory}", file=sys.stderr)
        return []
    results: list[dict[str, Any]] = []
    for p in sorted(directory.rglob("*summary*.json")):
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[H2F] WARNING: could not read {p}: {exc}", file=sys.stderr)
            continue
        results.append(_extract_run(data, p, label))
    print(f"[H2F] {label}: {len(results)} run(s) found.", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_by_config(runs: list[dict]) -> dict[str, Any]:
    if not runs:
        return {"n_runs": 0}

    def _col(key: str) -> list:
        return [r.get(key) for r in runs]

    macros  = [r.get("best_dev_macro_f1") for r in runs if r.get("best_dev_macro_f1") is not None]
    accs    = [r.get("best_dev_acc")       for r in runs if r.get("best_dev_acc")       is not None]
    sup_rs  = [r.get("v7_support_recall")  for r in runs if r.get("v7_support_recall")  is not None]
    ne_rs   = [r.get("v7_ne_recall")       for r in runs if r.get("v7_ne_recall")       is not None]
    ref_rs  = [r.get("v7_refute_recall")   for r in runs if r.get("v7_refute_recall")   is not None]

    bad_sups  = [r.get("bad_SUP")           for r in runs if r.get("bad_SUP")           is not None]
    frame_sups = [r.get("frame_SUP")        for r in runs if r.get("frame_SUP")         is not None]
    lr_sups   = [r.get("location_role_SUP") for r in runs if r.get("location_role_SUP") is not None]
    pred_sups = [r.get("predicate_axis_SUP") for r in runs if r.get("predicate_axis_SUP") is not None]
    miss_sups = [r.get("missing_SUP")       for r in runs if r.get("missing_SUP")       is not None]
    ctrl_nne  = [r.get("control_non_ne")    for r in runs if r.get("control_non_ne")    is not None]

    return {
        "n_runs": len(runs),
        "macro_mean": _mean(macros),
        "macro_std":  _std(macros),
        "acc_mean":   _mean(accs),
        "acc_std":    _std(accs),
        "SUP_r_mean": _mean(sup_rs),
        "NE_r_mean":  _mean(ne_rs),
        "REF_r_mean": _mean(ref_rs),
        "bad_SUP_total": sum(bad_sups) if bad_sups else None,
        "bad_SUP_mean":  _mean(bad_sups),
        "frame_SUP_total": sum(frame_sups) if frame_sups else None,
        "frame_SUP_mean":  _mean(frame_sups),
        "location_role_SUP_total": sum(lr_sups) if lr_sups else None,
        "location_role_SUP_mean":  _mean(lr_sups),
        "predicate_SUP_total": sum(pred_sups) if pred_sups else None,
        "predicate_SUP_mean":  _mean(pred_sups),
        "missing_SUP_total": sum(miss_sups) if miss_sups else None,
        "control_non_ne_mean": _mean(ctrl_nne),
    }


def _per_intervention_support_by_config(
    learned: list[dict],
    product: list[dict],
    hybrid: list[dict],
) -> dict[str, Any]:
    """Return mean SUPPORT count per intervention for each config."""
    result: dict[str, Any] = {}
    configs = {"learned": learned, "product": product, "hybrid": hybrid}
    for iv in ALL_INTERVENTIONS:
        result[iv] = {}
        for label, runs in configs.items():
            vals = []
            for r in runs:
                d = r.get("prediction_dists", {}).get(iv)
                if d is not None:
                    vals.append(d["SUPPORT"])
            result[iv][label] = {
                "n": len(vals),
                "sup_mean": _mean(vals),
                "sup_total": sum(vals) if vals else None,
            }
    return result


# ---------------------------------------------------------------------------
# Paired seed delta
# ---------------------------------------------------------------------------

def _paired_delta(
    runs_a: list[dict],
    runs_b: list[dict],
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """Compute per-seed deltas between config A and config B."""
    a_by_seed = {r["seed"]: r for r in runs_a if r.get("seed") is not None}
    b_by_seed = {r["seed"]: r for r in runs_b if r.get("seed") is not None}
    common = sorted(set(a_by_seed) & set(b_by_seed))

    rows: list[dict] = []
    for seed in common:
        a, b = a_by_seed[seed], b_by_seed[seed]

        def _delta(key: str) -> "float | None":
            va, vb = a.get(key), b.get(key)
            if va is not None and vb is not None:
                try:
                    return round(float(va) - float(vb), 6)
                except (TypeError, ValueError):
                    pass
            return None

        def _int_delta(key: str) -> "int | None":
            va, vb = _int_or_none(a.get(key)), _int_or_none(b.get(key))
            return va - vb if va is not None and vb is not None else None

        rows.append({
            "seed": seed,
            f"{label_a}_macro": a.get("best_dev_macro_f1"),
            f"{label_b}_macro": b.get("best_dev_macro_f1"),
            "macro_delta": _delta("best_dev_macro_f1"),
            "acc_delta":   _delta("best_dev_acc"),
            "SUP_r_delta": _delta("v7_support_recall"),
            "NE_r_delta":  _delta("v7_ne_recall"),
            "bad_SUP_delta":           _int_delta("bad_SUP"),
            "frame_SUP_delta":         _int_delta("frame_SUP"),
            "location_role_SUP_delta": _int_delta("location_role_SUP"),
            "predicate_SUP_delta":     _int_delta("predicate_axis_SUP"),
            "missing_SUP_delta":       _int_delta("missing_SUP"),
        })

    macro_deltas = [r["macro_delta"] for r in rows if r.get("macro_delta") is not None]
    bad_deltas   = [r["bad_SUP_delta"] for r in rows if r.get("bad_SUP_delta") is not None]

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_paired": len(rows),
        "paired_rows": rows,
        "macro_delta_mean": _mean(macro_deltas),
        "bad_SUP_delta_mean": _mean(bad_deltas),
    }


# ---------------------------------------------------------------------------
# Learned seed specialization
# ---------------------------------------------------------------------------

def _learned_specialization(runs: list[dict]) -> dict[str, Any]:
    if not runs:
        return {"n_seeds": 0, "ranking": [], "notes": {}}

    def _sort_key(r: dict) -> tuple:
        macro = r.get("best_dev_macro_f1")
        bad   = r.get("bad_SUP")
        lr    = r.get("location_role_SUP")
        return (
            -(macro if macro is not None else -1.0),
             bad if bad is not None else 99_999,
             lr  if lr  is not None else 99_999,
        )

    ranked = sorted(runs, key=_sort_key)

    ranking = []
    for i, r in enumerate(ranked, 1):
        ranking.append({
            "rank": i,
            "seed": r.get("seed"),
            "macro": r.get("best_dev_macro_f1"),
            "acc":   r.get("best_dev_acc"),
            "SUP_r": r.get("v7_support_recall"),
            "NE_r":  r.get("v7_ne_recall"),
            "bad_SUP":           r.get("bad_SUP"),
            "frame_SUP":         r.get("frame_SUP"),
            "location_role_SUP": r.get("location_role_SUP"),
            "predicate_SUP":     r.get("predicate_axis_SUP"),
            "missing_SUP":       r.get("missing_SUP"),
        })

    # Notes: best by axis
    def _best_seed(key: str, minimize: bool = False) -> "int | None":
        valid = [r for r in runs if r.get(key) is not None]
        if not valid:
            return None
        s = min(valid, key=lambda r: r[key]) if minimize else max(valid, key=lambda r: r[key])
        return s.get("seed")

    notes = {
        "best_macro_seed":         _best_seed("best_dev_macro_f1"),
        "lowest_bad_SUP_seed":     _best_seed("bad_SUP", minimize=True),
        "lowest_location_role_SUP_seed": _best_seed("location_role_SUP", minimize=True),
        "best_NE_r_seed":          _best_seed("v7_ne_recall"),
    }

    return {"n_seeds": len(runs), "ranking": ranking, "notes": notes}


# ---------------------------------------------------------------------------
# Interpretation flags
# ---------------------------------------------------------------------------

def _interpretation_flags(
    learned: list[dict],
    product: list[dict],
    hybrid:  list[dict],
    agg: dict[str, Any],
) -> dict[str, Any]:
    prod_agg = agg.get("product") or {}
    hyb_agg  = agg.get("hybrid") or {}
    lrn_agg  = agg.get("learned") or {}

    prod_macro_mean = prod_agg.get("macro_mean")
    prod_bad_mean   = prod_agg.get("bad_SUP_mean")
    hyb_macro_mean  = hyb_agg.get("macro_mean")
    hyb_bad_total   = hyb_agg.get("bad_SUP_total")
    prod_bad_total  = prod_agg.get("bad_SUP_total")

    # learned_has_local_signal
    learned_has_local_signal = False
    if product and learned:
        for lr in learned:
            lm = lr.get("best_dev_macro_f1")
            lb = lr.get("bad_SUP")
            if (
                (lm is not None and prod_macro_mean is not None and lm >= prod_macro_mean - 0.01)
                or (lb is not None and prod_bad_mean is not None and lb <= prod_bad_mean + 5)
            ):
                learned_has_local_signal = True
                break

    # learned_unstable
    lrn_macros = [r.get("best_dev_macro_f1") for r in learned if r.get("best_dev_macro_f1") is not None]
    lrn_bads   = [r.get("bad_SUP") for r in learned if r.get("bad_SUP") is not None]
    macro_spread = (max(lrn_macros) - min(lrn_macros)) if len(lrn_macros) >= 2 else None
    bad_spread   = (max(lrn_bads)   - min(lrn_bads))   if len(lrn_bads)   >= 2 else None
    learned_unstable = bool(
        (macro_spread is not None and macro_spread >= 0.05)
        or (bad_spread is not None and bad_spread >= 40)
    )

    # hybrid_has_local_gain: any hybrid seed beats same-seed product in macro or bad_SUP
    prod_by_seed = {r["seed"]: r for r in product if r.get("seed") is not None}
    hybrid_has_local_gain = False
    for hr in hybrid:
        seed = hr.get("seed")
        pr = prod_by_seed.get(seed)
        if pr is None:
            continue
        hm = hr.get("best_dev_macro_f1")
        pm = pr.get("best_dev_macro_f1")
        hb = hr.get("bad_SUP")
        pb = pr.get("bad_SUP")
        if (hm is not None and pm is not None and hm > pm) or (
            hb is not None and pb is not None and hb < pb
        ):
            hybrid_has_local_gain = True
            break

    # hybrid_not_final
    hybrid_not_final = bool(
        (hyb_macro_mean is not None and prod_macro_mean is not None and hyb_macro_mean < prod_macro_mean)
        and (hyb_bad_total is not None and prod_bad_total is not None and hyb_bad_total > prod_bad_total)
    )

    # hard_axis: which axis contributes most false SUPPORT in product runs
    axis_totals = {
        "location_role": sum(r.get("location_role_SUP") or 0 for r in product),
        "predicate":     sum(r.get("predicate_axis_SUP") or 0 for r in product),
        "missing":       sum(r.get("missing_SUP") or 0 for r in product),
    }
    other = sum(r.get("entity_SUP") or 0 for r in product) + \
            sum(r.get("event_SUP") or 0 for r in product) + \
            sum(r.get("title_SUP") or 0 for r in product)
    axis_totals["other"] = other
    hard_axis = max(axis_totals, key=lambda k: axis_totals[k]) if any(
        axis_totals.values()
    ) else "unknown"

    return {
        "learned_has_local_signal": learned_has_local_signal,
        "learned_unstable": learned_unstable,
        "macro_spread": round(macro_spread, 6) if macro_spread is not None else None,
        "bad_SUP_spread": int(bad_spread) if bad_spread is not None else None,
        "hybrid_has_local_gain": hybrid_has_local_gain,
        "hybrid_not_final": hybrid_not_final,
        "hard_axis": hard_axis,
        "axis_false_SUP_totals_product": axis_totals,
    }


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any, fmt: str = ".4f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def _fmt_int(v: Any) -> str:
    if v is None:
        return "N/A"
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return str(v)


def _aggregate_table_md(agg: dict[str, Any]) -> str:
    headers = [
        "config", "n", "macro_mean", "macro_std", "acc_mean",
        "SUP_r", "NE_r", "REF_r",
        "bad_SUP_total", "frame_SUP_total", "lr_SUP_total", "pred_SUP_total", "miss_SUP_total",
    ]
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    hrow = "| " + " | ".join(headers) + " |"
    rows = []
    for cfg in ("learned", "product", "hybrid"):
        a = agg.get(cfg) or {}
        rows.append(
            f"| {cfg} | {a.get('n_runs','0')} "
            f"| {_fmt(a.get('macro_mean'))} | {_fmt(a.get('macro_std'))} "
            f"| {_fmt(a.get('acc_mean'))} "
            f"| {_fmt(a.get('SUP_r_mean'))} | {_fmt(a.get('NE_r_mean'))} | {_fmt(a.get('REF_r_mean'))} "
            f"| {_fmt_int(a.get('bad_SUP_total'))} "
            f"| {_fmt_int(a.get('frame_SUP_total'))} "
            f"| {_fmt_int(a.get('location_role_SUP_total'))} "
            f"| {_fmt_int(a.get('predicate_SUP_total'))} "
            f"| {_fmt_int(a.get('missing_SUP_total'))} |"
        )
    return "\n".join([hrow, sep] + rows) + "\n"


def _per_run_table_md(all_runs: list[dict]) -> str:
    if not all_runs:
        return "_No runs found._\n"
    headers = [
        "config", "seed", "macro", "acc", "SUP_r", "NE_r",
        "bad_SUP", "frame_SUP", "lr_SUP", "pred_SUP", "miss_SUP",
    ]
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    hrow = "| " + " | ".join(headers) + " |"
    rows = []
    for r in sorted(all_runs, key=lambda x: (x.get("mode",""), x.get("seed") or 0)):
        rows.append(
            f"| {r.get('mode','?')} | {r.get('seed','?')} "
            f"| {_fmt(r.get('best_dev_macro_f1'))} | {_fmt(r.get('best_dev_acc'))} "
            f"| {_fmt(r.get('v7_support_recall'))} | {_fmt(r.get('v7_ne_recall'))} "
            f"| {_fmt_int(r.get('bad_SUP'))} "
            f"| {_fmt_int(r.get('frame_SUP'))} "
            f"| {_fmt_int(r.get('location_role_SUP'))} "
            f"| {_fmt_int(r.get('predicate_axis_SUP'))} "
            f"| {_fmt_int(r.get('missing_SUP'))} |"
        )
    return "\n".join([hrow, sep] + rows) + "\n"


def _per_intervention_md(per_int: dict[str, Any]) -> str:
    headers = ["intervention", "learned_sup_mean", "product_sup_mean", "hybrid_sup_mean"]
    sep  = "|" + "|".join(["---"] * len(headers)) + "|"
    hrow = "| " + " | ".join(headers) + " |"
    rows = []
    for iv in ALL_INTERVENTIONS:
        d = per_int.get(iv, {})
        lm = _fmt(d.get("learned", {}).get("sup_mean"))
        pm = _fmt(d.get("product", {}).get("sup_mean"))
        hm = _fmt(d.get("hybrid",  {}).get("sup_mean"))
        rows.append(f"| {iv} | {lm} | {pm} | {hm} |")
    return "\n".join([hrow, sep] + rows) + "\n"


def _delta_table_md(delta: dict[str, Any], label_a: str, label_b: str) -> str:
    rows_data = delta.get("paired_rows", [])
    if not rows_data:
        return f"_No paired seeds found between {label_a} and {label_b}._\n"
    headers = [
        "seed", f"{label_a}_macro", f"{label_b}_macro", "macro_delta",
        "NE_r_delta", "bad_SUP_delta", "frame_SUP_delta", "lr_SUP_delta",
        "pred_SUP_delta", "miss_SUP_delta",
    ]
    sep  = "|" + "|".join(["---"] * len(headers)) + "|"
    hrow = "| " + " | ".join(headers) + " |"
    rows = []
    for row in rows_data:
        rows.append(
            f"| {row.get('seed','?')} "
            f"| {_fmt(row.get(f'{label_a}_macro'))} "
            f"| {_fmt(row.get(f'{label_b}_macro'))} "
            f"| {_fmt(row.get('macro_delta'))} "
            f"| {_fmt(row.get('NE_r_delta'))} "
            f"| {_fmt_int(row.get('bad_SUP_delta'))} "
            f"| {_fmt_int(row.get('frame_SUP_delta'))} "
            f"| {_fmt_int(row.get('location_role_SUP_delta'))} "
            f"| {_fmt_int(row.get('predicate_SUP_delta'))} "
            f"| {_fmt_int(row.get('missing_SUP_delta'))} |"
        )
    return "\n".join([hrow, sep] + rows) + "\n"


def _specialization_md(spec: dict[str, Any]) -> str:
    ranking = spec.get("ranking", [])
    if not ranking:
        return "_No learned runs found._\n"
    headers = ["rank", "seed", "macro", "NE_r", "bad_SUP", "lr_SUP", "pred_SUP", "miss_SUP"]
    sep  = "|" + "|".join(["---"] * len(headers)) + "|"
    hrow = "| " + " | ".join(headers) + " |"
    rows = [
        f"| {r['rank']} | {r.get('seed','?')} "
        f"| {_fmt(r.get('macro'))} | {_fmt(r.get('NE_r'))} "
        f"| {_fmt_int(r.get('bad_SUP'))} "
        f"| {_fmt_int(r.get('location_role_SUP'))} "
        f"| {_fmt_int(r.get('predicate_SUP'))} "
        f"| {_fmt_int(r.get('missing_SUP'))} |"
        for r in ranking
    ]
    notes = spec.get("notes", {})
    notes_md = "\n".join(
        f"- **{k}**: seed {v}" for k, v in notes.items() if v is not None
    ) or "_Not enough data to determine._"
    return "\n".join([hrow, sep] + rows) + "\n\n**Notes:**\n" + notes_md + "\n"


def _interpretation_md(flags: dict[str, Any], agg: dict[str, Any]) -> str:
    lines: list[str] = []

    hard = flags.get("hard_axis", "unknown")
    axis_totals = flags.get("axis_false_SUP_totals_product") or {}
    lines.append(
        f"**Dominant false-SUPPORT axis (product runs):** `{hard}` "
        f"(location_role={axis_totals.get('location_role',0)}, "
        f"predicate={axis_totals.get('predicate',0)}, "
        f"missing={axis_totals.get('missing',0)}, "
        f"other={axis_totals.get('other',0)})."
    )

    if flags.get("learned_unstable"):
        spread = flags.get("macro_spread")
        bad_sp = flags.get("bad_SUP_spread")
        lines.append(
            f"Learned gate is unstable: macro spread={_fmt(spread)}, "
            f"bad_SUP spread={_fmt_int(bad_sp)}. "
            "High seed variance prevents learned from being a reliable production configuration."
        )
    else:
        lines.append(
            "Learned gate shows moderate seed variance. "
            "Instability may be reducible with targeted regularization."
        )

    if flags.get("learned_has_local_signal"):
        lines.append(
            "At least one learned seed matches or approaches product macro / bad_SUP. "
            "Learned should be preserved as a diagnostic and specialization branch, "
            "not discarded, in case axis-level signal emerges with stabilization."
        )
    else:
        lines.append(
            "No learned seed reaches product-level performance on macro or bad_SUP. "
            "Learned gate is currently weaker than product on all seeds in these runs."
        )

    if flags.get("hybrid_has_local_gain"):
        lines.append(
            "At least one hybrid seed beats the same-seed product on macro or bad_SUP. "
            "Naive residual injection is still not safe as a default, but the learned residual "
            "contains local correction signal worth investigating further."
        )
    else:
        lines.append(
            "No hybrid seed beats its same-seed product counterpart on either macro or bad_SUP. "
            "Residual injection does not provide a consistent per-seed advantage."
        )

    if flags.get("hybrid_not_final"):
        lines.append(
            "Hybrid aggregate macro is below product and hybrid bad_SUP_total exceeds product. "
            "product_power=0.90 remains the stable final Stage27 v7-H1 configuration."
        )

    lines.append(
        "Gate behavior should be decomposed by intervention axis rather than ranked solely "
        "by aggregate macro-F1. Location/role frame mismatch is the primary false-SUPPORT "
        "failure mode; any improvement mechanism should be evaluated on this axis specifically."
    )

    return "\n\n".join(f"- {ln}" for ln in lines) + "\n"


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def _build_markdown(
    all_runs: list[dict],
    agg: dict[str, Any],
    per_int: dict[str, Any],
    delta_lp: dict[str, Any],
    delta_hp: dict[str, Any],
    spec: dict[str, Any],
    flags: dict[str, Any],
    inputs: dict[str, str],
) -> str:
    decision_lines = [
        "- **Final configuration:** product_power=0.90 remains the stable final Stage27 v7-H1 configuration.",
        "- **Learned status:** preserve as diagnostic / specialization branch if local axis-level signal is observed.",
        "- **Hybrid status:** preserve as negative-but-informative residual experiment unless aggregate improves.",
        "- **Next recommended stage:** Stage27-H3 final evidence package, unless H2F reveals a sharply specialized axis worth isolating.",
    ]

    return f"""\
# Stage27-H2F Gate-Axis Decomposition

## Objective

Compare learned / product_power=0.90 / product_learned_residual (hybrid) v7-H1 entitlement
gates by intervention axis rather than aggregate macro-F1 alone. Determine whether any
gate configuration shows specialized axis-level advantages that aggregate metrics obscure.

## Inputs

| Parameter | Value |
|---|---|
| learned_dir | {inputs.get("learned_dir", "N/A")} |
| product_dir | {inputs.get("product_dir", "N/A")} |
| hybrid_dir | {inputs.get("hybrid_dir", "N/A")} |
| output_md | {inputs.get("output_md", "N/A")} |
| output_json | {inputs.get("output_json", "N/A")} |

## Method

Each directory is scanned recursively for `*summary*.json` files. Per-run metrics and
intervention prediction distributions are extracted robustly from multiple possible
nesting locations (`best_dev_interventions`, `runs.single.best_dev_interventions`, or
the first run object under `runs` that contains `best_dev_interventions`).

Axis buckets:
- **frame_axis**: location_swap, role_swap, entity_swap, event_swap, title_name_swap
- **location_role_axis**: location_swap, role_swap
- **predicate_axis**: predicate_swap
- **evidence_missing_axis**: evidence_deletion, evidence_truncation, irrelevant_evidence
- **control_axis**: none, paraphrase, polarity_flip

`bad_SUP` = SUPPORT counts over frame_axis + predicate_axis + evidence_missing_axis.

## Aggregate Comparison

{_aggregate_table_md(agg)}

## Per-Run Axis Table

{_per_run_table_md(all_runs)}

## Per-Intervention SUPPORT Decomposition

Mean SUPPORT prediction count per intervention type, by configuration.

{_per_intervention_md(per_int)}

## Paired Seed Delta: Learned vs Product

{_delta_table_md(delta_lp, "learned", "product")}

## Paired Seed Delta: Hybrid vs Product

{_delta_table_md(delta_hp, "hybrid", "product")}

## Learned Seed Specialization

Learned seeds ranked by macro_f1 desc, bad_SUP asc, location_role_SUP asc.

{_specialization_md(spec)}

## Interpretation

{_interpretation_md(flags, agg)}

## Decision

{chr(10).join(decision_lines)}

## Remaining Risks

- Results are based on the controlled no-time validation setting
  (`controlled_v5_v3_without_time_swap.jsonl`). Generalization beyond this setting is
  not established.
- time_swap was excluded because earlier Stage12 analysis identified it as
  corrupted/problematic.
- Axis decomposition is only as reliable as the intervention coverage in the
  summary JSONs. Missing intervention blocks are reported as N/A and excluded from
  aggregates.
- T4-safe frozen-encoder setting used max_length=64. Claims should be framed as
  controlled-setting evidence unless full-encoder runs confirm the same ordering.
- Paired delta analysis requires the same seed number to appear in both configs.
  Mismatched seed sets will produce fewer or no paired rows.
"""


def _build_json_report(
    all_runs: list[dict],
    agg: dict[str, Any],
    per_int: dict[str, Any],
    delta_lp: dict[str, Any],
    delta_hp: dict[str, Any],
    spec: dict[str, Any],
    flags: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    def _run_slice(r: dict) -> dict:
        return {k: r.get(k) for k in (
            "filename", "mode", "seed", "dir_label",
            "best_dev_macro_f1", "best_dev_acc",
            "v7_support_recall", "v7_ne_recall", "v7_refute_recall",
            "v7_h1_entitlement_decision_signal",
            "v7_h1_entitlement_product_power",
            "v7_h1_hybrid_residual_beta",
            "bad_SUP", "frame_SUP", "location_role_SUP",
            "predicate_axis_SUP", "missing_SUP",
            "location_SUP", "role_SUP", "entity_SUP",
            "event_SUP", "title_SUP", "predicate_SUP",
            "deletion_SUP", "truncation_SUP", "irrelevant_SUP",
            "control_non_ne", "control_support", "control_refute",
            "mean_diag_probs",
        )}

    return {
        "stage": "Stage27-H2F",
        "objective": (
            "Compare learned / product_power=0.90 / product_learned_residual v7-H1 gates "
            "by intervention axis to identify axis-level specialization obscured by aggregate metrics."
        ),
        "input_dirs": inputs,
        "per_run_table": [_run_slice(r) for r in all_runs],
        "aggregate_by_config": agg,
        "per_intervention_support_by_config": per_int,
        "paired_seed_delta_tables": {
            "learned_minus_product": delta_lp,
            "hybrid_minus_product": delta_hp,
        },
        "learned_seed_specialization": spec,
        "interpretation_flags": flags,
        "decision": {
            "final_config":
                "product_power=0.90 remains the stable final Stage27 v7-H1 configuration.",
            "learned_status":
                "preserve as diagnostic / specialization branch if local axis-level signal is observed.",
            "hybrid_status":
                "preserve as negative-but-informative residual experiment unless aggregate improves.",
            "next_recommended_stage": (
                "Stage27-H3 final evidence package, unless H2F reveals a sharply "
                "specialized axis worth isolating."
            ),
        },
        "remaining_risks": [
            "Results are based on the controlled no-time validation setting. "
            "Generalization beyond this setting is not established.",
            "time_swap was excluded because earlier Stage12 analysis identified it as "
            "corrupted/problematic.",
            "Axis decomposition is only as reliable as the intervention coverage in the "
            "summary JSONs. Missing blocks are reported as null.",
            "T4-safe frozen-encoder setting (max_length=64). Claims should be framed as "
            "controlled-setting evidence unless full-encoder runs confirm the same ordering.",
            "Paired delta analysis requires matching seed numbers across configs; "
            "mismatched seed sets produce fewer or no paired rows.",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage27-H2F: Gate-axis decomposition - compare learned / product / hybrid "
            "v7-H1 entitlement gates by intervention axis rather than aggregate macro-F1."
        )
    )
    p.add_argument(
        "--learned-dir",
        type=Path,
        default=Path("results/h2a_learned"),
        help="Directory of learned-gate per-seed *summary*.json files (default: results/h2a_learned).",
    )
    p.add_argument(
        "--product-dir",
        type=Path,
        default=Path("results/h2b_product"),
        help="Directory of product_power=0.90 per-seed *summary*.json files (default: results/h2b_product).",
    )
    p.add_argument(
        "--hybrid-dir",
        type=Path,
        default=Path("results/h2e_hybrid"),
        help="Directory of product_learned_residual per-seed *summary*.json files (default: results/h2e_hybrid).",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/stage27_h2f_gate_axis_decomposition.md"),
        help="Output markdown report path.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/stage27_h2f_gate_axis_decomposition.json"),
        help="Output JSON report path.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    args = parse_args(argv)

    inputs = {
        "learned_dir": str(args.learned_dir),
        "product_dir": str(args.product_dir),
        "hybrid_dir":  str(args.hybrid_dir),
        "output_md":   str(args.output_md),
        "output_json": str(args.output_json),
    }

    learned_runs = _scan_dir(args.learned_dir, "learned")
    product_runs = _scan_dir(args.product_dir, "product")
    hybrid_runs  = _scan_dir(args.hybrid_dir,  "hybrid")
    all_runs = learned_runs + product_runs + hybrid_runs

    agg = {
        "learned": _aggregate_by_config(learned_runs),
        "product": _aggregate_by_config(product_runs),
        "hybrid":  _aggregate_by_config(hybrid_runs),
    }

    per_int = _per_intervention_support_by_config(learned_runs, product_runs, hybrid_runs)
    delta_lp = _paired_delta(learned_runs, product_runs, "learned", "product")
    delta_hp = _paired_delta(hybrid_runs,  product_runs, "hybrid",  "product")
    spec  = _learned_specialization(learned_runs)
    flags = _interpretation_flags(learned_runs, product_runs, hybrid_runs, agg)

    md     = _build_markdown(all_runs, agg, per_int, delta_lp, delta_hp, spec, flags, inputs)
    report = _build_json_report(all_runs, agg, per_int, delta_lp, delta_hp, spec, flags, inputs)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    args.output_md.write_text(md, encoding="utf-8")
    print(f"[H2F] Wrote: {args.output_md}", file=sys.stderr)

    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[H2F] Wrote: {args.output_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
