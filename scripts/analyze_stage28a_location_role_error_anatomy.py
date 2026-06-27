"""Stage28-A: Location/Role Error Anatomy analyzer.

Inspects prediction JSONs and summary JSONs from learned/product/hybrid runs to
determine whether location/role frame mismatch errors are structured enough to
justify a future location-role-specific boundary module (Stage28-B/C).

Usage:
    python scripts/analyze_stage28a_location_role_error_anatomy.py \\
        --learned-dir  /kaggle/working/stage27_h2a_learned_rerun_notimeswap \\
        --product-dir  /kaggle/working/stage27_h2b_0p90_rerun_notimeswap \\
        --hybrid-dir   /kaggle/working/stage27_h2e_product_learned_residual_beta0p2_3seed_notimeswap \\
        --output-md    reports/stage28a_location_role_error_anatomy.md \\
        --output-json  reports/stage28a_location_role_error_anatomy.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FOCUS_INTERVENTIONS: tuple[str, ...] = ("location_swap", "role_swap")

ALL_INTERVENTIONS: tuple[str, ...] = (
    "none", "paraphrase", "polarity_flip",
    "location_swap", "role_swap", "predicate_swap",
    "entity_swap", "event_swap", "title_name_swap",
    "evidence_deletion", "evidence_truncation", "irrelevant_evidence",
)

PREDICTION_LABELS: tuple[str, ...] = ("REFUTE", "NOT_ENTITLED", "SUPPORT")

PROB_KEYS: tuple[str, ...] = (
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "entitlement_prob",
    "v7_h1_entitlement_for_decision",
    "product_base",
    "learned_residual",
    "support_score",
    "refute_score",
    "ne_score",
)

_ID_KEYS     = ("id", "example_id", "uid", "index", "idx", "record_id")
_INTERV_KEYS = (
    "intervention", "intervention_type", "perturbation",
    "probe_type", "stage15_probe_type", "controlled_intervention",
)
_GOLD_KEYS  = ("gold", "label", "gold_label", "y_true", "target")
_PRED_KEYS  = ("pred", "prediction", "predicted_label", "y_pred")
_TEXT_KEYS  = ("claim", "evidence", "premise", "hypothesis", "sentence1", "sentence2")

_PREDICTION_LIKE_FIELDS = set(
    _ID_KEYS + _INTERV_KEYS + _GOLD_KEYS + _PRED_KEYS + PROB_KEYS
)

_LABEL_NORM: dict[str, str] = {
    "supports": "SUPPORT", "support": "SUPPORT", "2": "SUPPORT",
    "refutes": "REFUTE",   "refute": "REFUTE",   "0": "REFUTE",
    "not_entitled": "NOT_ENTITLED", "ne": "NOT_ENTITLED",
    "not_enough_info": "NOT_ENTITLED", "1": "NOT_ENTITLED",
}

_RE_SEED = re.compile(r"seed(\d+)", re.IGNORECASE)


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


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

def _normalize_label(raw: Any) -> "str | None":
    if raw is None:
        return None
    s = str(raw).strip().lower()
    return _LABEL_NORM.get(s)


def _extract_label(rec: dict, keys: tuple) -> "tuple[Any, str | None]":
    """Return (raw_value, normalized_label) from first matching key."""
    for k in keys:
        v = rec.get(k)
        if v is not None:
            return v, _normalize_label(v)
    return None, None


# ---------------------------------------------------------------------------
# Summary JSON extraction
# ---------------------------------------------------------------------------

def _resolve_best_dev_interventions(data: dict) -> dict:
    """Find best_dev_interventions at multiple possible nesting locations."""
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


def _pred_dist_from_block(block: "dict | None") -> "dict | None":
    if block is None:
        return None
    pd = block.get("prediction_distribution")
    if isinstance(pd, dict):
        return {lbl: int(pd.get(lbl, 0)) for lbl in PREDICTION_LABELS}
    # Legacy flat counts
    result: dict[str, int] = {}
    for lbl in PREDICTION_LABELS:
        for key in (lbl, lbl.lower(), f"{lbl.lower()}_predictions", f"{lbl}_count"):
            v = block.get(key)
            if v is not None:
                try:
                    result[lbl] = int(v)
                    break
                except (TypeError, ValueError):
                    pass
        if lbl not in result:
            result[lbl] = 0
    return result if any(v > 0 for v in result.values()) else None


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


def _extract_summary_run(data: dict, path: Path, dir_label: str) -> dict[str, Any]:
    """Extract aggregate metrics and per-intervention counts from a summary JSON."""
    config: dict = data.get("configuration") or {}
    seed = _infer_seed(data, path)

    interventions = _resolve_best_dev_interventions(data)

    def _cnt(iv: str, lbl: str) -> "int | None":
        block = interventions.get(iv) if interventions else None
        pd = _pred_dist_from_block(block)
        return pd[lbl] if pd else None

    loc_sup  = _cnt("location_swap", "SUPPORT")
    loc_ne   = _cnt("location_swap", "NOT_ENTITLED")
    loc_ref  = _cnt("location_swap", "REFUTE")
    role_sup = _cnt("role_swap", "SUPPORT")
    role_ne  = _cnt("role_swap", "NOT_ENTITLED")
    role_ref = _cnt("role_swap", "REFUTE")

    lr_sup = (
        (loc_sup or 0) + (role_sup or 0)
        if (loc_sup is not None or role_sup is not None) else None
    )
    lr_ne = (
        (loc_ne or 0) + (role_ne or 0)
        if (loc_ne is not None or role_ne is not None) else None
    )
    lr_total = (
        (lr_sup or 0) + (lr_ne or 0) + ((loc_ref or 0) + (role_ref or 0))
        if lr_sup is not None or lr_ne is not None else None
    )
    lr_false_sup_rate = (
        round(lr_sup / lr_total, 6) if lr_total and lr_total > 0 else None
    )

    return {
        "filename": path.name,
        "source_file": str(path),
        "dir_label": dir_label,
        "seed": seed,
        "config": dir_label,
        "best_dev_macro_f1": data.get("best_dev_macro_f1"),
        "best_dev_acc":      data.get("best_dev_acc"),
        "v7_support_recall": data.get("v7_support_recall"),
        "v7_ne_recall":      data.get("v7_ne_recall"),
        "v7_refute_recall":  data.get("v7_refute_recall"),
        "v7_h1_entitlement_decision_signal": (
            data.get("v7_h1_entitlement_decision_signal")
            or config.get("v7_h1_entitlement_decision_signal")
        ),
        "v7_h1_entitlement_product_power": (
            data.get("v7_h1_entitlement_product_power")
            or config.get("v7_h1_entitlement_product_power")
        ),
        "v7_h1_hybrid_residual_beta": (
            data.get("v7_h1_hybrid_residual_beta")
            or config.get("v7_h1_hybrid_residual_beta")
        ),
        "location_SUPPORT": loc_sup,
        "location_NE":      loc_ne,
        "location_REFUTE":  loc_ref,
        "role_SUPPORT":     role_sup,
        "role_NE":          role_ne,
        "role_REFUTE":      role_ref,
        "location_role_SUPPORT": lr_sup,
        "location_role_NE":      lr_ne,
        "location_role_total":   lr_total,
        "location_role_false_support_rate": lr_false_sup_rate,
        "has_intervention_data": interventions != {},
    }


# ---------------------------------------------------------------------------
# Prediction JSON extraction
# ---------------------------------------------------------------------------

def _looks_like_prediction(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    return bool(set(d.keys()) & _PREDICTION_LIKE_FIELDS)


def _extract_predictions_recursive(data: Any, depth: int = 0) -> list[dict]:
    """Recursively extract a list of prediction-like dicts from arbitrary JSON."""
    if depth > 6:
        return []

    if isinstance(data, list):
        flat = [x for x in data if isinstance(x, dict)]
        if flat and sum(1 for d in flat if _looks_like_prediction(d)) >= len(flat) // 2:
            return flat
        results: list[dict] = []
        for item in data:
            results.extend(_extract_predictions_recursive(item, depth + 1))
        return results

    if isinstance(data, dict):
        # Priority keys
        for key in ("predictions", "dev_predictions", "records"):
            if key in data and isinstance(data[key], (list, dict)):
                result = _extract_predictions_recursive(data[key], depth + 1)
                if result:
                    return result
        # Runs nesting
        runs = data.get("runs")
        if isinstance(runs, dict):
            for v in runs.values():
                result = _extract_predictions_recursive(v, depth + 1)
                if result:
                    return result
        # Generic descent into list/dict values (skip scalar-heavy config keys)
        skip = {"configuration", "metrics", "best_dev_interventions",
                "stage", "objective", "decision", "remaining_risks"}
        for k, v in data.items():
            if k in skip:
                continue
            if isinstance(v, (list, dict)):
                result = _extract_predictions_recursive(v, depth + 1)
                if result:
                    return result

    return []


def _extract_record_fields(rec: dict, seed: "int | None") -> dict[str, Any]:
    """Normalize one prediction record to a common schema."""
    # Example ID
    ex_id = None
    for k in _ID_KEYS:
        v = rec.get(k)
        if v is not None:
            ex_id = str(v)
            break

    # Intervention
    interv = None
    for k in _INTERV_KEYS:
        v = rec.get(k)
        if v is not None:
            interv = str(v).lower().strip()
            break

    # Labels
    raw_gold, norm_gold = _extract_label(rec, _GOLD_KEYS)
    raw_pred, norm_pred = _extract_label(rec, _PRED_KEYS)

    # Text
    text: dict[str, Any] = {}
    for k in _TEXT_KEYS:
        if rec.get(k) is not None:
            text[k] = str(rec[k])[:200]

    # Probabilities
    probs: dict[str, "float | None"] = {}
    for k in PROB_KEYS:
        v = rec.get(k)
        if isinstance(v, (int, float)):
            probs[k] = float(v)
        elif isinstance(v, list) and v:
            # logits/probs may be a list
            probs[k] = None
            probs[f"{k}_raw"] = v[:4]  # type: ignore[assignment]
        else:
            probs[k] = None

    # Also handle nested prob dicts
    for k in ("probs", "logits"):
        nested = rec.get(k)
        if isinstance(nested, dict):
            for lbl in PREDICTION_LABELS:
                if lbl in nested:
                    probs[f"{k}_{lbl}"] = float(nested[lbl])

    return {
        "example_id": ex_id,
        "seed": seed,
        "intervention": interv,
        "gold_raw": raw_gold,
        "gold_normalized": norm_gold,
        "pred_raw": raw_pred,
        "pred_normalized": norm_pred,
        "text": text or None,
        **probs,
    }


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def _scan_dir(directory: Path, label: str) -> dict[str, Any]:
    """Return {'summaries': [...], 'predictions': [...]} for one directory."""
    summaries: list[dict] = []
    predictions: list[dict] = []   # list of {seed, records: [...]}

    if not directory.exists():
        print(f"[S28A] WARNING: {label} dir not found: {directory}", file=sys.stderr)
        return {"summaries": summaries, "predictions": predictions}

    for p in sorted(directory.rglob("*")):
        if not p.is_file() or p.suffix != ".json":
            continue
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[S28A] WARNING: could not read {p}: {exc}", file=sys.stderr)
            continue

        stem = p.stem.lower()
        if "summary" in stem:
            summaries.append(_extract_summary_run(data, p, label))
        elif "prediction" in stem or "pred" in stem:
            seed = _infer_seed({}, p)
            raw_records = _extract_predictions_recursive(data)
            if raw_records:
                records = [_extract_record_fields(r, seed) for r in raw_records]
                predictions.append({"seed": seed, "path": str(p), "records": records})
        else:
            # Try as summary first, then prediction
            if isinstance(data, dict) and "best_dev_macro_f1" in data:
                summaries.append(_extract_summary_run(data, p, label))
            else:
                raw_records = _extract_predictions_recursive(data)
                if raw_records:
                    seed = _infer_seed({}, p)
                    records = [_extract_record_fields(r, seed) for r in raw_records]
                    predictions.append({
                        "seed": seed, "path": str(p), "records": records,
                    })

    print(
        f"[S28A] {label}: {len(summaries)} summary file(s), "
        f"{len(predictions)} prediction file(s).",
        file=sys.stderr,
    )
    return {"summaries": summaries, "predictions": predictions}


# ---------------------------------------------------------------------------
# Output 1: per_run_location_role_summary
# ---------------------------------------------------------------------------

def _per_run_lr_summary(summaries: list[dict]) -> list[dict]:
    rows = []
    for s in sorted(summaries, key=lambda x: (x.get("seed") or 0)):
        rows.append({
            "config":    s.get("config"),
            "seed":      s.get("seed"),
            "macro":     s.get("best_dev_macro_f1"),
            "acc":       s.get("best_dev_acc"),
            "SUP_r":     s.get("v7_support_recall"),
            "NE_r":      s.get("v7_ne_recall"),
            "location_SUPPORT": s.get("location_SUPPORT"),
            "location_NE":      s.get("location_NE"),
            "location_REFUTE":  s.get("location_REFUTE"),
            "role_SUPPORT":     s.get("role_SUPPORT"),
            "role_NE":          s.get("role_NE"),
            "role_REFUTE":      s.get("role_REFUTE"),
            "location_role_SUPPORT": s.get("location_role_SUPPORT"),
            "location_role_NE":      s.get("location_role_NE"),
            "location_role_total":   s.get("location_role_total"),
            "location_role_false_support_rate": s.get("location_role_false_support_rate"),
        })
    return rows


# ---------------------------------------------------------------------------
# Output 2: aggregate_location_role_by_config
# ---------------------------------------------------------------------------

def _aggregate_lr_by_config(summaries: list[dict], label: str) -> dict[str, Any]:
    if not summaries:
        return {"config": label, "n_runs": 0}

    def _col(key: str) -> list:
        return [s.get(key) for s in summaries if s.get(key) is not None]

    loc_sups  = _col("location_SUPPORT")
    role_sups = _col("role_SUPPORT")
    lr_sups   = _col("location_role_SUPPORT")
    lr_nes    = _col("location_role_NE")
    macros    = _col("best_dev_macro_f1")
    ne_rs     = _col("v7_ne_recall")

    loc_total  = sum(loc_sups)  if loc_sups  else None
    role_total = sum(role_sups) if role_sups else None
    lr_total_s = sum(lr_sups)   if lr_sups   else None

    lr_balance = (
        round(loc_total / max(role_total, 1), 4)
        if loc_total is not None and role_total is not None else None
    )

    return {
        "config": label,
        "n_runs": len(summaries),
        "location_SUPPORT_total": loc_total,
        "location_SUPPORT_mean":  _mean(loc_sups),
        "role_SUPPORT_total":     role_total,
        "role_SUPPORT_mean":      _mean(role_sups),
        "location_role_SUPPORT_total": lr_total_s,
        "location_role_SUPPORT_mean":  _mean(lr_sups),
        "location_role_balance":  lr_balance,
        "location_role_NE_total": sum(lr_nes) if lr_nes else None,
        "location_role_NE_mean":  _mean(lr_nes),
        "macro_mean": _mean(macros),
        "macro_std":  _std(macros),
        "NE_r_mean":  _mean(ne_rs),
    }


# ---------------------------------------------------------------------------
# Output 3: seed_paired_location_role_deltas
# ---------------------------------------------------------------------------

def _paired_lr_deltas(
    summaries_a: list[dict],
    summaries_b: list[dict],
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    a_by_seed = {s["seed"]: s for s in summaries_a if s.get("seed") is not None}
    b_by_seed = {s["seed"]: s for s in summaries_b if s.get("seed") is not None}
    common = sorted(set(a_by_seed) & set(b_by_seed))

    rows = []
    for seed in common:
        a, b = a_by_seed[seed], b_by_seed[seed]

        def _d(k: str) -> "float | None":
            va, vb = a.get(k), b.get(k)
            if va is not None and vb is not None:
                try:
                    return round(float(va) - float(vb), 6)
                except (TypeError, ValueError):
                    pass
            return None

        def _di(k: str) -> "int | None":
            va, vb = _int_or_none(a.get(k)), _int_or_none(b.get(k))
            return va - vb if va is not None and vb is not None else None

        rows.append({
            "seed": seed,
            f"{label_a}_macro":            a.get("best_dev_macro_f1"),
            f"{label_b}_macro":            b.get("best_dev_macro_f1"),
            "macro_delta":                 _d("best_dev_macro_f1"),
            "NE_r_delta":                  _d("v7_ne_recall"),
            "location_SUPPORT_delta":      _di("location_SUPPORT"),
            "role_SUPPORT_delta":          _di("role_SUPPORT"),
            "location_role_SUPPORT_delta": _di("location_role_SUPPORT"),
            "location_role_NE_delta":      _di("location_role_NE"),
        })

    macro_deltas = [r["macro_delta"] for r in rows if r.get("macro_delta") is not None]
    lr_deltas    = [r["location_role_SUPPORT_delta"] for r in rows
                    if r.get("location_role_SUPPORT_delta") is not None]

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_paired": len(rows),
        "paired_rows": rows,
        "macro_delta_mean": _mean(macro_deltas),
        "location_role_SUPPORT_delta_mean": _mean(lr_deltas),
    }


# ---------------------------------------------------------------------------
# Output 4: overlap analysis
# ---------------------------------------------------------------------------

def _focus_records(records: list[dict]) -> list[dict]:
    """Filter to location_swap / role_swap records only."""
    return [r for r in records if r.get("intervention") in FOCUS_INTERVENTIONS]


def _overlap_analysis(
    pred_by_config: dict[str, list[dict]],
    max_examples: int,
) -> dict[str, Any]:
    """Compare per-example predictions between configs, grouped by seed."""
    # pred_by_config: {"learned": [{seed, path, records},...], ...}

    # Build {config: {seed: {example_id: norm_pred}}} for focus interventions
    index: dict[str, dict[int, dict[str, str]]] = {}
    has_ids = True

    for config, pred_files in pred_by_config.items():
        index[config] = {}
        for pf in pred_files:
            seed = pf.get("seed") or 0
            recs = _focus_records(pf.get("records", []))
            by_id: dict[str, str] = {}
            for r in recs:
                eid = r.get("example_id")
                pred = r.get("pred_normalized")
                if eid is None:
                    has_ids = False
                elif pred is not None:
                    by_id[eid] = pred
            if by_id:
                if seed not in index[config]:
                    index[config][seed] = {}
                index[config][seed].update(by_id)

    if not has_ids or not index:
        return {
            "overlap_available": False,
            "reason": "No stable example id found across prediction files.",
        }

    results: dict[str, Any] = {
        "overlap_available": True,
        "pairs": [],
    }

    product_idx = index.get("product", {})
    for other_label in ("learned", "hybrid"):
        other_idx = index.get(other_label, {})
        common_seeds = sorted(set(product_idx) & set(other_idx))
        pair_rows = []

        total_rescued = 0
        total_new_errors = 0

        for seed in common_seeds:
            prod_preds  = product_idx[seed]
            other_preds = other_idx[seed]
            common_ids  = set(prod_preds) & set(other_preds)

            rescued   = [eid for eid in common_ids
                         if prod_preds[eid] == "SUPPORT"
                         and other_preds[eid] == "NOT_ENTITLED"]
            new_errs  = [eid for eid in common_ids
                         if prod_preds[eid] == "NOT_ENTITLED"
                         and other_preds[eid] == "SUPPORT"]

            total_rescued    += len(rescued)
            total_new_errors += len(new_errs)

            pair_rows.append({
                "seed": seed,
                "n_common_ids": len(common_ids),
                "product_false_support_count": sum(
                    1 for p in prod_preds.values() if p == "SUPPORT"
                ),
                f"{other_label}_false_support_count": sum(
                    1 for p in other_preds.values() if p == "SUPPORT"
                ),
                f"{other_label}_rescued_product_false_support": len(rescued),
                f"{other_label}_new_false_support": len(new_errs),
                f"{other_label}_rescued_example_ids": rescued[:max_examples],
                f"{other_label}_new_error_example_ids": new_errs[:max_examples],
            })

        results["pairs"].append({
            "comparison": f"{other_label}_minus_product",
            "n_paired_seeds": len(common_seeds),
            "total_rescued":    total_rescued,
            "total_new_errors": total_new_errors,
            "per_seed": pair_rows,
        })

    return results


# ---------------------------------------------------------------------------
# Output 5: diagnostic probability analysis
# ---------------------------------------------------------------------------

def _diagnostic_prob_analysis(
    pred_by_config: dict[str, list[dict]],
) -> dict[str, Any]:
    """For location/role records, compute mean probs grouped by config/interv/pred."""
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for config, pred_files in pred_by_config.items():
        for pf in pred_files:
            for r in _focus_records(pf.get("records", [])):
                interv = r.get("intervention") or "unknown"
                pred   = r.get("pred_normalized") or "unknown"
                groups[(config, interv, pred)].append(r)

    rows = []
    for (config, interv, pred), recs_for_group in sorted(groups.items()):
        row: dict[str, Any] = {
            "config": config,
            "intervention": interv,
            "predicted_label": pred,
            "n_records": len(recs_for_group),
        }
        for k in PROB_KEYS:
            vals = [r[k] for r in recs_for_group if r.get(k) is not None]
            row[f"mean_{k}"] = _mean(vals) if vals else None
        rows.append(row)

    return {
        "available": bool(rows),
        "note": (
            "Computed from prediction JSON files. Null when prediction files absent."
        ),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Interpretation flags
# ---------------------------------------------------------------------------

def _interpretation_flags(
    agg_by_config: dict[str, Any],
    overlap: dict[str, Any],
) -> dict[str, Any]:
    prod = agg_by_config.get("product") or {}
    loc_t  = prod.get("location_SUPPORT_total")
    role_t = prod.get("role_SUPPORT_total")

    location_role_dominates = bool(
        loc_t is not None and role_t is not None
        and (loc_t + role_t) > 0
    )

    role_harder   = bool(role_t is not None and loc_t is not None and role_t > loc_t)
    loc_harder    = bool(loc_t  is not None and role_t is not None and loc_t  > role_t)
    both_high     = bool(
        loc_t is not None and role_t is not None
        and loc_t > 0 and role_t > 0
        and not (role_harder or loc_harder)
    )

    learned_rescues  = False
    hybrid_rescues   = False
    learned_new_errs = False
    hybrid_new_errs  = False

    if overlap.get("overlap_available"):
        for pair in overlap.get("pairs", []):
            comp  = pair.get("comparison", "")
            resc  = pair.get("total_rescued", 0)
            new_e = pair.get("total_new_errors", 0)
            if "learned" in comp:
                if resc > 0:
                    learned_rescues = True
                if resc > 0 and new_e > resc:
                    learned_new_errs = True
            if "hybrid" in comp:
                if resc > 0:
                    hybrid_rescues = True
                if resc > 0 and new_e > resc:
                    hybrid_new_errs = True

    # frame_prob_overconfidence: check if prob analysis has data
    frame_overconfidence = None  # set to True/False by caller if probs available

    return {
        "location_role_dominates_product_errors": location_role_dominates,
        "role_harder_than_location": role_harder,
        "location_harder_than_role": loc_harder,
        "both_location_and_role_significant": both_high,
        "learned_rescues_product_cases": learned_rescues,
        "hybrid_rescues_product_cases":  hybrid_rescues,
        "learned_introduces_new_errors": learned_new_errs,
        "hybrid_introduces_new_errors":  hybrid_new_errs,
        "frame_prob_overconfidence_suspected": frame_overconfidence,
    }


def _check_frame_overconfidence(
    diag: dict[str, Any],
) -> "bool | None":
    """True if product location/role false SUPPORT records have mean frame_prob >= 0.5."""
    for row in diag.get("rows", []):
        if (
            row.get("config") == "product"
            and row.get("predicted_label") == "SUPPORT"
            and row.get("intervention") in FOCUS_INTERVENTIONS
        ):
            mfp = row.get("mean_frame_prob")
            if mfp is not None:
                return float(mfp) >= 0.5
    return None


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _per_run_table_md(rows: list[dict]) -> str:
    if not rows:
        return "_No summary runs found._\n"
    headers = [
        "config", "seed", "macro", "NE_r",
        "loc_SUP", "loc_NE", "role_SUP", "role_NE",
        "lr_SUP", "lr_NE", "lr_total", "lr_false_sup_rate",
    ]
    sep  = "|" + "|".join(["---"] * len(headers)) + "|"
    hrow = "| " + " | ".join(headers) + " |"
    table_rows = [
        f"| {r.get('config','?')} | {r.get('seed','?')} "
        f"| {_fmt(r.get('macro'))} | {_fmt(r.get('NE_r'))} "
        f"| {_fmt_int(r.get('location_SUPPORT'))} "
        f"| {_fmt_int(r.get('location_NE'))} "
        f"| {_fmt_int(r.get('role_SUPPORT'))} "
        f"| {_fmt_int(r.get('role_NE'))} "
        f"| {_fmt_int(r.get('location_role_SUPPORT'))} "
        f"| {_fmt_int(r.get('location_role_NE'))} "
        f"| {_fmt_int(r.get('location_role_total'))} "
        f"| {_fmt(r.get('location_role_false_support_rate'))} |"
        for r in rows
    ]
    return "\n".join([hrow, sep] + table_rows) + "\n"


def _aggregate_table_md(agg_by_config: dict[str, Any]) -> str:
    headers = [
        "config", "n", "macro_mean", "macro_std", "NE_r_mean",
        "loc_SUP_total", "role_SUP_total", "lr_SUP_total", "lr_balance",
        "lr_NE_total",
    ]
    sep  = "|" + "|".join(["---"] * len(headers)) + "|"
    hrow = "| " + " | ".join(headers) + " |"
    rows = []
    for cfg in ("learned", "product", "hybrid"):
        a = agg_by_config.get(cfg) or {}
        rows.append(
            f"| {cfg} | {a.get('n_runs', 0)} "
            f"| {_fmt(a.get('macro_mean'))} | {_fmt(a.get('macro_std'))} "
            f"| {_fmt(a.get('NE_r_mean'))} "
            f"| {_fmt_int(a.get('location_SUPPORT_total'))} "
            f"| {_fmt_int(a.get('role_SUPPORT_total'))} "
            f"| {_fmt_int(a.get('location_role_SUPPORT_total'))} "
            f"| {_fmt(a.get('location_role_balance'))} "
            f"| {_fmt_int(a.get('location_role_NE_total'))} |"
        )
    return "\n".join([hrow, sep] + rows) + "\n"


def _delta_table_md(delta: dict[str, Any]) -> str:
    rows_data = delta.get("paired_rows", [])
    if not rows_data:
        la = delta.get("label_a", "A")
        lb = delta.get("label_b", "B")
        return f"_No paired seeds found between {la} and {lb}._\n"
    la, lb = delta.get("label_a", "A"), delta.get("label_b", "B")
    headers = [
        "seed", f"{la}_macro", f"{lb}_macro", "macro_delta", "NE_r_delta",
        "loc_SUP_delta", "role_SUP_delta", "lr_SUP_delta",
    ]
    sep  = "|" + "|".join(["---"] * len(headers)) + "|"
    hrow = "| " + " | ".join(headers) + " |"
    rows = [
        f"| {r.get('seed','?')} "
        f"| {_fmt(r.get(f'{la}_macro'))} "
        f"| {_fmt(r.get(f'{lb}_macro'))} "
        f"| {_fmt(r.get('macro_delta'))} "
        f"| {_fmt(r.get('NE_r_delta'))} "
        f"| {_fmt_int(r.get('location_SUPPORT_delta'))} "
        f"| {_fmt_int(r.get('role_SUPPORT_delta'))} "
        f"| {_fmt_int(r.get('location_role_SUPPORT_delta'))} |"
        for r in rows_data
    ]
    return "\n".join([hrow, sep] + rows) + "\n"


def _overlap_md(overlap: dict[str, Any], max_ex: int) -> str:
    if not overlap.get("overlap_available"):
        reason = overlap.get("reason", "Overlap analysis not available.")
        return f"_Not available: {reason}_\n"

    lines = []
    for pair in overlap.get("pairs", []):
        comp      = pair.get("comparison", "")
        n_seeds   = pair.get("n_paired_seeds", 0)
        rescued   = pair.get("total_rescued", 0)
        new_errs  = pair.get("total_new_errors", 0)
        lines.append(f"### {comp}")
        lines.append(f"Paired seeds: {n_seeds}  |  Total rescued: {rescued}  |  Total new errors: {new_errs}")
        lines.append("")
        for seed_row in pair.get("per_seed", []):
            seed = seed_row.get("seed")
            nc   = seed_row.get("n_common_ids", 0)
            lines.append(f"**Seed {seed}** ({nc} common location/role IDs)")
            for k, v in seed_row.items():
                if k in ("seed", "n_common_ids"):
                    continue
                if isinstance(v, list):
                    display = v[:max_ex]
                    lines.append(f"  - {k}: {display}")
                else:
                    lines.append(f"  - {k}: {v}")
            lines.append("")

    return "\n".join(lines) + "\n"


def _diag_prob_md(diag: dict[str, Any]) -> str:
    if not diag.get("available"):
        return "_Not available: no prediction JSON files found._\n"
    rows = diag.get("rows", [])
    if not rows:
        return "_No location/role records found in prediction files._\n"

    present_prob_keys = [
        k for k in PROB_KEYS
        if any(row.get(f"mean_{k}") is not None for row in rows)
    ]
    if not present_prob_keys:
        return "_Probability fields not present in prediction records._\n"

    headers = ["config", "intervention", "pred_label", "n"] + [
        f"mean_{k[:10]}" for k in present_prob_keys
    ]
    sep  = "|" + "|".join(["---"] * len(headers)) + "|"
    hrow = "| " + " | ".join(headers) + " |"
    table_rows = []
    for row in rows:
        cells = [
            row.get("config", "?"),
            row.get("intervention", "?"),
            row.get("predicted_label", "?"),
            str(row.get("n_records", 0)),
        ] + [_fmt(row.get(f"mean_{k}")) for k in present_prob_keys]
        table_rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([hrow, sep] + table_rows) + "\n"


def _interpretation_md(flags: dict[str, Any], agg_by_config: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append(
        "This is a diagnostic anatomy stage, not a new training stage. "
        "The purpose is to determine whether location/role errors are structured "
        "enough to justify Stage28-B/C development."
    )

    prod = agg_by_config.get("product") or {}
    loc_t  = prod.get("location_SUPPORT_total")
    role_t = prod.get("role_SUPPORT_total")

    if flags.get("location_harder_than_role") and loc_t is not None:
        lines.append(
            f"Location_swap produces more false SUPPORT than role_swap in product runs "
            f"(location={loc_t}, role={role_t}). "
            "The hard frame mismatch is location-heavy. "
            "Stage28-B should prioritize location frame-boundary signal."
        )
    elif flags.get("role_harder_than_location") and role_t is not None:
        lines.append(
            f"Role_swap produces more false SUPPORT than location_swap in product runs "
            f"(location={loc_t}, role={role_t}). "
            "Role semantics need special handling. "
            "Stage28-B should investigate role-specific entitlement boundaries."
        )
    elif flags.get("both_location_and_role_significant"):
        lines.append(
            f"Both location_swap and role_swap contribute substantially to false SUPPORT "
            f"(location={loc_t}, role={role_t}). "
            "Location/role should be treated as a joint frame-boundary problem in Stage28-B."
        )
    else:
        lines.append(
            "Location/role false SUPPORT distribution is unclear from available data. "
            "Collect more structured diagnostic examples before module design."
        )

    fooc = flags.get("frame_prob_overconfidence_suspected")
    if fooc is True:
        lines.append(
            "frame_prob_overconfidence_suspected=True: product false SUPPORT on "
            "location/role interventions has mean frame_prob >= 0.5. "
            "The frame_prob head is assigning confident entitlement to swapped-location/role "
            "claims, suggesting the frame boundary signal is insufficiently discriminative "
            "at max_length=64 with frozen encoder."
        )
    elif fooc is False:
        lines.append(
            "frame_prob_overconfidence_suspected=False: product false SUPPORT records "
            "have mean frame_prob < 0.5. The entitlement gate may be correcting a weak "
            "frame signal but cannot fully compensate."
        )

    overlap_avail = flags.get("learned_rescues_product_cases") is not None
    if overlap_avail:
        if flags.get("learned_rescues_product_cases"):
            if not flags.get("learned_introduces_new_errors"):
                lines.append(
                    "Learned gate rescues some product false SUPPORT cases on location/role "
                    "without introducing more new errors. This is positive evidence for a "
                    "constrained specialist gate design in Stage28-B."
                )
            else:
                lines.append(
                    "Learned gate rescues some product false SUPPORT cases but introduces "
                    "more new errors than it rescues. A naive learned specialist gate is "
                    "not safe; constrained gating is required."
                )
        else:
            lines.append(
                "Learned gate does not rescue product false SUPPORT cases on location/role. "
                "No clean rescue pattern exists. Recommend freezing Stage27 and designing "
                "a new diagnostic dataset before any model changes."
            )
    else:
        lines.append(
            "Overlap analysis not available (no prediction files with stable IDs). "
            "Cannot determine whether learned/hybrid rescues product false SUPPORT cases. "
            "Stage28-B should collect structured predictions before module design."
        )

    lines.append(
        "product_power=0.90 remains the stable controlled-setting baseline. "
        "No Stage28-A finding changes the Stage27 final configuration."
    )

    return "\n\n".join(f"- {ln}" for ln in lines) + "\n"


def _decision_md(flags: dict[str, Any], agg_by_config: dict[str, Any]) -> str:
    overlap_avail = agg_by_config.get("product", {}).get("n_runs", 0) > 0

    if not overlap_avail:
        next_stage = "Stage28-B diagnostic dataset construction"
        rationale  = "Insufficient data from prediction files to determine rescue patterns."
    elif flags.get("learned_rescues_product_cases") and not flags.get(
        "learned_introduces_new_errors"
    ):
        next_stage = "Stage28-B specialist gate design"
        rationale  = (
            "Learned gate shows rescue without net new errors on location/role axis. "
            "Constrained specialist gate is worth building."
        )
    else:
        next_stage = "Stage28-B diagnostic dataset construction"
        rationale  = (
            "No clean rescue pattern confirmed. Diagnostic dataset collection should "
            "precede model changes."
        )

    return (
        f"- **Stage27 final baseline:** product_power=0.90 (macro_mean=0.9511, "
        f"bad_SUP=44, location_role_SUP=33, missing_SUP=0, controlled no-time setting).\n"
        f"- **Whether to build specialist gate next:** "
        f"{'yes - constrained specialist gate' if next_stage == 'Stage28-B specialist gate design' else 'no - collect diagnostic data first'}.\n"
        f"- **Recommended next stage:** {next_stage}.\n"
        f"- **Rationale:** {rationale}\n"
    )


def _build_markdown(
    per_run_rows: list[dict],
    agg_by_config: dict[str, Any],
    delta_lp: dict[str, Any],
    delta_hp: dict[str, Any],
    overlap: dict[str, Any],
    diag: dict[str, Any],
    flags: dict[str, Any],
    inputs: dict[str, str],
    max_ex: int,
) -> str:
    return f"""\
# Stage28-A Location/Role Error Anatomy

## Objective

Determine whether location/role frame mismatch errors in Stage27 v7-H1 predictions
are structured enough to justify a future location-role-specific boundary module
(Stage28-B/C). This is a diagnostic anatomy stage, not a new training stage.
product_power=0.90 remains the stable controlled-setting baseline.

## Inputs

| Parameter | Value |
|---|---|
| learned_dir | {inputs.get('learned_dir', 'N/A')} |
| product_dir | {inputs.get('product_dir', 'N/A')} |
| hybrid_dir | {inputs.get('hybrid_dir', 'N/A')} |
| output_md | {inputs.get('output_md', 'N/A')} |
| output_json | {inputs.get('output_json', 'N/A')} |
| max_examples_per_section | {max_ex} |

## Method

Each directory is scanned recursively for `*summary.json` files (aggregate intervention
counts) and `*prediction*.json` / `*pred*.json` files (per-example records).

Summary JSONs provide per-run location/role SUPPORT/NE/REFUTE counts extracted from
`best_dev_interventions.location_swap` and `best_dev_interventions.role_swap` blocks.

Prediction JSONs are parsed schema-robustly: list-of-dicts, `predictions` key,
`dev_predictions` key, `records` key, or nested under `runs` objects. Records are
normalized to a common schema; labels are normalized to REFUTE / NOT_ENTITLED / SUPPORT.

Focus interventions: `location_swap`, `role_swap`.

## Per-Run Location/Role Summary

{_per_run_table_md(per_run_rows)}

## Aggregate Location/Role Comparison

`lr_balance` = location_SUPPORT_total / max(role_SUPPORT_total, 1). Values > 1.0
indicate location_swap is the harder axis.

{_aggregate_table_md(agg_by_config)}

## Paired Seed Deltas

### Learned vs Product

{_delta_table_md(delta_lp)}

### Hybrid vs Product

{_delta_table_md(delta_hp)}

## Overlap Analysis

Example-level comparison between configs on location/role records.
Requires stable example IDs across prediction files.

{_overlap_md(overlap, max_ex)}

## Diagnostic Probability Analysis

Mean probability values for location/role records, grouped by config and prediction outcome.
Only populated when prediction JSON files with probability fields are available.

{_diag_prob_md(diag)}

## Interpretation

{_interpretation_md(flags, agg_by_config)}

## Decision

{_decision_md(flags, agg_by_config)}

## Remaining Risks

- All source data is from the controlled no-time validation setting
  (`controlled_v5_v3_without_time_swap.jsonl`). No OOD or time_swap claim is made.
- Summary JSONs may aggregate multiple seeds; if seed is not inferred from filename,
  the seed field is null and pairing is not possible.
- Prediction JSONs may not exist if only summary JSONs were saved. In that case,
  overlap analysis and diagnostic probability analysis report null.
- Example IDs must be stable across product/learned/hybrid files for overlap to work.
  If IDs differ between runs, overlap_available=false.
- T4-safe frozen encoder (max_length=64) is the scope of all Stage27 evidence.
  Full-encoder behavior may differ.
- frame_prob_overconfidence_suspected requires probability fields in prediction JSONs.
  If those fields are absent, the flag remains null.
"""


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def _build_json_report(
    per_run_rows: list[dict],
    agg_by_config: dict[str, Any],
    delta_lp: dict[str, Any],
    delta_hp: dict[str, Any],
    overlap: dict[str, Any],
    diag: dict[str, Any],
    flags: dict[str, Any],
    inputs: dict[str, str],
    max_ex: int,
) -> dict[str, Any]:
    prod = agg_by_config.get("product") or {}
    overlap_avail = prod.get("n_runs", 0) > 0

    if flags.get("learned_rescues_product_cases") and not flags.get(
        "learned_introduces_new_errors"
    ):
        next_stage = "Stage28-B specialist gate design"
        build_specialist = True
    else:
        next_stage = "Stage28-B diagnostic dataset construction"
        build_specialist = False

    return {
        "stage": "Stage28-A",
        "objective": (
            "Determine whether location/role frame mismatch errors are structured enough "
            "to justify a future location-role-specific boundary module. "
            "This is a diagnostic anatomy stage, not a new training stage."
        ),
        "input_dirs": inputs,
        "per_run_location_role_summary": per_run_rows,
        "aggregate_location_role_by_config": agg_by_config,
        "seed_paired_location_role_deltas": {
            "learned_minus_product": delta_lp,
            "hybrid_minus_product":  delta_hp,
        },
        "overlap_analysis": overlap,
        "diagnostic_probability_analysis": diag,
        "interpretation_flags": flags,
        "decision": {
            "stage27_final_baseline": {
                "config": "product_power=0.90",
                "macro_mean": 0.951136,
                "bad_SUP_total": 44,
                "location_role_SUP_total": 33,
                "missing_SUP_total": 0,
                "dataset": "controlled_v5_v3_without_time_swap.jsonl",
            },
            "whether_to_build_specialist_gate_next": build_specialist,
            "recommended_next_stage": next_stage,
        },
        "remaining_risks": [
            "All source data from controlled no-time validation only. No OOD or time_swap claim.",
            "Prediction JSONs may be absent; overlap and probability analyses report null if so.",
            "Example IDs must be stable across configs for overlap analysis to work.",
            "T4-safe frozen encoder (max_length=64) is the scope; full-encoder may differ.",
            "frame_prob_overconfidence_suspected remains null when probability fields absent.",
            "Seed inference from filename may fail if naming convention differs.",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage28-A: Location/Role Error Anatomy - inspect prediction and summary "
            "JSONs from learned/product/hybrid runs to assess whether location/role "
            "frame mismatch errors are structured enough to justify Stage28-B/C."
        )
    )
    p.add_argument(
        "--learned-dir",
        type=Path,
        default=Path("/kaggle/working/stage27_h2a_learned_rerun_notimeswap"),
        help="Directory of learned-gate *summary.json and *predictions.json files.",
    )
    p.add_argument(
        "--product-dir",
        type=Path,
        default=Path("/kaggle/working/stage27_h2b_0p90_rerun_notimeswap"),
        help="Directory of product_power=0.90 *summary.json and *predictions.json files.",
    )
    p.add_argument(
        "--hybrid-dir",
        type=Path,
        default=Path(
            "/kaggle/working/"
            "stage27_h2e_product_learned_residual_beta0p2_3seed_notimeswap"
        ),
        help="Directory of hybrid (product_learned_residual beta=0.2) files.",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/stage28a_location_role_error_anatomy.md"),
        help="Output markdown report path.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/stage28a_location_role_error_anatomy.json"),
        help="Output JSON report path.",
    )
    p.add_argument(
        "--max-examples-per-section",
        type=int,
        default=20,
        dest="max_examples_per_section",
        help="Maximum number of example IDs / text snippets per overlap section (default: 20).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    args = parse_args(argv)
    max_ex = args.max_examples_per_section

    inputs = {
        "learned_dir": str(args.learned_dir),
        "product_dir": str(args.product_dir),
        "hybrid_dir":  str(args.hybrid_dir),
        "output_md":   str(args.output_md),
        "output_json": str(args.output_json),
        "max_examples_per_section": max_ex,
    }

    learned_data = _scan_dir(args.learned_dir, "learned")
    product_data = _scan_dir(args.product_dir, "product")
    hybrid_data  = _scan_dir(args.hybrid_dir,  "hybrid")

    # Per-run location/role summary rows
    all_summary_rows: list[dict] = []
    for data, label in (
        (learned_data, "learned"),
        (product_data, "product"),
        (hybrid_data,  "hybrid"),
    ):
        all_summary_rows.extend(_per_run_lr_summary(data["summaries"]))

    # Aggregate by config
    agg_by_config = {
        "learned": _aggregate_lr_by_config(learned_data["summaries"], "learned"),
        "product": _aggregate_lr_by_config(product_data["summaries"], "product"),
        "hybrid":  _aggregate_lr_by_config(hybrid_data["summaries"],  "hybrid"),
    }

    # Paired deltas
    delta_lp = _paired_lr_deltas(
        learned_data["summaries"], product_data["summaries"], "learned", "product"
    )
    delta_hp = _paired_lr_deltas(
        hybrid_data["summaries"], product_data["summaries"], "hybrid", "product"
    )

    # Prediction-level analysis
    pred_by_config: dict[str, list[dict]] = {
        "learned": learned_data["predictions"],
        "product": product_data["predictions"],
        "hybrid":  hybrid_data["predictions"],
    }

    overlap = _overlap_analysis(pred_by_config, max_ex)
    diag    = _diagnostic_prob_analysis(pred_by_config)

    flags = _interpretation_flags(agg_by_config, overlap)
    flags["frame_prob_overconfidence_suspected"] = _check_frame_overconfidence(diag)

    # Build reports
    md     = _build_markdown(
        all_summary_rows, agg_by_config, delta_lp, delta_hp,
        overlap, diag, flags, inputs, max_ex,
    )
    report = _build_json_report(
        all_summary_rows, agg_by_config, delta_lp, delta_hp,
        overlap, diag, flags, inputs, max_ex,
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    args.output_md.write_text(md, encoding="utf-8")
    print(f"[S28A] Wrote: {args.output_md}", file=sys.stderr)

    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[S28A] Wrote: {args.output_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
