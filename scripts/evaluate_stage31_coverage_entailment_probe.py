"""Stage31-B: Coverage/Entailment probe evaluation (diagnostic only, no training)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_NAMES = ["REFUTE", "NOT_ENTITLED", "SUPPORT"]
LABEL_TO_INT = {"REFUTE": 0, "NOT_ENTITLED": 1, "SUPPORT": 2}
INT_TO_LABEL = {0: "REFUTE", 1: "NOT_ENTITLED", 2: "SUPPORT"}

PRED_FIELD_CANDIDATES = [
    "pred_label", "prediction", "pred", "final_pred", "label_pred",
    "predicted_label", "prediction_label", "pred_final_label",
    "predicted_final_label", "final_prediction", "final_label_pred",
    "predicted_final",
]

PREDICTION_WRAPPER_KEYS = [
    "predictions",
    "records",
    "examples",
    "items",
    "data",
    "per_example",
    "per_example_predictions",
    "external_predictions",
]

ID_FIELD_CANDIDATES = ["id", "pair_id", "example_id"]

COVERAGE_FAILURE_GROUPS = {
    "some_to_all_not_entitled",
    "general_to_specific_not_entitled",
    "also_to_only_not_entitled",
    "part_to_whole_not_entitled",
}
SUPPORT_ENTAILMENT_GROUPS = {
    "all_to_some_support",
    "specific_to_general_support",
    "only_to_base_support",
    "whole_to_part_support",
}
REFUTE_GROUPS = {
    "none_to_some_refute",
    "some_to_none_refute",
}

DIAGNOSTIC_COLUMNS = [
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "base_entitlement",
    "entitlement_after_location_cap",
    "entitlement_after_temporal_cap",
    "entitlement_for_decision",
    "support_energy",
    "refute_energy",
    "support_score",
    "refute_score",
    "ne_score",
    "temporal_mismatch_fused_prob",
    "temporal_preservation_prob",
    "effective_temporal_penalty",
]

OBSERVED_STAGE31B_RESULT = {
    "total_accuracy": 0.445,
    "macro_f1": 0.3607,
    "coverage_failure_predicted_support": 6,
    "support_entailment_predicted_ne": 61,
    "refute_case_predicted_support": 5,
    "refute_case_predicted_ne": 27,
    "interpretation": (
        "The current proxy stack is conservative but under-structured. It can "
        "often suppress over-claims, but it cannot reliably preserve valid "
        "SUPPORT under weakening/generalization/part-inclusion entailment."
    ),
}

# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_probe(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if "id" not in row and "pair_id" in row:
                    row["id"] = row["pair_id"]
                rows.append(row)
    return rows


def validate_probe(rows: list[dict]) -> None:
    missing = [i for i, row in enumerate(rows) if "id" not in row]
    if missing:
        raise ValueError(
            f"Probe rows must contain 'id' or fallback 'pair_id'. "
            f"Missing in rows: {missing[:5]}"
        )
    ids = [r["id"] for r in rows]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        raise ValueError(f"Duplicate IDs in probe file: {dup[:5]}")


# ---------------------------------------------------------------------------
# Prediction normalization
# ---------------------------------------------------------------------------

_STR_MAP: dict[str, str] = {
    "refute": "REFUTE", "refutes": "REFUTE", "0": "REFUTE",
    "not_entitled": "NOT_ENTITLED", "ne": "NOT_ENTITLED",
    "not enough info": "NOT_ENTITLED", "not_enough_info": "NOT_ENTITLED", "1": "NOT_ENTITLED",
    "support": "SUPPORT", "supports": "SUPPORT", "2": "SUPPORT",
}


def normalize_pred(raw: Any) -> str:
    if isinstance(raw, int):
        if raw not in INT_TO_LABEL:
            raise ValueError(f"Unknown numeric prediction: {raw!r}")
        return INT_TO_LABEL[raw]
    if isinstance(raw, float) and raw == int(raw):
        return normalize_pred(int(raw))
    s = str(raw).strip()
    mapped = _STR_MAP.get(s.lower())
    if mapped is None:
        raise ValueError(
            f"Cannot normalize prediction value {raw!r}. "
            f"Expected one of: REFUTE/REFUTES/0, NOT_ENTITLED/NE/NOT ENOUGH INFO/1, SUPPORT/SUPPORTS/2"
        )
    return mapped


def detect_pred_field(columns: list[str]) -> str:
    for candidate in PRED_FIELD_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"No prediction field found. Expected one of {PRED_FIELD_CANDIDATES}. "
        f"Got columns: {columns}"
    )


# ---------------------------------------------------------------------------
# Prediction file loading
# ---------------------------------------------------------------------------

def load_predictions_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def load_predictions_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_predictions_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return extract_prediction_records(data)


def _is_list_of_dicts(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def extract_prediction_records(data: Any) -> list[dict]:
    if _is_list_of_dicts(data):
        return data
    if isinstance(data, dict):
        for key in PREDICTION_WRAPPER_KEYS:
            value = data.get(key)
            if _is_list_of_dicts(value):
                return value
        list_candidates = [
            value for value in data.values()
            if _is_list_of_dicts(value)
        ]
        if list_candidates:
            return max(list_candidates, key=len)
        keys = ", ".join(sorted(str(k) for k in data.keys()))
        raise ValueError(
            "JSON prediction wrapper did not contain a list of prediction objects. "
            f"Checked keys {PREDICTION_WRAPPER_KEYS}. Top-level keys: [{keys}]"
        )
    raise ValueError(
        "JSON prediction file must be a list of objects or an object containing "
        f"a list of objects under one of {PREDICTION_WRAPPER_KEYS}."
    )


def load_prediction_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_predictions_csv(path)
    if suffix == ".jsonl":
        return load_predictions_jsonl(path)
    if suffix == ".json":
        return load_predictions_json(path)
    raise ValueError(f"Unsupported prediction file format: {suffix}. Use .csv, .jsonl, or .json.")


def prediction_row_id(row: dict) -> Any:
    for field in ID_FIELD_CANDIDATES:
        value = row.get(field)
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def macro_f1(golds: list[str], preds: list[str]) -> float:
    f1s = []
    for label in LABEL_NAMES:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1s.append(safe_div(2 * prec * rec, prec + rec))
    return sum(f1s) / len(f1s)


def confusion_matrix(golds: list[str], preds: list[str]) -> dict[str, dict[str, int]]:
    mat: dict[str, dict[str, int]] = {g: {p: 0 for p in LABEL_NAMES} for g in LABEL_NAMES}
    for g, p in zip(golds, preds):
        mat[g][p] += 1
    return mat


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(probe_rows: list[dict], pred_map: dict[str, str],
             diag_map: dict[str, dict]) -> dict:
    golds, preds = [], []
    for row in probe_rows:
        golds.append(row["label"])
        preds.append(pred_map[row["id"]])

    total_acc = safe_div(sum(g == p for g, p in zip(golds, preds)), len(golds))
    mf1 = macro_f1(golds, preds)
    cm = confusion_matrix(golds, preds)

    # group-level
    group_gold: dict[str, list[str]] = defaultdict(list)
    group_pred: dict[str, list[str]] = defaultdict(list)
    for row in probe_rows:
        g = row["group"]
        group_gold[g].append(row["label"])
        group_pred[g].append(pred_map[row["id"]])

    group_metrics: dict[str, dict] = {}
    for grp in sorted(group_gold.keys()):
        gg = group_gold[grp]
        gp = group_pred[grp]
        acc = safe_div(sum(a == b for a, b in zip(gg, gp)), len(gg))
        gold_counts = Counter(gg)
        pred_counts = Counter(gp)
        group_metrics[grp] = {
            "n": len(gg),
            "accuracy": round(acc, 4),
            "gold_counts": dict(gold_counts),
            "pred_counts": dict(pred_counts),
        }

    # failure modes
    cov_fail_pred_support = 0
    sup_ent_pred_ne = 0
    refute_pred_support = 0
    refute_pred_ne = 0
    unexpected_refute_on_support_or_ne = 0

    for row in probe_rows:
        grp = row["group"]
        gold = row["label"]
        pred = pred_map[row["id"]]
        if grp in COVERAGE_FAILURE_GROUPS and pred == "SUPPORT":
            cov_fail_pred_support += 1
        if grp in SUPPORT_ENTAILMENT_GROUPS and pred == "NOT_ENTITLED":
            sup_ent_pred_ne += 1
        if grp in REFUTE_GROUPS:
            if pred == "SUPPORT":
                refute_pred_support += 1
            elif pred == "NOT_ENTITLED":
                refute_pred_ne += 1
        if gold in ("SUPPORT", "NOT_ENTITLED") and pred == "REFUTE":
            unexpected_refute_on_support_or_ne += 1

    failure_modes = {
        "coverage_failure_predicted_support": cov_fail_pred_support,
        "support_entailment_predicted_ne": sup_ent_pred_ne,
        "refute_case_predicted_support": refute_pred_support,
        "refute_case_predicted_ne": refute_pred_ne,
        "unexpected_refute_on_support_or_ne": unexpected_refute_on_support_or_ne,
    }

    # coverage_entailment owner summary
    owner_rows = [r for r in probe_rows if r.get("expected_owner") == "coverage_entailment"]
    owner_golds = [r["label"] for r in owner_rows]
    owner_preds = [pred_map[r["id"]] for r in owner_rows]
    owner_acc = safe_div(sum(g == p for g, p in zip(owner_golds, owner_preds)), len(owner_golds))

    # diagnostic column means
    diag_means: dict[str, dict[str, float]] = {}
    if diag_map:
        for grp in sorted(group_gold.keys()):
            grp_diag: dict[str, list[float]] = defaultdict(list)
            for row in probe_rows:
                if row["group"] != grp:
                    continue
                row_diag = diag_map.get(row["id"], {})
                for col in DIAGNOSTIC_COLUMNS:
                    if col in row_diag:
                        try:
                            grp_diag[col].append(float(row_diag[col]))
                        except (TypeError, ValueError):
                            pass
            diag_means[grp] = {
                col: round(sum(vals) / len(vals), 5)
                for col, vals in grp_diag.items() if vals
            }

    return {
        "total_accuracy": round(total_acc, 4),
        "macro_f1": round(mf1, 4),
        "confusion_matrix": cm,
        "group_metrics": group_metrics,
        "owner_accuracy": round(owner_acc, 4),
        "owner_n": len(owner_rows),
        "failure_modes": failure_modes,
        "diagnostic_column_means": diag_means,
    }


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

def build_interpretation(eval_results: dict | None, dry_run: bool) -> str:
    if dry_run or eval_results is None:
        return (
            "No model predictions were evaluated (dry-run mode). "
            "Run with --predictions-file to obtain a Coverage/Entailment diagnostic."
        )
    fm = eval_results["failure_modes"]
    lines = []

    cov_fail = fm["coverage_failure_predicted_support"]
    sup_ne = fm["support_entailment_predicted_ne"]
    ref_sup = fm["refute_case_predicted_support"]
    ref_ne = fm["refute_case_predicted_ne"]

    if cov_fail > 0:
        lines.append(
            f"Over-entitlement detected: {cov_fail} coverage-failure examples "
            "(gold=NOT_ENTITLED) were predicted SUPPORT. This points to missing "
            "coverage-failure ownership: frame/predicate/polarity compatibility can still "
            "override quantifier-scope, specificity, or inclusion failures."
        )
    else:
        lines.append(
            "Coverage-failure groups (gold=NOT_ENTITLED) were NOT incorrectly predicted SUPPORT. "
            "The proxy stack appears to handle downward quantifier failures correctly."
        )

    if sup_ne > 0:
        lines.append(
            f"Over-conservatism detected: {sup_ne} support-entailment examples "
            "(gold=SUPPORT) were predicted NOT_ENTITLED. This points to missing "
            "entailment-preservation ownership: valid weakening, generalization, and "
            "part-inclusion cases are being collapsed into NOT_ENTITLED."
        )
    else:
        lines.append(
            "Support-entailment groups (gold=SUPPORT) were NOT incorrectly predicted NOT_ENTITLED. "
            "Entailment preservation appears intact."
        )

    if ref_sup > 0 or ref_ne > 0:
        lines.append(
            f"Refute groups show failure: {ref_sup} predicted SUPPORT, {ref_ne} predicted NOT_ENTITLED. "
            "Polarity/refute direction is also failing or being suppressed by entitlement."
        )
    else:
        lines.append("Refute groups were handled correctly; polarity appears reliable.")

    return " ".join(lines)


def next_step_recommendation(eval_results: dict | None, dry_run: bool) -> str:
    if dry_run or eval_results is None:
        return (
            "Run with --predictions-file to determine next steps. "
            "If predictions reveal systematic failure, proceed to Stage31-C."
        )
    fm = eval_results["failure_modes"]
    strong_failure = (
        fm["coverage_failure_predicted_support"] > 0
        or fm["support_entailment_predicted_ne"] > 0
        or fm["refute_case_predicted_support"] > 0
    )
    if strong_failure:
        return (
            "Stage31-C should add a directional Coverage/Entailment owner that explicitly "
            "models quantifier-scope, specificity, weakening/generalization, and part/whole "
            "direction. The observed pattern calls for ownership, not merely another cap."
        )
    return (
        "The current proxy stack handles Coverage/Entailment cases acceptably. "
        "Stage31 can be documented as lower priority; proceed to Stage32 relation/role nuance."
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_confusion_matrix_md(cm: dict[str, dict[str, int]]) -> str:
    header = "| Gold \\ Pred | " + " | ".join(LABEL_NAMES) + " |"
    sep = "|" + "---|" * (len(LABEL_NAMES) + 1)
    rows = [header, sep]
    for gold in LABEL_NAMES:
        cells = " | ".join(str(cm[gold].get(p, 0)) for p in LABEL_NAMES)
        rows.append(f"| {gold} | {cells} |")
    return "\n".join(rows)


def write_markdown(
    path: Path,
    probe_file: str,
    row_count: int,
    label_dist: dict[str, int],
    group_counts: dict[str, int],
    dry_run: bool,
    predictions_file: str | None,
    eval_results: dict | None,
    interpretation: str,
    next_step: str,
    run_name: str,
) -> None:
    lines = []
    lines.append("# Stage31-B Coverage/Entailment Probe Evaluation")
    lines.append("")
    lines.append("## Purpose")
    lines.append(
        "Diagnostic evaluation of the ContraMamba proxy stack on directional Coverage/Entailment "
        "cases. Tests whether the current architecture treats quantifier-scope failures as "
        "NOT_ENTITLED or incorrectly resolves them as SUPPORT due to frame/predicate/polarity "
        "compatibility."
    )
    lines.append("")
    lines.append("## Probe")
    lines.append(f"- **File:** `{probe_file}`")
    lines.append(f"- **Row count:** {row_count}")
    lines.append(f"- **Mode:** {'Dry-run (no predictions)' if dry_run else f'Prediction-file evaluation: `{predictions_file}`'}")
    lines.append(f"- **Run name:** `{run_name}`")
    lines.append("")
    lines.append("## Label Distribution")
    lines.append("| Label | Count |")
    lines.append("|---|---|")
    for lbl in LABEL_NAMES:
        lines.append(f"| {lbl} | {label_dist.get(lbl, 0)} |")
    lines.append("")
    lines.append("## Group Counts")
    lines.append("| Group | Count |")
    lines.append("|---|---|")
    for grp, cnt in sorted(group_counts.items()):
        lines.append(f"| {grp} | {cnt} |")
    lines.append("")

    if eval_results is not None:
        lines.append("## Overall Metrics")
        lines.append(f"- **Total accuracy:** {eval_results['total_accuracy']:.4f}")
        lines.append(f"- **Macro-F1:** {eval_results['macro_f1']:.4f}")
        lines.append(f"- **coverage_entailment owner accuracy:** {eval_results['owner_accuracy']:.4f} (n={eval_results['owner_n']})")
        lines.append("")
        lines.append("## Confusion Matrix")
        lines.append(format_confusion_matrix_md(eval_results["confusion_matrix"]))
        lines.append("")
        lines.append("## Group-Level Accuracy")
        lines.append("| Group | N | Accuracy |")
        lines.append("|---|---|---|")
        for grp, gm in sorted(eval_results["group_metrics"].items()):
            lines.append(f"| {grp} | {gm['n']} | {gm['accuracy']:.4f} |")
        lines.append("")
        lines.append("## Group-Level Prediction Counts")
        lines.append("| Group | Gold Counts | Pred Counts |")
        lines.append("|---|---|---|")
        for grp, gm in sorted(eval_results["group_metrics"].items()):
            gc = ", ".join(f"{k}={v}" for k, v in sorted(gm["gold_counts"].items()))
            pc = ", ".join(f"{k}={v}" for k, v in sorted(gm["pred_counts"].items()))
            lines.append(f"| {grp} | {gc} | {pc} |")
        lines.append("")

        diag_means = eval_results.get("diagnostic_column_means", {})
        if any(diag_means.values()):
            available_cols = sorted({col for grp_v in diag_means.values() for col in grp_v})
            if available_cols:
                lines.append("## Diagnostic Column Means by Group")
                header_cols = " | ".join(available_cols)
                lines.append(f"| Group | {header_cols} |")
                lines.append("|" + "---|" * (len(available_cols) + 1))
                for grp in sorted(diag_means.keys()):
                    vals = " | ".join(
                        f"{diag_means[grp].get(col, '')}" for col in available_cols
                    )
                    lines.append(f"| {grp} | {vals} |")
                lines.append("")

        lines.append("## Failure Mode Summary")
        fm = eval_results["failure_modes"]
        lines.append("| Failure Mode | Count |")
        lines.append("|---|---|")
        for k, v in fm.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.append("## Interpretation")
    lines.append(interpretation)
    lines.append("")
    lines.append("## Observed Stage31-B Diagnostic Pattern")
    lines.append(
        f"Current observed result: total_accuracy={OBSERVED_STAGE31B_RESULT['total_accuracy']}, "
        f"macro_f1={OBSERVED_STAGE31B_RESULT['macro_f1']}, "
        f"coverage_failure_predicted_support={OBSERVED_STAGE31B_RESULT['coverage_failure_predicted_support']}, "
        f"support_entailment_predicted_ne={OBSERVED_STAGE31B_RESULT['support_entailment_predicted_ne']}, "
        f"refute_case_predicted_support={OBSERVED_STAGE31B_RESULT['refute_case_predicted_support']}, "
        f"refute_case_predicted_ne={OBSERVED_STAGE31B_RESULT['refute_case_predicted_ne']}."
    )
    lines.append(OBSERVED_STAGE31B_RESULT["interpretation"])
    lines.append(
        "Stage31-C should add a directional Coverage/Entailment owner, not merely another cap."
    )
    lines.append("")
    lines.append("## Leakage Policy")
    lines.append(
        "This probe is **diagnostic-only**. It must not be used for training, fine-tuning, "
        "calibration, threshold selection, checkpoint selection, model selection, "
        "or any form of model optimisation."
    )
    lines.append("")
    lines.append("## Next-Step Recommendation")
    lines.append(next_step)
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(
    path: Path,
    run_name: str,
    probe_file: str,
    predictions_file: str | None,
    dry_run: bool,
    row_count: int,
    label_dist: dict[str, int],
    group_counts: dict[str, int],
    eval_results: dict | None,
    interpretation: str,
    next_step: str,
) -> None:
    out: dict[str, Any] = {
        "stage": "31b",
        "title": "Stage31-B Coverage/Entailment Probe Evaluation",
        "run_name": run_name,
        "probe_file": probe_file,
        "predictions_file": predictions_file,
        "dry_run": dry_run,
        "row_count": row_count,
        "label_distribution": label_dist,
        "group_counts": group_counts,
        "metrics": (
            {
                "total_accuracy": eval_results["total_accuracy"],
                "macro_f1": eval_results["macro_f1"],
                "owner_accuracy": eval_results["owner_accuracy"],
                "owner_n": eval_results["owner_n"],
            }
            if eval_results else None
        ),
        "confusion_matrix": eval_results["confusion_matrix"] if eval_results else None,
        "group_metrics": eval_results["group_metrics"] if eval_results else None,
        "failure_modes": eval_results["failure_modes"] if eval_results else None,
        "diagnostic_column_means": eval_results.get("diagnostic_column_means") if eval_results else None,
        "interpretation": interpretation,
        "observed_stage31b_result": OBSERVED_STAGE31B_RESULT,
        "leakage_policy": (
            "This probe is diagnostic-only. Do not use for training, calibration, "
            "threshold selection, checkpoint selection, or model selection."
        ),
        "next_step_recommendation": next_step,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage31-B: Evaluate Coverage/Entailment probe (diagnostic only)."
    )
    parser.add_argument(
        "--probe-file",
        default="data/stage31_coverage_entailment_probe.jsonl",
        help="Path to the Stage31-A probe JSONL file.",
    )
    parser.add_argument(
        "--predictions-file",
        default=None,
        help="Path to a CSV/JSONL/JSON file containing model predictions.",
    )
    parser.add_argument(
        "--output-md",
        default="reports/stage31b_coverage_entailment_eval_report.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--output-json",
        default="reports/stage31b_coverage_entailment_eval_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--run-name",
        default="stage31b_coverage_entailment_eval",
        help="Run identifier written into reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Summarise probe only; do not require predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dry_run and args.predictions_file is None:
        print(
            "ERROR: Either --predictions-file or --dry-run is required.",
            file=sys.stderr,
        )
        sys.exit(1)

    probe_path = REPO_ROOT / args.probe_file
    if not probe_path.exists():
        print(f"ERROR: Probe file not found: {probe_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading probe: {probe_path}")
    probe_rows = load_probe(probe_path)
    validate_probe(probe_rows)
    print(f"  {len(probe_rows)} rows loaded.")

    label_dist: dict[str, int] = dict(Counter(r["label"] for r in probe_rows))
    group_counts: dict[str, int] = dict(Counter(r["group"] for r in probe_rows))

    eval_results: dict | None = None

    if not args.dry_run:
        pred_path = Path(args.predictions_file)
        if not pred_path.is_absolute():
            pred_path = REPO_ROOT / pred_path
        if not pred_path.exists():
            print(f"ERROR: Predictions file not found: {pred_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Loading predictions: {pred_path}")
        raw_preds = load_prediction_file(pred_path)

        # detect prediction field
        if not raw_preds:
            print("ERROR: Prediction file is empty.", file=sys.stderr)
            sys.exit(1)
        columns = list(dict.fromkeys(col for row in raw_preds for col in row.keys()))
        pred_field = detect_pred_field(columns)
        if pred_field != "pred_label":
            for row in raw_preds:
                if "pred_label" not in row and pred_field in row:
                    row["pred_label"] = row[pred_field]
        print(f"  Using prediction field: {pred_field!r}")

        # check for duplicate prediction IDs
        pred_ids = [prediction_row_id(r) for r in raw_preds]
        none_ids = [i for i, v in enumerate(pred_ids) if v is None]
        if none_ids:
            print(
                f"ERROR: {len(none_ids)} prediction rows are missing an ID field. "
                f"Expected one of {ID_FIELD_CANDIDATES}.",
                file=sys.stderr,
            )
            sys.exit(1)
        dup_pred_ids = [k for k, v in Counter(pred_ids).items() if v > 1]
        if dup_pred_ids:
            print(
                f"ERROR: Duplicate IDs in prediction file: {dup_pred_ids[:5]}",
                file=sys.stderr,
            )
            sys.exit(1)

        # match IDs exactly
        probe_id_set = {r["id"] for r in probe_rows}
        pred_id_set = set(pred_ids)
        missing = probe_id_set - pred_id_set
        extra = pred_id_set - probe_id_set
        if missing or extra:
            print(
                f"ERROR: Prediction IDs do not match probe IDs exactly. "
                f"Missing from predictions: {len(missing)}, Extra in predictions: {len(extra)}.",
                file=sys.stderr,
            )
            if missing:
                print(f"  Sample missing: {sorted(missing)[:5]}", file=sys.stderr)
            if extra:
                print(f"  Sample extra: {sorted(extra)[:5]}", file=sys.stderr)
            sys.exit(1)

        # build maps
        pred_map: dict[str, str] = {}
        diag_map: dict[str, dict] = {}
        for row in raw_preds:
            rid = prediction_row_id(row)
            try:
                pred_map[rid] = normalize_pred(row["pred_label"])
            except ValueError as exc:
                print(f"ERROR in row id={rid!r}: {exc}", file=sys.stderr)
                sys.exit(1)
            diag_row = {
                col: row[col] for col in DIAGNOSTIC_COLUMNS if col in row
            }
            if diag_row:
                diag_map[rid] = diag_row

        print(f"  {len(pred_map)} predictions loaded.")
        eval_results = evaluate(probe_rows, pred_map, diag_map)

    interpretation = build_interpretation(eval_results, args.dry_run)
    next_step = next_step_recommendation(eval_results, args.dry_run)

    # -----------------------------------------------------------------------
    # Print summary to stdout
    # -----------------------------------------------------------------------
    print("\n--- Stage31-B Summary ---")
    print(f"Probe rows : {len(probe_rows)}")
    print(f"Label dist : {label_dist}")
    print(f"Groups     : {len(group_counts)}")
    if eval_results:
        print(f"Accuracy   : {eval_results['total_accuracy']:.4f}")
        print(f"Macro-F1   : {eval_results['macro_f1']:.4f}")
        print(f"Failure modes: {eval_results['failure_modes']}")
    else:
        print("(dry-run: no predictions evaluated)")
    print("")

    # -----------------------------------------------------------------------
    # Write reports
    # -----------------------------------------------------------------------
    md_path = REPO_ROOT / args.output_md
    json_path = REPO_ROOT / args.output_json

    write_markdown(
        path=md_path,
        probe_file=args.probe_file,
        row_count=len(probe_rows),
        label_dist=label_dist,
        group_counts=group_counts,
        dry_run=args.dry_run,
        predictions_file=args.predictions_file,
        eval_results=eval_results,
        interpretation=interpretation,
        next_step=next_step,
        run_name=args.run_name,
    )
    write_json(
        path=json_path,
        run_name=args.run_name,
        probe_file=args.probe_file,
        predictions_file=args.predictions_file,
        dry_run=args.dry_run,
        row_count=len(probe_rows),
        label_dist=label_dist,
        group_counts=group_counts,
        eval_results=eval_results,
        interpretation=interpretation,
        next_step=next_step,
    )
    print(f"Markdown report -> {md_path}")
    print(f"JSON report     -> {json_path}")


if __name__ == "__main__":
    main()
