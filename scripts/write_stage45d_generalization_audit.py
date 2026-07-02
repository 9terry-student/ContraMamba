"""Stage45D: generalization/regression audit for Stage45C recovery-weight selection.

Reporting/aggregation-only. Scans a results directory for Stage45C/Stage45D-style
JSON train reports, groups them by internal holdout (field, family), compares each
recovery-weight configuration against the baseline (support_w=0.0, ne_w=0.0) config
within its holdout group, flags regressions/gains, and writes:

  - results/stage45d_generalization_audit.csv   (flat per-report row + deltas/flags)
  - results/stage45d_generalization_audit.md    (human-readable audit report)
  - results/stage45d_generalization_summary.json (machine-readable summary + recommendation)

`ne_rate_shift_large` (a large absolute swing in NOT_ENTITLED prediction rate vs.
baseline) is purely diagnostic and direction-agnostic: a large *decrease* in NE
rate can be a genuine improvement (less over-rejection), not a regression. It only
contributes to `catastrophic_regression` via the narrower `harmful_ne_rate_shift`
flag, which additionally requires a large NE-rate swing to co-occur with real harm
on accuracy, macro-F1, SUPPORT recall, or REFUTE recall.

This script does not train, evaluate, or run any Kaggle/OOD/experiment commands. It
only reads existing JSON files and writes report files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_CSV_OUTPUT = ROOT / "results" / "stage45d_generalization_audit.csv"
DEFAULT_MD_OUTPUT = ROOT / "results" / "stage45d_generalization_audit.md"
DEFAULT_SUMMARY_JSON_OUTPUT = ROOT / "results" / "stage45d_generalization_summary.json"

# A JSON file must satisfy ALL of: contain "stage45c" or "stage45d", contain
# "holdout", and contain "train_report" (filename, case-insensitive). This is
# an AND predicate, not an "any token" match, to avoid ingesting unrelated
# train reports from other stages that happen to share one token (e.g. a
# generic "*_holdout_train_report.json" from a different stage).
STAGE45_TOKENS: tuple[str, ...] = ("stage45c", "stage45d")
REQUIRED_NAME_TOKENS: tuple[str, ...] = ("holdout", "train_report")

# Compact schema fields this audit understands. All are optional; a report is
# still included (with missing fields as None) if at least one is present.
FIELDS: tuple[str, ...] = (
    "field",
    "family",
    "support_w",
    "ne_w",
    "acc",
    "macro_all3",
    "macro_present",
    "ne_pred_rate",
    "support_recall",
    "refute_recall",
    "pred",
    "stage44b2_decision",
)

# Nested wrapper keys checked as a fallback if a field is not present at the
# JSON top level (some producers may nest the compact summary under a subkey).
NESTED_WRAPPER_KEYS: tuple[str, ...] = ("summary", "result", "metrics", "stage45c", "stage45d")

CONFIG_BASELINE = "baseline"
CONFIG_RECOVERY_STABLE = "recovery_w01_ne01"
CONFIG_RECOVERY_PARAPHRASE = "recovery_w010_ne020"
SUMMARY_CONFIGS: tuple[str, ...] = (
    CONFIG_BASELINE,
    CONFIG_RECOVERY_STABLE,
    CONFIG_RECOVERY_PARAPHRASE,
)

DELTA_FIELDS: tuple[str, ...] = (
    "delta_acc",
    "delta_macro_all3",
    "delta_support_recall",
    "delta_refute_recall",
    "delta_ne_pred_rate",
)
FLAG_FIELDS: tuple[str, ...] = (
    "improves_over_baseline",
    "support_recovery_gain",
    "refute_regression",
    "ne_rate_shift_large",
    "harmful_ne_rate_shift",
    "catastrophic_regression",
)

REFUTE_REGRESSION_THRESHOLD = 0.02
NE_RATE_SHIFT_THRESHOLD = 0.08
ACC_CATASTROPHIC_THRESHOLD = 0.03
MACRO_CATASTROPHIC_THRESHOLD = 0.03
SUPPORT_RECALL_HARM_THRESHOLD = 0.03
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Discovery and parsing
# ---------------------------------------------------------------------------


def _is_stage45_holdout_train_report(path: Path) -> bool:
    """True only if filename is .json AND names a Stage45C/D holdout train report.

    Predicate (all required):
      is_json
      AND ("stage45c" in name_lower OR "stage45d" in name_lower)
      AND "holdout" in name_lower
      AND "train_report" in name_lower
    """
    if path.suffix.lower() != ".json":
        return False
    name = path.name.lower()
    if not any(token in name for token in STAGE45_TOKENS):
        return False
    return all(token in name for token in REQUIRED_NAME_TOKENS)


def discover_report_files(results_dir: Path) -> list[Path]:
    """Find candidate Stage45C/Stage45D JSON train reports under results_dir."""
    if not results_dir.exists():
        return []
    return [
        path
        for path in sorted(results_dir.rglob("*.json"))
        if _is_stage45_holdout_train_report(path)
    ]


def extract_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the compact Stage45C/D schema fields from one report payload.

    Checks the JSON top level first, then falls back to a short list of
    common nested wrapper keys, since some producers may nest the compact
    summary fields under a subkey rather than at the top level.
    """
    candidates: list[dict[str, Any]] = [payload]
    for wrapper_key in NESTED_WRAPPER_KEYS:
        nested = payload.get(wrapper_key)
        if isinstance(nested, dict):
            candidates.append(nested)

    extracted: dict[str, Any] = {}
    for field in FIELDS:
        value = None
        for candidate in candidates:
            candidate_value = candidate.get(field)
            if candidate_value is not None:
                value = candidate_value
                break
        extracted[field] = value
    return extracted


def _round_or_none(value: Any) -> float | None:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def normalize_config_name(support_w: Any, ne_w: Any) -> str:
    """Map a (support_w, ne_w) pair to a readable config name."""
    sw = _round_or_none(support_w)
    nw = _round_or_none(ne_w)
    if sw is None or nw is None:
        return f"w{support_w}_ne{ne_w}"
    if sw == 0.0 and nw == 0.0:
        return CONFIG_BASELINE
    if sw == 0.1 and nw == 0.1:
        return CONFIG_RECOVERY_STABLE
    if sw == 0.1 and nw == 0.2:
        return CONFIG_RECOVERY_PARAPHRASE
    return f"w{support_w}_ne{ne_w}"


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_reports(files: list[Path]) -> list[dict[str, Any]]:
    """Read and flatten every discovered report file into one row per file.

    Files that fail to parse as JSON objects are silently skipped (this is a
    best-effort scan over a results directory that may contain unrelated or
    partially-written files).
    """
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        extracted = extract_fields(payload)
        extracted["source_file"] = _relative_to_root(path)
        extracted["config_name"] = normalize_config_name(
            extracted.get("support_w"), extracted.get("ne_w")
        )
        rows.append(extracted)
    return rows


# ---------------------------------------------------------------------------
# Grouping, deltas, and decision flags
# ---------------------------------------------------------------------------


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    def norm(value: Any) -> tuple[bool, str]:
        return (value is None, "" if value is None else str(value))

    return (
        norm(row.get("field")),
        norm(row.get("family")),
        norm(row.get("config_name")),
    )


def build_holdout_groups(
    rows: list[dict[str, Any]],
) -> dict[tuple[Any, Any], list[dict[str, Any]]]:
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("field"), row.get("family"))
        groups.setdefault(key, []).append(row)
    return groups


def _safe_delta(value: Any, baseline_value: Any) -> float | None:
    if value is None or baseline_value is None:
        return None
    try:
        return float(value) - float(baseline_value)
    except (TypeError, ValueError):
        return None


def _blank_delta_and_flags(entry: dict[str, Any]) -> None:
    for key in DELTA_FIELDS + FLAG_FIELDS:
        entry[key] = None


def analyze_holdout_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach delta_* fields and decision flags to every row in one holdout group.

    Deltas/flags are computed relative to the group's baseline row (support_w=0,
    ne_w=0) when one exists. The baseline row itself, and any row in a group with
    no baseline present, gets None for every delta/flag field.
    """
    baseline_row = next(
        (row for row in rows if row["config_name"] == CONFIG_BASELINE), None
    )

    analyzed: list[dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        if baseline_row is None or row is baseline_row:
            _blank_delta_and_flags(entry)
            analyzed.append(entry)
            continue

        delta_acc = _safe_delta(row.get("acc"), baseline_row.get("acc"))
        delta_macro_all3 = _safe_delta(row.get("macro_all3"), baseline_row.get("macro_all3"))
        delta_support_recall = _safe_delta(
            row.get("support_recall"), baseline_row.get("support_recall")
        )
        delta_refute_recall = _safe_delta(
            row.get("refute_recall"), baseline_row.get("refute_recall")
        )
        delta_ne_pred_rate = _safe_delta(
            row.get("ne_pred_rate"), baseline_row.get("ne_pred_rate")
        )

        entry["delta_acc"] = delta_acc
        entry["delta_macro_all3"] = delta_macro_all3
        entry["delta_support_recall"] = delta_support_recall
        entry["delta_refute_recall"] = delta_refute_recall
        entry["delta_ne_pred_rate"] = delta_ne_pred_rate

        refute_regression = (
            delta_refute_recall is not None
            and delta_refute_recall < -REFUTE_REGRESSION_THRESHOLD
        )
        # Diagnostic only: a large absolute swing in NE prediction rate, in
        # either direction. A large *decrease* (less over-rejection) can be a
        # genuine improvement, so this flag alone must not imply harm.
        ne_rate_shift_large = (
            delta_ne_pred_rate is not None
            and abs(delta_ne_pred_rate) > NE_RATE_SHIFT_THRESHOLD
        )
        # harmful_ne_rate_shift narrows ne_rate_shift_large to cases where the
        # large NE-rate swing co-occurs with real harm on another core metric,
        # so a beneficial NE-rate drop (accuracy/macro/support/refute all
        # improving) is not counted as catastrophic on its own.
        harmful_ne_rate_shift = bool(
            ne_rate_shift_large
            and (
                (delta_acc is not None and delta_acc < -ACC_CATASTROPHIC_THRESHOLD)
                or (
                    delta_macro_all3 is not None
                    and delta_macro_all3 < -MACRO_CATASTROPHIC_THRESHOLD
                )
                or (
                    delta_support_recall is not None
                    and delta_support_recall < -SUPPORT_RECALL_HARM_THRESHOLD
                )
                or refute_regression
            )
        )
        catastrophic_regression = bool(
            (delta_acc is not None and delta_acc < -ACC_CATASTROPHIC_THRESHOLD)
            or (
                delta_macro_all3 is not None
                and delta_macro_all3 < -MACRO_CATASTROPHIC_THRESHOLD
            )
            or refute_regression
            or harmful_ne_rate_shift
        )

        entry["improves_over_baseline"] = bool(
            delta_macro_all3 is not None
            and delta_acc is not None
            and delta_macro_all3 >= 0.0
            and delta_acc >= 0.0
        )
        entry["support_recovery_gain"] = bool(
            delta_support_recall is not None and delta_support_recall > 0.0
        )
        entry["refute_regression"] = refute_regression
        entry["ne_rate_shift_large"] = ne_rate_shift_large
        entry["harmful_ne_rate_shift"] = harmful_ne_rate_shift
        entry["catastrophic_regression"] = catastrophic_regression
        analyzed.append(entry)
    return analyzed


def analyze_all_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = build_holdout_groups(rows)
    analyzed: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: (k[0] is None, str(k[0]), k[1] is None, str(k[1]))):
        analyzed.extend(analyze_holdout_group(groups[key]))
    analyzed.sort(key=_sort_key)
    return analyzed


# ---------------------------------------------------------------------------
# Overall selection summary and recommendation
# ---------------------------------------------------------------------------


def _average(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def compute_overall_summary(analyzed_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for config in SUMMARY_CONFIGS:
        config_rows = [row for row in analyzed_rows if row["config_name"] == config]
        summary[config] = {
            "holdout_groups_seen": len(config_rows),
            "groups_improved_over_baseline": sum(
                1 for row in config_rows if row.get("improves_over_baseline") is True
            ),
            "groups_with_support_recovery_gain": sum(
                1 for row in config_rows if row.get("support_recovery_gain") is True
            ),
            "groups_with_refute_regression": sum(
                1 for row in config_rows if row.get("refute_regression") is True
            ),
            "groups_with_ne_rate_shift_large": sum(
                1 for row in config_rows if row.get("ne_rate_shift_large") is True
            ),
            "groups_with_harmful_ne_rate_shift": sum(
                1 for row in config_rows if row.get("harmful_ne_rate_shift") is True
            ),
            "groups_with_catastrophic_regression": sum(
                1 for row in config_rows if row.get("catastrophic_regression") is True
            ),
            "avg_delta_acc": _average([row.get("delta_acc") for row in config_rows]),
            "avg_delta_macro_all3": _average(
                [row.get("delta_macro_all3") for row in config_rows]
            ),
            "avg_delta_support_recall": _average(
                [row.get("delta_support_recall") for row in config_rows]
            ),
            "avg_delta_refute_recall": _average(
                [row.get("delta_refute_recall") for row in config_rows]
            ),
            "avg_delta_ne_pred_rate": _average(
                [row.get("delta_ne_pred_rate") for row in config_rows]
            ),
        }
    return summary


def _find_group_row(
    rows: list[dict[str, Any]], field: Any, family: Any, config_name: str
) -> dict[str, Any] | None:
    for row in rows:
        if (
            row.get("field") == field
            and row.get("family") == family
            and row["config_name"] == config_name
        ):
            return row
    return None


def _is_best_on_group(
    rows: list[dict[str, Any]], field: Any, family: Any, config_name: str
) -> bool:
    """True if config_name has the highest macro_all3 (falling back to acc) among
    all configs present for this (field, family) holdout group."""
    group_rows = [
        row for row in rows if row.get("field") == field and row.get("family") == family
    ]
    scored = [
        (
            row["config_name"],
            row.get("macro_all3") if row.get("macro_all3") is not None else row.get("acc"),
        )
        for row in group_rows
    ]
    scored = [(name, score) for name, score in scored if score is not None]
    if not scored:
        return False
    best_name, _ = max(scored, key=lambda item: item[1])
    return best_name == config_name


def build_recommendation(
    analyzed_rows: list[dict[str, Any]], summary: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    has_any_data = len(analyzed_rows) > 0

    stable = summary.get(CONFIG_RECOVERY_STABLE, {})
    specialized = summary.get(CONFIG_RECOVERY_PARAPHRASE, {})

    stable_cat = stable.get("groups_with_catastrophic_regression", 0) or 0
    specialized_cat = specialized.get("groups_with_catastrophic_regression", 0) or 0

    primary_none_stable = _find_group_row(
        analyzed_rows, "primary_failure_type", "none", CONFIG_RECOVERY_STABLE
    )
    stable_preserves_primary_none = bool(
        primary_none_stable is not None
        and (
            primary_none_stable.get("improves_over_baseline") is True
            or (
                primary_none_stable.get("delta_macro_all3") is not None
                and primary_none_stable["delta_macro_all3"] >= -_EPS
                and primary_none_stable.get("catastrophic_regression") is not True
            )
        )
    )

    avg_macro_stable = stable.get("avg_delta_macro_all3")
    stable_macro_ok = avg_macro_stable is not None and avg_macro_stable >= -_EPS

    stable_no_refute_regression = (stable.get("groups_with_refute_regression", 0) or 0) == 0

    recommend_stable_default = bool(
        has_any_data
        and (stable_cat == 0 or stable_cat < specialized_cat)
        and stable_preserves_primary_none
        and stable_macro_ok
        and stable_no_refute_regression
    )

    specialized_best_on_paraphrase = _is_best_on_group(
        analyzed_rows, "intervention_type", "paraphrase", CONFIG_RECOVERY_PARAPHRASE
    )
    primary_none_specialized = _find_group_row(
        analyzed_rows, "primary_failure_type", "none", CONFIG_RECOVERY_PARAPHRASE
    )
    specialized_regresses_primary_none = bool(
        primary_none_specialized is not None
        and (
            primary_none_specialized.get("catastrophic_regression") is True
            or (
                primary_none_specialized.get("delta_macro_all3") is not None
                and primary_none_specialized["delta_macro_all3"] < 0.0
            )
        )
    )
    mark_specialized_paraphrase_only = bool(
        has_any_data
        and specialized_best_on_paraphrase
        and (specialized_regresses_primary_none or specialized_cat > stable_cat)
    )

    other_configs = sorted(
        {
            row["config_name"]
            for row in analyzed_rows
            if row["config_name"] not in SUMMARY_CONFIGS
        }
    )

    return {
        "has_any_data": has_any_data,
        "recommend_stable_default": recommend_stable_default,
        "mark_paraphrase_specialized": mark_specialized_paraphrase_only,
        "stable_config": CONFIG_RECOVERY_STABLE,
        "specialized_config": CONFIG_RECOVERY_PARAPHRASE,
        "other_observed_configs": other_configs,
        "reasoning": {
            "stable_catastrophic_regression_groups": stable_cat,
            "specialized_catastrophic_regression_groups": specialized_cat,
            "stable_preserves_or_improves_primary_failure_type_none": (
                stable_preserves_primary_none
            ),
            "stable_avg_delta_macro_all3": avg_macro_stable,
            "stable_has_no_refute_regression": stable_no_refute_regression,
            "specialized_is_best_on_intervention_type_paraphrase": (
                specialized_best_on_paraphrase
            ),
            "specialized_regresses_on_primary_failure_type_none": (
                specialized_regresses_primary_none
            ),
        },
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

CSV_COLUMNS: tuple[str, ...] = (
    "source_file",
    "field",
    "family",
    "config_name",
    "support_w",
    "ne_w",
    "acc",
    "macro_all3",
    "macro_present",
    "ne_pred_rate",
    "support_recall",
    "refute_recall",
    "pred",
    "stage44b2_decision",
) + DELTA_FIELDS + FLAG_FIELDS


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _comparison_table_md(rows: list[dict[str, Any]]) -> list[str]:
    columns = (
        "field",
        "family",
        "config_name",
        "support_w",
        "ne_w",
        "acc",
        "macro_all3",
        "macro_present",
        "ne_pred_rate",
        "support_recall",
        "refute_recall",
        "stage44b2_decision",
    )
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return lines


def _delta_table_md(rows: list[dict[str, Any]]) -> list[str]:
    columns = ("field", "family", "config_name") + DELTA_FIELDS + FLAG_FIELDS
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        if row["config_name"] == CONFIG_BASELINE:
            continue
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return lines


def _summary_table_md(summary: dict[str, dict[str, Any]]) -> list[str]:
    columns = (
        "config",
        "holdout_groups_seen",
        "groups_improved_over_baseline",
        "groups_with_support_recovery_gain",
        "groups_with_refute_regression",
        "groups_with_ne_rate_shift_large",
        "groups_with_harmful_ne_rate_shift",
        "groups_with_catastrophic_regression",
        "avg_delta_acc",
        "avg_delta_macro_all3",
        "avg_delta_support_recall",
        "avg_delta_refute_recall",
        "avg_delta_ne_pred_rate",
    )
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for config in SUMMARY_CONFIGS:
        stats = summary.get(config, {})
        values = [config] + [_fmt(stats.get(column)) for column in columns[1:]]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _recommendation_md(recommendation: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if not recommendation["has_any_data"]:
        lines.append(
            "No Stage45C/Stage45D train report JSON files were found under the scanned "
            "results directory. This audit produced no comparison data; run the "
            "Stage45C/Stage45D holdout trainings and re-run this script to populate the "
            "audit."
        )
        return lines

    reasoning = recommendation["reasoning"]

    lines.append("**Stable global default:**")
    if recommendation["recommend_stable_default"]:
        lines.append(
            f"- `{CONFIG_RECOVERY_STABLE}` (support_w=0.1, ne_w=0.1) is recommended as the "
            "stable global default. It has no more catastrophic regressions than "
            f"`{CONFIG_RECOVERY_PARAPHRASE}` "
            f"({reasoning['stable_catastrophic_regression_groups']} vs. "
            f"{reasoning['specialized_catastrophic_regression_groups']}), preserves or "
            "improves the `primary_failure_type=none` holdout, has a non-negative average "
            f"delta_macro_all3 ({_fmt(reasoning['stable_avg_delta_macro_all3'])}), and shows "
            "no refute_recall regression across holdout groups."
        )
    else:
        lines.append(
            f"- `{CONFIG_RECOVERY_STABLE}` (support_w=0.1, ne_w=0.1) does NOT currently meet "
            "the stable-default criteria; see `reasoning` in "
            "`results/stage45d_generalization_summary.json` for the specific condition(s) "
            "that failed."
        )

    lines.append("")
    lines.append("**Paraphrase-specialized diagnostic setting:**")
    if recommendation["mark_paraphrase_specialized"]:
        lines.append(
            f"- `{CONFIG_RECOVERY_PARAPHRASE}` (support_w=0.1, ne_w=0.2) is marked as a "
            "paraphrase-specialized diagnostic setting, not a general default: it is the "
            "best-performing config on the `intervention_type=paraphrase` holdout, but "
            "regresses on `primary_failure_type=none` and/or shows more catastrophic "
            f"regressions than `{CONFIG_RECOVERY_STABLE}` "
            f"({reasoning['specialized_catastrophic_regression_groups']} vs. "
            f"{reasoning['stable_catastrophic_regression_groups']})."
        )
    else:
        lines.append(
            f"- `{CONFIG_RECOVERY_PARAPHRASE}` (support_w=0.1, ne_w=0.2) does not currently "
            "meet the paraphrase-specialized criteria in the scanned data; see `reasoning` "
            "in `results/stage45d_generalization_summary.json`."
        )

    lines.append("")
    lines.append("**Dropped settings:**")
    other = recommendation.get("other_observed_configs") or []
    if other:
        lines.append(
            "- The following observed configs are neither the stable default nor the "
            "paraphrase-specialized setting and are not recommended for further use: "
            + ", ".join(f"`{name}`" for name in other)
        )
    else:
        lines.append("- None. Only baseline and the two named recovery configs were observed.")

    return lines


def render_markdown(
    analyzed_rows: list[dict[str, Any]],
    summary: dict[str, dict[str, Any]],
    recommendation: dict[str, Any],
    files_scanned: int,
) -> str:
    lines: list[str] = [
        "# Stage45D Generalization / Regression Audit",
        "",
        (
            "Stage45C compared internal-only SUPPORT entitlement recovery weight settings "
            "(baseline support_w=0.0/ne_w=0.0, stable candidate support_w=0.1/ne_w=0.1, and "
            "a paraphrase-specialized candidate support_w=0.1/ne_w=0.2) on the "
            "`intervention_type=paraphrase` and `primary_failure_type=none` internal family "
            "holdouts. This Stage45D audit aggregates every Stage45C/Stage45D-style train "
            "report found under the scanned results directory, compares each recovery "
            "config against its holdout group's baseline, and checks whether the "
            "`recovery_w01_ne01` candidate generalizes across other internal holdout "
            "families without catastrophic regression. This is a reporting/aggregation-only "
            "pass over existing JSON reports; it does not train or evaluate anything."
        ),
        "",
        f"Report files scanned: {files_scanned}. Rows parsed: {len(analyzed_rows)}.",
        "",
        "## Per-Holdout Comparison Table",
        "",
    ]
    if analyzed_rows:
        lines.extend(_comparison_table_md(analyzed_rows))
    else:
        lines.append(
            "_No Stage45C/Stage45D train report JSON files were found under the scanned "
            "results directory._"
        )
    lines.extend(["", "## Delta Table (vs. baseline within each holdout group)", ""])
    delta_rows = [row for row in analyzed_rows if row["config_name"] != CONFIG_BASELINE]
    if delta_rows:
        lines.extend(_delta_table_md(analyzed_rows))
    else:
        lines.append("_No non-baseline configs with a matching baseline row were found._")
    lines.extend(["", "## Overall Selection Summary", ""])
    lines.extend(_summary_table_md(summary))
    lines.extend(["", "## Recommendation", ""])
    lines.extend(_recommendation_md(recommendation))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage45D: build a generalization/regression audit for Stage45C internal "
            "SUPPORT recovery weight selection by scanning existing Stage45C/Stage45D "
            "train report JSON files. Reporting/aggregation-only; does not train or "
            "evaluate anything."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to scan (recursively) for Stage45C/Stage45D train report JSON files. "
        "Default: results/",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help="Output CSV path. Default: results/stage45d_generalization_audit.csv",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=DEFAULT_MD_OUTPUT,
        help="Output Markdown path. Default: results/stage45d_generalization_audit.md",
    )
    parser.add_argument(
        "--summary-json-output",
        type=Path,
        default=DEFAULT_SUMMARY_JSON_OUTPUT,
        help="Output summary JSON path. Default: results/stage45d_generalization_summary.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    files = discover_report_files(args.results_dir)
    raw_rows = load_reports(files)
    analyzed_rows = analyze_all_rows(raw_rows)
    summary = compute_overall_summary(analyzed_rows)
    recommendation = build_recommendation(analyzed_rows, summary)

    write_csv(analyzed_rows, args.csv_output)

    markdown = render_markdown(analyzed_rows, summary, recommendation, len(files))
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text(markdown, encoding="utf-8")

    summary_payload = {
        "decision": (
            "STAGE45D_GENERALIZATION_AUDIT_READY"
            if analyzed_rows
            else "STAGE45D_GENERALIZATION_AUDIT_NO_DATA"
        ),
        "results_dir_scanned": str(args.results_dir),
        "report_files_scanned": len(files),
        "report_files": [_relative_to_root(path) for path in files],
        "rows_parsed": len(analyzed_rows),
        "holdout_groups": sorted(
            {(row.get("field"), row.get("family")) for row in analyzed_rows},
            key=lambda pair: (pair[0] is None, str(pair[0]), pair[1] is None, str(pair[1])),
        ),
        "overall_summary": summary,
        "recommendation": recommendation,
        "leakage_policy": {
            "scope": "internal_stage45c_stage45d_train_reports_only",
            "training_run": False,
            "evaluation_run": False,
            "kaggle_commands_run": False,
            "external_data_used": False,
        },
    }
    args.summary_json_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_output.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary_payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
