"""Stage46: freeze the Stage45D-selected internal SUPPORT recovery configuration.

Reporting/freeze-only. Reads results/stage45d_generalization_summary.json (and,
optionally, the Stage45D CSV/Markdown audit for richer reporting), and — only if
that summary genuinely recommends recovery_w01_ne01 as the stable global default —
freezes the selection and writes:

  - results/stage46_selected_recovery_config.json
  - results/stage46_recovery_selection_summary.json
  - results/stage46_recovery_selection_report.md

If the Stage45D summary is missing, malformed, has no data, or does not recommend
the stable default, this script does not fabricate a selection: it writes an
honest blocked/no-selection report to the same three output paths instead.

This script does not train, evaluate, or run any Kaggle/OOD/experiment commands,
and it never modifies the Stage45D output files it reads.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STAGE45D_SUMMARY_JSON = ROOT / "results" / "stage45d_generalization_summary.json"
DEFAULT_STAGE45D_AUDIT_CSV = ROOT / "results" / "stage45d_generalization_audit.csv"
DEFAULT_STAGE45D_AUDIT_MD = ROOT / "results" / "stage45d_generalization_audit.md"

DEFAULT_SELECTED_CONFIG_JSON = ROOT / "results" / "stage46_selected_recovery_config.json"
DEFAULT_SELECTION_SUMMARY_JSON = ROOT / "results" / "stage46_recovery_selection_summary.json"
DEFAULT_SELECTION_REPORT_MD = ROOT / "results" / "stage46_recovery_selection_report.md"

STAGE46_DECISION_FROZEN = "STAGE46_RECOVERY_SELECTION_FROZEN"
STAGE46_DECISION_BLOCKED = "STAGE46_RECOVERY_SELECTION_BLOCKED"

CONFIG_STABLE = "recovery_w01_ne01"
CONFIG_SPECIALIZED = "recovery_w010_ne020"

# Canonical (support_w, ne_w) for the two named recovery configs, matching the
# normalize_config_name() convention established in
# scripts/write_stage45d_generalization_audit.py. The Stage45D summary JSON
# only stores aggregated metrics (not per-row weights), so these are fixed by
# the config-name convention rather than re-derived at runtime.
STABLE_SUPPORT_W = 0.1
STABLE_NE_W = 0.1
SPECIALIZED_SUPPORT_W = 0.1
SPECIALIZED_NE_W = 0.2

CAVEATS: tuple[str, ...] = (
    "Stage45D used reconstructed/internal Stage45C train-report JSONs where applicable.",
    "Stage46 is a reporting/freeze stage only.",
    "No training or evaluation is performed by this script.",
)

KEY_METRIC_FIELDS: tuple[str, ...] = (
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


# ---------------------------------------------------------------------------
# Loading Stage45D inputs
# ---------------------------------------------------------------------------


def load_stage45d_summary(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load the Stage45D summary JSON.

    Returns (summary, error_reason). error_reason is None on success; otherwise
    summary is None and error_reason explains why the file could not be used.
    """
    if not path.exists():
        return None, f"stage45d_summary_missing: no file found at {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, f"stage45d_summary_malformed: failed to parse {path} ({exc})"
    if not isinstance(payload, dict):
        return None, f"stage45d_summary_malformed: {path} did not contain a JSON object"
    return payload, None


def load_stage45d_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Best-effort load of the Stage45D audit CSV for optional richer reporting.

    Returns an empty list if the file is missing or unreadable; this is purely
    supplemental and never affects the freeze/block decision.
    """
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Gating: decide whether Stage45D genuinely supports a frozen selection
# ---------------------------------------------------------------------------


def validate_stage45d_summary(summary: dict[str, Any]) -> str | None:
    """Return None if `summary` supports freezing recovery_w01_ne01, else a reason string."""
    recommendation = summary.get("recommendation")
    if not isinstance(recommendation, dict):
        return "stage45d_summary_malformed: missing 'recommendation' object"

    if not recommendation.get("has_any_data"):
        return "stage45d_no_data: Stage45D found no comparison data (has_any_data is false)"

    rows_parsed = summary.get("rows_parsed")
    if not isinstance(rows_parsed, int) or rows_parsed <= 0:
        return "stage45d_no_data: Stage45D rows_parsed is zero or missing"

    if recommendation.get("stable_config") != CONFIG_STABLE:
        return (
            "stage45d_unexpected_stable_config: Stage45D's recommendation.stable_config "
            f"is {recommendation.get('stable_config')!r}, not {CONFIG_STABLE!r}"
        )

    if recommendation.get("recommend_stable_default") is not True:
        return (
            "stage45d_does_not_recommend_stable_default: "
            "Stage45D recommendation.recommend_stable_default is not true"
        )

    return None


# ---------------------------------------------------------------------------
# Building outputs
# ---------------------------------------------------------------------------


def _format_holdout_groups(summary: dict[str, Any]) -> list[str]:
    groups = summary.get("holdout_groups") or []
    formatted: list[str] = []
    for group in groups:
        if isinstance(group, (list, tuple)) and len(group) == 2:
            formatted.append(f"{group[0]}={group[1]}")
        else:
            formatted.append(str(group))
    return formatted


def build_selected_config(
    summary: dict[str, Any], summary_path: Path
) -> dict[str, Any]:
    recommendation = summary.get("recommendation") or {}
    dropped_configs = list(recommendation.get("other_observed_configs") or [])

    return {
        "stage": "Stage46",
        "decision": STAGE46_DECISION_FROZEN,
        "selected_stable_default": {
            "config_name": CONFIG_STABLE,
            "support_w": STABLE_SUPPORT_W,
            "ne_w": STABLE_NE_W,
            "role": "stable_global_default",
        },
        "diagnostic_runner_up": {
            "config_name": CONFIG_SPECIALIZED,
            "support_w": SPECIALIZED_SUPPORT_W,
            "ne_w": SPECIALIZED_NE_W,
            "role": "paraphrase_specialized_diagnostic_runner_up",
        },
        "dropped_configs": dropped_configs,
        "source_summary": {
            "path": _relative_to_root(summary_path),
            "decision": summary.get("decision"),
            "rows_parsed": summary.get("rows_parsed"),
            "holdout_groups": _format_holdout_groups(summary),
        },
        "caveats": list(CAVEATS),
    }


def build_blocked_config(reason: str, summary_path: Path) -> dict[str, Any]:
    return {
        "stage": "Stage46",
        "decision": STAGE46_DECISION_BLOCKED,
        "selected_stable_default": None,
        "diagnostic_runner_up": None,
        "dropped_configs": [],
        "source_summary": {
            "path": _relative_to_root(summary_path),
            "decision": None,
            "rows_parsed": None,
            "holdout_groups": [],
        },
        "block_reason": reason,
        "caveats": list(CAVEATS),
    }


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_selection_summary(
    selected_config: dict[str, Any], summary: dict[str, Any] | None
) -> dict[str, Any]:
    overall_summary = (summary or {}).get("overall_summary") or {}
    recommendation = (summary or {}).get("recommendation") or {}

    key_metrics = {
        CONFIG_STABLE: {
            field: (overall_summary.get(CONFIG_STABLE) or {}).get(field)
            for field in KEY_METRIC_FIELDS
        },
        CONFIG_SPECIALIZED: {
            field: (overall_summary.get(CONFIG_SPECIALIZED) or {}).get(field)
            for field in KEY_METRIC_FIELDS
        },
    }

    return {
        "decision": selected_config["decision"],
        "selected_config": selected_config.get("selected_stable_default"),
        "diagnostic_runner_up": selected_config.get("diagnostic_runner_up"),
        "dropped_configs": selected_config.get("dropped_configs", []),
        "key_metrics": key_metrics,
        "recommendation_reasoning": recommendation.get("reasoning", {}),
        "leakage_policy": {
            "training_run": False,
            "evaluation_run": False,
            "external_data_used": False,
            "scope": "stage45d_internal_report_aggregation_only",
        },
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

FINAL_RECOMMENDATION_SENTENCE = (
    "Stage46 freezes recovery_w01_ne01 (support_w=0.1, ne_w=0.1) as the stable global "
    "recovery setting. recovery_w010_ne020 is retained only as a paraphrase-specialized "
    "diagnostic / runner-up and is not selected as the global default."
)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _evidence_table_md(key_metrics: dict[str, dict[str, Any]]) -> list[str]:
    columns = ("config",) + KEY_METRIC_FIELDS
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for config_name in (CONFIG_STABLE, CONFIG_SPECIALIZED):
        stats = key_metrics.get(config_name, {})
        values = [config_name] + [_fmt(stats.get(field)) for field in KEY_METRIC_FIELDS]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_frozen_markdown(
    selected_config: dict[str, Any], selection_summary: dict[str, Any]
) -> str:
    source = selected_config["source_summary"]
    dropped = selected_config.get("dropped_configs") or []
    stable = selected_config["selected_stable_default"]
    specialized = selected_config["diagnostic_runner_up"]

    lines: list[str] = [
        "# Stage46 Recovery Selection Freeze",
        "",
        (
            "Stage46 is a reporting/freeze-only stage. It reads the Stage45D "
            "generalization/regression audit and, only when that audit genuinely "
            f"recommends `{CONFIG_STABLE}` as the stable global default, freezes the "
            "internal SUPPORT entitlement recovery selection and exports final "
            "selection artifacts. It performs no training, evaluation, or further "
            "experiments."
        ),
        "",
        "## Selected Stable Default",
        "",
        f"- Config name: `{stable['config_name']}`",
        f"- support_w: {stable['support_w']}",
        f"- ne_w: {stable['ne_w']}",
        f"- Role: `{stable['role']}`",
        "",
        "## Diagnostic Runner-Up",
        "",
        f"- Config name: `{specialized['config_name']}`",
        f"- support_w: {specialized['support_w']}",
        f"- ne_w: {specialized['ne_w']}",
        f"- Role: `{specialized['role']}`",
        "",
        "## Dropped Settings",
        "",
    ]
    if dropped:
        for name in dropped:
            lines.append(f"- `{name}`")
    else:
        lines.append("- None recorded in the source Stage45D summary.")

    lines.extend(["", "## Stage45D Evidence", "", f"- Source summary: `{source['path']}`"])
    lines.append(f"- Stage45D decision: `{source['decision']}`")
    lines.append(f"- Rows parsed: {source['rows_parsed']}")
    lines.append(
        "- Holdout groups: "
        + ", ".join(f"`{group}`" for group in source["holdout_groups"])
    )
    lines.append("")
    lines.extend(_evidence_table_md(selection_summary["key_metrics"]))

    lines.extend(["", "## Caveats / Provenance", ""])
    for caveat in selected_config.get("caveats") or []:
        lines.append(f"- {caveat}")

    lines.extend(["", "## Final Recommendation", "", FINAL_RECOMMENDATION_SENTENCE, ""])
    return "\n".join(lines)


def render_blocked_markdown(selected_config: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Stage46 Recovery Selection Freeze",
        "",
        (
            "Stage46 is a reporting/freeze-only stage. It reads the Stage45D "
            "generalization/regression audit and, only when that audit genuinely "
            f"recommends `{CONFIG_STABLE}` as the stable global default, freezes the "
            "internal SUPPORT entitlement recovery selection."
        ),
        "",
        "## Decision",
        "",
        f"`{selected_config['decision']}`",
        "",
        "No recovery configuration was selected or frozen. Stage46 does not fabricate a "
        "selection when the Stage45D evidence does not support one.",
        "",
        "## Block Reason",
        "",
        str(selected_config.get("block_reason")),
        "",
        "## Source Summary",
        "",
        f"- Path checked: `{selected_config['source_summary']['path']}`",
        "",
        "## Next Step",
        "",
        (
            "Re-run the Stage45D generalization audit "
            "(`scripts/write_stage45d_generalization_audit.py`) against a results "
            "directory that contains real Stage45C/Stage45D holdout train report JSON "
            "files, confirm it reports `recommend_stable_default: true` for "
            f"`{CONFIG_STABLE}`, and then re-run this script."
        ),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage46: freeze the Stage45D-selected internal SUPPORT recovery "
            "configuration and export final selection artifacts. "
            "Reporting/freeze-only; does not train or evaluate anything."
        )
    )
    parser.add_argument(
        "--stage45d-summary-json",
        type=Path,
        default=DEFAULT_STAGE45D_SUMMARY_JSON,
        help="Path to the Stage45D summary JSON. "
        "Default: results/stage45d_generalization_summary.json",
    )
    parser.add_argument(
        "--stage45d-audit-csv",
        type=Path,
        default=DEFAULT_STAGE45D_AUDIT_CSV,
        help="Optional path to the Stage45D audit CSV (supplemental only). "
        "Default: results/stage45d_generalization_audit.csv",
    )
    parser.add_argument(
        "--stage45d-audit-md",
        type=Path,
        default=DEFAULT_STAGE45D_AUDIT_MD,
        help="Optional path to the Stage45D audit Markdown (supplemental only). "
        "Default: results/stage45d_generalization_audit.md",
    )
    parser.add_argument(
        "--selected-config-json",
        type=Path,
        default=DEFAULT_SELECTED_CONFIG_JSON,
        help="Output path for the frozen/blocked selected-config JSON. "
        "Default: results/stage46_selected_recovery_config.json",
    )
    parser.add_argument(
        "--selection-summary-json",
        type=Path,
        default=DEFAULT_SELECTION_SUMMARY_JSON,
        help="Output path for the selection summary JSON. "
        "Default: results/stage46_recovery_selection_summary.json",
    )
    parser.add_argument(
        "--selection-report-md",
        type=Path,
        default=DEFAULT_SELECTION_REPORT_MD,
        help="Output path for the selection Markdown report. "
        "Default: results/stage46_recovery_selection_report.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    summary, load_error = load_stage45d_summary(args.stage45d_summary_json)
    # Supplemental only; never affects the freeze/block decision.
    load_stage45d_csv_rows(args.stage45d_audit_csv)

    block_reason = load_error
    if block_reason is None:
        assert summary is not None
        block_reason = validate_stage45d_summary(summary)

    if block_reason is not None:
        selected_config = build_blocked_config(block_reason, args.stage45d_summary_json)
        selection_summary = {
            "decision": selected_config["decision"],
            "selected_config": None,
            "diagnostic_runner_up": None,
            "dropped_configs": [],
            "key_metrics": {},
            "recommendation_reasoning": {},
            "leakage_policy": {
                "training_run": False,
                "evaluation_run": False,
                "external_data_used": False,
                "scope": "stage45d_internal_report_aggregation_only",
            },
            "block_reason": block_reason,
        }
        markdown = render_blocked_markdown(selected_config)
    else:
        assert summary is not None
        selected_config = build_selected_config(summary, args.stage45d_summary_json)
        selection_summary = build_selection_summary(selected_config, summary)
        markdown = render_frozen_markdown(selected_config, selection_summary)

    for path in (
        args.selected_config_json,
        args.selection_summary_json,
        args.selection_report_md,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    args.selected_config_json.write_text(
        json.dumps(selected_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.selection_summary_json.write_text(
        json.dumps(selection_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.selection_report_md.write_text(markdown, encoding="utf-8")

    print(json.dumps(selected_config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
