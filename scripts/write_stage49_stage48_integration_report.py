"""Stage49: verify the Stage48 runner integration against the Stage47 frozen config.

Reporting/verification stage only. Reads results/stage47_selected_recovery_config_check.json
and statically inspects scripts/train_controlled_v6b_minimal.py (source text only, no
import/execution) to confirm Stage48's optional --use-stage47-selected-recovery-config /
--stage47-recovery-config-path integration is present and wired to the expected Stage45C
override targets (stage45c_support_recovery_weight, stage45c_entitled_ne_penalty_weight).

If both the Stage47 config and the Stage48 runner source match the expected frozen
selection (recovery_w01_ne01: support_w=0.1, ne_w=0.1), this script writes a ready
report to:

  - results/stage49_stage48_integration_report.json
  - results/stage49_stage48_integration_report.md

If the Stage47 file is missing/malformed/unexpected, or the runner source is missing
any of the expected Stage48 integration markers, this script does not fabricate a
selection: it writes an honest blocked report to the same two output paths instead.

This script does not train, evaluate, import the runner module, or modify any file
it reads (including scripts/train_controlled_v6b_minimal.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STAGE47_CONFIG_JSON = ROOT / "results" / "stage47_selected_recovery_config_check.json"
DEFAULT_RUNNER_PY = ROOT / "scripts" / "train_controlled_v6b_minimal.py"
DEFAULT_REPORT_JSON = ROOT / "results" / "stage49_stage48_integration_report.json"
DEFAULT_REPORT_MD = ROOT / "results" / "stage49_stage48_integration_report.md"

STAGE47_DECISION_READY = "STAGE47_SELECTED_RECOVERY_CONFIG_READY"

STAGE49_DECISION_READY = "STAGE49_STAGE48_INTEGRATION_READY"
STAGE49_DECISION_BLOCKED = "STAGE49_STAGE48_INTEGRATION_BLOCKED"

CONFIG_NAME = "recovery_w01_ne01"
SELECTED_SUPPORT_W = 0.1
SELECTED_NE_W = 0.1

# Source markers that must be present in the Stage48 runner for the integration
# to be considered wired correctly. Checked as plain substrings of the file text.
RUNNER_FLAG_MARKERS = {
    "use_stage47_selected_recovery_config": "--use-stage47-selected-recovery-config",
    "stage47_recovery_config_path": "--stage47-recovery-config-path",
}
RUNNER_OTHER_MARKERS = [
    "load_stage47_selected_recovery_weights",
    "stage45c_support_recovery_weight",
    "stage45c_entitled_ne_penalty_weight",
    "STAGE47_SELECTED_RECOVERY_CONFIG_READY",
]

LEAKAGE_POLICY: dict[str, Any] = {
    "training_run": False,
    "evaluation_run": False,
    "external_data_used": False,
    "scope": "static_stage48_integration_check_only",
}


# ---------------------------------------------------------------------------
# Loading inputs
# ---------------------------------------------------------------------------


def load_stage47_config(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load the Stage47 selected-recovery-config-check JSON.

    Returns (config, error_reason). error_reason is None on success; otherwise
    config is None and error_reason explains why the file could not be used.
    """
    if not path.exists():
        return None, f"stage47_config_missing: no file found at {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, f"stage47_config_malformed: failed to parse {path} ({exc})"
    if not isinstance(payload, dict):
        return None, f"stage47_config_malformed: {path} did not contain a JSON object"
    return payload, None


def load_runner_source(path: Path) -> tuple[str | None, str | None]:
    """Read the Stage48 runner file as plain text (no import/execution).

    Returns (source_text, error_reason).
    """
    if not path.exists():
        return None, f"runner_missing: no file found at {path}"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"runner_unreadable: failed to read {path} ({exc})"
    return text, None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_stage47_config(config: dict[str, Any]) -> str | None:
    """Return None if `config` matches the expected frozen Stage47 selection, else a reason."""
    if config.get("decision") != STAGE47_DECISION_READY:
        return (
            "stage47_not_ready: Stage47 decision is "
            f"{config.get('decision')!r}, not {STAGE47_DECISION_READY!r}"
        )
    if config.get("selected_config_name") != CONFIG_NAME:
        return (
            "stage47_unexpected_config_name: selected_config_name is "
            f"{config.get('selected_config_name')!r}, not {CONFIG_NAME!r}"
        )
    if config.get("selected_support_w") != SELECTED_SUPPORT_W:
        return (
            "stage47_unexpected_support_w: selected_support_w is "
            f"{config.get('selected_support_w')!r}, not {SELECTED_SUPPORT_W!r}"
        )
    if config.get("selected_ne_w") != SELECTED_NE_W:
        return (
            "stage47_unexpected_ne_w: selected_ne_w is "
            f"{config.get('selected_ne_w')!r}, not {SELECTED_NE_W!r}"
        )
    return None


def detect_runner_flags(source: str) -> dict[str, bool]:
    return {key: marker in source for key, marker in RUNNER_FLAG_MARKERS.items()}


def validate_runner_source(source: str) -> str | None:
    """Return None if the runner source contains every expected Stage48 marker, else a reason."""
    missing: list[str] = []
    for marker in RUNNER_FLAG_MARKERS.values():
        if marker not in source:
            missing.append(marker)
    for marker in RUNNER_OTHER_MARKERS:
        if marker not in source:
            missing.append(marker)
    if missing:
        return "runner_missing_markers: " + ", ".join(sorted(missing))
    return None


# ---------------------------------------------------------------------------
# Building outputs
# ---------------------------------------------------------------------------


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_ready_report(
    runner_flags_detected: dict[str, bool],
    stage47_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    return {
        "stage": "Stage49",
        "decision": STAGE49_DECISION_READY,
        "selected_config_name": CONFIG_NAME,
        "selected_support_w": SELECTED_SUPPORT_W,
        "selected_ne_w": SELECTED_NE_W,
        "runner_flags_detected": dict(runner_flags_detected),
        "overridden_runner_args": {
            "stage45c_support_recovery_weight": SELECTED_SUPPORT_W,
            "stage45c_entitled_ne_penalty_weight": SELECTED_NE_W,
        },
        "source_files": {
            "stage47_config": _relative_to_root(stage47_path),
            "runner": _relative_to_root(runner_path),
        },
        "leakage_policy": dict(LEAKAGE_POLICY),
    }


def build_blocked_report(
    reason: str,
    stage47_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    return {
        "stage": "Stage49",
        "decision": STAGE49_DECISION_BLOCKED,
        "block_reason": reason,
        "selected_config_name": None,
        "selected_support_w": None,
        "selected_ne_w": None,
        "runner_flags_detected": {key: False for key in RUNNER_FLAG_MARKERS},
        "overridden_runner_args": {
            "stage45c_support_recovery_weight": None,
            "stage45c_entitled_ne_penalty_weight": None,
        },
        "source_files": {
            "stage47_config": _relative_to_root(stage47_path),
            "runner": _relative_to_root(runner_path),
        },
        "leakage_policy": dict(LEAKAGE_POLICY),
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_ready_markdown(report: dict[str, Any]) -> str:
    flags = report.get("runner_flags_detected") or {}
    overrides = report.get("overridden_runner_args") or {}
    sources = report.get("source_files") or {}
    lines: list[str] = [
        "# Stage49 Stage48 Integration Verification Report",
        "",
        (
            "Stage49 confirms that Stage48 can consume the Stage47 frozen "
            "recovery config. This is a reporting/verification stage only: it "
            "performs no training or evaluation."
        ),
        "",
        "## Result",
        "",
        f"`{report['decision']}`",
        "",
        (
            f"Stable default recovery setting is `{report['selected_config_name']}`, "
            f"support_w={report['selected_support_w']}, ne_w={report['selected_ne_w']}."
        ),
        "",
        (
            "The runner override targets are `stage45c_support_recovery_weight` and "
            "`stage45c_entitled_ne_penalty_weight`."
        ),
        "",
        "## Detected Runner Flags",
        "",
    ]
    for key, present in flags.items():
        lines.append(f"- `{key}`: {present}")
    lines.extend(
        [
            "",
            "## Overridden Runner Args",
            "",
            f"- `stage45c_support_recovery_weight`: {overrides.get('stage45c_support_recovery_weight')}",
            f"- `stage45c_entitled_ne_penalty_weight`: {overrides.get('stage45c_entitled_ne_penalty_weight')}",
            "",
            "## Sources",
            "",
            f"- Stage47 config: `{sources.get('stage47_config')}`",
            f"- Runner: `{sources.get('runner')}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_blocked_markdown(report: dict[str, Any]) -> str:
    sources = report.get("source_files") or {}
    lines: list[str] = [
        "# Stage49 Stage48 Integration Verification Report",
        "",
        (
            "Stage49 confirms that Stage48 can consume the Stage47 frozen "
            "recovery config before downstream runs rely on it. This is a "
            "reporting/verification stage only: it performs no training or "
            "evaluation."
        ),
        "",
        "## Decision",
        "",
        f"`{report['decision']}`",
        "",
        "No integration was confirmed. Stage49 does not fabricate a selection "
        "when the Stage47 file or the Stage48 runner source does not match the "
        "expected frozen integration.",
        "",
        "## Block Reason",
        "",
        str(report.get("block_reason")),
        "",
        "## Sources Checked",
        "",
        f"- Stage47 config: `{sources.get('stage47_config')}`",
        f"- Runner: `{sources.get('runner')}`",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage49: verify the Stage48 runner integration against the Stage47 "
            "frozen recovery config and write a small verification report. "
            "Reporting/verification-only; does not train, evaluate, or import "
            "the runner module."
        )
    )
    parser.add_argument(
        "--stage47-config-json",
        type=Path,
        default=DEFAULT_STAGE47_CONFIG_JSON,
        help="Path to the Stage47 selected recovery config-check JSON. "
        "Default: results/stage47_selected_recovery_config_check.json",
    )
    parser.add_argument(
        "--runner-py",
        type=Path,
        default=DEFAULT_RUNNER_PY,
        help="Path to the Stage48 runner source file. "
        "Default: scripts/train_controlled_v6b_minimal.py",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_REPORT_JSON,
        help="Output path for the Stage49 integration report JSON. "
        "Default: results/stage49_stage48_integration_report.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=DEFAULT_REPORT_MD,
        help="Output path for the Stage49 integration report Markdown. "
        "Default: results/stage49_stage48_integration_report.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config, config_error = load_stage47_config(args.stage47_config_json)
    source, source_error = load_runner_source(args.runner_py)

    block_reason = config_error or source_error
    if block_reason is None:
        assert config is not None
        block_reason = validate_stage47_config(config)
    if block_reason is None:
        assert source is not None
        block_reason = validate_runner_source(source)

    if block_reason is not None:
        report = build_blocked_report(block_reason, args.stage47_config_json, args.runner_py)
        markdown = render_blocked_markdown(report)
    else:
        assert source is not None
        runner_flags_detected = detect_runner_flags(source)
        report = build_ready_report(
            runner_flags_detected, args.stage47_config_json, args.runner_py
        )
        markdown = render_ready_markdown(report)

    for path in (args.report_json, args.report_md):
        path.parent.mkdir(parents=True, exist_ok=True)

    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.report_md.write_text(markdown, encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
