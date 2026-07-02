"""Stage50: write a usage manifest for the Stage48 frozen recovery config integration.

Reporting-only. Reads results/stage47_selected_recovery_config_check.json and
results/stage49_stage48_integration_report.json and, only if both confirm the
frozen recovery_w01_ne01 selection (support_w=0.1, ne_w=0.1) is ready and the
Stage48 runner integration is verified, writes a small usage manifest telling
downstream runs how to consume it:

  - results/stage50_stage48_usage_manifest.json
  - results/stage50_stage48_usage_manifest.md

If either input file is missing, malformed, or does not match the expected
frozen values, this script does not fabricate a manifest: it writes an honest
blocked report to the same two output paths instead.

This script does not train, evaluate, or modify any file it reads (including
scripts/train_controlled_v6b_minimal.py, which it does not touch at all).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STAGE47_CONFIG_JSON = ROOT / "results" / "stage47_selected_recovery_config_check.json"
DEFAULT_STAGE49_REPORT_JSON = ROOT / "results" / "stage49_stage48_integration_report.json"
DEFAULT_MANIFEST_JSON = ROOT / "results" / "stage50_stage48_usage_manifest.json"
DEFAULT_MANIFEST_MD = ROOT / "results" / "stage50_stage48_usage_manifest.md"

STAGE47_DECISION_READY = "STAGE47_SELECTED_RECOVERY_CONFIG_READY"
STAGE49_DECISION_READY = "STAGE49_STAGE48_INTEGRATION_READY"

STAGE50_DECISION_READY = "STAGE50_STAGE48_USAGE_MANIFEST_READY"
STAGE50_DECISION_BLOCKED = "STAGE50_STAGE48_USAGE_MANIFEST_BLOCKED"

CONFIG_NAME = "recovery_w01_ne01"
SELECTED_SUPPORT_W = 0.1
SELECTED_NE_W = 0.1

DIAGNOSTIC_CONFIG_NAME = "recovery_w010_ne020"
DIAGNOSTIC_SUPPORT_W = 0.1
DIAGNOSTIC_NE_W = 0.2
DIAGNOSTIC_ROLE = "paraphrase_specialized_diagnostic_runner_up"

REQUIRED_RUNNER_FLAG = "--use-stage47-selected-recovery-config"
DEFAULT_CONFIG_PATH = "results/stage47_selected_recovery_config_check.json"

LEAKAGE_POLICY: dict[str, Any] = {
    "training_run": False,
    "evaluation_run": False,
    "external_data_used": False,
    "scope": "static_usage_manifest_only",
}


# ---------------------------------------------------------------------------
# Loading inputs
# ---------------------------------------------------------------------------


def load_json_file(path: Path, label: str) -> tuple[dict[str, Any] | None, str | None]:
    """Load a JSON object file. Returns (payload, error_reason)."""
    if not path.exists():
        return None, f"{label}_missing: no file found at {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, f"{label}_malformed: failed to parse {path} ({exc})"
    if not isinstance(payload, dict):
        return None, f"{label}_malformed: {path} did not contain a JSON object"
    return payload, None


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
    if config.get("diagnostic_config_name") != DIAGNOSTIC_CONFIG_NAME:
        return (
            "stage47_unexpected_diagnostic_config_name: diagnostic_config_name is "
            f"{config.get('diagnostic_config_name')!r}, not {DIAGNOSTIC_CONFIG_NAME!r}"
        )
    if config.get("diagnostic_support_w") != DIAGNOSTIC_SUPPORT_W:
        return (
            "stage47_unexpected_diagnostic_support_w: diagnostic_support_w is "
            f"{config.get('diagnostic_support_w')!r}, not {DIAGNOSTIC_SUPPORT_W!r}"
        )
    if config.get("diagnostic_ne_w") != DIAGNOSTIC_NE_W:
        return (
            "stage47_unexpected_diagnostic_ne_w: diagnostic_ne_w is "
            f"{config.get('diagnostic_ne_w')!r}, not {DIAGNOSTIC_NE_W!r}"
        )
    return None


def validate_stage49_report(report: dict[str, Any]) -> str | None:
    """Return None if `report` confirms the Stage48 integration is ready, else a reason."""
    if report.get("decision") != STAGE49_DECISION_READY:
        return (
            "stage49_not_ready: Stage49 decision is "
            f"{report.get('decision')!r}, not {STAGE49_DECISION_READY!r}"
        )
    if report.get("selected_config_name") != CONFIG_NAME:
        return (
            "stage49_unexpected_config_name: selected_config_name is "
            f"{report.get('selected_config_name')!r}, not {CONFIG_NAME!r}"
        )
    if report.get("selected_support_w") != SELECTED_SUPPORT_W:
        return (
            "stage49_unexpected_support_w: selected_support_w is "
            f"{report.get('selected_support_w')!r}, not {SELECTED_SUPPORT_W!r}"
        )
    if report.get("selected_ne_w") != SELECTED_NE_W:
        return (
            "stage49_unexpected_ne_w: selected_ne_w is "
            f"{report.get('selected_ne_w')!r}, not {SELECTED_NE_W!r}"
        )
    return None


# ---------------------------------------------------------------------------
# Building outputs
# ---------------------------------------------------------------------------


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_ready_manifest(
    stage47_config: dict[str, Any],
    stage47_path: Path,
    stage49_path: Path,
) -> dict[str, Any]:
    dropped_configs = list(stage47_config.get("dropped_configs") or [])
    return {
        "stage": "Stage50",
        "decision": STAGE50_DECISION_READY,
        "selected_config_name": CONFIG_NAME,
        "selected_support_w": SELECTED_SUPPORT_W,
        "selected_ne_w": SELECTED_NE_W,
        "required_runner_flag": REQUIRED_RUNNER_FLAG,
        "default_config_path": DEFAULT_CONFIG_PATH,
        "overridden_runner_args": {
            "stage45c_support_recovery_weight": SELECTED_SUPPORT_W,
            "stage45c_entitled_ne_penalty_weight": SELECTED_NE_W,
        },
        "diagnostic_runner_up": {
            "config_name": DIAGNOSTIC_CONFIG_NAME,
            "support_w": DIAGNOSTIC_SUPPORT_W,
            "ne_w": DIAGNOSTIC_NE_W,
            "role": DIAGNOSTIC_ROLE,
        },
        "dropped_configs": dropped_configs,
        "source_files": {
            "stage47_config": _relative_to_root(stage47_path),
            "stage49_report": _relative_to_root(stage49_path),
        },
        "leakage_policy": dict(LEAKAGE_POLICY),
    }


def build_blocked_manifest(
    reason: str,
    stage47_path: Path,
    stage49_path: Path,
) -> dict[str, Any]:
    return {
        "stage": "Stage50",
        "decision": STAGE50_DECISION_BLOCKED,
        "block_reason": reason,
        "selected_config_name": None,
        "selected_support_w": None,
        "selected_ne_w": None,
        "required_runner_flag": None,
        "default_config_path": None,
        "overridden_runner_args": {
            "stage45c_support_recovery_weight": None,
            "stage45c_entitled_ne_penalty_weight": None,
        },
        "diagnostic_runner_up": None,
        "dropped_configs": [],
        "source_files": {
            "stage47_config": _relative_to_root(stage47_path),
            "stage49_report": _relative_to_root(stage49_path),
        },
        "leakage_policy": dict(LEAKAGE_POLICY),
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_ready_markdown(manifest: dict[str, Any]) -> str:
    overrides = manifest.get("overridden_runner_args") or {}
    diagnostic = manifest.get("diagnostic_runner_up") or {}
    dropped = manifest.get("dropped_configs") or []
    sources = manifest.get("source_files") or {}
    lines: list[str] = [
        "# Stage50 Stage48 Usage Manifest",
        "",
        (
            "Stage50 records how downstream runs should use the Stage47/48 frozen "
            "recovery config. This is a reporting stage only: it performs no "
            "training or evaluation."
        ),
        "",
        "## Result",
        "",
        f"`{manifest['decision']}`",
        "",
        "## How To Use",
        "",
        (
            f"Use `{manifest['required_runner_flag']}` for stable default recovery "
            "runs. Do not manually retype support_w/ne_w unless intentionally "
            "doing an ablation."
        ),
        "",
        (
            f"This loads `{manifest['selected_config_name']}` "
            f"(support_w={manifest['selected_support_w']}, ne_w={manifest['selected_ne_w']}) "
            f"from `{manifest['default_config_path']}` and overrides "
            f"`stage45c_support_recovery_weight={overrides.get('stage45c_support_recovery_weight')}` "
            f"and `stage45c_entitled_ne_penalty_weight={overrides.get('stage45c_entitled_ne_penalty_weight')}`."
        ),
        "",
        (
            f"`{diagnostic.get('config_name')}` (support_w={diagnostic.get('support_w')}, "
            f"ne_w={diagnostic.get('ne_w')}) is diagnostic only, not the global default."
        ),
        "",
        "## Dropped Configs",
        "",
    ]
    if dropped:
        for name in dropped:
            lines.append(f"- `{name}`")
    else:
        lines.append("- None recorded in the source Stage47 file.")

    lines.extend(
        [
            "",
            "## Sources",
            "",
            f"- Stage47 config: `{sources.get('stage47_config')}`",
            f"- Stage49 report: `{sources.get('stage49_report')}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_blocked_markdown(manifest: dict[str, Any]) -> str:
    sources = manifest.get("source_files") or {}
    lines: list[str] = [
        "# Stage50 Stage48 Usage Manifest",
        "",
        (
            "Stage50 records how downstream runs should use the Stage47/48 frozen "
            "recovery config. This is a reporting stage only: it performs no "
            "training or evaluation."
        ),
        "",
        "## Decision",
        "",
        f"`{manifest['decision']}`",
        "",
        "No usage manifest was confirmed. Stage50 does not fabricate config "
        "values when the Stage47 or Stage49 inputs are missing or do not match "
        "the expected frozen selection.",
        "",
        "## Block Reason",
        "",
        str(manifest.get("block_reason")),
        "",
        "## Sources Checked",
        "",
        f"- Stage47 config: `{sources.get('stage47_config')}`",
        f"- Stage49 report: `{sources.get('stage49_report')}`",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage50: write a usage manifest for the Stage48 frozen recovery "
            "config integration, gated on Stage47 readiness and Stage49 "
            "verification. Reporting-only; does not train or evaluate anything."
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
        "--stage49-report-json",
        type=Path,
        default=DEFAULT_STAGE49_REPORT_JSON,
        help="Path to the Stage49 Stage48 integration report JSON. "
        "Default: results/stage49_stage48_integration_report.json",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=DEFAULT_MANIFEST_JSON,
        help="Output path for the Stage50 usage manifest JSON. "
        "Default: results/stage50_stage48_usage_manifest.json",
    )
    parser.add_argument(
        "--manifest-md",
        type=Path,
        default=DEFAULT_MANIFEST_MD,
        help="Output path for the Stage50 usage manifest Markdown. "
        "Default: results/stage50_stage48_usage_manifest.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    stage47_config, stage47_error = load_json_file(args.stage47_config_json, "stage47_config")
    stage49_report, stage49_error = load_json_file(args.stage49_report_json, "stage49_report")

    block_reason = stage47_error or stage49_error
    if block_reason is None:
        assert stage47_config is not None
        block_reason = validate_stage47_config(stage47_config)
    if block_reason is None:
        assert stage49_report is not None
        block_reason = validate_stage49_report(stage49_report)

    if block_reason is not None:
        manifest = build_blocked_manifest(
            block_reason, args.stage47_config_json, args.stage49_report_json
        )
        markdown = render_blocked_markdown(manifest)
    else:
        assert stage47_config is not None
        manifest = build_ready_manifest(
            stage47_config, args.stage47_config_json, args.stage49_report_json
        )
        markdown = render_ready_markdown(manifest)

    for path in (args.manifest_json, args.manifest_md):
        path.parent.mkdir(parents=True, exist_ok=True)

    args.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.manifest_md.write_text(markdown, encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
