"""Stage47: load/validate the Stage46 frozen recovery selection for reuse.

Reporting/helper stage only. Reads results/stage46_selected_recovery_config.json
and, only if it genuinely reflects the frozen Stage46 decision (stable default
recovery_w01_ne01 with support_w=0.1/ne_w=0.1, diagnostic runner-up
recovery_w010_ne020 with support_w=0.1/ne_w=0.2), writes a small ready-to-use
config-check report so downstream scripts do not need to hardcode or retype
these weights:

  - results/stage47_selected_recovery_config_check.json
  - results/stage47_selected_recovery_config_check.md

If the Stage46 file is missing, malformed, or does not match the expected
frozen values, this script does not fabricate a selection: it writes an
honest blocked report to the same two output paths instead.

This script does not train, evaluate, or modify the Stage46 file it reads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STAGE46_CONFIG_JSON = ROOT / "results" / "stage46_selected_recovery_config.json"
DEFAULT_CHECK_JSON = ROOT / "results" / "stage47_selected_recovery_config_check.json"
DEFAULT_CHECK_MD = ROOT / "results" / "stage47_selected_recovery_config_check.md"

STAGE46_DECISION_FROZEN = "STAGE46_RECOVERY_SELECTION_FROZEN"

STAGE47_DECISION_READY = "STAGE47_SELECTED_RECOVERY_CONFIG_READY"
STAGE47_DECISION_BLOCKED = "STAGE47_SELECTED_RECOVERY_CONFIG_BLOCKED"

CONFIG_STABLE = "recovery_w01_ne01"
CONFIG_SPECIALIZED = "recovery_w010_ne020"

STABLE_SUPPORT_W = 0.1
STABLE_NE_W = 0.1
SPECIALIZED_SUPPORT_W = 0.1
SPECIALIZED_NE_W = 0.2

LEAKAGE_POLICY: dict[str, Any] = {
    "training_run": False,
    "evaluation_run": False,
    "external_data_used": False,
    "scope": "stage46_frozen_config_read_only",
}


# ---------------------------------------------------------------------------
# Loading the Stage46 selected-config JSON
# ---------------------------------------------------------------------------


def load_stage46_config(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load the Stage46 selected-config JSON.

    Returns (config, error_reason). error_reason is None on success; otherwise
    config is None and error_reason explains why the file could not be used.
    """
    if not path.exists():
        return None, f"stage46_config_missing: no file found at {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, f"stage46_config_malformed: failed to parse {path} ({exc})"
    if not isinstance(payload, dict):
        return None, f"stage46_config_malformed: {path} did not contain a JSON object"
    return payload, None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_stage46_config(config: dict[str, Any]) -> str | None:
    """Return None if `config` matches the frozen Stage46 selection, else a reason string."""
    if config.get("decision") != STAGE46_DECISION_FROZEN:
        return (
            "stage46_not_frozen: Stage46 decision is "
            f"{config.get('decision')!r}, not {STAGE46_DECISION_FROZEN!r}"
        )

    stable = config.get("selected_stable_default")
    if not isinstance(stable, dict):
        return "stage46_config_malformed: missing 'selected_stable_default' object"
    if stable.get("config_name") != CONFIG_STABLE:
        return (
            "stage46_unexpected_stable_config: selected_stable_default.config_name is "
            f"{stable.get('config_name')!r}, not {CONFIG_STABLE!r}"
        )
    if stable.get("support_w") != STABLE_SUPPORT_W:
        return (
            "stage46_unexpected_stable_support_w: selected_stable_default.support_w is "
            f"{stable.get('support_w')!r}, not {STABLE_SUPPORT_W!r}"
        )
    if stable.get("ne_w") != STABLE_NE_W:
        return (
            "stage46_unexpected_stable_ne_w: selected_stable_default.ne_w is "
            f"{stable.get('ne_w')!r}, not {STABLE_NE_W!r}"
        )

    diagnostic = config.get("diagnostic_runner_up")
    if not isinstance(diagnostic, dict):
        return "stage46_config_malformed: missing 'diagnostic_runner_up' object"
    if diagnostic.get("config_name") != CONFIG_SPECIALIZED:
        return (
            "stage46_unexpected_diagnostic_config: diagnostic_runner_up.config_name is "
            f"{diagnostic.get('config_name')!r}, not {CONFIG_SPECIALIZED!r}"
        )
    if diagnostic.get("support_w") != SPECIALIZED_SUPPORT_W:
        return (
            "stage46_unexpected_diagnostic_support_w: diagnostic_runner_up.support_w is "
            f"{diagnostic.get('support_w')!r}, not {SPECIALIZED_SUPPORT_W!r}"
        )
    if diagnostic.get("ne_w") != SPECIALIZED_NE_W:
        return (
            "stage46_unexpected_diagnostic_ne_w: diagnostic_runner_up.ne_w is "
            f"{diagnostic.get('ne_w')!r}, not {SPECIALIZED_NE_W!r}"
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


def build_ready_check(config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    return {
        "stage": "Stage47",
        "decision": STAGE47_DECISION_READY,
        "selected_config_name": CONFIG_STABLE,
        "selected_support_w": STABLE_SUPPORT_W,
        "selected_ne_w": STABLE_NE_W,
        "diagnostic_config_name": CONFIG_SPECIALIZED,
        "diagnostic_support_w": SPECIALIZED_SUPPORT_W,
        "diagnostic_ne_w": SPECIALIZED_NE_W,
        "dropped_configs": list(config.get("dropped_configs") or []),
        "source_file": _relative_to_root(source_path),
        "leakage_policy": dict(LEAKAGE_POLICY),
    }


def build_blocked_check(reason: str, source_path: Path) -> dict[str, Any]:
    return {
        "stage": "Stage47",
        "decision": STAGE47_DECISION_BLOCKED,
        "block_reason": reason,
        "selected_config_name": None,
        "selected_support_w": None,
        "selected_ne_w": None,
        "diagnostic_config_name": None,
        "diagnostic_support_w": None,
        "diagnostic_ne_w": None,
        "dropped_configs": [],
        "source_file": _relative_to_root(source_path),
        "leakage_policy": dict(LEAKAGE_POLICY),
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_ready_markdown(check: dict[str, Any]) -> str:
    dropped = check.get("dropped_configs") or []
    lines: list[str] = [
        "# Stage47 Selected Recovery Config Check",
        "",
        (
            "Stage47 confirms that the Stage46 frozen recovery selection is readable "
            "and valid. This is a reporting/helper stage only: it does not train, "
            "evaluate, or fabricate any configuration values."
        ),
        "",
        "## Result",
        "",
        f"`{check['decision']}`",
        "",
        (
            f"Use `{check['selected_config_name']}` "
            f"(support_w={check['selected_support_w']}, ne_w={check['selected_ne_w']}) "
            "as the stable global default."
        ),
        "",
        (
            f"Keep `{check['diagnostic_config_name']}` "
            f"(support_w={check['diagnostic_support_w']}, ne_w={check['diagnostic_ne_w']}) "
            "only as a paraphrase-specialized diagnostic / runner-up, not as the default."
        ),
        "",
        "Do not use dropped configs.",
        "",
        "## Dropped Settings",
        "",
    ]
    if dropped:
        for name in dropped:
            lines.append(f"- `{name}`")
    else:
        lines.append("- None recorded in the source Stage46 file.")

    lines.extend(
        [
            "",
            "## Source",
            "",
            f"- Stage46 file: `{check['source_file']}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_blocked_markdown(check: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Stage47 Selected Recovery Config Check",
        "",
        (
            "Stage47 confirms that the Stage46 frozen recovery selection is readable "
            "and valid before downstream scripts reuse it. This is a reporting/helper "
            "stage only."
        ),
        "",
        "## Decision",
        "",
        f"`{check['decision']}`",
        "",
        "No recovery configuration was confirmed. Stage47 does not fabricate a "
        "selection when the Stage46 file is missing or does not match the frozen values.",
        "",
        "## Block Reason",
        "",
        str(check.get("block_reason")),
        "",
        "## Source",
        "",
        f"- Stage46 file checked: `{check['source_file']}`",
        "",
        "## Next Step",
        "",
        (
            "Re-run Stage46 (`scripts/write_stage46_recovery_selection_report.py`) "
            f"against real Stage45D evidence, confirm it writes decision "
            f"`{STAGE46_DECISION_FROZEN}` with `{CONFIG_STABLE}` "
            f"(support_w={STABLE_SUPPORT_W}, ne_w={STABLE_NE_W}) as the stable default "
            f"and `{CONFIG_SPECIALIZED}` (support_w={SPECIALIZED_SUPPORT_W}, "
            f"ne_w={SPECIALIZED_NE_W}) as the diagnostic runner-up, and then re-run "
            "this script."
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
            "Stage47: validate the Stage46 frozen recovery selection and write a "
            "small config-check report for downstream scripts to reuse. "
            "Reporting/helper-only; does not train or evaluate anything."
        )
    )
    parser.add_argument(
        "--stage46-config-json",
        type=Path,
        default=DEFAULT_STAGE46_CONFIG_JSON,
        help="Path to the Stage46 selected recovery config JSON. "
        "Default: results/stage46_selected_recovery_config.json",
    )
    parser.add_argument(
        "--check-json",
        type=Path,
        default=DEFAULT_CHECK_JSON,
        help="Output path for the Stage47 config-check JSON. "
        "Default: results/stage47_selected_recovery_config_check.json",
    )
    parser.add_argument(
        "--check-md",
        type=Path,
        default=DEFAULT_CHECK_MD,
        help="Output path for the Stage47 config-check Markdown. "
        "Default: results/stage47_selected_recovery_config_check.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config, load_error = load_stage46_config(args.stage46_config_json)

    block_reason = load_error
    if block_reason is None:
        assert config is not None
        block_reason = validate_stage46_config(config)

    if block_reason is not None:
        check = build_blocked_check(block_reason, args.stage46_config_json)
        markdown = render_blocked_markdown(check)
    else:
        assert config is not None
        check = build_ready_check(config, args.stage46_config_json)
        markdown = render_ready_markdown(check)

    for path in (args.check_json, args.check_md):
        path.parent.mkdir(parents=True, exist_ok=True)

    args.check_json.write_text(
        json.dumps(check, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.check_md.write_text(markdown, encoding="utf-8")

    print(json.dumps(check, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
