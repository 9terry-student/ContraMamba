#!/usr/bin/env python3
"""Build the fail-closed Stage191-B six-run deterministic replay manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import subprocess
import traceback
from pathlib import Path
from typing import Any, Iterable

STAGE191_TRAINING_SEEDS = (174, 175, 176)
SEEDS = STAGE191_TRAINING_SEEDS
ARMS = ("baseline", "intervention")
TRAINING_COMMIT = "bee2f5ad452d1d9f57b30f444d18835dbffdbecf"
STAGE190_COMMIT = "ac0b9032b94436ce8ac8134c650d389134faebd4"
STAGE191A_COMMIT = "a6700aa6344e81ef02f0696e117c3d9ffbbaa10c"
TRAINER189_SHA = "24b01c5799c762772fe1700204afae59f8566898f65e7f3eefa4ac57ac6f126f"
DATA_SHA = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
SIDECAR_SHA = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
READY189 = "STAGE189A_THREE_SEED_MARGIN_REPLICATION_AND_POSTHOC_REFERENCE_SPEC_READY"
CLOSED190 = "STAGE190C_MARGIN_GRADIENT_HEAD_LOCAL_OR_NONCONFLICTING"
REQUIRED191A = "STAGE191A_DETERMINISTIC_REPLAY_REQUIRED"
BLOCKED = "STAGE191B_DETERMINISTIC_REPLAY_IMPLEMENTATION_BLOCKED"
READY = "STAGE191B_DETERMINISTIC_REPLAY_MANIFEST_READY"
SELECTED = {"seed174_baseline":20,"seed174_intervention":20,"seed175_baseline":20,"seed175_intervention":19,"seed176_baseline":20,"seed176_intervention":19}
SELECTED_CHECKPOINT_SHA256 = {
    "seed174_baseline": "8e31dbd1459a67e65571ea1926a6e1a5f49f1ae2e57deb8455b41617f9ed972c",
    "seed174_intervention": "66cfb4fd91c29dfc6d4f243e701103e6b82e6ccac810a3b7a17d5b05310a57b3",
    "seed175_baseline": "5baa306161f204dbc984681a0b18c22484e3724aea4bfbdc9858e6f434ea1c0a",
    "seed175_intervention": "c00700d170e11fcc0376e2fe0ca7bc8037a76330330ed1b99fa21a35775a8018",
    "seed176_baseline": "8bcae6880e68cf8f34b9fb86f3f987e26872511636af33a64f6cc012857e51ea",
    "seed176_intervention": "539ff0c226e6f862abf99c9b4ceaf883989cee9985c5a5cf438801ecb87620a5",
}
STAGE191_RUN_KEYS = {
    "seed174_baseline",
    "seed174_intervention",
    "seed175_baseline",
    "seed175_intervention",
    "seed176_baseline",
    "seed176_intervention",
}
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
FLAGS = ("--stage191-trajectory-replay-observability", "--stage191-save-trajectory-state-capsules")
STAGE191_FORBIDDEN_PATH_OPTIONS = (
    "ood_data",
    "output_ood_json",
    "output_ood_predictions_json",
    "external_data",
    "external_output_dir",
    "external_eval_jsonl",
    "external_eval_name",
    "stage43_external_factver_jsonl",
    "stage57_bridge_train_jsonl",
    "stage66_bridge_train_jsonl",
    "stage75_bridge_train_jsonl",
    "stage80a_bridge_train_jsonl",
)
STAGE191_LIST_ABSENT_OPTIONS = (
    "stage43_external_factver_jsonl",
    "external_eval_jsonl",
    "external_eval_name",
)
STAGE191_SCALAR_ABSENT_OPTIONS = (
    "ood_data",
    "output_ood_json",
    "output_ood_predictions_json",
    "external_data",
    "external_output_dir",
    "stage57_bridge_train_jsonl",
    "stage66_bridge_train_jsonl",
    "stage75_bridge_train_jsonl",
    "stage80a_bridge_train_jsonl",
)
STAGE191_EXTERNAL_ENABLE_FLAGS = (
    "enable_external_eval",
    "enable_stage43_external_eval",
    "stage43_external_enable_shadow_export",
)
STAGE191_BRIDGE_MODE_OPTIONS = (
    "stage57_bridge_train_mode",
    "stage66_bridge_train_mode",
    "stage75_bridge_train_mode",
    "stage80a_bridge_train_mode",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--stage189a-dir", type=Path, required=True)
    p.add_argument("--stage190a-dir", type=Path, required=True)
    p.add_argument("--stage190c-dir", type=Path, required=True)
    p.add_argument("--stage191a-dir", type=Path, required=True)
    p.add_argument("--current-replay-git-commit", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle: return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for row in rows: writer.writerow({k: json.dumps(row.get(k), sort_keys=True) if isinstance(row.get(k),(dict,list)) else row.get(k) for k in fields})


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def is_exact_sha256(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def git(repo: Path, *parts: str, binary: bool=False) -> Any:
    result = subprocess.run(["git", *parts], cwd=repo, check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode("utf-8", errors="strict").strip()


def option_map(argv: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}; index = 0
    while index < len(argv):
        token = argv[index]
        if not isinstance(token, str) or not token.startswith("--"): raise ValueError(f"invalid Stage189 argv token: {token!r}")
        if "=" in token:
            raise ValueError(f"unsupported equals-form Stage189 option: {token!r}")
        key = token[2:].replace("-", "_")
        if key in result:
            raise ValueError(f"duplicate Stage189 option: {token}")
        if index + 1 < len(argv) and isinstance(argv[index+1], str) and not argv[index+1].startswith("--"):
            result[key] = argv[index+1]; index += 2
        else: result[key] = True; index += 1
    return result


def rewrite_exact_outputs(
    argv: list[str], historical_run_dir: Path, replay_run_dir: Path
) -> tuple[list[str], dict[str, Any]]:
    if any(flag in argv for flag in FLAGS):
        raise ValueError("historical argv already contains a Stage191 flag")
    rewritten = list(argv)
    changes: list[dict[str, Any]] = []
    output_options = (
        ("--output-json", "training_report.json"),
        ("--output-predictions-json", "clean_dev_predictions.json"),
        ("--stage115-clean-dev-scalar-output-jsonl", "clean_dev_scalars.jsonl"),
    )
    for option, filename in output_options:
        positions = [index for index, token in enumerate(argv) if token == option]
        if any(token.startswith(option + "=") for token in argv):
            raise ValueError(f"{option}: equals form is not the historical token form")
        if len(positions) != 1:
            raise ValueError(f"{option}: expected exactly one occurrence, found {len(positions)}")
        value_index = positions[0] + 1
        if value_index >= len(argv) or argv[value_index].startswith("--"):
            raise ValueError(f"{option}: value token is absent")
        expected = (historical_run_dir / filename).resolve()
        observed = Path(argv[value_index]).resolve()
        if observed != expected:
            raise ValueError(f"{option}: expected {expected}, observed {observed}")
        replacement = str((replay_run_dir / filename).resolve())
        rewritten[value_index] = replacement
        changes.append({
            "option": option, "value_index": value_index,
            "original": argv[value_index], "replacement": replacement,
        })
    rewritten.extend(FLAGS)
    expected_indexes = sorted(change["value_index"] for change in changes)
    changed_indexes = [
        index for index in range(len(argv)) if argv[index] != rewritten[index]
    ]
    unchanged = (
        changed_indexes == expected_indexes
        and all(
            argv[index] == rewritten[index]
            for index in range(len(argv)) if index not in expected_indexes
        )
    )
    appended = rewritten[len(argv):]
    passed = unchanged and appended == list(FLAGS)
    audit = {
        "passed": passed,
        "original_length": len(argv),
        "replay_length": len(rewritten),
        "output_value_changes": changes,
        "changed_existing_token_indexes": changed_indexes,
        "expected_changed_value_indexes": expected_indexes,
        "all_other_tokens_unchanged_and_order_preserved": unchanged,
        "appended_tokens": appended,
        "expected_appended_tokens": list(FLAGS),
    }
    if not passed:
        raise ValueError(f"argv-difference audit failed: {audit}")
    return rewritten, audit


def source_contract(trainer_text: str, provenance_path: Path) -> dict[str, Any]:
    flag_blocks: dict[str, bool] = {}
    for flag in FLAGS:
        pattern = (
            r"parser\.add_argument\(\s*" + re.escape(json.dumps(flag))
            + r"\s*,\s*action=\"store_true\"\s*,\s*default=False\s*,"
        )
        flag_blocks[flag] = re.search(pattern, trainer_text, re.MULTILINE) is not None

    contract_start = trainer_text.index("            _stage191_write_contract(")
    contract_end = trainer_text.index("        # Stage26-F extended", contract_start)
    export_start = trainer_text.index("                _stage191_export_epoch(")
    export_end = trainer_text.index("            # Stage44-B", export_start)
    contract_call_block = trainer_text[contract_start:contract_end]
    export_call_block = trainer_text[export_start:export_end]
    contract_guard_start = trainer_text.rindex(
        "        if args.stage191_trajectory_replay_observability:",
        0,
        contract_start,
    )
    export_guard_start = trainer_text.rindex(
        "            if args.stage191_trajectory_replay_observability:",
        0,
        export_start,
    )
    contract_guard_block = trainer_text[contract_guard_start:contract_end]
    export_guard_block = trainer_text[export_guard_start:export_end]
    export_function_start = trainer_text.index("def _stage191_export_epoch(")
    export_function_end = trainer_text.index("\ndef _stage187_", export_function_start)
    export_function_block = trainer_text[export_function_start:export_function_end]
    runtime_context_start = trainer_text.index("    _stage191_runtime_context:")
    runtime_context_end = trainer_text.index(
        "    # Wrap v5 training", runtime_context_start
    )
    runtime_context_block = trainer_text[runtime_context_start:runtime_context_end]
    runtime_validator_start = trainer_text.index(
        "def _stage191_validate_runtime_contract("
    )
    runtime_validator_end = trainer_text.index(
        "\ndef _stage191_write_contract(", runtime_validator_start
    )
    runtime_validator_block = trainer_text[
        runtime_validator_start:runtime_validator_end
    ]
    contract_function_start = trainer_text.index("def _stage191_write_contract(")
    contract_function_end = trainer_text.index(
        "\ndef _stage191_export_epoch(", contract_function_start
    )
    contract_function_block = trainer_text[
        contract_function_start:contract_function_end
    ]
    selection_score_start = trainer_text.index(
        "            raw_stage191_selection_score = dev_metrics[select_metric]"
    )
    selection_score_end = trainer_text.index(
        "            _stage191_best_epoch_before = best_epoch",
        selection_score_start,
    )
    selection_score_block = trainer_text[
        selection_score_start:selection_score_end
    ]
    invalid_late_symbols = (
        "_stage174a_provenance_record",
        "_stage174a_parsed_args",
        "_stage174a_resolved_runtime_config",
        "compatible_positive_margin_sidecar_audit",
    )
    invalid_by_block = {
        "stage191_contract_call": [
            symbol for symbol in invalid_late_symbols if symbol in contract_call_block
        ],
        "stage191_export_call": [
            symbol for symbol in invalid_late_symbols if symbol in export_call_block
        ],
        "stage191_runtime_context": [
            symbol for symbol in invalid_late_symbols if symbol in runtime_context_block
        ],
    }
    main_function_start = trainer_text.index("def main(argv: list[str] | None = None) -> int:")
    main_args_start = trainer_text.index(
        "    args = parser.parse_args(argv)", main_function_start
    )
    if main_args_start >= runtime_context_start:
        raise ValueError("Stage191 main pre-context boundaries are invalid")
    main_pre_context_block = trainer_text[main_args_start:runtime_context_start]
    main_assignment_needles = {
        "args": "    args = parser.parse_args(argv)",
        "resolved_split_seed": "    resolved_split_seed =",
        "stage187_sidecar_audit": "    _stage187_sidecar_audit:",
        "max_length": "    max_length = max(",
        "model_initialization": "        model = None",
        "model_fallback_guard": "    if model is None:",
        "model_moved_to_device": "    model = model.to(device)",
    }
    main_assignment_positions = {
        name: main_pre_context_block.find(needle)
        for name, needle in main_assignment_needles.items()
    }
    main_assignment_evidence = {
        name: {
            "position_in_main_pre_context_block": position,
            "runtime_context_boundary_offset": len(main_pre_context_block),
            "distance_before_runtime_context_boundary": (
                len(main_pre_context_block) - position if position >= 0 else None
            ),
            "present_before_runtime_context": 0 <= position < len(main_pre_context_block),
        }
        for name, position in main_assignment_positions.items()
    }
    model_fallback_completed_before_move = (
        0 <= main_assignment_positions["model_initialization"]
        < main_assignment_positions["model_fallback_guard"]
        < main_assignment_positions["model_moved_to_device"]
        < len(main_pre_context_block)
    )
    main_scope_assignment_contract = {
        "block_begins_with_args_assignment": main_pre_context_block.startswith(
            "    args = parser.parse_args(argv)"
        ),
        "main_pre_context_block_sha256": hashlib.sha256(
            main_pre_context_block.encode("utf-8")
        ).hexdigest(),
        "runtime_context_boundary_offset": len(main_pre_context_block),
        "assignments": main_assignment_evidence,
        "model_construction_fallback_completed_before_device_move": (
            model_fallback_completed_before_move
        ),
        "passed": (
            main_pre_context_block.startswith("    args = parser.parse_args(argv)")
            and all(
                evidence["present_before_runtime_context"]
                for evidence in main_assignment_evidence.values()
            )
            and model_fallback_completed_before_move
        ),
    }
    position_evidence = {
        "runtime_context_before_nested_trainer": runtime_context_start < trainer_text.index("    def run_training_v6b(", runtime_context_start),
        "nested_trainer_before_contract_call": trainer_text.index("    def run_training_v6b(", runtime_context_start) < contract_start,
        "contract_uses_exact_stage187_audit": "_stage187_sidecar_audit" in contract_call_block,
        "contract_uses_runtime_context": "_stage191_runtime_context" in contract_call_block,
        "export_uses_runtime_context": "_stage191_runtime_context" in export_call_block,
    }
    def_use_evidence = {
        "invalid_late_references_by_exact_block": invalid_by_block,
        "main_scope_assignment_contract": main_scope_assignment_contract,
        "assignment_order": position_evidence,
        "contract_call_block_sha256": hashlib.sha256(contract_call_block.encode("utf-8")).hexdigest(),
        "export_call_block_sha256": hashlib.sha256(export_call_block.encode("utf-8")).hexdigest(),
        "runtime_context_block_sha256": hashlib.sha256(runtime_context_block.encode("utf-8")).hexdigest(),
        "passed": (
            not any(invalid_by_block.values())
            and main_scope_assignment_contract["passed"]
            and all(position_evidence.values())
        ),
    }
    statement_boundaries = {
        "frame_finite_check_then_gold_counts": (
            '        raise RuntimeError(\n'
            '            "Stage191 frame_logit values must all be finite"\n'
            '        )\n\n'
            '    gold_counts = {'
        ) in export_function_block,
        "prediction_path_then_jsonl_write": (
            '    prediction_path = (\n'
            '        output_dir\n'
            '        / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl"\n'
            '    )\n'
            '    _stage191_write_jsonl(prediction_path, rows)'
        ) in export_function_block,
        "merged_frame_gold_form_absent": (
            'finite")    gold_counts = {' not in export_function_block
        ),
        "merged_prediction_write_form_absent": (
            '.jsonl"    _stage191_write_jsonl' not in export_function_block
        ),
        "export_function_block_sha256": hashlib.sha256(
            export_function_block.encode("utf-8")
        ).hexdigest(),
    }
    statement_boundaries["passed"] = all(
        value
        for key, value in statement_boundaries.items()
        if key != "export_function_block_sha256"
    )

    bridge_mode_parser_defaults: dict[str, dict[str, Any]] = {}
    for name in STAGE191_BRIDGE_MODE_OPTIONS:
        option = "--" + name.replace("_", "-")
        option_position = trainer_text.index(json.dumps(option))
        parser_block_start = trainer_text.rindex(
            "    parser.add_argument(", 0, option_position
        )
        parser_block_end = trainer_text.index("\n    )", option_position) + len("\n    )")
        parser_block = trainer_text[parser_block_start:parser_block_end]
        default_is_none = 'default="none"' in parser_block
        bridge_mode_parser_defaults[name] = {
            "option": option,
            "default": "none" if default_is_none else None,
            "default_is_exact_none": default_is_none,
            "parser_block_sha256": hashlib.sha256(
                parser_block.encode("utf-8")
            ).hexdigest(),
        }

    exact_absent_parser_defaults = {
        "stage43_external_factver_jsonl": "[]",
        "external_eval_jsonl": "[]",
        "external_eval_name": "[]",
    }
    absent_parser_default_evidence: dict[str, dict[str, Any]] = {}
    for name, expected_source in exact_absent_parser_defaults.items():
        option = "--" + name.replace("_", "-")
        option_position = trainer_text.index(json.dumps(option))
        parser_block_start = trainer_text.rindex(
            "    parser.add_argument(", 0, option_position
        )
        parser_block_end = trainer_text.index(
            "\n    )", option_position
        ) + len("\n    )")
        parser_block = trainer_text[parser_block_start:parser_block_end]
        exact_default = f"default={expected_source}," in parser_block
        absent_parser_default_evidence[name] = {
            "option": option,
            "required_default_source": expected_source,
            "exact_default_present": exact_default,
            "parser_block_sha256": hashlib.sha256(
                parser_block.encode("utf-8")
            ).hexdigest(),
            "passed": exact_default,
        }

    def extract_string_tuple(name: str) -> tuple[tuple[str, ...], str]:
        block_start = trainer_text.index(name + " = (")
        block_end = trainer_text.index("\n)", block_start) + 2
        block = trainer_text[block_start:block_end]
        return tuple(re.findall(r'    "([a-z0-9_]+)",', block)), block

    trainer_forbidden_paths, forbidden_path_block = extract_string_tuple(
        "STAGE191_FORBIDDEN_PATH_OPTIONS"
    )
    trainer_list_absent, list_absent_block = extract_string_tuple(
        "STAGE191_LIST_ABSENT_OPTIONS"
    )
    trainer_scalar_absent, scalar_absent_block = extract_string_tuple(
        "STAGE191_SCALAR_ABSENT_OPTIONS"
    )
    trainer_external_flags, external_flag_block = extract_string_tuple(
        "STAGE191_EXTERNAL_ENABLE_FLAGS"
    )
    trainer_bridge_modes, bridge_mode_block = extract_string_tuple(
        "STAGE191_BRIDGE_MODE_OPTIONS"
    )
    absent_partition_exact = (
        set(STAGE191_LIST_ABSENT_OPTIONS).isdisjoint(
            STAGE191_SCALAR_ABSENT_OPTIONS
        )
        and set(STAGE191_LIST_ABSENT_OPTIONS).union(
            STAGE191_SCALAR_ABSENT_OPTIONS
        ) == set(STAGE191_FORBIDDEN_PATH_OPTIONS)
    )
    runtime_option_parity = {
        "forbidden_path_options_required": list(STAGE191_FORBIDDEN_PATH_OPTIONS),
        "forbidden_path_options_observed": list(trainer_forbidden_paths),
        "list_absent_options_required": list(STAGE191_LIST_ABSENT_OPTIONS),
        "list_absent_options_observed": list(trainer_list_absent),
        "scalar_absent_options_required": list(STAGE191_SCALAR_ABSENT_OPTIONS),
        "scalar_absent_options_observed": list(trainer_scalar_absent),
        "external_enable_flags_required": list(STAGE191_EXTERNAL_ENABLE_FLAGS),
        "external_enable_flags_observed": list(trainer_external_flags),
        "bridge_mode_options_required": list(STAGE191_BRIDGE_MODE_OPTIONS),
        "bridge_mode_options_observed": list(trainer_bridge_modes),
        "forbidden_path_block_sha256": hashlib.sha256(forbidden_path_block.encode("utf-8")).hexdigest(),
        "list_absent_block_sha256": hashlib.sha256(list_absent_block.encode("utf-8")).hexdigest(),
        "scalar_absent_block_sha256": hashlib.sha256(scalar_absent_block.encode("utf-8")).hexdigest(),
        "external_flag_block_sha256": hashlib.sha256(external_flag_block.encode("utf-8")).hexdigest(),
        "bridge_mode_block_sha256": hashlib.sha256(bridge_mode_block.encode("utf-8")).hexdigest(),
        "absent_partition_exact": absent_partition_exact,
        "passed": (
            trainer_forbidden_paths == STAGE191_FORBIDDEN_PATH_OPTIONS
            and trainer_list_absent == STAGE191_LIST_ABSENT_OPTIONS
            and trainer_scalar_absent == STAGE191_SCALAR_ABSENT_OPTIONS
            and trainer_external_flags == STAGE191_EXTERNAL_ENABLE_FLAGS
            and trainer_bridge_modes == STAGE191_BRIDGE_MODE_OPTIONS
            and absent_partition_exact
        ),
    }

    runtime_absence_contract = {
        "scalar_values_require_exact_none": (
            "observed_absent_values[name] is None"
            in runtime_validator_block
        ),
        "list_values_require_exact_list_type": (
            "type(observed_absent_values[name]) is list"
            in runtime_validator_block
        ),
        "list_values_require_exact_empty_list": (
            "observed_absent_values[name] == []"
            in runtime_validator_block
        ),
        "external_flags_require_exact_false": (
            "value is False for value in observed_external_flags.values()"
            in runtime_validator_block
        ),
        "failure_records_required_and_observed": (
            "required_absent_values={required_absent_values!r}"
            in runtime_validator_block
            and "observed_absent_values={observed_absent_values!r}"
            in runtime_validator_block
            and "observed_external_flags={observed_external_flags!r}"
            in runtime_validator_block
        ),
        "runtime_validator_block_sha256": hashlib.sha256(
            runtime_validator_block.encode("utf-8")
        ).hexdigest(),
    }
    runtime_absence_contract["passed"] = all(
        value
        for key, value in runtime_absence_contract.items()
        if key != "runtime_validator_block_sha256"
    )

    provenance_bytes = provenance_path.read_bytes()
    provenance_text = provenance_bytes.decode("utf-8", errors="strict")
    helper_start = provenance_text.index("def git_info(")
    helper_end = provenance_text.index("\ndef command_string", helper_start)
    helper_block = provenance_text[helper_start:helper_end]
    definition_proof = (
        "def git_info(repo_root: Path) -> dict[str, Any]:" in helper_block
        and 'cwd=str(repo_root)' in helper_block
        and "git_info as provenance_git_info" in trainer_text
        and "provenance_git_info(ROOT)" in runtime_context_block
    )
    git_commit_field_proof = (
        '"git_commit": None' in helper_block
        and 'info["git_commit"] =' in helper_block
        and "return info" in helper_block
    )
    provenance_helper_contract = {
        "inspected_path": str(provenance_path.resolve()),
        "source_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        "helper_name": "git_info",
        "definition_proof": definition_proof,
        "git_commit_field_proof": git_commit_field_proof,
        "helper_block_sha256": hashlib.sha256(helper_block.encode("utf-8")).hexdigest(),
        "passed": definition_proof and git_commit_field_proof,
    }

    raw_assignment = (
        "raw_stage191_selection_score = dev_metrics[select_metric]"
    )
    bool_check = "isinstance(raw_stage191_selection_score, bool)"
    exact_type_check = (
        "type(raw_stage191_selection_score) not in (int, float)"
    )
    finite_check = (
        "not math.isfinite(float(raw_stage191_selection_score))"
    )
    raw_conversion = "score = float(raw_stage191_selection_score)"
    selection_score_contract = {
        "raw_assignment_present": raw_assignment in selection_score_block,
        "stage191_enabled_gate_present": (
            "if args.stage191_trajectory_replay_observability:"
            in selection_score_block
        ),
        "raw_bool_rejected": bool_check in selection_score_block,
        "raw_exact_type_checked": exact_type_check in selection_score_block,
        "raw_finite_checked": finite_check in selection_score_block,
        "raw_checked_before_float_conversion": (
            selection_score_block.index(raw_assignment)
            < selection_score_block.index(bool_check)
            < selection_score_block.index(raw_conversion)
        ),
        "disabled_path_preserves_original_conversion": (
            "else:\n                score = float(dev_metrics[select_metric])"
            in selection_score_block
        ),
        "selection_score_block_sha256": hashlib.sha256(
            selection_score_block.encode("utf-8")
        ).hexdigest(),
    }
    selection_score_contract["passed"] = all(
        value
        for key, value in selection_score_contract.items()
        if key != "selection_score_block_sha256"
    )

    runtime_commit_contract = {
        "source_provenance_exact_dict_check": (
            "if type(_stage191_source_provenance) is not dict:"
            in runtime_context_block
        ),
        "commit_requires_exact_string_type": (
            "type(_stage191_validated_source_commit) is not str"
            in runtime_context_block
        ),
        "exact_40_lowercase_hex_validation": (
            'r"[0-9a-f]{40}", _stage191_validated_source_commit'
            in runtime_context_block
        ),
        "validation_precedes_context_population": (
            runtime_context_block.index(
                "type(_stage191_validated_source_commit) is not str"
            )
            < runtime_context_block.index(
                'r"[0-9a-f]{40}", _stage191_validated_source_commit'
            )
            < runtime_context_block.index(
                "        _stage191_runtime_context = {"
            )
        ),
        "validated_commit_stored_in_context": (
            '"trainer_source_commit": _stage191_validated_source_commit'
            in runtime_context_block
        ),
        "contract_call_uses_validated_context_commit": (
            '_stage191_runtime_context["trainer_source_commit"]'
            in contract_call_block
        ),
        "contract_json_uses_validated_commit_parameter": (
            '"trainer_source_commit": trainer_source_commit'
            in contract_function_block
        ),
        "runtime_context_block_sha256": hashlib.sha256(
            runtime_context_block.encode("utf-8")
        ).hexdigest(),
        "contract_function_block_sha256": hashlib.sha256(
            contract_function_block.encode("utf-8")
        ).hexdigest(),
    }
    runtime_commit_contract["passed"] = all(
        value
        for key, value in runtime_commit_contract.items()
        if not key.endswith("_sha256")
    )

    seed_constant_line = (
        "STAGE191_TRAINING_SEEDS = "
        f"{STAGE191_TRAINING_SEEDS!r}"
    )
    seed_constant_position = trainer_text.index(seed_constant_line)
    seed_constant_block = trainer_text[
        seed_constant_position:seed_constant_position + len(seed_constant_line)
    ]
    raw_seed_gate = (
        "type(args.seed) is not int or args.seed not in "
        "STAGE191_TRAINING_SEEDS"
    )
    nested_seed_checks = (
        "type(seed) is not int",
        "seed != args.seed",
        "seed not in STAGE191_TRAINING_SEEDS",
    )
    training_seed_contract = {
        "required_builder_training_seeds": list(STAGE191_TRAINING_SEEDS),
        "observed_trainer_training_seeds": [174, 175, 176]
        if seed_constant_block == seed_constant_line else None,
        "trainer_builder_seed_tuples_exactly_identical": (
            seed_constant_block == seed_constant_line
            and STAGE191_TRAINING_SEEDS == (174, 175, 176)
        ),
        "trainer_seed_constant_occurs_exactly_once": (
            trainer_text.count(seed_constant_line) == 1
        ),
        "raw_args_seed_type_membership_gate": (
            raw_seed_gate in runtime_validator_block
        ),
        "nested_contract_seed_equality_gate": all(
            check in contract_guard_block for check in nested_seed_checks
        ),
        "nested_export_seed_equality_gate": all(
            check in export_guard_block for check in nested_seed_checks
        ),
        "contract_authorized_seed_list_written": (
            '"authorized_training_seeds": list(STAGE191_TRAINING_SEEDS)'
            in contract_function_block
        ),
        "contract_seed_authorized_true_written": (
            '"training_seed_authorized": True'
            in contract_function_block
        ),
        "contract_validated_current_seed_written": (
            '"training_seed": seed' in contract_function_block
        ),
        "trainer_seed_constant_block_sha256": hashlib.sha256(
            seed_constant_block.encode("utf-8")
        ).hexdigest(),
        "contract_guard_block_sha256": hashlib.sha256(
            contract_guard_block.encode("utf-8")
        ).hexdigest(),
        "export_guard_block_sha256": hashlib.sha256(
            export_guard_block.encode("utf-8")
        ).hexdigest(),
    }
    training_seed_contract["passed"] = all(
        value
        for key, value in training_seed_contract.items()
        if not key.endswith("_sha256")
        and key not in {
            "required_builder_training_seeds",
            "observed_trainer_training_seeds",
        }
    ) and (
        training_seed_contract["required_builder_training_seeds"]
        == training_seed_contract["observed_trainer_training_seeds"]
    )

    top_level_re_import_count = len(
        re.findall(r"(?m)^import re$", trainer_text)
    )
    re_dependency_contract = {
        "top_level_import_re_present": top_level_re_import_count == 1,
        "top_level_import_re_count": top_level_re_import_count,
        "re_fullmatch_used_for_stage191_commit_validation": (
            "or re.fullmatch("
            in runtime_context_block
            and 'r"[0-9a-f]{40}", _stage191_validated_source_commit'
            in runtime_context_block
        ),
    }
    re_dependency_contract["passed"] = (
        re_dependency_contract["top_level_import_re_present"]
        and re_dependency_contract[
            "re_fullmatch_used_for_stage191_commit_validation"
        ]
    )

    proofs = {
        "capsule_requires_observability": (
            "args.stage191_save_trajectory_state_capsules\n"
            "        and not args.stage191_trajectory_replay_observability"
        ) in trainer_text,
        "authoritative_logits_used": 'logits = dev_output.get("logits")' in trainer_text,
        "loss_logits_used_false": '"loss_logits_used": False' in trainer_text,
        "extra_forward_pass_false": '"extra_forward_pass_performed": False' in trainer_text,
        "training_semantics_changed_false": '"training_semantics_changed": False' in trainer_text,
    }
    return {
        "flag_blocks": flag_blocks,
        "proofs": proofs,
        "stage191_def_use_audit": def_use_evidence,
        "stage191_export_statement_boundaries": statement_boundaries,
        "stage191_runtime_option_parity": runtime_option_parity,
        "stage191_runtime_absence_contract": runtime_absence_contract,
        "bridge_mode_parser_defaults": bridge_mode_parser_defaults,
        "absent_parser_default_evidence": absent_parser_default_evidence,
        "provenance_helper_contract": provenance_helper_contract,
        "selection_score_contract": selection_score_contract,
        "runtime_commit_contract": runtime_commit_contract,
        "re_dependency_contract": re_dependency_contract,
        "training_seed_contract": training_seed_contract,
        "passed": (
            all(flag_blocks.values())
            and all(proofs.values())
            and def_use_evidence["passed"]
            and statement_boundaries["passed"]
            and runtime_option_parity["passed"]
            and runtime_absence_contract["passed"]
            and all(
                evidence["default_is_exact_none"]
                for evidence in bridge_mode_parser_defaults.values()
            )
            and all(
                evidence["passed"]
                for evidence in absent_parser_default_evidence.values()
            )
            and provenance_helper_contract["passed"]
            and selection_score_contract["passed"]
            and runtime_commit_contract["passed"]
            and re_dependency_contract["passed"]
            and training_seed_contract["passed"]
        ),
    }

def exact_file_identity(repo: Path, commit: str, path: Path) -> dict[str, Any]:
    rel = path.resolve().relative_to(repo).as_posix()
    working = path.read_bytes()
    blob = git(repo, "show", f"{commit}:{rel}", binary=True)
    working_sha256 = hashlib.sha256(working).hexdigest()
    commit_blob_sha256 = hashlib.sha256(blob).hexdigest()
    unstaged = subprocess.run(
        ["git", "diff", "--quiet", "--", rel], cwd=repo
    ).returncode == 0
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", rel], cwd=repo
    ).returncode == 0
    sha256_equal = working_sha256 == commit_blob_sha256
    passed = working == blob and sha256_equal and unstaged and staged
    return {
        "path": str(path),
        "working_sha256": working_sha256,
        "commit_blob_sha256": commit_blob_sha256,
        "bytes_equal": working == blob,
        "sha256_equal": sha256_equal,
        "no_unstaged_difference": unstaged,
        "no_staged_difference": staged,
        "passed": passed,
    }

def main() -> int:
    args = parse_args(); repo = args.repo_root.resolve(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    authoritative_sidecar=(repo/"reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl").resolve()
    blockers: list[str] = []; gates: list[dict[str, Any]] = []; matrix: list[dict[str, Any]] = []; manifests: list[dict[str, Any]] = []; argv_audits: list[dict[str, Any]] = []; prediction_references: list[dict[str, Any]] = []; selected_prediction_reference_identity_by_run: dict[str, Any] = {}; identities: list[dict[str, Any]] = []; exception = None
    trainer = (repo / "scripts/train_controlled_v6b_minimal.py").resolve(); builder = Path(__file__).resolve()
    provenance_helper = (repo / "scripts/run_provenance.py").resolve()
    trainer_sha: str | None = None
    builder_sha: str | None = None
    provenance_helper_sha: str | None = None
    commits = {"stage189_training_commit": TRAINING_COMMIT, "stage190_diagnostic_commit": STAGE190_COMMIT, "stage191a_diagnostic_commit": STAGE191A_COMMIT, "stage191b_replay_commit": args.current_replay_git_commit.strip()}
    def gate(name: str, required: Any, observed: Any, passed: bool, reason: str) -> None:
        gates.append({"gate":name,"required":required,"observed":observed,"passed":passed,"blocking_reason":"" if passed else reason})
        if not passed: blockers.append(reason)
    try:
        trainer_sha = sha(trainer)
        builder_sha = sha(builder)
        provenance_helper_sha = sha(provenance_helper)
        supplied = args.current_replay_git_commit.strip(); head = git(repo,"rev-parse","HEAD")
        gate("current_head", supplied, head, head == supplied, "current HEAD differs from --current-replay-git-commit")
        identities = [
            exact_file_identity(repo, supplied, trainer),
            exact_file_identity(repo, supplied, builder),
            exact_file_identity(repo, supplied, provenance_helper),
        ]
        gate("trainer_commit_identity", True, identities[0]["passed"], identities[0]["passed"], "trainer bytes/diff state differs from current replay commit")
        gate("builder_commit_identity", True, identities[1]["passed"], identities[1]["passed"], "builder bytes/diff state differs from current replay commit")
        gate(
            "provenance_helper_commit_identity",
            True,
            identities[2],
            identities[2]["passed"],
            "run_provenance.py bytes/diff state differs from current replay commit",
        )
        trainer_text = trainer.read_text(encoding="utf-8")
        trainer_source_contract = source_contract(
            trainer_text, provenance_helper
        )
        gate("default_off_stage191_source_contract", True, trainer_source_contract, trainer_source_contract["passed"], "trainer Stage191 source contract is incomplete")
        s191 = read_json(args.stage191a_dir.resolve() / "stage191a_trajectory_feasibility_report.json")
        required191 = {"decision":REQUIRED191A,"runnable":False,"blocking_reasons":[],"identity_closure_blocking_reasons":[],"artifact_validity_blocking_reasons":[],"identity_and_closure_gates_passed":True,"replay_specification_required":True,"next_stage_specification_authorized":True,"training_authorized":False,"replay_execution_authorized":False,"model_advancement_decision":False,"current_stage191_diagnostic_git_commit":STAGE191A_COMMIT}
        observed191 = {key:s191.get(key) for key in required191}; gate("stage191a_frozen_contract",required191,observed191,observed191==required191,"Stage191-A contract mismatch")
        missing_by_run = s191.get("missing_clean_metric_fields_by_run") or {}
        expected_missing = ["support_recall","false_entitlement_total","polarity_error_total","clean_dev_ce"]
        missing_ok = set(missing_by_run)=={f"seed{s}_{a}" for s in SEEDS for a in ARMS} and all(set(v)==set(expected_missing) for v in missing_by_run.values())
        gate("stage191a_exact_missing_metrics",expected_missing,missing_by_run,missing_ok,"Stage191-A missing-metric inventory mismatch")
        run_keys={f"seed{s}_{a}" for s in SEEDS for a in ARMS}
        margin_complete=s191.get("margin_history_complete_by_run") or {}
        clean_complete=s191.get("clean_metric_history_complete_by_run") or {}
        checkpoint_complete=s191.get("epoch_checkpoint_history_complete_by_run") or {}
        history_ok=(set(margin_complete)==run_keys and all(value is True for value in margin_complete.values()) and set(clean_complete)==run_keys and all(value is False for value in clean_complete.values()) and set(checkpoint_complete)==run_keys and all(value is False for value in checkpoint_complete.values()))
        gate("stage191a_six_run_history_closure",True,history_ok,history_ok,"Stage191-A six-run history closure mismatch")
        s190 = read_json(args.stage190c_dir.resolve() / "stage190c_gradient_conflict_report.json")
        required190 = {
            "decision": CLOSED190,
            "blocking_reasons": [],
            "diagnostic_only": True,
            "model_advancement_decision": False,
            "significance_testing_performed": False,
            "causality_claimed": False,
            "intervention_support_conflict_by_seed": {"174": False, "175": False, "176": False},
            "intervention_shared_margin_gradient_fraction_by_seed": {
                "174": 0.9300744063636249,
                "175": 0.916385071914493,
                "176": 0.9091867800401853,
            },
            "qualified_shared_conflict_seeds": [],
            "authorized_next_design_class": "Investigate checkpoint-selection and optimization-trajectory effects.",
        }
        observed190 = {key: s190.get(key) for key in required190}
        closure_ok = observed190 == required190
        gate("stage190c_authoritative_runtime_contract", required190, observed190, closure_ok, "Stage190-C runtime contract mismatch")
        s190a = read_json(args.stage190a_dir.resolve() / "stage190a_manifest_report.json")
        required190a = {"decision":"STAGE190A_GRADIENT_CONFLICT_DIAGNOSTIC_MANIFEST_READY","runnable":True,"blocking_reasons":[],"manifest_count":6,"training_git_commit":TRAINING_COMMIT,"diagnostic_git_commit":STAGE190_COMMIT}
        observed190a = {key:s190a.get(key) for key in required190a}
        gate("stage190a_frozen_contract",required190a,observed190a,observed190a==required190a,"Stage190-A manifest contract mismatch")
        s189_report = read_json(args.stage189a_dir.resolve() / "stage189a_manifest_report.json")
        identity189 = {"decision":READY189,"current_git_commit":TRAINING_COMMIT,"trainer_sha256":TRAINER189_SHA,"dataset_sha256":DATA_SHA,"sidecar_semantic_sha256":SIDECAR_SHA}
        observed189={k:s189_report.get(k) for k in identity189}; gate("stage189_identity",identity189,observed189,observed189==identity189,"Stage189 identity mismatch")
        frozen_key_observed = {
            "selected_epoch_keys": sorted(SELECTED),
            "selected_checkpoint_sha256_keys": sorted(SELECTED_CHECKPOINT_SHA256),
        }
        frozen_key_required = sorted(STAGE191_RUN_KEYS)
        frozen_key_coverage_ok = (
            set(SELECTED) == STAGE191_RUN_KEYS
            and set(SELECTED_CHECKPOINT_SHA256) == STAGE191_RUN_KEYS
            and len(SELECTED) == 6
            and len(SELECTED_CHECKPOINT_SHA256) == 6
        )
        gate(
            "exact_frozen_run_key_coverage",
            {
                "selected_epoch_keys": frozen_key_required,
                "selected_checkpoint_sha256_keys": frozen_key_required,
            },
            frozen_key_observed,
            frozen_key_coverage_ok,
            "frozen Stage191 run-key coverage is not exact",
        )
        frozen_sha_format_ok = all(
            is_exact_sha256(value)
            for value in SELECTED_CHECKPOINT_SHA256.values()
        )
        gate(
            "frozen_selected_checkpoint_sha256_format",
            "six exact lowercase 64-hex SHA256 values",
            SELECTED_CHECKPOINT_SHA256,
            frozen_sha_format_ok,
            "frozen selected-checkpoint SHA256 map contains an invalid value",
        )
        checkpoint_shas: set[str] = set()
        for seed in SEEDS:
            for arm in ARMS:
                key=f"seed{seed}_{arm}"; historical=read_json(args.stage189a_dir.resolve()/f"stage189a_seed{seed}_{arm}_manifest.json")
                argv=historical.get("argv"); run_raw=historical.get("run_directory")
                if not isinstance(argv,list) or not argv or not isinstance(run_raw,str) or not run_raw: raise ValueError(f"{key}: authoritative Stage189 argv/run_directory absent")
                run_dir=Path(run_raw).resolve(); provenance=read_json(run_dir/"run_provenance.json"); training=read_json(run_dir/"training_report.json")
                parsed=provenance.get("parsed_args") or {}; source=provenance.get("source_provenance") or {}; data=((provenance.get("data_provenance") or {}).get("main_data") or {}); selected=((provenance.get("finalization") or {}).get("selected_checkpoint") or {})
                selected_sha=selected.get("sha256")
                training_path=(run_dir/"training_report.json").resolve()
                provenance_path=(run_dir/"run_provenance.json").resolve()
                checkpoint_path=(run_dir/"selected_checkpoint.pt").resolve()
                selected_prediction_path=(run_dir/"clean_dev_predictions.json").resolve()
                training_sha=sha(training_path)
                provenance_sha=sha(provenance_path)
                current_checkpoint_sha=sha(checkpoint_path)
                stage190_manifest=read_json(args.stage190a_dir.resolve()/f"stage190a_seed{seed}_{arm}_manifest.json")
                artifact_hashes=stage190_manifest.get("artifact_hashes")
                external_contract=stage190_manifest.get("external_use_contract")
                arm_runtime_contract=stage190_manifest.get("arm_runtime_contract")

                frozen_checkpoint_sha = SELECTED_CHECKPOINT_SHA256[key]
                stage190_checkpoint_sha = stage190_manifest.get("checkpoint_sha256")
                checkpoint_base_identity_ok = (
                    is_exact_sha256(frozen_checkpoint_sha)
                    and is_exact_sha256(current_checkpoint_sha)
                    and is_exact_sha256(selected_sha)
                    and is_exact_sha256(stage190_checkpoint_sha)
                    and current_checkpoint_sha == frozen_checkpoint_sha
                    and selected_sha == frozen_checkpoint_sha
                    and stage190_checkpoint_sha == frozen_checkpoint_sha
                )
                validated_checkpoint_sha = (
                    frozen_checkpoint_sha if checkpoint_base_identity_ok else None
                )
                checkpoint_identity_evidence = {
                    "frozen_sha256": frozen_checkpoint_sha,
                    "current_selected_checkpoint_sha256": current_checkpoint_sha,
                    "stage189_provenance_selected_checkpoint_sha256": selected_sha,
                    "stage190a_checkpoint_sha256": stage190_checkpoint_sha,
                    "stage191b_manifest_checkpoint_sha256": validated_checkpoint_sha,
                    "stage191b_matrix_checkpoint_sha256": validated_checkpoint_sha,
                }
                checkpoint_identity_ok = (
                    checkpoint_base_identity_ok
                    and all(
                        is_exact_sha256(value)
                        and value == frozen_checkpoint_sha
                        for value in checkpoint_identity_evidence.values()
                    )
                )

                prediction_file_regular = selected_prediction_path.is_file()
                current_prediction_sha = (
                    sha(selected_prediction_path) if prediction_file_regular else None
                )
                stage190_prediction_path_raw = stage190_manifest.get(
                    "clean_dev_predictions_path"
                )
                stage190_prediction_path = (
                    str(Path(stage190_prediction_path_raw).resolve())
                    if type(stage190_prediction_path_raw) is str
                    and stage190_prediction_path_raw
                    else None
                )
                stage190_prediction_sha = (
                    artifact_hashes.get("clean_dev_predictions")
                    if type(artifact_hashes) is dict else None
                )
                prediction_identity_ok = (
                    prediction_file_regular
                    and type(artifact_hashes) is dict
                    and is_exact_sha256(current_prediction_sha)
                    and is_exact_sha256(stage190_prediction_sha)
                    and stage190_prediction_path == str(selected_prediction_path)
                    and current_prediction_sha == stage190_prediction_sha
                )
                validated_prediction_path = (
                    str(selected_prediction_path) if prediction_identity_ok else None
                )
                validated_prediction_sha = (
                    current_prediction_sha if prediction_identity_ok else None
                )
                prediction_identity_evidence = {
                    "exact_historical_prediction_path": str(selected_prediction_path),
                    "file_exists_and_is_regular": prediction_file_regular,
                    "stage190a_clean_dev_predictions_path": stage190_prediction_path,
                    "current_clean_dev_predictions_sha256": current_prediction_sha,
                    "stage190a_clean_dev_predictions_sha256": stage190_prediction_sha,
                    "artifact_hashes_exact_dict": type(artifact_hashes) is dict,
                    "validated_prediction_path": validated_prediction_path,
                    "validated_prediction_sha256": validated_prediction_sha,
                    "passed": prediction_identity_ok,
                }
                selected_prediction_reference_identity_by_run[key] = (
                    prediction_identity_evidence
                )
                gate(
                    key+"_selected_prediction_reference_identity",
                    {
                        "path": str(selected_prediction_path),
                        "sha256": validated_prediction_sha,
                    },
                    prediction_identity_evidence,
                    prediction_identity_ok,
                    f"{key}: selected prediction reference differs from Stage190-A",
                )
                if prediction_identity_ok:
                    prediction_references.append({
                        "run": key,
                        "original_selected_prediction_path": validated_prediction_path,
                        "original_selected_prediction_sha256": validated_prediction_sha,
                    })

                required_stage190_run={
                    "runnable": True,
                    "blocking_reasons": [],
                    "seed": seed,
                    "arm": arm,
                    "training_seed": seed,
                    "split_seed": 174,
                    "training_git_commit": TRAINING_COMMIT,
                    "diagnostic_git_commit": STAGE190_COMMIT,
                    "fixed_split_identity": {"split_seed": 174, "train": 2880, "dev": 720},
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": frozen_checkpoint_sha,
                    "external_use_contract_passed": True,
                    "arm_runtime_contract_passed": True,
                    "training_report_sha256": training_sha,
                    "run_provenance_sha256": provenance_sha,
                    "selected_checkpoint_sha256": frozen_checkpoint_sha,
                    "clean_dev_predictions_path": str(selected_prediction_path),
                    "clean_dev_predictions_sha256": validated_prediction_sha,
                }
                observed_stage190_run={
                    "runnable": stage190_manifest.get("runnable"),
                    "blocking_reasons": stage190_manifest.get("blocking_reasons"),
                    "seed": stage190_manifest.get("seed"),
                    "arm": stage190_manifest.get("arm"),
                    "training_seed": stage190_manifest.get("training_seed"),
                    "split_seed": stage190_manifest.get("split_seed"),
                    "training_git_commit": stage190_manifest.get("training_git_commit"),
                    "diagnostic_git_commit": stage190_manifest.get("diagnostic_git_commit"),
                    "fixed_split_identity": stage190_manifest.get("fixed_split_identity"),
                    "checkpoint_path": str(Path(stage190_manifest.get("checkpoint_path", "")).resolve()),
                    "checkpoint_sha256": stage190_checkpoint_sha,
                    "external_use_contract_passed": external_contract.get("passed") if type(external_contract) is dict else None,
                    "arm_runtime_contract_passed": arm_runtime_contract.get("passed") if type(arm_runtime_contract) is dict else None,
                    "training_report_sha256": artifact_hashes.get("training_report") if type(artifact_hashes) is dict else None,
                    "run_provenance_sha256": artifact_hashes.get("run_provenance") if type(artifact_hashes) is dict else None,
                    "selected_checkpoint_sha256": stage190_checkpoint_sha,
                    "clean_dev_predictions_path": stage190_prediction_path,
                    "clean_dev_predictions_sha256": stage190_prediction_sha,
                }
                stage190_identity_ok=(
                    observed_stage190_run==required_stage190_run
                    and type(external_contract) is dict
                    and type(arm_runtime_contract) is dict
                    and type(artifact_hashes) is dict
                    and artifact_hashes.get("run_provenance")==provenance_sha
                    and prediction_identity_ok
                    and checkpoint_identity_ok
                )
                gate(key+"_stage190a_runtime_contract",required_stage190_run,observed_stage190_run,stage190_identity_ok,f"{key}: Stage190-A per-run runtime contract mismatch")
                historical_map=option_map(argv)
                identity_ok=(historical.get("seed")==seed and historical.get("arm")==arm and historical.get("training_seed")==seed and historical.get("split_seed")==174 and historical.get("trainer_sha256")==TRAINER189_SHA and historical.get("dataset_sha256")==DATA_SHA and source.get("git_commit")==TRAINING_COMMIT and source.get("trainer_sha256")==TRAINER189_SHA and data.get("sha256")==DATA_SHA and parsed.get("seed")==seed and parsed.get("split_seed")==174 and parsed.get("epochs")==20 and training["runs"]["single"]["best_epoch"]==SELECTED[key] and training["runs"]["single"]["final_epoch"]==20 and selected_sha==validated_checkpoint_sha and current_checkpoint_sha==validated_checkpoint_sha)
                gate(key+"_identity",True,identity_ok,identity_ok,f"{key}: Stage189 run/checkpoint identity mismatch")
                runtime_ok=historical_map.get("backbone")=="mamba" and historical_map.get("device")=="cuda" and historical_map.get("model_name")=="state-spaces/mamba-130m-hf"
                gate(key+"_cuda_mamba_runtime",True,runtime_ok,runtime_ok,f"{key}: exact CUDA/Mamba runtime arguments absent")
                present_forbidden_paths = [
                    name for name in STAGE191_FORBIDDEN_PATH_OPTIONS
                    if name in historical_map
                ]
                present_external_flags = [
                    name for name in STAGE191_EXTERNAL_ENABLE_FLAGS
                    if name in historical_map
                ]
                bridge_mode_evidence: dict[str, dict[str, Any]] = {}
                for mode_name in STAGE191_BRIDGE_MODE_OPTIONS:
                    present = mode_name in historical_map
                    parser_default = trainer_source_contract[
                        "bridge_mode_parser_defaults"
                    ][mode_name]
                    observed_mode = (
                        historical_map.get(mode_name)
                        if present else parser_default.get("default")
                    )
                    bridge_mode_evidence[mode_name] = {
                        "option_present": present,
                        "observed_or_parser_default": observed_mode,
                        "evidence_source": (
                            "historical_argv" if present else "trainer_parser_default"
                        ),
                        "parser_default_is_exact_none": parser_default[
                            "default_is_exact_none"
                        ],
                        "passed": (
                            observed_mode == "none"
                            and (present or parser_default["default_is_exact_none"])
                        ),
                    }
                internal_only_evidence = {
                    "forbidden_path_options": list(STAGE191_FORBIDDEN_PATH_OPTIONS),
                    "present_forbidden_path_options": present_forbidden_paths,
                    "external_enable_flags": list(STAGE191_EXTERNAL_ENABLE_FLAGS),
                    "present_external_enable_flags": present_external_flags,
                    "bridge_modes": bridge_mode_evidence,
                }
                internal_only = (
                    not present_forbidden_paths
                    and not present_external_flags
                    and all(
                        item["passed"] for item in bridge_mode_evidence.values()
                    )
                )
                internal_only_evidence["passed"] = internal_only
                gate(key+"_internal_only",True,internal_only_evidence,internal_only,f"{key}: external/OOD/bridge contract mismatch")
                if arm=="baseline":
                    sidecar_required={
                        "compatible_positive_margin_weight": "0.0",
                        "controlled_integrity_sidecar_path_present": False,
                        "expected_integrity_sidecar_semantic_sha256_present": False,
                    }
                    sidecar_observed={
                        "compatible_positive_margin_weight": historical_map.get("compatible_positive_margin_weight"),
                        "controlled_integrity_sidecar_path_present": "controlled_integrity_sidecar_path" in historical_map,
                        "expected_integrity_sidecar_semantic_sha256_present": "expected_integrity_sidecar_semantic_sha256" in historical_map,
                    }
                else:
                    configured_sidecar=historical_map.get("controlled_integrity_sidecar_path")
                    sidecar_required={
                        "compatible_positive_margin_weight": "0.05",
                        "compatible_positive_margin_logit": "0.0",
                        "controlled_integrity_sidecar_path": str(authoritative_sidecar),
                        "expected_integrity_sidecar_semantic_sha256": SIDECAR_SHA,
                    }
                    sidecar_observed={
                        "compatible_positive_margin_weight": historical_map.get("compatible_positive_margin_weight"),
                        "compatible_positive_margin_logit": historical_map.get("compatible_positive_margin_logit"),
                        "controlled_integrity_sidecar_path": str(((Path(configured_sidecar) if Path(configured_sidecar).is_absolute() else repo/Path(configured_sidecar)).resolve())) if isinstance(configured_sidecar,str) else None,
                        "expected_integrity_sidecar_semantic_sha256": historical_map.get("expected_integrity_sidecar_semantic_sha256"),
                    }
                sidecar_ok=sidecar_observed==sidecar_required
                gate(key+"_sidecar_contract",sidecar_required,sidecar_observed,sidecar_ok,f"{key}: arm sidecar access contract mismatch")
                replay_dir=(output/key).resolve()
                replay_argv, argv_audit=rewrite_exact_outputs(list(argv),run_dir,replay_dir)
                argv_audit={"run":key,**argv_audit}
                argv_audits.append(argv_audit)
                gate(key+"_argv_difference_audit",True,argv_audit,argv_audit["passed"],f"{key}: argv-difference audit failed")
                manifest={"stage":"Stage191-B","run":key,"seed":seed,"arm":arm,"training_seed":seed,"split_seed":174,"runnable":True,"blocking_reasons":[],"diagnostic_replay_only":True,"replay_execution_authorized":True,"training_for_model_advancement_authorized":False,"model_advancement_decision":False,"external_data_used":False,"commit_identities":commits,"trainer_path":str(trainer),"trainer_sha256":trainer_sha,"builder_sha256":builder_sha,"original_stage189_run_directory":str(run_dir),"original_selected_epoch":SELECTED[key],"original_selected_checkpoint_sha256":validated_checkpoint_sha,"original_selected_prediction_path":validated_prediction_path,"original_selected_prediction_sha256":validated_prediction_sha,"original_argv":argv,"argv":replay_argv,"argv_difference_audit":argv_audit,"command_argv":["python",str(trainer),*replay_argv],"command":shlex.join(["python",str(trainer),*replay_argv]),"replay_output_directory":str(replay_dir),"expected_trajectory_rows":20,"expected_prediction_rows_per_epoch":720,"expected_state_capsules":20,"logits_source":"output[\"logits\"]","baseline_sidecar_training_access":False if arm=="baseline" else None,"intervention_sidecar_exact_access_contract":sidecar_required if arm=="intervention" else None}
                matrix_row={"run":key,"seed":seed,"arm":arm,"split_seed":174,"selected_epoch":SELECTED[key],"selected_checkpoint_sha256":validated_checkpoint_sha,"original_selected_prediction_path":validated_prediction_path,"original_selected_prediction_sha256":validated_prediction_sha,"replay_output_directory":str(replay_dir),"expected_trajectory_rows":20,"expected_prediction_rows_per_epoch":720,"expected_state_capsules":20,"runnable":True}
                checkpoint_identity_evidence[
                    "stage191b_manifest_checkpoint_sha256"
                ] = manifest["original_selected_checkpoint_sha256"]
                checkpoint_identity_evidence[
                    "stage191b_matrix_checkpoint_sha256"
                ] = matrix_row["selected_checkpoint_sha256"]
                checkpoint_identity_ok = (
                    checkpoint_identity_ok
                    and manifest["original_selected_checkpoint_sha256"]
                    == frozen_checkpoint_sha
                    and matrix_row["selected_checkpoint_sha256"]
                    == frozen_checkpoint_sha
                )
                gate(
                    key+"_frozen_selected_checkpoint_sha256",
                    frozen_checkpoint_sha,
                    checkpoint_identity_evidence,
                    checkpoint_identity_ok,
                    f"{key}: selected-checkpoint SHA256 differs from frozen map",
                )
                if checkpoint_identity_ok:
                    checkpoint_shas.add(validated_checkpoint_sha)
                manifests.append(manifest); matrix.append(matrix_row)
        checkpoint_map_gate_count = sum(
            item["gate"].endswith("_frozen_selected_checkpoint_sha256")
            for item in gates
        )
        prediction_reference_gate_count = sum(
            item["gate"].endswith("_selected_prediction_reference_identity")
            for item in gates
        )
        gate(
            "exactly_six_frozen_checkpoint_map_gates",
            6,
            checkpoint_map_gate_count,
            checkpoint_map_gate_count == 6,
            "exactly six frozen checkpoint-map gates were not emitted",
        )
        gate(
            "exactly_six_prediction_reference_gates",
            6,
            prediction_reference_gate_count,
            prediction_reference_gate_count == 6,
            "exactly six prediction-reference gates were not emitted",
        )
        gate("six_distinct_selected_checkpoints",6,len(checkpoint_shas),len(checkpoint_shas)==6,"six selected checkpoint SHA values are not exact and distinct")
        gate("exactly_six_manifests",6,len(manifests),len(manifests)==6,"exactly six replay manifests were not built")
        gate(
            "exactly_six_prediction_references",
            6,
            len(prediction_references),
            len(prediction_references) == 6,
            "exactly six validated prediction references were not built",
        )
        prediction_identity_key_coverage = (
            set(selected_prediction_reference_identity_by_run)
            == STAGE191_RUN_KEYS
        )
        gate(
            "selected_prediction_reference_key_coverage",
            sorted(STAGE191_RUN_KEYS),
            sorted(selected_prediction_reference_identity_by_run),
            prediction_identity_key_coverage,
            "selected prediction reference identity keys are not exact",
        )
    except BaseException as exc:
        exception={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        gate("fail_closed_exception","no exception",exception,False,f"{type(exc).__name__}: {exc}")
    ready=not blockers and len(manifests)==6
    if not ready: manifests=[]; matrix=[]
    decision=READY if ready else BLOCKED
    equivalence=[
        {"gate":"epochs_exact_1_through_20","comparison":"exact","required":True},
        {"gate":"historical_macro_f1_each_epoch","comparison":"exact","required":True},
        {"gate":"normalized_three_label_prediction_counts_each_epoch","comparison":"exact","required":True},
        {"gate":"margin_enabled_eligible_active_each_epoch","comparison":"exact","required":True},
        {"gate":"numeric_margin_values","comparison":"absolute_and_relative_tolerance_1e-12_only_if_exact_not_guaranteed","required":True},
        {"gate":"selected_epoch_and_final_epoch","comparison":"exact","required":True},
        {"gate":"original_selected_checkpoint_predictions","comparison":"exact_row_by_row","required":True,"source":"per-run original_selected_prediction_path and original_selected_prediction_sha256"},
        {"gate":"no_external_ood_bridge","comparison":"exact_false","required":True},
        {"gate":"baseline_sidecar_training_access","comparison":"exact_false","required":True},
    ]
    metric_contract=[{"metric":m,"source":"output[\"logits\"]","rows":720} for m in ("clean_dev_ce","clean_accuracy","clean_macro_f1","support_recall","false_entitlement_total","false_not_entitled_total","polarity_error_total","prediction_counts","gold_counts","confusion_matrix")]
    capsule_contract=[{"field":f,"required":True} for f in ("epoch","trainable_parameters_only","all_named_buffers","names_dtype_shape_metadata","trainable_state_sha256","buffer_state_sha256","training_seed","split_seed","arm","model_construction_provenance")]
    report={"stage":"Stage191-B","decision":decision,"runnable":ready,"blocking_reasons":blockers,"diagnostic_replay_only":True,"replay_execution_authorized":ready,"training_for_model_advancement_authorized":False,"model_advancement_decision":False,"external_data_used":False,"commit_identities":commits,"authorized_training_seeds":list(STAGE191_TRAINING_SEEDS),"frozen_selected_checkpoint_sha256_by_run":SELECTED_CHECKPOINT_SHA256,"selected_prediction_reference_identity_by_run":selected_prediction_reference_identity_by_run,"current_trainer_sha256":trainer_sha,"current_builder_sha256":builder_sha,"current_provenance_helper_sha256":provenance_helper_sha,"current_replay_file_identities":identities,"six_run_matrix":matrix,"argv_difference_audit":argv_audits,"original_selected_prediction_references":prediction_references,"original_selected_epochs":SELECTED,"expected_trajectory_rows_per_run":20,"expected_prediction_rows_per_epoch":720,"expected_state_capsules_per_run":20,"logits_source":"output[\"logits\"]","missing_stage191a_metrics_now_instrumented":["support_recall","false_entitlement_total","polarity_error_total","clean_dev_ce"],"equivalence_gates":equivalence,"baseline_sidecar_non_access_contract":True,"frozen_selected_checkpoint_sha256_gates":[item for item in gates if item["gate"].endswith("_frozen_selected_checkpoint_sha256")],"selected_prediction_reference_gates":[item for item in gates if item["gate"].endswith("_selected_prediction_reference_identity")],"intervention_sidecar_exact_access_contract":{"weight":0.05,"margin_logit":0.0,"sidecar_path":str(authoritative_sidecar),"semantic_sha256":SIDECAR_SHA},"identity_gates":gates,"exception":exception}
    write_json(output/"stage191b_deterministic_replay_manifest_report.json",report)
    md=f"# Stage191-B deterministic replay manifest report\n\nDecision: `{decision}`\n\n- Runnable: {str(ready).lower()}\n- Diagnostic replay only: true\n- Replay execution authorized: {str(ready).lower()}\n- Model advancement authorized: false\n- Blocking reasons: {blockers if blockers else 'none'}\n"
    (output/"stage191b_deterministic_replay_manifest_report.md").write_text(md,encoding="utf-8")
    if ready:
        for manifest in manifests: write_json(output/f"stage191b_{manifest['run']}_replay_manifest.json",manifest)
    matrix_fields=["run","seed","arm","split_seed","selected_epoch","selected_checkpoint_sha256","original_selected_prediction_path","original_selected_prediction_sha256","replay_output_directory","expected_trajectory_rows","expected_prediction_rows_per_epoch","expected_state_capsules","runnable"]
    write_csv(output/"stage191b_six_run_replay_matrix.csv",matrix_fields,matrix)
    write_csv(output/"stage191b_metric_contract.csv",["metric","source","rows"],metric_contract)
    write_csv(output/"stage191b_state_capsule_contract.csv",["field","required"],capsule_contract)
    write_csv(output/"stage191b_equivalence_gate.csv",["gate","comparison","required","source"],equivalence)
    write_csv(output/"stage191b_precommitted_gate.csv",["gate","required","observed","passed","blocking_reason"],gates)
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
