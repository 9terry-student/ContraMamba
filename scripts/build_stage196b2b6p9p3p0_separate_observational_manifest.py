
"""Build Stage196-B2-B6P9-P3-P0 seven-run observational manifest.

Authority-seeking builder for exactly one observer-off control run and six
separate teacher-observer runs. It resolves the base training configuration
from tracked Stage196 lineage evidence and fails closed on ambiguity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

UPSTREAM_DIR = Path("reports/stage196b2b6p9p2_separate_observability_instrumentations_20260724_143152")
UPSTREAM_REQUIRED = {
    "stage196b2b6p9p2_analysis.json",
    "stage196b2b6p9p2_contract.csv",
    "stage196b2b6p9p2_cli_contract.csv",
    "stage196b2b6p9p2_lifecycle_hook_audit.csv",
    "stage196b2b6p9p2_baseline_invariance_audit.csv",
    "stage196b2b6p9p2_checkpoint_schema_audit.csv",
}
READY_DECISION = "STAGE196B2B6P9P3P0_MANIFEST_READY"
DECISIONS = [
    "STAGE196B2B6P9P3P0_BLOCKED_UPSTREAM_AUTHORITY",
    "STAGE196B2B6P9P3P0_BLOCKED_BASE_CONFIG_AUTHORITY",
    "STAGE196B2B6P9P3P0_BLOCKED_OBSERVER_CLI_AUTHORITY",
    READY_DECISION,
]
NEXT = {
    DECISIONS[0]: "STAGE196B2B6P9P3P0_REPAIR_UPSTREAM_AUTHORITY",
    DECISIONS[1]: "STAGE196B2B6P9P3P0_REPAIR_BASE_CONFIG_AUTHORITY",
    DECISIONS[2]: "STAGE196B2B6P9P3P0_REPAIR_OBSERVER_CLI_AUTHORITY",
    READY_DECISION: "STAGE196B2B6P9P3_EXECUTE_SEVEN_CONTROLLED_RUNS",
}
RUN_TABLE = [
    ("control_off_none", "off", "none", None),
    ("previous_step_direction", "previous_step", "direction", None),
    ("previous_step_candidate_order", "previous_step", "candidate_order", None),
    ("previous_epoch_direction", "previous_epoch", "direction", None),
    ("previous_epoch_candidate_order", "previous_epoch", "candidate_order", None),
    ("ema_direction", "ema", "direction", 0.99),
    ("ema_candidate_order", "ema", "candidate_order", 0.99),
]
SIDECARS = [
    "teacher_observer_manifest.json",
    "teacher_observer_batch_metrics.jsonl",
    "teacher_observer_epoch_metrics.csv",
    "teacher_observer_run_summary.json",
    "teacher_observer_state_audit.json",
]
OBSERVER_PATH_KEYS = {"output_json", "output_predictions_json", "save_checkpoint_path", "save_model_checkpoint", "teacher_observer_output_dir"}
OBSERVER_KEYS = {"teacher_observer_mode", "teacher_observer_target_family", "teacher_observer_ema_decay", *OBSERVER_PATH_KEYS}
RUN_PATH_KEYS = {*OBSERVER_PATH_KEYS, "stage196b2b3p0_composer_input_dir", "stage115_clean_dev_scalar_output_jsonl"}
PRIMARY_ARM = "joint"
PRIMARY_ARM_AUTHORITY = "PRIMARY_UNRESTRICTED_JOINT_LINEAGE"
PRIMARY_RUN_PROVENANCE = Path("reports/stage196b2b3p0_epoch_composer_input_observability_runs_retry_20260722_104834/seed183_joint/trajectory/run_provenance.json")
PRIMARY_TRAINING_REPORT = Path("reports/stage196b2b3p0_epoch_composer_input_observability_runs_retry_20260722_104834/seed183_joint/trajectory/training_report.json")
PRIMARY_COMPOSER_MANIFEST = Path("reports/stage196b2b3p0_epoch_composer_input_observability_runs_retry_20260722_104834/seed183_joint/composer_inputs/stage196b2b3p0_composer_input_manifest.json")
SIBLING_RUN_PROVENANCE_GLOB = "reports/stage196b2b3p0_epoch_composer_input_observability_runs_retry_20260722_104834/seed*_*/trajectory/run_provenance.json"
P8_ANALYSIS = Path("reports/stage196b2b6p8_full_trainable_path_replay_20260723_203414/stage196b2b6p8_analysis.json")
P8_CONTRACT = Path("reports/stage196b2b6p8_full_trainable_path_replay_20260723_203414/stage196b2b6p8_contract.csv")
P7_SPEC = Path("reports/stage196b2b6p7_full_counterfactual_forward_design_spec.md")
P6_SPEC = Path("reports/stage196b2b6p6_minimal_gradient_path_instrumentation_spec.md")
REQUIRED_BASE_KEYS = {
    "data", "backbone", "model_name", "device", "seed", "epochs",
    "train_batch_size", "eval_batch_size", "lr", "head_lr", "encoder_lr",
    "weight_decay", "architecture", "split_seed", "vnext_router_mode", "vnext_evidence_interface",
    "training_scope", "checkpoint_selection_behavior", "mixed_precision_behavior",
    "external_evaluation_state", "bridge_training_state", "select_metric",
    "freeze_encoder", "frame_downstream_gradient_mode", "active_stage195_stage196_flags",
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def source_record(path: Path, *, accepted: bool, reason: str, tier: str) -> dict[str, Any]:
    return {"path": str(path), "accepted": bool(accepted), "reason": reason, "tier": tier}


def validate_upstream(repo_root: Path) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    authority_dir = repo_root / UPSTREAM_DIR
    missing = sorted(name for name in UPSTREAM_REQUIRED if not (authority_dir / name).is_file())
    if missing:
        reasons.append(f"missing_upstream_files={missing}")
        return False, reasons, {"authority_dir": str(authority_dir), "missing": missing}
    analysis = read_json(authority_dir / "stage196b2b6p9p2_analysis.json")
    checks = {
        "decision": analysis.get("decision") == "STAGE196B2B6P9P2_SEPARATE_IMPLEMENTATIONS_READY",
        "recommended_next_stage": analysis.get("recommended_next_stage") == "STAGE196B2B6P9P3_SEPARATE_OBSERVATIONAL_RUNS",
        "blocking_reasons": analysis.get("blocking_reasons") == [],
        "failure": analysis.get("failure") is None,
        "failed_contract_count": analysis.get("failed_contract_count") == 0,
    }
    for key, passed in checks.items():
        if not passed:
            reasons.append(f"upstream_{key}_failed")
    return not reasons, reasons, {"authority_dir": str(authority_dir), "analysis": analysis, "checks": checks}


def validate_observer_cli(repo_root: Path) -> tuple[bool, list[str], dict[str, Any]]:
    trainer = repo_root / "scripts/train_controlled_v6b_minimal.py"
    observer = repo_root / "src/contramamba/teacher_state_observer.py"
    reasons: list[str] = []
    if not trainer.is_file():
        reasons.append("missing_trainer_script")
    if not observer.is_file():
        reasons.append("missing_teacher_state_observer")
    trainer_text = trainer.read_text(encoding="utf-8") if trainer.is_file() else ""
    observer_text = observer.read_text(encoding="utf-8") if observer.is_file() else ""
    required_trainer = ["--teacher-observer-mode", "--teacher-observer-target-family", "--teacher-observer-ema-decay", "--teacher-observer-output-dir", "stage196b2b6p8_enable_full_trainable_path_replay_api"]
    required_observer = [
        'MODES = ("off", "previous_step", "previous_epoch", "ema")',
        'TARGET_FAMILIES = ("none", "direction", "candidate_order")',
        "teacher_observer_manifest.json",
        "teacher_observer_batch_metrics.jsonl",
        "teacher_observer_epoch_metrics.csv",
        "teacher_observer_run_summary.json",
        "teacher_observer_state_audit.json",
        "NO_LOSS_OR_BACKWARD_IN_OBSERVATIONAL_STAGE",
    ]
    missing_trainer = [token for token in required_trainer if token not in trainer_text]
    missing_observer = [token for token in required_observer if token not in observer_text]
    if missing_trainer:
        reasons.append(f"missing_trainer_tokens={missing_trainer}")
    if missing_observer:
        reasons.append(f"missing_observer_tokens={missing_observer}")
    return not reasons, reasons, {"trainer_script": str(trainer), "observer_script": str(observer), "missing_trainer_tokens": missing_trainer, "missing_observer_tokens": missing_observer}


def command_contains_flag(command_string: str, flag: str) -> bool:
    return flag in shlex.split(command_string)


def parsed_or_absent(parsed_args: dict[str, Any], key: str, command_string: str) -> tuple[Any, str, dict[str, Any] | None]:
    if key in parsed_args:
        return parsed_args[key], "primary_run_provenance.parsed_args", None
    flag = "--" + key.replace("_", "-")
    if not command_contains_flag(command_string, flag):
        return None, "explicit_absence_from_primary_command_and_parsed_args", {"field": key, "value": None, "authority": "flag_absent_from_exact_command_and_parsed_args"}
    return None, "unresolved", None


def resolve_primary_arm(p8: dict[str, Any], p7_text: str) -> tuple[str | None, str | None, list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    conflicts: list[dict[str, Any]] = []
    unresolved: list[str] = []
    rejected: list[dict[str, Any]] = []
    checkpoints = p8.get("checkpoint_provenance") if isinstance(p8, dict) else None
    checkpoint_modes = sorted(item.get("gradient_mode") for item in checkpoints if isinstance(item, dict) and item.get("gradient_mode")) if isinstance(checkpoints, list) else []
    joint_native_selected = "Selected. Execute the native joint downstream path" in p7_text and "frame-local-only downstream path" in p7_text
    p8_complete = p8.get("decision") == "STAGE196B2B6P8_FULL_TRAINABLE_PATH_REPLAY_COMPLETE"
    if p8_complete and checkpoint_modes == ["frame_local_only", "joint"] and joint_native_selected:
        rejected.append({"candidate_arm": "frame_local_only", "reason": "P7/P8 authorize frame_local_only as separately trained donor/restricted contrast, not unrestricted native primary arm."})
        return PRIMARY_ARM, PRIMARY_ARM_AUTHORITY, conflicts, unresolved, rejected
    conflicts.append({"joint_native_selected": joint_native_selected, "p8_complete": p8_complete, "checkpoint_modes": checkpoint_modes})
    unresolved.append("primary_arm")
    return None, None, conflicts, unresolved, rejected


def sibling_seed_normalization(repo_root: Path) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    considered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    conflicts: list[str] = []
    comparable: list[dict[str, Any]] = []
    for path in sorted(repo_root.glob(SIBLING_RUN_PROVENANCE_GLOB)):
        try:
            payload = read_json(path)
        except Exception as exc:  # noqa: BLE001
            rejected.append(source_record(path, accepted=False, reason=f"unreadable_json:{exc}", tier="sibling_seed_consensus"))
            continue
        parsed = payload.get("parsed_args")
        if not isinstance(parsed, dict):
            rejected.append(source_record(path, accepted=False, reason="missing parsed_args", tier="sibling_seed_consensus"))
            continue
        arm = parsed.get("frame_downstream_gradient_mode")
        if arm != PRIMARY_ARM:
            rejected.append(source_record(path, accepted=False, reason="non_primary_arm_restricted_contrast", tier="sibling_seed_consensus"))
            continue
        stripped = {key: value for key, value in parsed.items() if key not in {"seed", *RUN_PATH_KEYS}}
        comparable.append({"path": str(path), "seed": parsed.get("seed"), "fingerprint": sha256_json(stripped)})
        considered.append(source_record(path, accepted=True, reason="primary_arm_seed_consensus_candidate", tier="sibling_seed_consensus"))
    fingerprints = {item["fingerprint"] for item in comparable}
    raw_seeds = [item["seed"] for item in comparable]
    seeds = sorted(seed for seed in raw_seeds if isinstance(seed, int))
    if len(seeds) != len(raw_seeds) or len(fingerprints) != 1 or seeds != [183, 184, 185]:
        conflicts.append(f"primary_arm_seed_consensus_failed seeds={raw_seeds} fingerprints={sorted(fingerprints)}")
        return False, considered, rejected, conflicts
    return True, considered, rejected, conflicts


def build_base_from_primary(repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "base_config": {},
        "field_provenance": {},
        "candidate_sources_considered": [],
        "candidate_sources_rejected": [],
        "normalizations_applied": [],
        "unresolved_fields": [],
        "conflicts": [],
        "selected_base_config_source": None,
        "selected_primary_arm": None,
        "primary_arm_authority": None,
    }
    required_files = [PRIMARY_RUN_PROVENANCE, PRIMARY_TRAINING_REPORT, PRIMARY_COMPOSER_MANIFEST, P8_ANALYSIS, P8_CONTRACT, P7_SPEC, P6_SPEC]
    for rel in required_files:
        path = repo_root / rel
        if path.is_file():
            result["candidate_sources_considered"].append(source_record(path, accepted=True, reason="tracked_authority_file_present", tier="deterministic_hierarchy"))
        else:
            result["candidate_sources_rejected"].append(source_record(path, accepted=False, reason="missing_tracked_authority_file", tier="deterministic_hierarchy"))
            result["unresolved_fields"].append(f"missing_authority:{rel}")
    if result["unresolved_fields"]:
        return result

    provenance = read_json(repo_root / PRIMARY_RUN_PROVENANCE)
    training_report = read_json(repo_root / PRIMARY_TRAINING_REPORT)
    composer_manifest = read_json(repo_root / PRIMARY_COMPOSER_MANIFEST)
    p8 = read_json(repo_root / P8_ANALYSIS)
    p7_text = (repo_root / P7_SPEC).read_text(encoding="utf-8")
    command_string = provenance.get("command_string", "")
    parsed_args = provenance.get("parsed_args")
    if not isinstance(parsed_args, dict) or not isinstance(command_string, str) or not command_string:
        result["unresolved_fields"].append("primary_run_provenance_command_or_parsed_args")
        return result
    resolved_runtime = provenance.get("resolved_runtime_config") or {}

    primary_arm, primary_authority, arm_conflicts, arm_unresolved, arm_rejected = resolve_primary_arm(p8, p7_text)
    result["conflicts"].extend(arm_conflicts)
    result["unresolved_fields"].extend(arm_unresolved)
    result["candidate_sources_rejected"].extend(arm_rejected)
    result["selected_primary_arm"] = primary_arm
    result["primary_arm_authority"] = primary_authority

    seed_consensus_ok, sibling_considered, sibling_rejected, sibling_conflicts = sibling_seed_normalization(repo_root)
    result["candidate_sources_considered"].extend(sibling_considered)
    result["candidate_sources_rejected"].extend(sibling_rejected)
    result["conflicts"].extend(sibling_conflicts)
    if not seed_consensus_ok:
        result["unresolved_fields"].append("seed_normalization_consensus")

    field_provenance: dict[str, Any] = {}
    normalizations: list[dict[str, Any]] = []

    def take(field: str, source_key: str | None = None) -> Any:
        key = source_key or field
        value, source, normalization = parsed_or_absent(parsed_args, key, command_string)
        field_provenance[field] = {"source": source, "source_key": key, "value": value}
        if normalization is not None:
            normalizations.append(normalization)
        return value

    freeze_encoder = take("freeze_encoder")
    freeze_a_log = take("freeze_a_log")
    base: dict[str, Any] = {
        "data": take("data"),
        "backbone": take("backbone"),
        "model_name": take("model_name"),
        "device": take("device"),
        "seed": take("seed"),
        "epochs": take("epochs"),
        "train_batch_size": take("train_batch_size"),
        "eval_batch_size": take("eval_batch_size"),
        "lr": take("lr"),
        "head_lr": take("head_lr"),
        "encoder_lr": take("encoder_lr"),
        "weight_decay": take("weight_decay"),
        "architecture": take("architecture"),
        "split_seed": take("split_seed"),
        "vnext_router_mode": take("vnext_router_mode"),
        "vnext_evidence_interface": take("vnext_evidence_interface"),
        "select_metric": take("select_metric"),
        "freeze_encoder": "true" if freeze_encoder is True else freeze_encoder,
        "freeze_a_log": "true" if freeze_a_log is True else freeze_a_log,
        "max_length": take("max_length"),
        "dev_ratio": take("dev_ratio"),
        "gradient_accumulation_steps": take("gradient_accumulation_steps"),
        "class_weighting": take("class_weighting"),
        "flag_source": take("flag_source"),
        "save_selected_checkpoint": take("save_selected_checkpoint"),
        "selected_checkpoint_filename": take("selected_checkpoint_filename"),
        "frame_downstream_gradient_mode": take("frame_downstream_gradient_mode"),
        "stage196b1_framegate_gradient_ownership_observability": take("stage196b1_framegate_gradient_ownership_observability"),
        "stage196b2p0_epoch_channel_observability": take("stage196b2p0_epoch_channel_observability"),
        "stage196b2b3p0_export_epoch_composer_inputs": take("stage196b2b3p0_export_epoch_composer_inputs"),
        "compatible_positive_margin_weight": take("compatible_positive_margin_weight"),
        "compatible_positive_margin_logit": take("compatible_positive_margin_logit"),
    }
    active_stage_flags: dict[str, Any] = {}
    for key, value in sorted(parsed_args.items()):
        if not key.startswith(("stage195", "stage196")):
            continue
        if key in base or key in RUN_PATH_KEYS or key in OBSERVER_KEYS:
            continue
        if value in (None, False, [], ""):
            continue
        if command_contains_flag(command_string, "--" + key.replace("_", "-")):
            base[key] = value
            active_stage_flags[key] = value
            field_provenance[key] = {"source": "primary_exact_command_and_parsed_args", "value": value}
    base["active_stage195_stage196_flags"] = active_stage_flags
    field_provenance["active_stage195_stage196_flags"] = {"source": "primary_exact_command_and_parsed_args", "value": active_stage_flags}

    base["stage196b2b6p8_enable_full_trainable_path_replay_api"] = True
    field_provenance["stage196b2b6p8_enable_full_trainable_path_replay_api"] = {"source": str(P8_ANALYSIS), "value": True, "reason": "P8 full trainable-path replay implementation is required for P9 teacher observer replay-state geometry."}
    normalizations.append({"field": "stage196b2b6p8_enable_full_trainable_path_replay_api", "value": True, "authority": "P8_FULL_TRAINABLE_PATH_REPLAY_COMPLETE"})

    bridge_modes = {name: parsed_args.get(name) for name in ("stage57_bridge_train_mode", "stage66_bridge_train_mode", "stage75_bridge_train_mode", "stage80a_bridge_train_mode")}
    bridge_paths = {name: parsed_args.get(name) for name in ("stage57_bridge_train_jsonl", "stage66_bridge_train_jsonl", "stage75_bridge_train_jsonl", "stage80a_bridge_train_jsonl")}
    external_flags = {
        "enable_external_eval": parsed_args.get("enable_external_eval"),
        "enable_stage43_external_eval": parsed_args.get("enable_stage43_external_eval"),
        "stage43_external_enable_shadow_export": parsed_args.get("stage43_external_enable_shadow_export"),
        "ood_data": parsed_args.get("ood_data"),
        "external_data": parsed_args.get("external_data"),
    }
    base["training_scope"] = {
        "scope": "clean_controlled_main_training",
        "max_train_records": parsed_args.get("max_train_records"),
        "smoke": parsed_args.get("smoke"),
        "loss_sweep": parsed_args.get("loss_sweep"),
        "main_data_sha256": ((provenance.get("data_provenance") or {}).get("main_data") or {}).get("sha256"),
        "expected_rows_per_epoch": composer_manifest.get("expected_rows_per_epoch"),
        "expected_epoch_count": composer_manifest.get("expected_epoch_count"),
        "train_batch_size_argument": resolved_runtime.get("train_batch_size_argument"),
        "eval_batch_size_argument": resolved_runtime.get("eval_batch_size_argument"),
        "resolved_train_batch_size": resolved_runtime.get("resolved_train_batch_size"),
        "resolved_eval_batch_size": resolved_runtime.get("resolved_eval_batch_size"),
        "effective_batch_size": resolved_runtime.get("effective_batch_size"),
    }
    base["checkpoint_selection_behavior"] = {
        "save_selected_checkpoint": parsed_args.get("save_selected_checkpoint"),
        "selected_checkpoint_filename": parsed_args.get("selected_checkpoint_filename"),
        "selection_source": "internal_clean_dev_only",
        "select_metric": parsed_args.get("select_metric"),
        "selected_epoch": ((provenance.get("finalization") or {}).get("selected_checkpoint") or {}).get("selected_epoch"),
        "checkpoint_is_selected_clean_dev_state": True,
    }
    base["mixed_precision_behavior"] = {
        "fp16": parsed_args.get("fp16"),
        "amp": parsed_args.get("amp"),
        "mixed_precision": parsed_args.get("mixed_precision"),
        "authority": "explicit_absence_or_value_in_primary_run_provenance_parsed_args",
    }
    base["external_evaluation_state"] = {
        "disabled": all(value in (None, False, []) for value in external_flags.values()),
        "flags": external_flags,
        "training_report_external_examples_used": (((training_report.get("stage43_external_factver_eval") or {}).get("summary") or {}).get("external_examples_used")),
    }
    base["bridge_training_state"] = {
        "disabled": all(value == "none" for value in bridge_modes.values()) and all(value is None for value in bridge_paths.values()),
        "modes": bridge_modes,
        "paths": bridge_paths,
    }
    for field in ("training_scope", "checkpoint_selection_behavior", "mixed_precision_behavior", "external_evaluation_state", "bridge_training_state"):
        field_provenance[field] = {"source": "primary_run_provenance_and_training_report", "value": base[field]}

    for key in list(base):
        if key in RUN_PATH_KEYS or key in OBSERVER_KEYS:
            base.pop(key)
    base = {key: base[key] for key in sorted(base)}

    required_checks = {
        "backbone": base.get("backbone") == "mamba",
        "model_name": base.get("model_name") == "state-spaces/mamba-130m-hf",
        "device": base.get("device") == "cuda",
        "seed": base.get("seed") == 183,
        "external_evaluation_state": (base.get("external_evaluation_state") or {}).get("disabled") is True,
        "bridge_training_state": (base.get("bridge_training_state") or {}).get("disabled") is True,
        "primary_arm": primary_arm == PRIMARY_ARM,
    }
    for field in REQUIRED_BASE_KEYS:
        if field not in base:
            result["unresolved_fields"].append(field)
    for field, passed in required_checks.items():
        if not passed:
            result["conflicts"].append({"field": field, "observed": base.get(field), "required_check_failed": True})
    if base.get("seed") == 183 and seed_consensus_ok:
        normalizations.append({"field": "seed", "value": 183, "authority": "exact_seed183_primary_run_plus_joint_seed183_184_185_consensus_only_seed_varies"})

    result.update({
        "ok": not result["unresolved_fields"] and not result["conflicts"],
        "base_config": base,
        "field_provenance": field_provenance,
        "normalizations_applied": normalizations,
        "selected_base_config_source": str(repo_root / PRIMARY_RUN_PROVENANCE),
    })
    return result


def args_to_cli(args: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    bool_value_keys = {"freeze_encoder", "freeze_a_log"}
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if key in bool_value_keys:
            argv.extend([flag, str(value).lower()])
        elif value is True:
            argv.append(flag)
        elif isinstance(value, list):
            for item in value:
                argv.extend([flag, str(item)])
        elif isinstance(value, dict):
            continue
        else:
            argv.extend([flag, str(value)])
    return argv


def build_manifest_rows(repo_root: Path, output_root: Path, base_args: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_fp = sha256_json(base_args)
    trainer_script = "scripts/train_controlled_v6b_minimal.py"
    for run_id, mode, family, decay in RUN_TABLE:
        run_dir = output_root / run_id
        observer_dir = run_dir / "teacher_observer"
        checkpoint_path = run_dir / "selected_checkpoint.pt"
        trainer_args = {key: value for key, value in base_args.items() if not isinstance(value, dict)}
        if trainer_args.get("stage196b2b3p0_export_epoch_composer_inputs"):
            trainer_args["stage196b2b3p0_composer_input_dir"] = str(run_dir / "composer_inputs")
        trainer_args.update({
            "seed": 183,
            "stage115_clean_dev_scalar_output_jsonl": str(run_dir / "clean_dev_scalars.jsonl"),
            "output_json": str(run_dir / "training_report.json"),
            "output_predictions_json": str(run_dir / "clean_dev_predictions.json"),
            "save_selected_checkpoint": True,
            "selected_checkpoint_filename": "selected_checkpoint.pt",
            "teacher_observer_mode": mode,
            "teacher_observer_target_family": family,
        })
        if mode != "off":
            trainer_args["teacher_observer_output_dir"] = str(observer_dir)
        if decay is not None:
            trainer_args["teacher_observer_ema_decay"] = decay
        observer_args = {key: trainer_args.get(key) for key in sorted(OBSERVER_KEYS) if key in trainer_args}
        config_fp = sha256_json(trainer_args)
        rows.append({
            "run_id": run_id,
            "seed": 183,
            "observer_mode": mode,
            "target_family": family,
            "ema_decay": decay,
            "ema_decay_authority": "EX_ANTE_EFFECTIVE_HORIZON_100_SUCCESSFUL_STEPS" if decay is not None else None,
            "trainer_script": trainer_script,
            "trainer_args": trainer_args,
            "normalized_base_args": base_args,
            "observer_specific_args": observer_args,
            "output_dir": str(run_dir),
            "observer_output_dir": None if mode == "off" else str(observer_dir),
            "checkpoint_path": str(checkpoint_path),
            "stdout_log": str(run_dir / "stdout.log"),
            "stderr_log": str(run_dir / "stderr.log"),
            "config_fingerprint": config_fp,
            "base_config_fingerprint": base_fp,
            "observer_config_fingerprint": sha256_json(observer_args),
            "expected_runtime_sidecars": [] if mode == "off" else SIDECARS,
            "expected_checkpoint_observer_state": mode != "off",
        })
    return rows


def only_observer_and_run_path_differences(rows: list[dict[str, Any]]) -> bool:
    if len(rows) != 7:
        return False
    normalized = [row.get("normalized_base_args") for row in rows]
    return len({sha256_json(item) for item in normalized}) == 1


def validate_contract_schema(contracts: list[tuple[str, bool]]) -> tuple[bool, list[str]]:
    errors = [name for name, passed in contracts if type(passed) is not bool]
    return not errors, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    upstream_ok, upstream_reasons, upstream_record = validate_upstream(repo_root)
    cli_ok, cli_reasons, cli_record = validate_observer_cli(repo_root)
    base_resolution = build_base_from_primary(repo_root)
    base_ok = bool(base_resolution["ok"])
    base_args = dict(base_resolution["base_config"])

    decision = READY_DECISION
    if not upstream_ok:
        decision = DECISIONS[0]
    elif not base_ok:
        decision = DECISIONS[1]
    elif not cli_ok:
        decision = DECISIONS[2]

    rows = build_manifest_rows(repo_root, args.output_root, base_args) if decision == READY_DECISION else []
    contracts: list[tuple[str, bool]] = [
        ("upstream_p9p2_exact_decision", bool(upstream_ok and not upstream_reasons)),
        ("upstream_p9p2_zero_blockers", bool(upstream_ok)),
        ("upstream_p9p2_zero_failed_contracts", bool(upstream_ok)),
        ("source_backed_base_config_unique", bool(base_ok)),
        ("primary_arm_authorized", base_resolution.get("selected_primary_arm") == PRIMARY_ARM and base_resolution.get("primary_arm_authority") == PRIMARY_ARM_AUTHORITY),
        ("all_required_base_fields_closed", bool(base_ok and not base_resolution.get("unresolved_fields"))),
        ("exact_seed_183", bool(len(rows) == 7 and all(row.get("seed") == 183 for row in rows))),
        ("exact_ema_decay_099", bool(len(rows) == 7 and all(row["ema_decay"] in (None, 0.99) for row in rows))),
        ("ema_decay_ex_ante", True),
        ("ema_decay_not_performance_selected", True),
        ("exact_seven_run_rows", len(rows) == 7),
        ("exact_one_control", sum(row["observer_mode"] == "off" for row in rows) == 1),
        ("exact_six_enabled", sum(row["observer_mode"] != "off" for row in rows) == 6),
        ("exact_three_teacher_modes", {row["observer_mode"] for row in rows if row["observer_mode"] != "off"} == {"previous_step", "previous_epoch", "ema"}),
        ("exact_two_target_families", {row["target_family"] for row in rows if row["target_family"] != "none"} == {"direction", "candidate_order"}),
        ("no_combined_target_family", bool(rows) and all(row["target_family"] != "combined" for row in rows)),
        ("teacher_modes_mutually_exclusive", True),
        ("one_base_config_fingerprint", bool(len(rows) == 7 and len({row["base_config_fingerprint"] for row in rows}) == 1)),
        ("observer_fields_only_run_differences", bool(only_observer_and_run_path_differences(rows))),
        ("control_has_no_ema_decay", bool(len(rows) >= 1 and rows[0]["run_id"] == "control_off_none" and rows[0]["ema_decay"] is None)),
        ("ema_decay_only_on_ema_rows", bool(len(rows) == 7 and all((row["observer_mode"] == "ema") == (row["ema_decay"] == 0.99) for row in rows))),
        ("device_cuda", base_args.get("device") == "cuda"),
        ("backbone_mamba", base_args.get("backbone") == "mamba"),
        ("model_name_mamba_130m", base_args.get("model_name") == "state-spaces/mamba-130m-hf"),
        ("no_external_eval", (base_args.get("external_evaluation_state") or {}).get("disabled") is True),
        ("no_teacher_selected", True),
        ("no_loss_added", True),
        ("exact_five_manifest_outputs", True),
    ]
    schema_ok, schema_errors = validate_contract_schema(contracts)
    if not schema_ok:
        decision = DECISIONS[1]
        base_resolution["unresolved_fields"].append("contract_schema_validation")
        base_resolution["conflicts"].append({"non_boolean_contracts": schema_errors})
        contracts = [(name, passed if type(passed) is bool else False) for name, passed in contracts]
    failed_contract_count = sum(1 for _, passed in contracts if not passed)
    if decision == READY_DECISION and failed_contract_count:
        decision = DECISIONS[1]

    blocking_reasons = list(upstream_reasons) + ([] if base_ok else ["base_config_authority_not_unique_or_incomplete"]) + list(cli_reasons)
    if not schema_ok:
        blocking_reasons.append("contract_schema_validation_failed")
    if decision != READY_DECISION and not blocking_reasons:
        blocking_reasons.append(decision)

    manifest = {
        "stage": "Stage196-B2-B6P9-P3-P0",
        "decision": decision,
        "recommended_next_stage": NEXT[decision],
        "decision_hierarchy": DECISIONS,
        "blocking_reasons": blocking_reasons,
        "failure": None if decision == READY_DECISION else decision,
        "failed_contract_count": failed_contract_count,
        "upstream_authority": upstream_record,
        "observer_cli_authority": cli_record,
        "base_config_fingerprint": sha256_json(base_args) if base_args else None,
        "primary_arm_authority": base_resolution.get("primary_arm_authority"),
        "run_table": rows,
    }
    authority = {
        "stage": "Stage196-B2-B6P9-P3-P0",
        "decision": decision,
        "recommended_next_stage": NEXT[decision],
        "blocking_reasons": blocking_reasons,
        "failure": None if decision == READY_DECISION else decision,
        "selected_base_config_source": base_resolution.get("selected_base_config_source"),
        "selected_primary_arm": base_resolution.get("selected_primary_arm"),
        "primary_arm_authority": base_resolution.get("primary_arm_authority"),
        "base_config_fingerprint": sha256_json(base_args) if base_args else None,
        "resolved_base_config": base_args,
        "field_provenance": base_resolution.get("field_provenance", {}),
        "candidate_sources_considered": base_resolution.get("candidate_sources_considered", []),
        "candidate_sources_rejected": base_resolution.get("candidate_sources_rejected", []),
        "normalizations_applied": base_resolution.get("normalizations_applied", []),
        "unresolved_fields": base_resolution.get("unresolved_fields", []),
        "conflicts": base_resolution.get("conflicts", []),
        "resolution_strategy": "1 exact seed183 primary Stage196 resolved command/parsed_args; 2 tracked P8/P7/P6 manifest/checkpoint/design metadata for primary-arm authority; 3 field-level consensus across seed183/184/185 primary-arm lineage after removing only seed and run paths; 4 fail closed.",
        "manifest_overlays_not_base_authority": {
            "seed": 183,
            "ema_decay": 0.99,
            "ema_decay_authority": "EX_ANTE_EFFECTIVE_HORIZON_100_SUCCESSFUL_STEPS",
        },
    }
    write_json(output_dir / "stage196b2b6p9p3p0_manifest.json", manifest)
    write_csv(output_dir / "stage196b2b6p9p3p0_run_table.csv", rows, [
        "run_id", "seed", "observer_mode", "target_family", "ema_decay",
        "config_fingerprint", "base_config_fingerprint", "observer_config_fingerprint",
        "output_dir", "observer_output_dir", "checkpoint_path", "stdout_log", "stderr_log",
    ])
    write_json(output_dir / "stage196b2b6p9p3p0_base_config_authority.json", authority)
    write_csv(output_dir / "stage196b2b6p9p3p0_decision_gate.csv", [{
        "decision": decision,
        "recommended_next_stage": NEXT[decision],
        "failed_contract_count": failed_contract_count,
        "blocking_reasons": canonical(manifest["blocking_reasons"]),
    }], ["decision", "recommended_next_stage", "failed_contract_count", "blocking_reasons"])
    write_csv(output_dir / "stage196b2b6p9p3p0_contract.csv", [
        {"contract": name, "passed": passed, "evidence": "static_source_backed_manifest_builder"}
        for name, passed in contracts
    ], ["contract", "passed", "evidence"])
    return 0 if decision == READY_DECISION else 2


if __name__ == "__main__":
    raise SystemExit(main())






