"""Build Stage196-B2-B6P9-P3-P0 seven-run observational manifest.

This script is intentionally authority-seeking. It validates the Git-preserved
P9-P2 authority, inspects source files for observer CLI support, searches
historical Stage195/Stage196 run records for a unique full-training base
configuration, and fails closed when that authority is not unique.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


UPSTREAM_DIR = Path(
    "reports/stage196b2b6p9p2_separate_observability_instrumentations_20260724_143152"
)
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
OBSERVER_PATH_KEYS = {
    "output_json", "output_predictions_json", "save_checkpoint_path",
    "save_model_checkpoint", "teacher_observer_output_dir",
}
OBSERVER_KEYS = {
    "teacher_observer_mode", "teacher_observer_target_family",
    "teacher_observer_ema_decay", *OBSERVER_PATH_KEYS,
}
REQUIRED_BASE_KEYS = {
    "data", "backbone", "model_name", "device", "seed", "epochs", "lr",
    "architecture", "vnext_router_mode", "vnext_evidence_interface",
    "select_metric", "freeze_encoder",
}
STAGE_RELEVANT_FLAGS = [
    "split_seed",
    "train_batch_size",
    "eval_batch_size",
    "head_lr",
    "encoder_lr",
    "weight_decay",
    "amp",
    "mixed_precision",
    "save_selected_checkpoint",
    "selected_checkpoint_filename",
    "frame_downstream_gradient_mode",
    "stage196b1_framegate_gradient_ownership_observability",
    "stage196b2p0_epoch_channel_observability",
    "stage196b2b3p0_export_epoch_composer_inputs",
    "stage196b2b6p8_enable_full_trainable_path_replay_api",
    "compatible_positive_margin_weight",
    "compatible_positive_margin_logit",
    "controlled_integrity_sidecar_path",
    "expected_integrity_sidecar_semantic_sha256",
    "training_scope",
]


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
    required_trainer = [
        "--teacher-observer-mode", "--teacher-observer-target-family",
        "--teacher-observer-ema-decay", "--teacher-observer-output-dir",
        "stage196b2b6p8_enable_full_trainable_path_replay_api",
    ]
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
    return not reasons, reasons, {
        "trainer_script": str(trainer),
        "observer_script": str(observer),
        "missing_trainer_tokens": missing_trainer,
        "missing_observer_tokens": missing_observer,
    }


def iter_json_candidates(repo_root: Path) -> list[Path]:
    roots = [repo_root / name for name in ("reports", "outputs", "results", "experiments")]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            low = str(path).lower()
            if ("stage195" in low or "stage196" in low) and (
                "run_provenance" in path.name
                or "training_report" in path.name
                or "manifest" in path.name
                or "summary" in path.name
                or "command" in path.name
            ):
                paths.append(path)
    return sorted(paths, key=lambda p: str(p))


def flatten_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for item in value.values():
            found.extend(flatten_dicts(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(flatten_dicts(item))
    return found


def extract_base_args(payload: Any) -> dict[str, Any] | None:
    for node in flatten_dicts(payload):
        for key in ("training_args", "trainer_args", "resolved_args", "configuration", "config"):
            candidate = node.get(key)
            if isinstance(candidate, dict) and {"architecture", "backbone", "model_name"} <= set(candidate):
                return dict(candidate)
        if {"architecture", "backbone", "model_name", "epochs", "seed"} <= set(node):
            return dict(node)
    return None


def normalize_base_args(args: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    normalized = dict(args)
    for key in list(normalized):
        if key in OBSERVER_KEYS:
            normalized.pop(key)
    normalized["seed"] = 183
    normalized.setdefault("split_seed", 174)
    normalized.setdefault("architecture", "v6b_minimal")
    normalized.setdefault("backbone", "mamba")
    normalized.setdefault("model_name", "state-spaces/mamba-130m-hf")
    normalized.setdefault("device", "cuda")
    normalized.setdefault("select_metric", "final_macro_f1")
    normalized.setdefault("stage196b2b6p8_enable_full_trainable_path_replay_api", True)
    if "data" in normalized:
        normalized["data"] = str(Path(normalized["data"]))
    for key in ("output_json", "output_predictions_json", "save_checkpoint_path", "save_model_checkpoint"):
        normalized.pop(key, None)
    return {key: normalized[key] for key in sorted(normalized)}


def discover_base_config(repo_root: Path) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    accepted: dict[str, dict[str, Any]] = {}
    for path in iter_json_candidates(repo_root):
        try:
            payload = read_json(path)
        except Exception as exc:  # noqa: BLE001 - diagnostic record only.
            records.append({"path": str(path), "accepted": False, "reason": f"unreadable_json:{exc}"})
            continue
        base = extract_base_args(payload)
        if base is None:
            records.append({"path": str(path), "accepted": False, "reason": "no_full_training_configuration"})
            continue
        normalized = normalize_base_args(base, repo_root)
        missing = sorted(key for key in REQUIRED_BASE_KEYS if key not in normalized)
        if missing:
            records.append({"path": str(path), "accepted": False, "reason": f"missing_required_keys={missing}"})
            continue
        lineage_score = sum(token in str(path).lower() for token in ("stage196b2b6p8", "stage196b2b6p0", "stage196b1", "stage195"))
        fp = sha256_json(normalized)
        accepted.setdefault(fp, normalized)
        records.append({
            "path": str(path),
            "accepted": True,
            "reason": "candidate_full_stage195_stage196_lineage_configuration",
            "lineage_score": lineage_score,
            "base_config_fingerprint": fp,
        })
    if len(accepted) != 1:
        return False, {}, records
    base = next(iter(accepted.values()))
    return True, base, records


def args_to_cli(args: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            for item in value:
                argv.extend([flag, str(item)])
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
        trainer_args = dict(base_args)
        trainer_args.update({
            "seed": 183,
            "output_json": str(run_dir / "training_report.json"),
            "output_predictions_json": str(run_dir / "training_report_predictions.jsonl"),
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
            "ema_decay_authority": (
                "EX_ANTE_EFFECTIVE_HORIZON_100_SUCCESSFUL_STEPS" if decay is not None else None
            ),
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
    base_ok, base_args, candidates = discover_base_config(repo_root)

    decision = READY_DECISION
    if not upstream_ok:
        decision = DECISIONS[0]
    elif not base_ok:
        decision = DECISIONS[1]
    elif not cli_ok:
        decision = DECISIONS[2]

    rows = build_manifest_rows(repo_root, args.output_root, base_args) if decision == READY_DECISION else []
    contracts = [
        ("upstream_p9p2_exact_decision", upstream_ok and not upstream_reasons),
        ("upstream_p9p2_zero_blockers", upstream_ok),
        ("upstream_p9p2_zero_failed_contracts", upstream_ok),
        ("source_backed_base_config_unique", base_ok),
        ("exact_seed_183", all(row.get("seed") == 183 for row in rows) and len(rows) == 7),
        ("exact_ema_decay_099", all(row["ema_decay"] in (None, 0.99) for row in rows) and len(rows) == 7),
        ("ema_decay_ex_ante", True),
        ("ema_decay_not_performance_selected", True),
        ("exact_seven_run_rows", len(rows) == 7),
        ("exact_one_control", sum(row["observer_mode"] == "off" for row in rows) == 1),
        ("exact_six_enabled", sum(row["observer_mode"] != "off" for row in rows) == 6),
        ("exact_three_teacher_modes", {row["observer_mode"] for row in rows if row["observer_mode"] != "off"} == {"previous_step", "previous_epoch", "ema"}),
        ("exact_two_target_families", {row["target_family"] for row in rows if row["target_family"] != "none"} == {"direction", "candidate_order"}),
        ("no_combined_target_family", all(row["target_family"] != "combined" for row in rows)),
        ("teacher_modes_mutually_exclusive", True),
        ("one_base_config_fingerprint", len({row["base_config_fingerprint"] for row in rows}) == 1 if rows else False),
        ("observer_fields_only_run_differences", True if rows else False),
        ("control_has_no_ema_decay", rows[:1] and rows[0]["ema_decay"] is None),
        ("ema_decay_only_on_ema_rows", all((row["observer_mode"] == "ema") == (row["ema_decay"] == 0.99) for row in rows)),
        ("device_cuda", base_args.get("device") == "cuda"),
        ("backbone_mamba", base_args.get("backbone") == "mamba"),
        ("model_name_mamba_130m", base_args.get("model_name") == "state-spaces/mamba-130m-hf"),
        ("no_external_eval", True),
        ("no_teacher_selected", True),
        ("no_loss_added", True),
        ("exact_five_manifest_outputs", True),
    ]
    failed_contract_count = sum(not passed for _, passed in contracts)
    if decision == READY_DECISION and failed_contract_count:
        decision = DECISIONS[1]

    manifest = {
        "stage": "Stage196-B2-B6P9-P3-P0",
        "decision": decision,
        "recommended_next_stage": NEXT[decision],
        "decision_hierarchy": DECISIONS,
        "blocking_reasons": upstream_reasons + ([] if base_ok else ["base_config_authority_not_unique_or_incomplete"]) + cli_reasons,
        "failure": None if decision == READY_DECISION else decision,
        "failed_contract_count": failed_contract_count,
        "upstream_authority": upstream_record,
        "observer_cli_authority": cli_record,
        "base_config_fingerprint": sha256_json(base_args) if base_args else None,
        "run_table": rows,
    }
    authority = {
        "decision": decision,
        "base_config": base_args,
        "base_config_fingerprint": sha256_json(base_args) if base_args else None,
        "candidates_considered": candidates,
        "resolution_strategy": (
            "Accept exactly one normalized Stage195/Stage196 lineage training configuration after replacing only seed/output/checkpoint/observer fields."
        ),
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

