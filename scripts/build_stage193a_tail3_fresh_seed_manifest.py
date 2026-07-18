#!/usr/bin/env python3
"""Build the fail-closed Stage193-A fresh-seed tail3 replication manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

TRAINER_BLOB_COMMIT = "e83d8af756fa84b7a91c14e0910ae388b07b5f02"
STAGE192_COMMIT = "a768d848256f88a7a1a15cc02a058f4d7d0a35f7"
STAGE191_COMMIT = "0872e66ccb05ae8a166f5cabf4e084272dc49500"
SIDECAR_SHA = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
STAGE191_BASENAME = "stage191b_deterministic_replay_manifest_20260717_153524"
STAGE185_BASENAME = "stage185a_controlled_train_integrity_sidecar_20260715_141914"
READY = "STAGE193A_TAIL3_FRESH_SEED_MANIFEST_READY"
BLOCKED = "STAGE193A_TAIL3_FRESH_SEED_MANIFEST_BLOCKED"
STAGE191_READY = "STAGE191B_DETERMINISTIC_REPLAY_MANIFEST_READY"
STAGE192_CLOSED = "STAGE192A_NO_TRAJECTORY_STABLE_SELECTOR"
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
SEEDS = (177, 178, 179)
SOURCE_SEEDS = (174, 175, 176)
ARMS = ("baseline", "intervention")
RUNS = tuple(f"seed{seed}_{arm}" for seed in SEEDS for arm in ARMS)
SOURCE_RUNS = tuple(f"seed{seed}_{arm}" for seed in SOURCE_SEEDS for arm in ARMS)
OUTPUT_OPTIONS = {
    "--output-json": "training_report.json",
    "--output-predictions-json": "clean_dev_predictions.json",
    "--stage115-clean-dev-scalar-output-jsonl": "clean_dev_scalars.jsonl",
}
STAGE191_FLAGS = (
    "--stage191-trajectory-replay-observability",
    "--stage191-save-trajectory-state-capsules",
)
STAGE193_FLAG = "--stage193-tail3-fresh-seed-observability"
FORBIDDEN_OPTIONS = (
    "ood_data", "output_ood_json", "output_ood_predictions_json", "external_data",
    "external_output_dir", "external_eval_jsonl", "external_eval_name",
    "stage43_external_factver_jsonl", "stage57_bridge_train_jsonl",
    "stage66_bridge_train_jsonl", "stage75_bridge_train_jsonl",
    "stage80a_bridge_train_jsonl", "pair_contrastive_frame_data",
    "temporal_diagnostic_data", "v7_temporal_safety_data",
    "v7_temporal_mismatch_multihead_data", "v7_temporal_preservation_data",
    "v7_coverage_entailment_data",
)
FORBIDDEN_FLAGS = (
    "smoke", "loss_sweep", "enable_external_eval", "enable_stage43_external_eval",
    "stage43_external_enable_shadow_export", "use_pair_contrastive_frame_loss",
    "use_temporal_diagnostic_loss", "use_td_constrained_selection",
    "use_temporal_residual_adapter", "use_temporal_adapter_loss",
    "use_temporal_adapter_final_penalty", "use_temporal_channel",
    "use_temporal_channel_loss", "use_temporal_channel_gated_penalty",
    "v7_use_temporal_safety_head", "v7_use_temporal_safety_loss",
    "v7_use_temporal_mismatch_multihead", "v7_use_temporal_mismatch_multihead_loss",
    "v7_use_temporal_preservation_head", "v7_use_temporal_preservation_loss",
    "v7_use_temporal_preservation_aware_cap", "v7_use_coverage_entailment_head",
    "v7_use_coverage_entailment_loss",
)
BRIDGE_MODES = (
    "stage57_bridge_train_mode", "stage66_bridge_train_mode",
    "stage75_bridge_train_mode", "stage80a_bridge_train_mode",
)
AUX_MODES = ("v7_temporal_safety_cap_mode", "v7_temporal_mismatch_multihead_cap_mode")
STAGE192_OUTPUTS = {
    "stage192a_trajectory_stable_selection_report.json",
    "stage192a_trajectory_stable_selection_report.md",
    "stage192a_stage191d_closure_gate.csv", "stage192a_selector_definition.csv",
    "stage192a_selector_choice_by_seed.csv", "stage192a_selected_arm_metrics.csv",
    "stage192a_selector_aggregate_metrics.csv", "stage192a_pair_delta_by_selector.csv",
    "stage192a_perturbation_grid.csv", "stage192a_perturbation_summary.csv",
    "stage192a_selected_pair_transition_summary.csv",
    "stage192a_selected_pair_transition_by_gold.csv",
    "stage192a_temporal_ensemble_comparator.csv", "stage192a_precommitted_gate.csv",
}
OUTPUTS = {
    "json": "stage193a_tail3_fresh_seed_manifest_report.json",
    "md": "stage193a_tail3_fresh_seed_manifest_report.md",
    "jsonl": "stage193a_run_manifest.jsonl",
    "matrix": "stage193a_run_command_matrix.csv",
    "source": "stage193a_source_and_template_gate.csv",
    "gates": "stage193a_precommitted_gate.csv",
}
MATRIX_HEADER = ["run", "training_seed", "split_seed", "arm", "planned_run_directory",
                 "planned_output_json_path", "planned_selected_checkpoint_path",
                 "trainer_source_path", "trainer_blob_commit", "trainer_blob_sha256",
                 "runtime_repository_commit",
                 "command", "expected_trajectory_rows", "expected_prediction_exports",
                 "expected_prediction_rows_per_export", "expected_state_capsules"]
GATE_HEADER = ["gate", "required", "observed", "passed", "blocking_reason"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--stage191b-dir", type=Path, required=True)
    p.add_argument("--stage192a-dir", type=Path, required=True)
    p.add_argument("--stage185-sidecar-dir", type=Path, required=True)
    p.add_argument("--stage193b-run-root", type=Path, required=True)
    p.add_argument("--current-diagnostic-git-commit", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL row")
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError(f"{path}:{number}: row is not an object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value


def write_csv(path: Path, header: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in header})


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_int(value: Any) -> bool:
    return type(value) is int


def semantic_sidecar_sha(path: Path) -> tuple[str, int]:
    rows = read_jsonl(path)
    canonical = [{key: row[key] for key in sorted(row) if key != "created_at"} for row in rows]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(rows)


def git_call(repo: Path, arguments: list[str], *, binary: bool = False, dirty: bool = False) -> Any:
    result = subprocess.run(["git", *arguments], cwd=repo, check=False, capture_output=True, shell=False)
    if dirty:
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        return result.returncode
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout if binary else result.stdout.decode("utf-8", errors="strict").strip()


def source_identity(repo: Path, commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", commit or "") is None:
        raise ValueError("current diagnostic commit must be lowercase hexadecimal length 40")
    relatives = (
        "reports/stage193a_tail3_fresh_seed_replication_spec.md",
        "scripts/build_stage193a_tail3_fresh_seed_manifest.py",
        "scripts/analyze_stage193c_tail3_fresh_seed_replication.py",
    )
    files = {}
    for relative in relatives:
        current = (repo / relative).read_bytes()
        blob = git_call(repo, ["show", f"{commit}:{relative}"], binary=True)
        files[relative] = {
            "current_sha256": hashlib.sha256(current).hexdigest(),
            "commit_blob_sha256": hashlib.sha256(blob).hexdigest(),
            "bytes_equal": current == blob,
            "unstaged_clean": git_call(repo, ["diff", "--quiet", "--", relative], dirty=True) == 0,
            "staged_clean": git_call(repo, ["diff", "--cached", "--quiet", "--", relative], dirty=True) == 0,
        }
    trainer_rel = "scripts/train_controlled_v6b_minimal.py"
    trainer_current = (repo / trainer_rel).read_bytes()
    trainer_blob = git_call(repo, ["show", f"{TRAINER_BLOB_COMMIT}:{trainer_rel}"], binary=True)
    trainer = {"path": str((repo / trainer_rel).resolve()), "blob_commit": TRAINER_BLOB_COMMIT,
               "current_sha256": hashlib.sha256(trainer_current).hexdigest(),
               "commit_blob_sha256": hashlib.sha256(trainer_blob).hexdigest(),
               "bytes_equal": trainer_current == trainer_blob}
    head = git_call(repo, ["rev-parse", "HEAD"])
    passed = head == commit and trainer["bytes_equal"] and all(item["bytes_equal"] and item["unstaged_clean"] and item["staged_clean"] for item in files.values())
    return {"supplied_commit": commit, "repository_head": head, "files": files,
            "trainer": trainer, "passed": passed}


def establish_safe_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path]:
    repo = args.repo_root.resolve()
    if not repo.is_dir():
        raise ValueError("repo root is not a directory")
    reports = (repo / "reports").resolve()
    stage191 = args.stage191b_dir.resolve()
    stage192 = args.stage192a_dir.resolve()
    stage185 = args.stage185_sidecar_dir.resolve()
    run_root = args.stage193b_run_root.resolve()
    output = args.output_dir.resolve()
    if stage191 != (reports / STAGE191_BASENAME).resolve() or not stage191.is_dir():
        raise ValueError("Stage191-B directory is not the exact authoritative directory")
    if stage185 != (reports / STAGE185_BASENAME).resolve() or not stage185.is_dir():
        raise ValueError("Stage185 directory is not the exact authoritative directory")
    if stage192.parent != reports or not stage192.name.startswith("stage192a_trajectory_stable_selection_") or not stage192.is_dir():
        raise ValueError("Stage192-A must be an explicit existing immediate reports child")
    if output.parent != reports or not output.name.startswith("stage193a_tail3_fresh_seed_manifest_"):
        raise ValueError("Stage193-A output path is unsafe")
    if run_root.parent != reports or not run_root.name.startswith("stage193b_tail3_fresh_seed_runs_"):
        raise ValueError("Stage193-B planned run root is unsafe")
    paths = (stage191, stage192, stage185, run_root, output)
    if len(set(paths)) != len(paths):
        raise ValueError("frozen inputs, run root, and output must be distinct")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Stage193-A output exists and is nonempty")
    if run_root.exists() and (not run_root.is_dir() or any(run_root.iterdir())):
        raise ValueError("Stage193-B run root exists and is nonempty")
    return repo, stage191, stage192, stage185, run_root, output


def option_map(argv: Any) -> dict[str, Any]:
    if type(argv) is not list or any(type(token) is not str for token in argv):
        raise ValueError("argv must be a list of strings")
    result: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--") or "=" in token:
            raise ValueError(f"unsupported argv token: {token!r}")
        key = token[2:].replace("-", "_")
        if key in result:
            raise ValueError(f"duplicate argv option: {token}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            result[key] = argv[index + 1]
            index += 2
        else:
            result[key] = True
            index += 1
    return result


def normalized_template(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in STAGE191_FLAGS:
            index += 1
            continue
        if token == STAGE193_FLAG:
            raise ValueError("Stage191 source argv already contains Stage193 observability")
        if token == "--seed":
            normalized.extend((token, "<TRAINING_SEED>")); index += 2; continue
        if token in OUTPUT_OPTIONS:
            normalized.extend((token, f"<PATH:{token}>")); index += 2; continue
        normalized.append(token)
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            normalized.append(argv[index + 1]); index += 2
        else:
            index += 1
    return normalized

def rewrite_argv(source: list[str], seed: int, run_dir: Path) -> tuple[list[str], dict[str, Any]]:
    rewritten: list[str] = []
    changes: list[dict[str, Any]] = []
    index = 0
    seen_outputs: set[str] = set()
    while index < len(source):
        token = source[index]
        if token in STAGE191_FLAGS:
            changes.append({"kind": "remove_flag", "token": token, "source_index": index})
            index += 1
            continue
        if token == STAGE193_FLAG:
            raise ValueError("source argv unexpectedly contains Stage193 flag")
        if token == "--seed":
            if index + 1 >= len(source): raise ValueError("--seed value missing")
            rewritten.extend((token, str(seed)))
            changes.append({"kind": "training_seed", "from": source[index + 1], "to": str(seed)})
            index += 2; continue
        if token in OUTPUT_OPTIONS:
            if index + 1 >= len(source): raise ValueError(f"{token} value missing")
            replacement = str((run_dir / OUTPUT_OPTIONS[token]).resolve())
            rewritten.extend((token, replacement)); seen_outputs.add(token)
            changes.append({"kind": "output_path", "option": token, "from": source[index + 1], "to": replacement})
            index += 2; continue
        rewritten.append(token)
        if index + 1 < len(source) and not source[index + 1].startswith("--"):
            rewritten.append(source[index + 1]); index += 2
        else:
            index += 1
    if seen_outputs != set(OUTPUT_OPTIONS):
        raise ValueError("source argv output option set is not exact")
    if any(source.count(flag) != 1 for flag in STAGE191_FLAGS):
        raise ValueError("source argv must contain each Stage191 flag exactly once")
    rewritten.append(STAGE193_FLAG)
    changes.append({"kind": "add_flag", "token": STAGE193_FLAG})
    return rewritten, {"allowed_changes": changes, "all_other_tokens_preserved": True}


def validate_envelope(argv: list[str], arm: str, seed: int, sidecar: Path, run_dir: Path) -> dict[str, Any]:
    options = option_map(argv)
    required = {
        "architecture": "v6b_minimal", "backbone": "mamba", "device": "cuda",
        "model_name": "state-spaces/mamba-130m-hf",
        "data": "data/controlled_v5_v3_without_time_swap.jsonl",
        "seed": str(seed), "split_seed": "174", "epochs": "20",
        "select_metric": "final_macro_f1", "compatible_positive_margin_logit": "0.0",
        "selected_checkpoint_filename": "selected_checkpoint.pt",
        "stage193_tail3_fresh_seed_observability": True,
    }
    if any(options.get(key) != value for key, value in required.items()):
        raise ValueError(f"{seed}/{arm}: frozen training envelope mismatch")
    if options.get("use_temporal_comparator", True) is not True:
        raise ValueError(f"{seed}/{arm}: frozen temporal comparator was disabled")
    if "stage191_trajectory_replay_observability" in options or "stage191_save_trajectory_state_capsules" in options:
        raise ValueError(f"{seed}/{arm}: Stage191 observability/capsule flag remains")
    if options.get("max_train_records") is not None:
        raise ValueError(f"{seed}/{arm}: row truncation is configured")
    for key in FORBIDDEN_OPTIONS:
        value = options.get(key)
        if value not in (None, [], ""):
            raise ValueError(f"{seed}/{arm}: forbidden data/output option {key} is configured")
    if any(options.get(key) is True for key in FORBIDDEN_FLAGS):
        raise ValueError(f"{seed}/{arm}: forbidden auxiliary/external flag is enabled")
    if any(options.get(key, "none") != "none" for key in (*BRIDGE_MODES, *AUX_MODES)):
        raise ValueError(f"{seed}/{arm}: forbidden bridge/auxiliary mode is enabled")
    for token, filename in OUTPUT_OPTIONS.items():
        if Path(str(options[token[2:].replace('-', '_')])).resolve() != (run_dir / filename).resolve():
            raise ValueError(f"{seed}/{arm}: output path mismatch for {token}")
    if arm == "baseline":
        arm_ok = (options.get("compatible_positive_margin_weight") in ("0", "0.0") and
                  "controlled_integrity_sidecar_path" not in options and
                  "expected_integrity_sidecar_semantic_sha256" not in options)
    else:
        arm_ok = (options.get("compatible_positive_margin_weight") == "0.05" and
                  Path(str(options.get("controlled_integrity_sidecar_path", ""))).resolve() == sidecar and
                  options.get("expected_integrity_sidecar_semantic_sha256") == SIDECAR_SHA)
    if not arm_ok:
        raise ValueError(f"{seed}/{arm}: margin/sidecar arm contract mismatch")
    return {"required": required, "arm_contract_passed": arm_ok, "external_auxiliary_absent": True}


def validate_stage192(stage192: Path) -> dict[str, Any]:
    entries = {path.name for path in stage192.iterdir()}
    if entries != STAGE192_OUTPUTS or any(not (stage192 / name).is_file() for name in STAGE192_OUTPUTS):
        raise ValueError("Stage192-A exact fourteen-output set mismatch")
    report = read_json(stage192 / "stage192a_trajectory_stable_selection_report.json")
    required = {
        "stage": "Stage192-A", "decision": STAGE192_CLOSED, "runnable": True,
        "blocking_reasons": [], "diagnostic_only": True, "training_performed": False,
        "model_constructed": False, "external_data_used": False,
        "model_advancement_decision": False, "stage192b_training_authorized": False,
        "quality_preserving_selectors": [], "quality_tradeoff_selectors": [],
        "winning_selector": None, "current_diagnostic_git_commit": STAGE192_COMMIT,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError("Stage192-A closure mismatch")
    closure = report.get("global_closure_results")
    if type(closure) is not dict or not closure or any(value is not True for value in closure.values()):
        raise ValueError("Stage192-A global closure is not exactly all true")
    return report


def analyze(args: argparse.Namespace, source_gates: list[dict[str, Any]], gates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    repo, stage191, stage192, stage185, run_root, _ = establish_safe_paths(args)
    blockers: list[str] = []
    def gate(rows: list[dict[str, Any]], name: str, required: Any, observed: Any, passed: bool, reason: str) -> None:
        rows.append({"gate": name, "required": required, "observed": observed, "passed": passed,
                     "blocking_reason": "" if passed else reason})
        if not passed: blockers.append(f"{name}: {reason}")

    identity = source_identity(repo, args.current_diagnostic_git_commit)
    gate(source_gates, "diagnostic_and_trainer_source_identity", True, identity, identity["passed"], "source identity mismatch")
    stage192_report = validate_stage192(stage192)
    gate(source_gates, "stage192a_frozen_closure", STAGE192_CLOSED, stage192_report.get("decision"), True, "")

    sidecar = (stage185 / "stage185a_controlled_train_integrity_sidecar.jsonl").resolve()
    if not sidecar.is_file(): raise ValueError("authoritative Stage185 sidecar JSONL is absent")
    semantic_sha, sidecar_rows = semantic_sidecar_sha(sidecar)
    gate(source_gates, "stage185_semantic_sha256", SIDECAR_SHA, semantic_sha, semantic_sha == SIDECAR_SHA, "sidecar semantic SHA mismatch")

    main = read_json(stage191 / "stage191b_deterministic_replay_manifest_report.json")
    main_required = {
        "stage": "Stage191-B", "decision": STAGE191_READY, "runnable": True,
        "blocking_reasons": [], "diagnostic_replay_only": True,
        "replay_execution_authorized": True, "training_for_model_advancement_authorized": False,
        "model_advancement_decision": False, "external_data_used": False,
        "authorized_training_seeds": list(SOURCE_SEEDS), "expected_trajectory_rows_per_run": 20,
        "expected_prediction_rows_per_epoch": 720, "expected_state_capsules_per_run": 20,
        "logits_source": 'output["logits"]',
    }
    stage191_integer_contract = {"expected_trajectory_rows_per_run": 20,
        "expected_prediction_rows_per_epoch": 720, "expected_state_capsules_per_run": 20}
    authorized_source_seeds = main.get("authorized_training_seeds")
    stage191_ok = (all(main.get(key) == value for key, value in main_required.items()) and
        type(authorized_source_seeds) is list and all(exact_int(value) for value in authorized_source_seeds) and
        all(exact_int(main.get(key)) and main.get(key) == value
            for key, value in stage191_integer_contract.items()) and
        (main.get("commit_identities") or {}).get("stage191b_replay_commit") == STAGE191_COMMIT)
    gate(source_gates, "stage191b_frozen_closure", main_required, {key: main.get(key) for key in main_required}, stage191_ok, "Stage191-B closure mismatch")
    identities = [{key: row.get(key) for key in ("run", "seed", "arm", "split_seed")} for row in main.get("six_run_matrix", [])] if type(main.get("six_run_matrix")) is list else None
    expected_identities = [{"run": f"seed{seed}_{arm}", "seed": seed, "arm": arm, "split_seed": 174} for seed in SOURCE_SEEDS for arm in ARMS]
    source_matrix_integers_ok = (type(identities) is list and all(
        exact_int(row.get(key)) for row in identities for key in ("seed", "split_seed")))
    gate(source_gates, "stage191b_ordered_source_matrix", expected_identities, identities,
         identities == expected_identities and source_matrix_integers_ok, "Stage191-B source matrix mismatch")
    if blockers: raise ValueError("frozen source closure failed")

    source_by_arm: dict[str, list[list[str]]] = {arm: [] for arm in ARMS}
    source_hashes = {}
    for source_run in SOURCE_RUNS:
        source_seed = int(source_run[4:7]); arm = source_run.split("_", 1)[1]
        path = stage191 / f"stage191b_{source_run}_replay_manifest.json"
        manifest = read_json(path)
        required = {"stage": "Stage191-B", "run": source_run, "seed": source_seed,
                    "training_seed": source_seed, "split_seed": 174, "arm": arm,
                    "runnable": True, "blocking_reasons": [], "diagnostic_replay_only": True,
                    "replay_execution_authorized": True,
                    "training_for_model_advancement_authorized": False,
                    "model_advancement_decision": False, "external_data_used": False,
                    "expected_trajectory_rows": 20, "expected_prediction_rows_per_epoch": 720,
                    "expected_state_capsules": 20, "logits_source": 'output["logits"]'}
        if any(manifest.get(key) != value for key, value in required.items()):
            raise ValueError(f"{source_run}: Stage191 manifest contract mismatch")
        source_manifest_integer_contract = {"seed": source_seed, "training_seed": source_seed,
            "split_seed": 174, "expected_trajectory_rows": 20,
            "expected_prediction_rows_per_epoch": 720, "expected_state_capsules": 20}
        if any(not exact_int(manifest.get(key)) or manifest.get(key) != value
               for key, value in source_manifest_integer_contract.items()):
            raise ValueError(f"{source_run}: Stage191 manifest strict integer contract mismatch")
        if (manifest.get("commit_identities") or {}).get("stage191b_replay_commit") != STAGE191_COMMIT:
            raise ValueError(f"{source_run}: Stage191 commit mismatch")
        argv = manifest.get("argv")
        option_map(argv)
        source_by_arm[arm].append(argv)
        source_hashes[path.name] = file_sha256(path)
    templates = {}
    for arm in ARMS:
        normalized = [normalized_template(argv) for argv in source_by_arm[arm]]
        equivalent = all(item == normalized[0] for item in normalized[1:])
        gate(source_gates, f"{arm}_three_source_template_equivalence", True,
             {"equivalent": equivalent, "template": normalized[0]}, equivalent, "within-arm templates differ")
        templates[arm] = source_by_arm[arm][0]
    if blockers: raise ValueError("Stage191 template equivalence failed")

    manifests: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    trainer = (repo / "scripts/train_controlled_v6b_minimal.py").resolve()
    for run in RUNS:
        match = re.fullmatch(r"seed(177|178|179)_(baseline|intervention)", run)
        if not match: raise RuntimeError("internal frozen run identity failure")
        seed, arm = int(match.group(1)), match.group(2)
        run_dir = (run_root / run).resolve()
        if run_dir.exists(): raise ValueError(f"planned run directory already exists: {run_dir}")
        argv, audit = rewrite_argv(templates[arm], seed, run_dir)
        envelope = validate_envelope(argv, arm, seed, sidecar, run_dir)
        predictions = [str((run_dir / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl").resolve()) for epoch in range(1, 21)]
        generated_options = option_map(argv)
        checkpoint_filename = generated_options.get("selected_checkpoint_filename")
        if type(checkpoint_filename) is not str or Path(checkpoint_filename).name != checkpoint_filename:
            raise ValueError(f"{run}: selected checkpoint filename is not an exact filename")
        planned_checkpoint = (run_dir / checkpoint_filename).resolve()
        arm_contract = ({"compatible_positive_margin_weight": 0.0, "sidecar_access": False}
                        if arm == "baseline" else
                        {"compatible_positive_margin_weight": 0.05, "compatible_positive_margin_logit": 0.0,
                         "sidecar_path": str(sidecar), "sidecar_semantic_sha256": SIDECAR_SHA})
        row = {
            "stage": "Stage193-A", "run": run, "training_seed": seed, "split_seed": 174,
            "arm": arm, "canonical_labels": list(LABELS), "trainer_source_path": str(trainer),
            "trainer_blob_commit": TRAINER_BLOB_COMMIT,
            "trainer_blob_sha256": identity["trainer"]["current_sha256"],
            "stage193_runtime_repository_commit": args.current_diagnostic_git_commit, "argv": argv,
            "command_argv": ["python", str(trainer), *argv],
            "command": shlex.join(["python", str(trainer), *argv]),
            "planned_run_directory": str(run_dir),
            "planned_output_json_path": str((run_dir / "training_report.json").resolve()),
            "planned_selected_checkpoint_path": str(planned_checkpoint),
            "expected_trajectory_contract_path": str((run_dir / "stage191_trajectory_contract.json").resolve()),
            "expected_trajectory_ledger_path": str((run_dir / "stage191_trajectory_epoch_metrics.jsonl").resolve()),
            "expected_prediction_export_paths": predictions,
            "expected_trajectory_rows": 20, "expected_prediction_exports": 20,
            "expected_prediction_rows_per_export": 720, "expected_state_capsules": 0,
            "logits_source": 'output["logits"]', "arm_contract": arm_contract,
            "argv_mutation_audit": audit, "frozen_training_envelope": envelope,
            "runnable": True, "diagnostic_only": True,
            "exact_six_run_diagnostic_execution_authorized": True,
            "training_for_model_advancement_authorized": False,
            "model_advancement_decision": False, "subsequent_training_authorized": False,
            "external_data_used": False,
        }
        manifests.append(row)
        matrix.append({**{key: row[key] for key in ("run", "training_seed", "split_seed", "arm", "planned_run_directory", "planned_output_json_path", "planned_selected_checkpoint_path", "trainer_source_path", "trainer_blob_commit", "trainer_blob_sha256")},
                       "runtime_repository_commit": row["stage193_runtime_repository_commit"],
                       "command": row["command"], "expected_trajectory_rows": 20,
                       "expected_prediction_exports": 20, "expected_prediction_rows_per_export": 720,
                       "expected_state_capsules": 0})
    manifest_integer_contract = {"training_seed": None, "split_seed": 174,
        "expected_trajectory_rows": 20, "expected_prediction_exports": 20,
        "expected_prediction_rows_per_export": 720, "expected_state_capsules": 0}
    manifest_integer_ok = all(
        exact_int(row.get(key)) and (expected is None or row.get(key) == expected)
        for row in manifests for key, expected in manifest_integer_contract.items())
    gate(gates, "manifest_strict_non_bool_integer_contract", True, manifest_integer_ok,
         manifest_integer_ok, "manifest integer identity/cardinality contract mismatch")
    report_integer_contract = {"run_manifest_count": 6,
        "expected_trajectory_rows_per_run": 20, "expected_prediction_exports_per_run": 20,
        "expected_prediction_rows_per_export": 720, "expected_state_capsules_per_run": 0}
    report_integer_ok = all(exact_int(value) for value in report_integer_contract.values())
    gate(gates, "report_strict_non_bool_integer_contract", True, report_integer_ok,
         report_integer_ok, "report integer cardinality contract mismatch")
    gate(gates, "exact_fresh_run_order", list(RUNS), [row["run"] for row in manifests], [row["run"] for row in manifests] == list(RUNS), "run order mismatch")
    run_root_safe = not run_root.exists() or not any(run_root.iterdir())
    gate(gates, "stage193b_run_root_empty_or_absent", True, run_root_safe,
         run_root_safe, "run root is nonempty")
    authorization_ok = all(row["exact_six_run_diagnostic_execution_authorized"] and
        not row["training_for_model_advancement_authorized"] and
        not row["subsequent_training_authorized"] for row in manifests)
    gate(gates, "diagnostic_authorization_only", True, authorization_ok,
         authorization_ok, "authorization scope mismatch")
    if blockers: raise ValueError("precommitted manifest gates failed")
    report = {
        "stage": "Stage193-A", "decision": READY, "runnable": True, "blocking_reasons": [],
        "diagnostic_only": True, "exact_six_run_diagnostic_execution_authorized": True,
        "training_for_model_advancement_authorized": False, "model_advancement_decision": False,
        "subsequent_training_authorized": False, "external_data_used": False,
        "checkpoint_loaded": False, "model_loaded": False, "capsule_loaded": False,
        "statistical_significance_claim": False,
        "current_diagnostic_git_commit": args.current_diagnostic_git_commit,
        "source_identity": identity,
        "trainer_blob_commit": TRAINER_BLOB_COMMIT,
        "trainer_blob_sha256": identity["trainer"]["current_sha256"],
        "stage193_runtime_repository_commit": args.current_diagnostic_git_commit,
        "frozen_source_identities": {"trainer_blob_commit": TRAINER_BLOB_COMMIT,
            "stage192a_implementation_commit": STAGE192_COMMIT,
            "stage191b_replay_implementation_commit": STAGE191_COMMIT,
            "stage185_sidecar_semantic_sha256": SIDECAR_SHA,
            "stage191b_directory": str(stage191), "stage192a_directory": str(stage192),
            "stage185_sidecar_directory": str(stage185), "stage185_sidecar_path": str(sidecar),
            "stage185_sidecar_rows": sidecar_rows, "stage191_manifest_sha256": source_hashes},
        "stage193b_run_root": str(run_root), "ordered_runs": list(RUNS),
        "run_manifest_count": 6, "expected_trajectory_rows_per_run": 20,
        "expected_prediction_exports_per_run": 20, "expected_prediction_rows_per_export": 720,
        "expected_state_capsules_per_run": 0, "canonical_labels": list(LABELS),
        "logits_source": 'output["logits"]', "source_and_template_gates": source_gates,
        "precommitted_gates": gates, "exception": None,
    }
    if any(not exact_int(report.get(key)) or report.get(key) != value
           for key, value in report_integer_contract.items()):
        raise ValueError("Stage193-A report strict integer contract mismatch")
    return report, manifests, matrix


def blocked_report(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {"stage": "Stage193-A", "decision": BLOCKED, "runnable": False,
            "blocking_reasons": [f"{type(exc).__name__}: {exc}"], "diagnostic_only": True,
            "exact_six_run_diagnostic_execution_authorized": False,
            "training_for_model_advancement_authorized": False, "model_advancement_decision": False,
            "subsequent_training_authorized": False, "external_data_used": False,
            "checkpoint_loaded": False, "model_loaded": False, "capsule_loaded": False,
            "statistical_significance_claim": False,
            "current_diagnostic_git_commit": args.current_diagnostic_git_commit,
            "trainer_blob_commit": TRAINER_BLOB_COMMIT, "trainer_blob_sha256": None,
            "stage193_runtime_repository_commit": args.current_diagnostic_git_commit,
            "ordered_runs": list(RUNS), "exception": {"type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc()}}


def markdown(report: dict[str, Any]) -> str:
    return "\n".join(["# Stage193-A tail3 fresh-seed manifest report", "",
        f"Decision: `{report['decision']}`", "", f"- Runnable: {str(report['runnable']).lower()}",
        "- Diagnostic only: true", "- Checkpoint/model/capsule loading: none",
        f"- Trainer blob commit: `{report.get('trainer_blob_commit')}`",
        f"- Trainer blob SHA256: `{report.get('trainer_blob_sha256')}`",
        f"- Stage193 runtime repository commit: `{report.get('stage193_runtime_repository_commit')}`",
        "- External data used: false", "- Statistical-significance claim: false",
        "- Model advancement authorized: false", "- Subsequent training authorized: false",
        f"- Exact six-run diagnostic execution authorized: {str(report.get('exact_six_run_diagnostic_execution_authorized', False)).lower()}", "",
        "READY authorizes only the exact frozen six diagnostic runs; it does not authorize model advancement or further training.", ""])


def main() -> int:
    args = parse_args()
    try:
        *_, output = establish_safe_paths(args)
    except BaseException:
        traceback.print_exc(file=sys.stderr); return 2
    output.mkdir(parents=False, exist_ok=True)
    source_gates: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    try:
        report, manifests, matrix = analyze(args, source_gates, gates)
    except BaseException as exc:
        report = blocked_report(args, exc)
        gates.append({"gate": "fail_closed_exception", "required": "no exception",
                      "observed": {"type": type(exc).__name__, "message": str(exc)},
                      "passed": False, "blocking_reason": str(exc)})
    write_json(output / OUTPUTS["json"], report)
    (output / OUTPUTS["md"]).write_text(markdown(report), encoding="utf-8")
    write_jsonl(output / OUTPUTS["jsonl"], manifests if report["runnable"] else [])
    write_csv(output / OUTPUTS["matrix"], MATRIX_HEADER, matrix if report["runnable"] else [])
    write_csv(output / OUTPUTS["source"], GATE_HEADER, source_gates)
    write_csv(output / OUTPUTS["gates"], GATE_HEADER, gates)
    return 0 if report["runnable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
