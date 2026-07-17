#!/usr/bin/env python3
"""Analyze frozen Stage191 replay trajectories and state capsules; never build a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable


STAGE191B_COMMIT = "0872e66ccb05ae8a166f5cabf4e084272dc49500"
STAGE191B_DIRNAME = "stage191b_deterministic_replay_manifest_20260717_153524"
STAGE191B_READY = "STAGE191B_DETERMINISTIC_REPLAY_MANIFEST_READY"
STAGE190B_EXPORTED = "STAGE190B_GRADIENT_DIAGNOSTIC_EXPORTED"
SIDECAR_SHA256 = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
STAGE185_SIDECAR_RELATIVE_PATH = Path("reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl")
STAGE190_DIAGNOSTIC_COMMIT = "ac0b9032b94436ce8ac8134c650d389134faebd4"
STAGE190A_DIRNAME = "stage190a_gradient_conflict_manifest_20260717_113644"
STAGE190_GROUPING_SOURCE_RELATIVE_PATH = Path("scripts/run_stage190b_gradient_conflict_diagnostic.py")
STAGE190_GROUPING_SOURCE_SHA256 = "482037cc34ffdc3b4f8a26181e8408c423f1004194a6f914942dfec50e36bc02"
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
SEEDS = (174, 175, 176)
ARMS = ("baseline", "intervention")
EPOCHS = tuple(range(1, 21))
RUNS = tuple(f"seed{seed}_{arm}" for seed in SEEDS for arm in ARMS)
SELECTED = {
    "seed174_baseline": 20,
    "seed174_intervention": 20,
    "seed175_baseline": 20,
    "seed175_intervention": 19,
    "seed176_baseline": 20,
    "seed176_intervention": 19,
}
FROZEN_SELECTED_CHECKPOINT_SHA256 = {
    "seed174_baseline": "8e31dbd1459a67e65571ea1926a6e1a5f49f1ae2e57deb8455b41617f9ed972c",
    "seed174_intervention": "66cfb4fd91c29dfc6d4f243e701103e6b82e6ccac810a3b7a17d5b05310a57b3",
    "seed175_baseline": "5baa306161f204dbc984681a0b18c22484e3724aea4bfbdc9858e6f434ea1c0a",
    "seed175_intervention": "c00700d170e11fcc0376e2fe0ca7bc8037a76330330ed1b99fa21a35775a8018",
    "seed176_baseline": "8bcae6880e68cf8f34b9fb86f3f987e26872511636af33a64f6cc012857e51ea",
    "seed176_intervention": "539ff0c226e6f862abf99c9b4ceaf883989cee9985c5a5cf438801ecb87620a5",
}

BLOCKED = "STAGE191D_TRAJECTORY_DIAGNOSTIC_BLOCKED"
CONFIRMED = "STAGE191D_LATE_SUPPORT_NE_PHASE_FLIP_CONFIRMED"
REDISTRIBUTION = "STAGE191D_LATE_CLASS_REDISTRIBUTION_WITHOUT_PHASE_FLIP"
INCONCLUSIVE = "STAGE191D_TRAJECTORY_EFFECT_INCONCLUSIVE"

OUTPUTS = {
    "json": "stage191d_trajectory_phase_flip_report.json",
    "md": "stage191d_trajectory_phase_flip_report.md",
    "equivalence": "stage191d_stage191c_equivalence_gate.csv",
    "epochs": "stage191d_run_epoch_metrics.csv",
    "deltas": "stage191d_pair_delta_by_epoch.csv",
    "late": "stage191d_late_sign_reversal.csv",
    "within": "stage191d_within_run_transition_summary.csv",
    "within_gold": "stage191d_within_run_transition_by_gold.csv",
    "paired": "stage191d_paired_transition_summary.csv",
    "paired_gold": "stage191d_paired_transition_by_gold.csv",
    "instability": "stage191d_late_instability_metrics.csv",
    "state": "stage191d_state_step_summary.csv",
    "groups": "stage191d_parameter_group_step.csv",
    "precommit": "stage191d_precommitted_gate.csv",
}

METRICS = (
    "clean_dev_ce",
    "clean_accuracy",
    "clean_macro_f1",
    "support_recall",
    "false_entitlement_total",
    "false_not_entitled_total",
    "polarity_error_total",
    "pred_REFUTE",
    "pred_NOT_ENTITLED",
    "pred_SUPPORT",
)
INSTABILITY_SERIES = (
    "pred_SUPPORT",
    "pred_NOT_ENTITLED",
    "support_recall",
    "false_entitlement_total",
)
CLEAN_DEV_CE_REDUCTION_CONTRACT = {
    "row_source": "ordered epoch export final_ce",
    "row_count": 720,
    "dtype": "torch.float32",
    "device": "cpu",
    "reduction": "mean",
    "comparison": "exact equality",
}
MATRIX_FIELDS = tuple(f"{left}_to_{right}" for left in LABELS for right in LABELS)
TRANSITION_FIELDS = (
    *MATRIX_FIELDS,
    "unchanged_rows",
    "changed_rows",
    "not_entitled_to_support",
    "support_to_not_entitled",
    "refute_involved_transitions",
    "exclusive_not_entitled_support_changed_rows",
    "exclusive_not_entitled_support_fraction_of_changed",
    "corrections",
    "regressions",
    "wrong_to_different_wrong",
)

GROUPS = ("frame_head", "decision_head", "router_or_epistemic", "backbone", "other")
STAGE190_ORIGINAL_GROUPS = ("frame_head", "decision_head", "router_and_epistemic_heads", "backbone", "other_trainable")
STAGE190_TO_STAGE191D_ALIASES = {
    "frame_head": "frame_head",
    "decision_head": "decision_head",
    "router_and_epistemic_heads": "router_or_epistemic",
    "backbone": "backbone",
    "other_trainable": "other",
}
NONZERO_STAGE190_GROUP_JUSTIFICATION = "nonzero module-owned group"
ZERO_SIZE_STAGE190_GROUP_JUSTIFICATION = "zero-size because the selected checkpoint has no trainable parameters owned by this conceptual module set"

STAGE190_REQUIRED_SOURCE_SNIPPETS = (
    'GROUPS = ("frame_head", "decision_head", "router_and_epistemic_heads", "backbone", "other_trainable")',
    '"frame_head": module_parameter_ids(model.frame_gate.frame_classifier)',
    '"decision_head": module_parameter_ids(model.decision_head)',
    '"backbone": module_parameter_ids(model.mamba)',
    '"frame_gate", "predicate_coverage_head", "sufficiency_gate", "polarity_energy_head",',
    '"boundary_head", "frame_violation_head", "predicate_isolation_head",',
    '"preservation_entitlement_head", "temporal_diagnostic_head", "temporal_residual_adapter",',
    '"temporal_channel_v1", "router", "fusion", "gate")',
    'router_ids -= owner_ids["frame_head"] | owner_ids["decision_head"] | owner_ids["backbone"]',
    'owner_ids["router_and_epistemic_heads"] = router_ids',
    'owner_ids["other_trainable"] = all_ids - assigned',
    'if overlap or union != all_ids:',
    'if len(matches) != 1:',
)
FORBIDDEN_PATH_OPTIONS = (
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
EXTERNAL_FLAGS = (
    "enable_external_eval",
    "enable_stage43_external_eval",
    "stage43_external_enable_shadow_export",
)
BRIDGE_MODES = (
    "stage57_bridge_train_mode",
    "stage66_bridge_train_mode",
    "stage75_bridge_train_mode",
    "stage80a_bridge_train_mode",
)

CSV_HEADERS = {
    "equivalence": ["gate", "run", "required", "observed", "passed", "blocking_reason"],
    "epochs": ["run", "seed", "arm", "epoch", *METRICS, "selected_epoch", "replaced_selected_checkpoint"],
    "deltas": ["seed", "epoch", *[f"delta_{name}" for name in METRICS], "delta_selected_epoch", "delta_replaced_selected_checkpoint"],
    "late": ["seed", "delta_pred_SUPPORT_epoch19", "delta_pred_SUPPORT_epoch20", "sign_pred_SUPPORT_epoch19", "sign_pred_SUPPORT_epoch20", "pred_SUPPORT_sign_reversal", "delta_pred_NOT_ENTITLED_epoch19", "delta_pred_NOT_ENTITLED_epoch20", "sign_pred_NOT_ENTITLED_epoch19", "sign_pred_NOT_ENTITLED_epoch20", "pred_NOT_ENTITLED_sign_reversal", "delta_false_entitlement_epoch19", "delta_false_entitlement_epoch20", "sign_false_entitlement_epoch19", "sign_false_entitlement_epoch20", "false_entitlement_sign_reversal", "abs_delta_pred_REFUTE_epoch19", "abs_delta_pred_REFUTE_epoch20", "delta_polarity_error_epoch19", "delta_polarity_error_epoch20", "phase_flip_seed_conditions_pass"],
    "within": ["run", "seed", "arm", "comparison", "previous_epoch", "next_epoch", *TRANSITION_FIELDS],
    "within_gold": ["run", "seed", "arm", "comparison", "previous_epoch", "next_epoch", "gold_label", *TRANSITION_FIELDS],
    "paired": ["seed", "epoch", "previous_arm", "next_arm", *TRANSITION_FIELDS],
    "paired_gold": ["seed", "epoch", "previous_arm", "next_arm", "gold_label", *TRANSITION_FIELDS],
    "instability": ["seed", "series", "baseline_total_variation", "intervention_total_variation", "intervention_minus_baseline_total_variation", "baseline_amplitude", "intervention_amplitude", "intervention_minus_baseline_amplitude", "baseline_direction_reversals", "intervention_direction_reversals", "intervention_minus_baseline_direction_reversals", "baseline_selected_to_final_drift", "intervention_selected_to_final_drift", "intervention_minus_baseline_selected_to_final_drift"],
    "state": ["run", "seed", "arm", "previous_epoch", "next_epoch", "trainable_step_squared_l2", "trainable_step_l2", "prior_state_l2", "normalized_step_norm", "buffer_step_l2", "cosine_with_preceding_step", "path_length_epoch15_to_20", "direct_displacement_epoch15_to_20", "path_length_direct_displacement_ratio"],
    "groups": ["run", "seed", "arm", "previous_epoch", "next_epoch", "parameter_group", "squared_l2_step_norm", "l2_step_norm", "fraction_total_squared_step_norm", "parameter_count", "tensor_count", "grouping_contract"],
    "precommit": ["gate", "required", "observed", "passed"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage191b-dir", type=Path, required=True)
    parser.add_argument("--current-diagnostic-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def semantic_sidecar_sha256(path: Path) -> str:
    rows = read_jsonl(path)
    canonical = [{key: row[key] for key in sorted(row) if key != "created_at"} for row in rows]
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in headers})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def establish_safe_output(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    repo = args.repo_root.resolve()
    if not repo.is_dir():
        raise ValueError("repo root is not a directory")
    reports_dir = (repo / "reports").resolve()
    stage191b = args.stage191b_dir.resolve()
    expected_stage191b = (reports_dir / STAGE191B_DIRNAME).resolve()
    if stage191b != expected_stage191b or not stage191b.is_dir():
        raise ValueError(f"Stage191-B directory must resolve exactly to the existing directory {expected_stage191b}")
    output = args.output_dir.resolve()
    if output.parent.resolve() != reports_dir:
        raise ValueError("output directory parent must resolve exactly to <repo-root>/reports")
    if not output.name.startswith("stage191d_trajectory_phase_flip_"):
        raise ValueError("output directory basename must start with stage191d_trajectory_phase_flip_")
    frozen_report_inputs = [path.resolve() for path in reports_dir.iterdir() if path.is_dir() and re.match(r"^stage(?:189|190|191[abc])", path.name, re.IGNORECASE)]
    protected_roots = [stage191b, (repo / "scripts").resolve(), (repo / "data").resolve(), (repo / "src").resolve(), *frozen_report_inputs]
    if any(output == protected or protected in output.parents for protected in protected_roots):
        raise ValueError("output directory is or is inside a protected source/frozen-artifact directory")
    if output.exists():
        if not output.is_dir():
            raise ValueError("output path exists and is not a directory")
        if any(output.iterdir()):
            raise ValueError("output directory already exists and is nonempty")
    return repo, stage191b, output


def git_call(repo: Path, arguments: list[str], *, binary: bool = False, allow_dirty_code: bool = False) -> Any:
    result = subprocess.run(["git", *arguments], cwd=repo, check=False, capture_output=True, shell=False)
    if allow_dirty_code:
        if result.returncode not in (0, 1):
            raise RuntimeError(f"git {' '.join(arguments)} failed ({result.returncode}): {result.stderr.decode('utf-8', errors='replace')}")
        return result.returncode
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed ({result.returncode}): {result.stderr.decode('utf-8', errors='replace')}")
    return result.stdout if binary else result.stdout.decode("utf-8", errors="strict").strip()


def validate_diagnostic_source_identity(repo: Path, supplied_commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", supplied_commit or "") is None:
        raise ValueError("current diagnostic commit must be exact 40-character lowercase hexadecimal")
    head = git_call(repo, ["rev-parse", "HEAD"])
    files: dict[str, Any] = {}
    for relative in ("scripts/analyze_stage191d_trajectory_phase_flip.py", "reports/stage191d_trajectory_phase_flip_spec.md"):
        current_path = repo / Path(relative)
        current_bytes = current_path.read_bytes()
        blob_bytes = git_call(repo, ["show", f"{supplied_commit}:{relative}"], binary=True)
        unstaged_code = git_call(repo, ["diff", "--quiet", "--", relative], allow_dirty_code=True)
        staged_code = git_call(repo, ["diff", "--cached", "--quiet", "--", relative], allow_dirty_code=True)
        files[relative] = {
            "current_sha256": hashlib.sha256(current_bytes).hexdigest(),
            "commit_blob_sha256": hashlib.sha256(blob_bytes).hexdigest(),
            "current_bytes_equal_commit_blob": current_bytes == blob_bytes,
            "unstaged_clean": unstaged_code == 0,
            "staged_clean": staged_code == 0,
        }
    passed = head == supplied_commit and all(item["current_bytes_equal_commit_blob"] and item["unstaged_clean"] and item["staged_clean"] for item in files.values())
    return {"supplied_commit": supplied_commit, "repository_head": head, "files": files, "passed": passed}


def validate_stage190_grouping_source(repo: Path) -> dict[str, Any]:
    source_path = (repo / STAGE190_GROUPING_SOURCE_RELATIVE_PATH).resolve()
    expected_path = (repo / "scripts" / "run_stage190b_gradient_conflict_diagnostic.py").resolve()
    current_bytes = source_path.read_bytes()
    frozen_blob = git_call(repo, ["show", f"{STAGE190_DIAGNOSTIC_COMMIT}:scripts/run_stage190b_gradient_conflict_diagnostic.py"], binary=True)
    commit_blob_sha = hashlib.sha256(frozen_blob).hexdigest()
    observed_sha = hashlib.sha256(current_bytes).hexdigest()
    source_text = current_bytes.decode("utf-8", errors="strict")
    snippets = {snippet: snippet in source_text for snippet in STAGE190_REQUIRED_SOURCE_SNIPPETS}
    unstaged_clean = git_call(repo, ["diff", "--quiet", "--", "scripts/run_stage190b_gradient_conflict_diagnostic.py"], allow_dirty_code=True) == 0
    staged_clean = git_call(repo, ["diff", "--cached", "--quiet", "--", "scripts/run_stage190b_gradient_conflict_diagnostic.py"], allow_dirty_code=True) == 0
    current_equals_blob = current_bytes == frozen_blob
    passed = source_path == expected_path and source_path.is_file() and commit_blob_sha == STAGE190_GROUPING_SOURCE_SHA256 and observed_sha == commit_blob_sha and current_equals_blob and unstaged_clean and staged_clean and all(snippets.values())
    return {
        "authoritative_stage190_diagnostic_commit": STAGE190_DIAGNOSTIC_COMMIT,
        "authoritative_stage190_grouping_source_path": str(expected_path),
        "authoritative_stage190_grouping_commit_blob_sha256": commit_blob_sha,
        "observed_stage190_grouping_source_sha256": observed_sha,
        "current_source_equals_frozen_commit_blob": current_equals_blob,
        "unstaged_clean": unstaged_clean,
        "staged_clean": staged_clean,
        "source_contract_passed": passed,
        "original_stage190_group_names": list(STAGE190_ORIGINAL_GROUPS),
        "stage191d_reporting_groups": list(GROUPS),
        "alias_mapping": dict(STAGE190_TO_STAGE191D_ALIASES),
        "required_source_snippets_present": snippets,
    }

def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def exact_int(value: Any) -> bool:
    return type(value) is int


def exact_sha(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def option_map(argv: Any) -> dict[str, Any]:
    if type(argv) is not list or any(type(token) is not str for token in argv):
        raise ValueError("argv must be a list of strings")
    result: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--") or "=" in token:
            raise ValueError(f"unsupported argv token form: {token!r}")
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


def load_stage190_parameter_inventories(repo: Path) -> dict[str, Any]:
    stage190a = (repo / "reports" / STAGE190A_DIRNAME).resolve()
    if stage190a != (repo / "reports" / "stage190a_gradient_conflict_manifest_20260717_113644").resolve() or not stage190a.is_dir():
        raise ValueError("authoritative Stage190-A directory is absent or not exact")
    expected_headers = ["order", "parameter_name", "shape", "numel", "dtype", "group", "requires_grad"]
    by_run: dict[str, Any] = {}
    canonical_mapping: list[dict[str, Any]] | None = None
    canonical_mapping_sha256: str | None = None
    canonical_group_row_counts: dict[str, int] | None = None
    canonical_zero_size_groups: tuple[str, ...] | None = None
    overall_passed = True
    for seed in SEEDS:
        for arm in ARMS:
            run = f"seed{seed}_{arm}"
            evidence: dict[str, Any] = {"run": run, "conceptual_group_contract_passed": False, "inventory_group_row_counts": None, "observed_nonzero_inventory_groups": None, "observed_zero_size_inventory_groups": None, "zero_size_justifications": None, "zero_size_justification_contract_passed": False, "per_run_mapping_passed": False, "passed": False}
            try:
                manifest_path = stage190a / f"stage190a_seed{seed}_{arm}_manifest.json"
                manifest = read_json(manifest_path)
                manifest_ok = manifest_path.name == f"stage190a_{run}_manifest.json" and manifest.get("stage") == "Stage190-A" and exact_int(manifest.get("seed")) and exact_int(manifest.get("training_seed")) and exact_int(manifest.get("split_seed")) and manifest.get("seed") == seed and manifest.get("training_seed") == seed and manifest.get("arm") == arm and manifest.get("split_seed") == 174 and manifest.get("diagnostic_git_commit") == STAGE190_DIAGNOSTIC_COMMIT and manifest.get("checkpoint_sha256") == FROZEN_SELECTED_CHECKPOINT_SHA256[run] and type(manifest.get("checkpoint_path")) is str and manifest.get("runnable") is True and manifest.get("blocking_reasons") == []
                options = option_map(manifest.get("argv"))
                raw_output = options.get("output_dir")
                if type(raw_output) is not str:
                    raise ValueError("Stage190-A argv has no separately parsed --output-dir")
                output_dir = Path(raw_output).resolve()
                expected_output = (stage190a / f"stage190b_seed{seed}_{arm}").resolve()
                if output_dir != expected_output:
                    raise ValueError("Stage190-B output directory differs from exact manifest child")
                report_path = output_dir / "stage190b_gradient_report.json"
                inventory_path = output_dir / "stage190b_parameter_inventory.csv"
                report = read_json(report_path)
                parameter_group_contract = report.get("parameter_group_contract")
                zero_size_justifications = parameter_group_contract.get("zero_size_justifications") if type(parameter_group_contract) is dict else None
                conceptual_group_contract_passed = type(parameter_group_contract) is dict and parameter_group_contract.get("groups") == list(STAGE190_ORIGINAL_GROUPS) and parameter_group_contract.get("disjoint") is True and parameter_group_contract.get("exhaustive") is True and type(zero_size_justifications) is dict and set(zero_size_justifications) == set(STAGE190_ORIGINAL_GROUPS) and all(type(zero_size_justifications[group]) is str for group in STAGE190_ORIGINAL_GROUPS)
                evidence.update({"conceptual_group_contract_passed": conceptual_group_contract_passed, "zero_size_justifications": zero_size_justifications})
                report_contract_ok = report.get("decision") == STAGE190B_EXPORTED and report.get("blocking_reasons") == [] and exact_int(report.get("training_seed")) and exact_int(report.get("split_seed")) and report.get("training_seed") == seed and report.get("split_seed") == 174 and report.get("arm") == arm and report.get("diagnostic_git_commit") == STAGE190_DIAGNOSTIC_COMMIT and report.get("selected_checkpoint_sha256") == FROZEN_SELECTED_CHECKPOINT_SHA256[run] and type(report.get("selected_checkpoint_path")) is str and Path(report["selected_checkpoint_path"]).resolve() == Path(manifest["checkpoint_path"]).resolve() and conceptual_group_contract_passed
                with inventory_path.open("r", newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames != expected_headers:
                        raise ValueError("Stage190-B inventory header mismatch")
                    raw_rows = list(reader)
                if not raw_rows:
                    raise ValueError("Stage190-B inventory is empty")
                mapping: list[dict[str, Any]] = []
                names: set[str] = set()
                for expected_order, row in enumerate(raw_rows):
                    if set(row) != set(expected_headers) or not re.fullmatch(r"0|[1-9][0-9]*", row["order"] or "") or int(row["order"]) != expected_order:
                        raise ValueError("Stage190-B inventory parameter order mismatch")
                    name = row["parameter_name"]
                    if not name or name in names:
                        raise ValueError("Stage190-B inventory parameter_name is empty or duplicate")
                    names.add(name)
                    shape = json.loads(row["shape"])
                    if type(shape) is not list or any(not exact_int(value) or value < 0 for value in shape):
                        raise ValueError(f"Stage190-B inventory shape invalid for {name}")
                    if not re.fullmatch(r"0|[1-9][0-9]*", row["numel"] or ""):
                        raise ValueError(f"Stage190-B inventory numel invalid for {name}")
                    numel = int(row["numel"])
                    shape_numel = math.prod(shape) if shape else 1
                    if numel != shape_numel or not row["dtype"] or row["requires_grad"] != "True" or row["group"] not in STAGE190_ORIGINAL_GROUPS:
                        raise ValueError(f"Stage190-B inventory topology/group contract invalid for {name}")
                    mapping.append({"order": expected_order, "parameter_name": name, "shape": shape, "numel": numel, "dtype": row["dtype"], "stage190_group": row["group"], "stage191d_group": STAGE190_TO_STAGE191D_ALIASES[row["group"]]})
                inventory_group_row_counts = {group: 0 for group in STAGE190_ORIGINAL_GROUPS}
                for item in mapping:
                    inventory_group_row_counts[item["stage190_group"]] += 1
                observed_nonzero_inventory_groups = tuple(group for group in STAGE190_ORIGINAL_GROUPS if inventory_group_row_counts[group] > 0)
                observed_zero_size_inventory_groups = tuple(group for group in STAGE190_ORIGINAL_GROUPS if inventory_group_row_counts[group] == 0)
                zero_size_justification_contract_passed = conceptual_group_contract_passed and all(
                    zero_size_justifications[group] == (NONZERO_STAGE190_GROUP_JUSTIFICATION if inventory_group_row_counts[group] > 0 else ZERO_SIZE_STAGE190_GROUP_JUSTIFICATION)
                    for group in STAGE190_ORIGINAL_GROUPS
                )
                evidence.update({"inventory_group_row_counts": inventory_group_row_counts, "observed_nonzero_inventory_groups": list(observed_nonzero_inventory_groups), "observed_zero_size_inventory_groups": list(observed_zero_size_inventory_groups), "zero_size_justification_contract_passed": zero_size_justification_contract_passed})
                if not zero_size_justification_contract_passed:
                    raise ValueError("Stage190-B zero-size justification contract is inconsistent with inventory row counts")
                ordering_payload = [{"name": item["parameter_name"], "shape": item["shape"], "numel": item["numel"], "group": item["stage190_group"]} for item in mapping]
                ordering_sha = hashlib.sha256(json.dumps(ordering_payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
                mapping_sha = hashlib.sha256(json.dumps(mapping, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
                topology_ok = report.get("parameter_ordering_sha256") == ordering_sha and report.get("trainable_parameter_count") == len(mapping) and report.get("trainable_parameter_numel") == sum(item["numel"] for item in mapping)
                if not manifest_ok or not report_contract_ok or not topology_ok:
                    raise ValueError("Stage190 manifest/report/checkpoint/inventory topology contract mismatch")
                if canonical_mapping is None:
                    canonical_mapping = mapping
                    canonical_mapping_sha256 = mapping_sha
                    canonical_group_row_counts = inventory_group_row_counts
                    canonical_zero_size_groups = observed_zero_size_inventory_groups
                elif mapping != canonical_mapping or mapping_sha != canonical_mapping_sha256 or inventory_group_row_counts != canonical_group_row_counts or observed_zero_size_inventory_groups != canonical_zero_size_groups:
                    raise ValueError("Stage190 inventories do not share one canonical mapping, five-group row-count dictionary, and zero-size group set")
                evidence.update({"stage190a_manifest_path": str(manifest_path), "manifest_run_identity": run, "stage190b_output_dir": str(output_dir), "stage190b_report_path": str(report_path), "inventory_path": str(inventory_path), "inventory_file_sha256": file_sha256(inventory_path), "mapping_sha256": mapping_sha, "parameter_count": len(mapping), "parameter_numel": sum(item["numel"] for item in mapping), "selected_checkpoint_sha256": report.get("selected_checkpoint_sha256"), "manifest_contract_passed": manifest_ok, "report_contract_passed": report_contract_ok, "conceptual_group_contract_passed": conceptual_group_contract_passed, "inventory_group_row_counts": inventory_group_row_counts, "observed_nonzero_inventory_groups": list(observed_nonzero_inventory_groups), "observed_zero_size_inventory_groups": list(observed_zero_size_inventory_groups), "zero_size_justifications": zero_size_justifications, "zero_size_justification_contract_passed": zero_size_justification_contract_passed, "topology_checks_passed": topology_ok, "per_run_mapping_passed": True, "passed": True, "mapping": mapping})
            except BaseException as exc:
                overall_passed = False
                evidence.update({"error_type": type(exc).__name__, "error_message": str(exc)})
            by_run[run] = evidence
    passed = overall_passed and canonical_mapping is not None and set(by_run) == set(RUNS) and all(item.get("passed") is True for item in by_run.values())
    return {"authoritative_stage190a_directory": str(stage190a), "canonical_mapping_sha256": canonical_mapping_sha256, "canonical_mapping": canonical_mapping, "canonical_group_row_counts": canonical_group_row_counts, "canonical_zero_size_groups": list(canonical_zero_size_groups) if canonical_zero_size_groups is not None else None, "by_run": by_run, "passed": passed}

def validate_internal_argv(argv: Any, label: str) -> None:
    options = option_map(argv)
    present_paths = [name for name in FORBIDDEN_PATH_OPTIONS if name in options]
    present_flags = [name for name in EXTERNAL_FLAGS if name in options]
    invalid_modes = {name: options[name] for name in BRIDGE_MODES if name in options and options[name] != "none"}
    if present_paths or present_flags or invalid_modes:
        raise ValueError(f"{label}: external/OOD/bridge option present: paths={present_paths}, flags={present_flags}, modes={invalid_modes}")


def run_report(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("runs")) is not dict:
        raise ValueError(f"{label}: training report has no exact runs object")
    runs = value["runs"]
    if set(runs) != {"single"} or type(runs["single"]) is not dict:
        raise ValueError(f"{label}: training report runs must contain exactly single")
    report = runs["single"]
    required = {"final_epoch", "best_epoch", "v7_epoch_diagnostic_history", "compatible_positive_margin"}
    if not required.issubset(report):
        raise ValueError(f"{label}: runs.single schema missing {sorted(required - set(report))}")
    return report


def prediction_rows(value: Any, label: str) -> list[dict[str, Any]]:
    if type(value) is not dict or set(("metadata", "predictions")) - set(value):
        raise ValueError(f"{label}: prediction artifact schema mismatch")
    rows = value["predictions"]
    if type(rows) is not list or len(rows) != 720 or any(type(row) is not dict for row in rows):
        raise ValueError(f"{label}: predictions must be exactly 720 objects")
    return rows


def historical_pair(row: dict[str, Any], label: str) -> tuple[str, str]:
    gold = row.get("gold_final_label")
    prediction = row.get("pred_final_label")
    if gold not in LABELS or prediction not in LABELS:
        raise ValueError(f"{label}: historical row lacks exact canonical labels")
    return gold, prediction


def normalize_historical_counts(value: Any, label: str) -> dict[str, int]:
    if type(value) is not dict or any(key not in LABELS for key in value):
        raise ValueError(f"{label}: historical prediction distribution has unknown labels")
    result = {name: value.get(name, 0) for name in LABELS}
    if any(not exact_int(count) or count < 0 for count in result.values()) or sum(result.values()) != 720:
        raise ValueError(f"{label}: historical prediction distribution is invalid")
    return result


def trajectory_metric_row(run: str, seed: int, arm: str, row: dict[str, Any]) -> dict[str, Any]:
    counts = row.get("normalized_prediction_counts")
    if type(counts) is not dict or set(counts) != set(LABELS):
        raise ValueError(f"{run}: trajectory prediction counts do not have the exact canonical key set")
    if any(not exact_int(counts[name]) or counts[name] < 0 for name in LABELS) or sum(counts.values()) != 720:
        raise ValueError(f"{run}: trajectory prediction counts invalid")
    for name in ("false_entitlement_total", "false_not_entitled_total", "polarity_error_total"):
        if not exact_int(row.get(name)) or row[name] < 0:
            raise ValueError(f"{run}: trajectory {name} must be an exact non-bool nonnegative integer")
    output = {"run": run, "seed": seed, "arm": arm, "epoch": row["epoch"]}
    for name in METRICS[:-3]:
        value = row.get(name)
        if not finite(value):
            raise ValueError(f"{run} epoch {row['epoch']}: non-finite {name}")
        output[name] = value
    output.update({f"pred_{name}": counts[name] for name in LABELS})
    output["selected_epoch"] = SELECTED[run]
    output["replaced_selected_checkpoint"] = row.get("current_epoch_replaced_selected_checkpoint")
    if type(output["replaced_selected_checkpoint"]) is not bool:
        raise ValueError(f"{run}: replacement flag is not bool")
    return output


def exported_rows(path: Path, epoch: int, run: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != 720:
        raise ValueError(f"{run} epoch {epoch}: prediction export row count is {len(rows)}")
    for position, row in enumerate(rows):
        if not exact_int(row.get("epoch")) or not exact_int(row.get("dev_position")) or row.get("epoch") != epoch or row.get("dev_position") != position:
            raise ValueError(f"{run} epoch {epoch}: dev_position/epoch mismatch at {position}")
        if row.get("gold_final_label") not in LABELS or row.get("predicted_final_label") not in LABELS:
            raise ValueError(f"{run} epoch {epoch}: noncanonical row label at {position}")
        logits = row.get("final_logits")
        if type(logits) is not list or len(logits) != 3 or any(not finite(value) for value in logits):
            raise ValueError(f"{run} epoch {epoch}: invalid final_logits at {position}")
        if not finite(row.get("final_ce")):
            raise ValueError(f"{run} epoch {epoch}: invalid final_ce at {position}")
        if "frame_logit" not in row:
            raise ValueError(f"{run} epoch {epoch}: frame_logit field is absent at {position}")
        frame_logit = row["frame_logit"]
        if frame_logit is not None and not finite(frame_logit):
            raise ValueError(f"{run} epoch {epoch}: invalid frame_logit at {position}")
        if "source_row_id" in row and row["source_row_id"] is not None and type(row["source_row_id"]) not in (str, int):
            raise ValueError(f"{run} epoch {epoch}: invalid optional source_row_id at {position}")
    return rows


def recompute_prediction_metrics(rows: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    final_ce_values: list[float] = []
    for row in rows:
        matrix[row["gold_final_label"]][row["predicted_final_label"]] += 1
        final_ce_values.append(float(row["final_ce"]))
    if len(final_ce_values) != 720:
        raise ValueError("authoritative clean CE reduction requires exactly 720 ordered final_ce values")
    reconstructed_torch_float32_mean = torch.tensor(final_ce_values, dtype=torch.float32, device="cpu").mean().item()
    diagnostic_python_float64_mean = sum(final_ce_values) / len(final_ce_values)
    gold_counts = {gold: sum(matrix[gold].values()) for gold in LABELS}
    counts = {pred: sum(matrix[gold][pred] for gold in LABELS) for pred in LABELS}
    correct = sum(matrix[label][label] for label in LABELS)
    f1s = []
    for label in LABELS:
        tp = matrix[label][label]
        predicted = counts[label]
        gold = gold_counts[label]
        precision = tp / predicted if predicted else 0.0
        recall = tp / gold if gold else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "reconstructed_torch_float32_mean": reconstructed_torch_float32_mean,
        "diagnostic_python_float64_mean": diagnostic_python_float64_mean,
        "counts": counts,
        "gold_counts": gold_counts,
        "clean_accuracy": correct / 720,
        "clean_macro_f1": sum(f1s) / 3,
        "support_recall": matrix["SUPPORT"]["SUPPORT"] / gold_counts["SUPPORT"],
        "false_entitlement_total": matrix["NOT_ENTITLED"]["REFUTE"] + matrix["NOT_ENTITLED"]["SUPPORT"],
        "false_not_entitled_total": matrix["REFUTE"]["NOT_ENTITLED"] + matrix["SUPPORT"]["NOT_ENTITLED"],
        "polarity_error_total": matrix["REFUTE"]["SUPPORT"] + matrix["SUPPORT"]["REFUTE"],
    }

def close_enough(left: Any, right: Any, tolerance: float = 1e-7) -> bool:
    return finite(left) and finite(right) and math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def transition_summary(previous: list[dict[str, Any]], nxt: list[dict[str, Any]], gold_filter: str | None = None) -> dict[str, Any]:
    if len(previous) != 720 or len(nxt) != 720:
        raise ValueError("transition inputs are not 720 rows")
    matrix = {left: {right: 0 for right in LABELS} for left in LABELS}
    corrections = regressions = wrong_different = 0
    for position, (before, after) in enumerate(zip(previous, nxt)):
        if before.get("dev_position") != position or after.get("dev_position") != position:
            raise ValueError("transition dev_position alignment failed")
        gold = before.get("gold_final_label")
        if gold != after.get("gold_final_label") or gold not in LABELS:
            raise ValueError("transition gold alignment failed")
        if gold_filter is not None and gold != gold_filter:
            continue
        old = before.get("predicted_final_label")
        new = after.get("predicted_final_label")
        if old not in LABELS or new not in LABELS:
            raise ValueError("transition prediction is noncanonical")
        matrix[old][new] += 1
        corrections += old != gold and new == gold
        regressions += old == gold and new != gold
        wrong_different += old != gold and new != gold and old != new
    total = sum(sum(row.values()) for row in matrix.values())
    unchanged = sum(matrix[label][label] for label in LABELS)
    changed = total - unchanged
    exclusive = matrix["NOT_ENTITLED"]["SUPPORT"] + matrix["SUPPORT"]["NOT_ENTITLED"]
    result = {f"{left}_to_{right}": matrix[left][right] for left in LABELS for right in LABELS}
    result.update({
        "unchanged_rows": unchanged,
        "changed_rows": changed,
        "not_entitled_to_support": matrix["NOT_ENTITLED"]["SUPPORT"],
        "support_to_not_entitled": matrix["SUPPORT"]["NOT_ENTITLED"],
        "refute_involved_transitions": sum(matrix[left][right] for left in LABELS for right in LABELS if left != right and "REFUTE" in (left, right)),
        "exclusive_not_entitled_support_changed_rows": exclusive,
        "exclusive_not_entitled_support_fraction_of_changed": exclusive / changed if changed else None,
        "corrections": corrections,
        "regressions": regressions,
        "wrong_to_different_wrong": wrong_different,
    })
    return result


def sign(value: Any) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def series_stats(values: list[float], selected_value: float, final_value: float) -> dict[str, Any]:
    changes = [right - left for left, right in zip(values, values[1:])]
    nonzero_signs = [sign(value) for value in changes if sign(value) != 0]
    return {
        "total_variation": sum(abs(value) for value in changes),
        "amplitude": max(values) - min(values),
        "direction_reversals": sum(left != right for left, right in zip(nonzero_signs, nonzero_signs[1:])),
        "selected_to_final_drift": final_value - selected_value,
    }


def tensor_state_sha256(items: list[tuple[str, Any]], torch: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in items:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"state value {name} is not a tensor")
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(cpu.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_capsule(path: Path, torch: Any) -> dict[str, Any]:
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RuntimeError("installed torch.load lacks required tensor-only weights_only mode")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if type(value) is not dict:
        raise ValueError(f"{path}: capsule is not an object")
    return value


def validate_capsule(capsule: dict[str, Any], path: Path, epoch: int, seed: int, arm: str, trajectory: dict[str, Any], torch: Any, expected_parameter_names: tuple[str, ...] | None, expected_buffer_names: tuple[str, ...] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required = {"epoch", "trainable_parameters", "buffers", "parameter_metadata", "buffer_metadata", "trainable_state_sha256", "buffer_state_sha256", "training_seed", "split_seed", "arm", "model_construction_provenance"}
    if set(capsule) != required:
        raise ValueError(f"{path}: exact capsule keys mismatch")
    if not all(exact_int(capsule.get(name)) for name in ("epoch", "training_seed", "split_seed")) or type(capsule.get("arm")) is not str or capsule["epoch"] != epoch or capsule["training_seed"] != seed or capsule["split_seed"] != 174 or capsule["arm"] != arm:
        raise ValueError(f"{path}: capsule identity mismatch")
    parameters = capsule["trainable_parameters"]
    buffers = capsule["buffers"]
    parameter_metadata = capsule["parameter_metadata"]
    buffer_metadata = capsule["buffer_metadata"]
    if not all(type(value) is dict for value in (parameters, buffers, parameter_metadata, buffer_metadata)):
        raise ValueError(f"{path}: state/metadata mapping mismatch")
    parameter_names = tuple(parameters)
    buffer_names = tuple(buffers)
    if not parameter_names or len(parameter_names) != len(set(parameter_names)) or len(buffer_names) != len(set(buffer_names)):
        raise ValueError(f"{path}: empty/duplicate state names")
    if set(parameter_names) & set(buffer_names):
        raise ValueError(f"{path}: parameter and buffer key sets overlap")
    if set(parameter_metadata) != set(parameter_names) or set(buffer_metadata) != set(buffer_names):
        raise ValueError(f"{path}: metadata key set mismatch")
    if expected_parameter_names is not None and parameter_names != expected_parameter_names:
        raise ValueError(f"{path}: ordered parameter keys changed")
    if expected_buffer_names is not None and buffer_names != expected_buffer_names:
        raise ValueError(f"{path}: ordered buffer keys changed")
    for name, tensor in list(parameters.items()) + list(buffers.items()):
        metadata = parameter_metadata[name] if name in parameters else buffer_metadata[name]
        if type(metadata) is not dict or set(metadata) != {"dtype", "shape"} or type(metadata.get("dtype")) is not str or type(metadata.get("shape")) is not list or any(not exact_int(dimension) or dimension < 0 for dimension in metadata["shape"]):
            raise ValueError(f"{path}: metadata schema mismatch for {name}")
        if metadata["dtype"] != str(tensor.dtype) or metadata["shape"] != list(tensor.shape):
            raise ValueError(f"{path}: dtype/shape metadata mismatch for {name}")
    trainable_sha = tensor_state_sha256(list(parameters.items()), torch)
    buffer_sha = tensor_state_sha256(list(buffers.items()), torch)
    if trainable_sha != capsule["trainable_state_sha256"] or trainable_sha != trajectory.get("trainable_state_sha256"):
        raise ValueError(f"{path}: trainable-state SHA256 mismatch")
    if buffer_sha != capsule["buffer_state_sha256"] or buffer_sha != trajectory.get("buffer_state_sha256"):
        raise ValueError(f"{path}: buffer-state SHA256 mismatch")
    if file_sha256(path) != trajectory.get("capsule_file_sha256"):
        raise ValueError(f"{path}: capsule file SHA256 mismatch")
    if not isinstance(capsule["model_construction_provenance"], dict):
        raise ValueError(f"{path}: construction provenance missing")
    return parameter_names, buffer_names


def squared_norm(tensors: Iterable[Any], torch: Any) -> float:
    total = 0.0
    for tensor in tensors:
        value = tensor.detach().to(dtype=torch.complex128 if tensor.is_complex() else torch.float64)
        total += float((value.conj() * value).real.sum().item())
    return total


def difference_squared(left: dict[str, Any], right: dict[str, Any], torch: Any) -> float:
    total = 0.0
    for name in left:
        if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype:
            raise ValueError(f"state topology changed for {name}")
        dtype = torch.complex128 if left[name].is_complex() else torch.float64
        delta = right[name].detach().to(dtype=dtype) - left[name].detach().to(dtype=dtype)
        total += float((delta.conj() * delta).real.sum().item())
    return total


def analyze_capsules(run_data: dict[str, Any], state_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]], inventory_evidence: dict[str, Any]) -> dict[str, Any]:
    import torch

    run = run_data["run"]
    seed = run_data["seed"]
    arm = run_data["arm"]
    paths = run_data["capsule_paths"]
    trajectory = run_data["trajectory"]
    expected_parameters: tuple[str, ...] | None = None
    expected_buffers: tuple[str, ...] | None = None
    group_by_name: dict[str, str] = {}
    parameter_metadata: dict[str, Any] = {}
    step_squared: dict[int, float] = {}
    step_rows_by_epoch: dict[int, dict[str, Any]] = {}

    previous = load_capsule(paths[1], torch)
    expected_parameters, expected_buffers = validate_capsule(previous, paths[1], 1, seed, arm, trajectory[1], torch, None, None)
    parameter_metadata = previous["parameter_metadata"]
    inventory_mapping = inventory_evidence.get("mapping")
    if inventory_evidence.get("passed") is not True or type(inventory_mapping) is not list:
        raise ValueError(f"{run}: exact Stage190 inventory mapping is unavailable")
    inventory_names = tuple(item.get("parameter_name") for item in inventory_mapping if type(item) is dict)
    if inventory_names != expected_parameters or len(inventory_names) != len(set(inventory_names)):
        raise ValueError(f"{run}: capsule ordered trainable names differ from exact Stage190 inventory")
    group_by_name: dict[str, str] = {}
    for name, item in zip(expected_parameters, inventory_mapping):
        metadata = parameter_metadata[name]
        tensor = previous["trainable_parameters"][name]
        if metadata.get("shape") != item.get("shape") or metadata.get("dtype") != item.get("dtype") or int(tensor.numel()) != item.get("numel"):
            raise ValueError(f"{run}: capsule topology differs from Stage190 inventory for {name}")
        group = item.get("stage191d_group")
        if group not in GROUPS or item.get("stage190_group") not in STAGE190_TO_STAGE191D_ALIASES or STAGE190_TO_STAGE191D_ALIASES[item["stage190_group"]] != group:
            raise ValueError(f"{run}: Stage190 inventory group alias is invalid for {name}")
        group_by_name[name] = group
    if set(group_by_name) != set(expected_parameters):
        raise ValueError(f"{run}: exact Stage190 inventory ownership is not exhaustive")

    for epoch in range(2, 21):
        current = load_capsule(paths[epoch], torch)
        validate_capsule(current, paths[epoch], epoch, seed, arm, trajectory[epoch], torch, expected_parameters, expected_buffers)
        if current["parameter_metadata"] != parameter_metadata:
            raise ValueError(f"{run}: parameter metadata changed at epoch {epoch}")
        prior_state_sq = squared_norm(previous["trainable_parameters"].values(), torch)
        total_sq = difference_squared(previous["trainable_parameters"], current["trainable_parameters"], torch)
        buffer_sq = difference_squared(previous["buffers"], current["buffers"], torch)
        step_squared[epoch] = total_sq
        row = {
            "run": run,
            "seed": seed,
            "arm": arm,
            "previous_epoch": epoch - 1,
            "next_epoch": epoch,
            "trainable_step_squared_l2": total_sq,
            "trainable_step_l2": math.sqrt(total_sq),
            "prior_state_l2": math.sqrt(prior_state_sq),
            "normalized_step_norm": math.sqrt(total_sq / prior_state_sq) if prior_state_sq else None,
            "buffer_step_l2": math.sqrt(buffer_sq),
            "cosine_with_preceding_step": None,
            "path_length_epoch15_to_20": None,
            "direct_displacement_epoch15_to_20": None,
            "path_length_direct_displacement_ratio": None,
        }
        state_rows.append(row)
        step_rows_by_epoch[epoch] = row
        group_sq = {group: 0.0 for group in GROUPS}
        group_numel = {group: 0 for group in GROUPS}
        group_tensors = {group: 0 for group in GROUPS}
        for name in expected_parameters:
            group = group_by_name[name]
            left = previous["trainable_parameters"][name]
            right = current["trainable_parameters"][name]
            delta_sq = difference_squared({name: left}, {name: right}, torch)
            group_sq[group] += delta_sq
            group_numel[group] += int(left.numel())
            group_tensors[group] += 1
        if not math.isclose(sum(group_sq.values()), total_sq, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{run}: group squared steps do not sum to total")
        for group in GROUPS:
            fraction_total_squared = group_sq[group] / total_sq if total_sq else None
            if group_tensors[group] == 0 and (group_sq[group] != 0.0 or group_numel[group] != 0 or (total_sq != 0.0 and fraction_total_squared != 0.0)):
                raise ValueError(f"{run}: zero-size reporting group {group} has nonzero state-step evidence")
            group_rows.append({
                "run": run,
                "seed": seed,
                "arm": arm,
                "previous_epoch": epoch - 1,
                "next_epoch": epoch,
                "parameter_group": group,
                "squared_l2_step_norm": group_sq[group],
                "l2_step_norm": math.sqrt(group_sq[group]),
                "fraction_total_squared_step_norm": fraction_total_squared,
                "parameter_count": group_numel[group],
                "tensor_count": group_tensors[group],
                "grouping_contract": inventory_evidence["inventory_path"],
            })
        previous = current

    for epoch in range(3, 21):
        earlier = load_capsule(paths[epoch - 2], torch)
        later = load_capsule(paths[epoch], torch)
        two_step_sq = difference_squared(earlier["trainable_parameters"], later["trainable_parameters"], torch)
        denominator = math.sqrt(step_squared[epoch - 1] * step_squared[epoch])
        cosine = (two_step_sq - step_squared[epoch - 1] - step_squared[epoch]) / (2 * denominator) if denominator else None
        if cosine is not None:
            cosine = min(1.0, max(-1.0, cosine))
        step_rows_by_epoch[epoch]["cosine_with_preceding_step"] = cosine

    epoch15 = load_capsule(paths[15], torch)
    epoch20 = load_capsule(paths[20], torch)
    direct = math.sqrt(difference_squared(epoch15["trainable_parameters"], epoch20["trainable_parameters"], torch))
    path_length = sum(math.sqrt(step_squared[epoch]) for epoch in range(16, 21))
    ratio = path_length / direct if direct else (1.0 if path_length == 0 else None)
    step_rows_by_epoch[20].update({
        "path_length_epoch15_to_20": path_length,
        "direct_displacement_epoch15_to_20": direct,
        "path_length_direct_displacement_ratio": ratio,
    })
    return {
        "integrity_passed": True,
        "parameter_tensor_count": len(expected_parameters),
        "buffer_tensor_count": len(expected_buffers),
        "parameter_groups": {group: sum(owner == group for owner in group_by_name.values()) for group in GROUPS},
        "ownership_exhaustive_disjoint": set(group_by_name) == set(expected_parameters) and len(group_by_name) == len(expected_parameters) and all(owner in GROUPS for owner in group_by_name.values()),
        "stage190_inventory_path": inventory_evidence["inventory_path"],
        "stage190_inventory_file_sha256": inventory_evidence["inventory_file_sha256"],
        "stage190_mapping_sha256": inventory_evidence["mapping_sha256"],
        "stage190_mapping_passed": inventory_evidence["passed"],
        "path_length_epoch15_to_20": path_length,
        "direct_displacement_epoch15_to_20": direct,
        "path_length_direct_displacement_ratio": ratio,
    }


def add_gate(rows: list[dict[str, Any]], blockers: list[str], gate: str, run: str, required: Any, observed: Any, passed: bool, reason: str) -> None:
    rows.append({"gate": gate, "run": run, "required": required, "observed": observed, "passed": passed, "blocking_reason": "" if passed else reason})
    if not passed:
        blockers.append(f"{run + ': ' if run else ''}{gate}: {reason}")


def analyze(args: argparse.Namespace, tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    import torch

    repo = args.repo_root.resolve()
    stage191b = args.stage191b_dir.resolve()
    output = args.output_dir.resolve()
    blockers: list[str] = []
    gates = tables["equivalence"]
    expected_stage191b = (repo / "reports" / STAGE191B_DIRNAME).resolve()
    add_gate(gates, blockers, "exact_stage191b_input_directory", "", str(expected_stage191b), str(stage191b), stage191b == expected_stage191b, "Stage191-B input does not resolve to the frozen repository path")
    diagnostic_source_identity = validate_diagnostic_source_identity(repo, args.current_diagnostic_git_commit)
    add_gate(gates, blockers, "diagnostic_source_commit_identity", "", True, diagnostic_source_identity, diagnostic_source_identity["passed"], "diagnostic source files or repository HEAD differ from supplied commit")
    grouping_source_contract = validate_stage190_grouping_source(repo)
    add_gate(gates, blockers, "stage190_grouping_source_contract", "", True, grouping_source_contract, grouping_source_contract["source_contract_passed"], "authoritative Stage190 grouping source contract mismatch")
    stage190_inventory_contract = load_stage190_parameter_inventories(repo)
    stage190_inventory_summary = {
        "authoritative_stage190a_directory": stage190_inventory_contract["authoritative_stage190a_directory"],
        "canonical_mapping_sha256": stage190_inventory_contract["canonical_mapping_sha256"],
        "canonical_mapping": stage190_inventory_contract["canonical_mapping"],
        "canonical_group_row_counts": stage190_inventory_contract["canonical_group_row_counts"],
        "canonical_zero_size_groups": stage190_inventory_contract["canonical_zero_size_groups"],
        "canonical_parameter_count": len(stage190_inventory_contract["canonical_mapping"] or []),
        "by_run": {run: {key: value for key, value in evidence.items() if key != "mapping"} for run, evidence in stage190_inventory_contract["by_run"].items()},
        "passed": stage190_inventory_contract["passed"],
    }
    add_gate(gates, blockers, "stage190_exact_parameter_inventory_mapping", "", True, stage190_inventory_summary, stage190_inventory_contract["passed"], "exact six-run Stage190-B parameter inventory mapping failed")
    authoritative_sidecar = (repo / STAGE185_SIDECAR_RELATIVE_PATH).resolve()
    sidecar_file_ok = authoritative_sidecar == (repo / "reports" / "stage185a_controlled_train_integrity_sidecar_20260715_141914" / "stage185a_controlled_train_integrity_sidecar.jsonl").resolve() and authoritative_sidecar.is_file()
    add_gate(gates, blockers, "authoritative_stage185_sidecar_path", "", str(authoritative_sidecar), {"path": str(authoritative_sidecar), "is_regular_file": authoritative_sidecar.is_file()}, sidecar_file_ok, "frozen Stage185 sidecar path is absent or not a regular file")
    observed_sidecar_semantic_sha256 = semantic_sidecar_sha256(authoritative_sidecar) if sidecar_file_ok else None
    add_gate(gates, blockers, "authoritative_stage185_sidecar_semantic_sha256", "", SIDECAR_SHA256, observed_sidecar_semantic_sha256, observed_sidecar_semantic_sha256 == SIDECAR_SHA256, "frozen Stage185 sidecar semantic SHA256 mismatch")

    manifest_report = read_json(stage191b / "stage191b_deterministic_replay_manifest_report.json")
    report_contract = {
        "decision": manifest_report.get("decision"),
        "runnable": manifest_report.get("runnable"),
        "blocking_reasons": manifest_report.get("blocking_reasons"),
        "diagnostic_replay_only": manifest_report.get("diagnostic_replay_only"),
        "replay_execution_authorized": manifest_report.get("replay_execution_authorized"),
        "training_for_model_advancement_authorized": manifest_report.get("training_for_model_advancement_authorized"),
        "model_advancement_decision": manifest_report.get("model_advancement_decision"),
        "external_data_used": manifest_report.get("external_data_used"),
        "authorized_training_seeds": manifest_report.get("authorized_training_seeds"),
        "stage191b_commit": (manifest_report.get("commit_identities") or {}).get("stage191b_replay_commit"),
    }
    required_report_contract = {"decision": STAGE191B_READY, "runnable": True, "blocking_reasons": [], "diagnostic_replay_only": True, "replay_execution_authorized": True, "training_for_model_advancement_authorized": False, "model_advancement_decision": False, "external_data_used": False, "authorized_training_seeds": list(SEEDS), "stage191b_commit": STAGE191B_COMMIT}
    add_gate(gates, blockers, "stage191b_manifest_report_contract", "", required_report_contract, report_contract, report_contract == required_report_contract, "Stage191-B manifest report contract mismatch")
    expected_six_identities = [{"run": f"seed{seed}_{arm}", "seed": seed, "arm": arm, "split_seed": 174} for seed in SEEDS for arm in ARMS]
    raw_matrix = manifest_report.get("six_run_matrix")
    observed_six_identities = [{key: row.get(key) for key in ("run", "seed", "arm", "split_seed")} for row in raw_matrix] if type(raw_matrix) is list and all(type(row) is dict for row in raw_matrix) else None
    six_identities_ok = observed_six_identities == expected_six_identities and all(exact_int(row["seed"]) and exact_int(row["split_seed"]) for row in observed_six_identities or [])
    add_gate(gates, blockers, "stage191b_exact_six_run_identities", "", expected_six_identities, observed_six_identities, six_identities_ok, "Stage191-B six-run identities mismatch")
    observed_checkpoint_map = manifest_report.get("frozen_selected_checkpoint_sha256_by_run")
    add_gate(gates, blockers, "stage191b_frozen_selected_checkpoint_sha_map", "", FROZEN_SELECTED_CHECKPOINT_SHA256, observed_checkpoint_map, observed_checkpoint_map == FROZEN_SELECTED_CHECKPOINT_SHA256, "Stage191-B frozen checkpoint SHA map mismatch")
    prediction_reference_identities = manifest_report.get("selected_prediction_reference_identity_by_run")
    prediction_reference_keys_ok = type(prediction_reference_identities) is dict and set(prediction_reference_identities) == set(RUNS)
    add_gate(gates, blockers, "stage191b_selected_prediction_reference_identity_keys", "", sorted(RUNS), sorted(prediction_reference_identities) if type(prediction_reference_identities) is dict else None, prediction_reference_keys_ok, "selected prediction reference identity keys mismatch")
    run_data: dict[str, dict[str, Any]] = {}
    gold_reference: list[str] | None = None
    for seed in SEEDS:
        for arm in ARMS:
            run = f"seed{seed}_{arm}"
            run_dir = stage191b / run
            manifest_path = stage191b / f"stage191b_{run}_replay_manifest.json"
            manifest = read_json(manifest_path)
            manifest_contract = {
                "run": manifest.get("run"), "seed": manifest.get("seed"), "training_seed": manifest.get("training_seed"), "split_seed": manifest.get("split_seed"), "arm": manifest.get("arm"),
                "runnable": manifest.get("runnable"), "blocking_reasons": manifest.get("blocking_reasons"), "diagnostic_replay_only": manifest.get("diagnostic_replay_only"), "replay_execution_authorized": manifest.get("replay_execution_authorized"),
                "training_for_model_advancement_authorized": manifest.get("training_for_model_advancement_authorized"), "model_advancement_decision": manifest.get("model_advancement_decision"), "external_data_used": manifest.get("external_data_used"),
                "stage191b_commit": (manifest.get("commit_identities") or {}).get("stage191b_replay_commit"), "selected_epoch": manifest.get("original_selected_epoch"),
                "expected_trajectory_rows": manifest.get("expected_trajectory_rows"), "expected_prediction_rows_per_epoch": manifest.get("expected_prediction_rows_per_epoch"), "expected_state_capsules": manifest.get("expected_state_capsules"), "logits_source": manifest.get("logits_source"),
            }
            required_manifest = {"run": run, "seed": seed, "training_seed": seed, "split_seed": 174, "arm": arm, "runnable": True, "blocking_reasons": [], "diagnostic_replay_only": True, "replay_execution_authorized": True, "training_for_model_advancement_authorized": False, "model_advancement_decision": False, "external_data_used": False, "stage191b_commit": STAGE191B_COMMIT, "selected_epoch": SELECTED[run], "expected_trajectory_rows": 20, "expected_prediction_rows_per_epoch": 720, "expected_state_capsules": 20, "logits_source": 'output["logits"]'}
            manifest_types_ok = exact_int(manifest.get("seed")) and exact_int(manifest.get("training_seed")) and exact_int(manifest.get("split_seed")) and exact_int(manifest.get("original_selected_epoch")) and type(manifest.get("run")) is str and type(manifest.get("arm")) is str
            add_gate(gates, blockers, "per_run_manifest_contract", run, required_manifest, manifest_contract, manifest_contract == required_manifest and manifest_types_ok, "per-run manifest identity/authorization mismatch")
            checkpoint_identity_ok = manifest.get("original_selected_checkpoint_sha256") == FROZEN_SELECTED_CHECKPOINT_SHA256[run]
            add_gate(gates, blockers, "per_run_frozen_selected_checkpoint_sha256", run, FROZEN_SELECTED_CHECKPOINT_SHA256[run], manifest.get("original_selected_checkpoint_sha256"), checkpoint_identity_ok, "per-run selected checkpoint differs from frozen map")
            reference_identity = prediction_reference_identities.get(run) if type(prediction_reference_identities) is dict else None
            reference_identity = reference_identity if type(reference_identity) is dict else {}
            required_prediction_identity = {
                "path": reference_identity.get("validated_prediction_path"),
                "sha256": reference_identity.get("validated_prediction_sha256"),
                "passed": reference_identity.get("passed"),
            }
            observed_prediction_identity = {
                "path": manifest.get("original_selected_prediction_path"),
                "sha256": manifest.get("original_selected_prediction_sha256"),
                "passed": reference_identity.get("passed"),
            }
            prediction_identity_ok = reference_identity.get("passed") is True and exact_sha(reference_identity.get("validated_prediction_sha256")) and type(reference_identity.get("validated_prediction_path")) is str and observed_prediction_identity == required_prediction_identity
            add_gate(gates, blockers, "per_run_selected_prediction_reference_matches_main_report", run, required_prediction_identity, observed_prediction_identity, prediction_identity_ok, "per-run selected prediction identity differs from validated Stage191-B main-report identity")
            expected_run_dir = run_dir.resolve()
            observed_replay_dir = Path(str(manifest.get("replay_output_directory", ""))).resolve()
            add_gate(gates, blockers, "exact_replay_output_directory", run, str(expected_run_dir), str(observed_replay_dir), observed_replay_dir == expected_run_dir, "manifest replay directory is not the exact Stage191-B child")
            validate_internal_argv(manifest.get("original_argv"), f"{run} original argv")
            validate_internal_argv(manifest.get("argv"), f"{run} replay argv")

            exact_names = {"stage191_trajectory_contract.json", "stage191_trajectory_epoch_metrics.jsonl", "training_report.json", "clean_dev_predictions.json"}
            for epoch in EPOCHS:
                exact_names.add(f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl")
                exact_names.add(f"stage191_trajectory_state_epoch_{epoch:03d}.pt")
            observed_required = {path.name for path in run_dir.iterdir() if path.is_file() and (path.name in exact_names or path.name.startswith("stage191_dev_predictions_epoch_") or path.name.startswith("stage191_trajectory_state_epoch_"))}
            add_gate(gates, blockers, "exact_required_replay_files", run, sorted(exact_names), sorted(observed_required), observed_required == exact_names, "required replay file set/cardinality mismatch")

            contract = read_json(run_dir / "stage191_trajectory_contract.json")
            sidecar = contract.get("sidecar_runtime_configuration") if type(contract.get("sidecar_runtime_configuration")) is dict else {}
            contract_common = contract.get("trainer_source_commit") == STAGE191B_COMMIT and contract.get("authorized_training_seeds") == list(SEEDS) and exact_int(contract.get("training_seed")) and exact_int(contract.get("split_seed")) and exact_int(contract.get("epoch_count")) and exact_int(contract.get("expected_dev_rows")) and type(contract.get("arm")) is str and contract.get("training_seed") == seed and contract.get("split_seed") == 174 and contract.get("arm") == arm and contract.get("canonical_logit_column_labels") == list(LABELS) and contract.get("epoch_count") == 20 and contract.get("expected_dev_rows") == 720 and contract.get("external_data_used") is False and contract.get("training_semantics_changed") is False and contract.get("extra_forward_pass_performed") is False and contract.get("loss_logits_used") is False and contract.get("enabled_flags") == {"stage191_trajectory_replay_observability": True, "stage191_save_trajectory_state_capsules": True}
            if arm == "baseline":
                sidecar_ok = sidecar == {"validated": True, "enabled": False, "accessed": False, "configured_weight": 0.0, "configured_margin_logit": 0.0, "resolved_path": None, "expected_semantic_sha256": None, "observed_semantic_sha256": None}
                sidecar_ok = sidecar_ok and manifest.get("baseline_sidecar_training_access") is False
            else:
                manifest_sidecar = manifest.get("intervention_sidecar_exact_access_contract")
                manifest_sidecar = manifest_sidecar if type(manifest_sidecar) is dict else {}
                configured_path = manifest_sidecar.get("controlled_integrity_sidecar_path")
                sidecar_ok = sidecar.get("validated") is True and sidecar.get("enabled") is True and sidecar.get("accessed") is True and sidecar.get("configured_weight") == 0.05 and sidecar.get("configured_margin_logit") == 0.0 and type(sidecar.get("resolved_path")) is str and type(configured_path) is str and Path(sidecar["resolved_path"]).resolve() == authoritative_sidecar and Path(configured_path).resolve() == authoritative_sidecar and authoritative_sidecar.is_file() and manifest_sidecar.get("expected_integrity_sidecar_semantic_sha256") == SIDECAR_SHA256 and sidecar.get("expected_semantic_sha256") == SIDECAR_SHA256 and sidecar.get("observed_semantic_sha256") == SIDECAR_SHA256
            add_gate(gates, blockers, "trajectory_contract_and_sidecar", run, True, {"common": contract_common, "sidecar": sidecar}, contract_common and sidecar_ok, "trajectory or sidecar runtime contract mismatch")

            trajectory_rows = read_jsonl(run_dir / "stage191_trajectory_epoch_metrics.jsonl")
            trajectory = {row.get("epoch"): row for row in trajectory_rows}
            epochs_ok = len(trajectory_rows) == 20 and len(trajectory) == 20 and tuple(sorted(trajectory)) == EPOCHS and all(exact_int(row.get("epoch")) for row in trajectory_rows)
            add_gate(gates, blockers, "epochs_exact_1_through_20", run, list(EPOCHS), [row.get("epoch") for row in trajectory_rows], epochs_ok, "trajectory epochs are not exactly 1 through 20")
            if not epochs_ok:
                raise ValueError(f"{run}: cannot safely index invalid trajectory epochs")

            predictions: dict[int, list[dict[str, Any]]] = {}
            capsule_paths: dict[int, Path] = {}
            reconstructed_ce_evidence: list[dict[str, Any]] = []
            reconstructed_non_ce_evidence: list[dict[str, Any]] = []
            reconstructed_ce_passed = True
            reconstructed_non_ce_metrics_passed = True
            for epoch in EPOCHS:
                trajectory_row = trajectory[epoch]
                prediction_path = run_dir / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl"
                capsule_path = run_dir / f"stage191_trajectory_state_epoch_{epoch:03d}.pt"
                if Path(str(trajectory_row.get("prediction_export_path", ""))).resolve() != prediction_path.resolve():
                    raise ValueError(f"{run} epoch {epoch}: prediction export path mismatch")
                if Path(str(trajectory_row.get("capsule_path", ""))).resolve() != capsule_path.resolve():
                    raise ValueError(f"{run} epoch {epoch}: capsule path mismatch")
                if file_sha256(prediction_path) != trajectory_row.get("prediction_export_sha256"):
                    raise ValueError(f"{run} epoch {epoch}: prediction export SHA256 mismatch")
                if file_sha256(capsule_path) != trajectory_row.get("capsule_file_sha256"):
                    raise ValueError(f"{run} epoch {epoch}: capsule file SHA256 mismatch")
                rows = exported_rows(prediction_path, epoch, run)
                computed = recompute_prediction_metrics(rows, torch)
                trajectory_ce = trajectory_row.get("clean_dev_ce")
                if not finite(trajectory_ce):
                    raise ValueError(f"{run} epoch {epoch}: trajectory clean_dev_ce is not finite numeric")
                exact_float32_match = computed["reconstructed_torch_float32_mean"] == trajectory_ce
                reconstructed_ce_passed = reconstructed_ce_passed and exact_float32_match
                reconstructed_ce_evidence.append({
                    "epoch": epoch,
                    "final_ce_row_count": len(rows),
                    "authoritative_reduction_dtype": "torch.float32",
                    "authoritative_reduction_device": "cpu",
                    "reconstructed_torch_float32_mean": computed["reconstructed_torch_float32_mean"],
                    "trajectory_clean_dev_ce": trajectory_ce,
                    "exact_float32_match": exact_float32_match,
                    "diagnostic_python_float64_mean": computed["diagnostic_python_float64_mean"],
                    "diagnostic_python_float64_abs_diff": abs(computed["diagnostic_python_float64_mean"] - float(trajectory_ce)),
                })
                dense = trajectory_row.get("normalized_prediction_counts")
                trajectory_gold = trajectory_row.get("gold_counts")
                gold_ok = sum(computed["gold_counts"].values()) == 720 and computed["gold_counts"]["SUPPORT"] == 89 and type(trajectory_gold) is dict and set(trajectory_gold) == set(LABELS) and all(exact_int(trajectory_gold[label]) for label in LABELS) and trajectory_gold == computed["gold_counts"]
                integer_ok = computed["counts"] == dense and all(exact_int(trajectory_row.get(name)) and computed[name] == trajectory_row.get(name) for name in ("false_entitlement_total", "false_not_entitled_total", "polarity_error_total"))
                non_ce_float_ok = all(close_enough(computed[name], trajectory_row.get(name)) for name in ("clean_accuracy", "clean_macro_f1", "support_recall"))
                epoch_non_ce_reconstruction_ok = gold_ok and integer_ok and non_ce_float_ok
                reconstructed_non_ce_metrics_passed = reconstructed_non_ce_metrics_passed and epoch_non_ce_reconstruction_ok
                reconstructed_non_ce_evidence.append({"epoch": epoch, "strict_epoch_export_schema_passed": True, "prediction_counts_and_integer_errors_exact": integer_ok, "floating_metrics_within_tolerance": non_ce_float_ok, "gold_total": sum(computed["gold_counts"].values()), "gold_support": computed["gold_counts"]["SUPPORT"], "gold_topology_exact": gold_ok, "passed": epoch_non_ce_reconstruction_ok})
                predictions[epoch] = rows
                capsule_paths[epoch] = capsule_path
                tables["epochs"].append(trajectory_metric_row(run, seed, arm, trajectory_row))
            add_gate(gates, blockers, "reconstructed_clean_dev_ce_matches", run, CLEAN_DEV_CE_REDUCTION_CONTRACT, reconstructed_ce_evidence, reconstructed_ce_passed, "torch.float32 CPU mean of ordered exported final_ce differs exactly from trajectory clean_dev_ce")
            add_gate(gates, blockers, "epoch_export_non_ce_metrics_and_gold_topology", run, True, reconstructed_non_ce_evidence, reconstructed_non_ce_metrics_passed, "non-CE reconstructed metrics, strict epoch-export schema, or exact gold topology mismatch")
            golds = [row["gold_final_label"] for row in predictions[1]]
            if any([row["gold_final_label"] for row in predictions[epoch]] != golds for epoch in EPOCHS):
                raise ValueError(f"{run}: gold labels changed across epochs")
            if gold_reference is None:
                gold_reference = golds
            elif gold_reference != golds:
                raise ValueError(f"{run}: exact dev-position gold alignment differs across runs")

            replay_training = read_json(run_dir / "training_report.json")
            replay_single = run_report(replay_training, f"{run} replay")
            original_dir = Path(str(manifest.get("original_stage189_run_directory", ""))).resolve()
            original_checkpoint_path = (original_dir / "selected_checkpoint.pt").resolve()
            original_checkpoint_sha256 = file_sha256(original_checkpoint_path) if original_dir.is_dir() and original_checkpoint_path.is_file() else None
            checkpoint_artifact_evidence = {"original_stage189_run_directory": str(original_dir), "directory_exists": original_dir.is_dir(), "checkpoint_path": str(original_checkpoint_path), "checkpoint_is_regular_file": original_checkpoint_path.is_file(), "current_checkpoint_sha256": original_checkpoint_sha256, "frozen_checkpoint_sha256": FROZEN_SELECTED_CHECKPOINT_SHA256[run], "manifest_checkpoint_sha256": manifest.get("original_selected_checkpoint_sha256"), "stage191b_main_report_checkpoint_sha256": observed_checkpoint_map.get(run) if type(observed_checkpoint_map) is dict else None}
            checkpoint_artifact_exact = original_dir.is_dir() and original_checkpoint_path.parent == original_dir and original_checkpoint_path.name == "selected_checkpoint.pt" and original_checkpoint_path.is_file() and original_checkpoint_sha256 == FROZEN_SELECTED_CHECKPOINT_SHA256[run] == manifest.get("original_selected_checkpoint_sha256") == (observed_checkpoint_map.get(run) if type(observed_checkpoint_map) is dict else None)
            add_gate(gates, blockers, "original_selected_checkpoint_artifact_exact", run, FROZEN_SELECTED_CHECKPOINT_SHA256[run], checkpoint_artifact_evidence, checkpoint_artifact_exact, "original selected checkpoint artifact path or current SHA256 mismatch")
            original_training_path = original_dir / "training_report.json"
            original_predictions_path = Path(str(manifest.get("original_selected_prediction_path", ""))).resolve()
            if original_predictions_path != original_dir / "clean_dev_predictions.json":
                raise ValueError(f"{run}: original selected prediction path is not exact")
            if file_sha256(original_predictions_path) != manifest.get("original_selected_prediction_sha256"):
                raise ValueError(f"{run}: original selected prediction SHA256 mismatch")
            original_single = run_report(read_json(original_training_path), f"{run} original")
            clean_history_equal = replay_single["v7_epoch_diagnostic_history"] == original_single["v7_epoch_diagnostic_history"]
            replay_margin = replay_single["compatible_positive_margin"]
            original_margin = original_single["compatible_positive_margin"]
            margin_equal = type(replay_margin) is dict and type(original_margin) is dict and replay_margin.get("epoch_metrics") == original_margin.get("epoch_metrics")
            add_gate(gates, blockers, "clean_history_exact", run, "exact original Stage189 history", clean_history_equal, clean_history_equal, "clean history differs from original Stage189 report")
            add_gate(gates, blockers, "compatible_positive_margin_history_exact", run, "exact original Stage189 margin epoch_metrics", margin_equal, margin_equal, "margin history differs from original Stage189 report")
            history = original_single["v7_epoch_diagnostic_history"]
            history_ok = type(history) is list and len(history) == 20 and [row.get("epoch") for row in history if type(row) is dict] == list(EPOCHS)
            if not history_ok:
                raise ValueError(f"{run}: historical clean history topology mismatch")
            historical_macro_ok = all(history[epoch - 1].get("dev_final_macro_f1") == trajectory[epoch].get("clean_macro_f1") for epoch in EPOCHS)
            historical_counts_ok = all(normalize_historical_counts(history[epoch - 1].get("dev_prediction_distribution"), f"{run} epoch {epoch}") == trajectory[epoch].get("normalized_prediction_counts") for epoch in EPOCHS)
            add_gate(gates, blockers, "historical_macro_f1_exact_all_epochs", run, True, historical_macro_ok, historical_macro_ok, "historical macro-F1 differs")
            add_gate(gates, blockers, "historical_sparse_counts_normalized_exact_all_epochs", run, True, historical_counts_ok, historical_counts_ok, "historical prediction counts differ")
            selection_ok = all(exact_int(value) for value in (replay_single.get("best_epoch"), original_single.get("best_epoch"), replay_single.get("final_epoch"), original_single.get("final_epoch"))) and replay_single.get("best_epoch") == original_single.get("best_epoch") == SELECTED[run] and replay_single.get("final_epoch") == original_single.get("final_epoch") == 20
            add_gate(gates, blockers, "selected_and_final_epoch_exact", run, {"selected": SELECTED[run], "final": 20}, {"replay_selected": replay_single.get("best_epoch"), "original_selected": original_single.get("best_epoch"), "replay_final": replay_single.get("final_epoch"), "original_final": original_single.get("final_epoch")}, selection_ok, "selected/final epoch mismatch")
            original_prediction_artifact = read_json(original_predictions_path)
            replay_prediction_artifact = read_json(run_dir / "clean_dev_predictions.json")
            original_prediction_rows = prediction_rows(original_prediction_artifact, f"{run} original selected")
            replay_clean_rows = prediction_rows(replay_prediction_artifact, f"{run} replay selected")
            original_pairs = [historical_pair(row, f"{run} original row {index}") for index, row in enumerate(original_prediction_rows)]
            replay_pairs = [historical_pair(row, f"{run} replay row {index}") for index, row in enumerate(replay_clean_rows)]
            selected_pairs = [(row["gold_final_label"], row["predicted_final_label"]) for row in predictions[SELECTED[run]]]
            artifact_exact = original_prediction_artifact == replay_prediction_artifact
            selected_pair_equivalence = original_pairs == replay_pairs == selected_pairs
            add_gate(gates, blockers, "selected_clean_prediction_artifact_exact", run, True, artifact_exact, artifact_exact, "complete original and replay clean prediction artifacts differ structurally")
            add_gate(gates, blockers, "selected_epoch_export_label_pairs_exact", run, True, selected_pair_equivalence, selected_pair_equivalence, "selected epoch export label pairs differ from clean selected predictions in fixed row order")
            file_hashes_ok = all(exact_sha(trajectory[epoch].get("prediction_export_sha256")) and exact_sha(trajectory[epoch].get("capsule_file_sha256")) for epoch in EPOCHS)
            add_gate(gates, blockers, "trajectory_export_and_capsule_sha256_exact", run, True, file_hashes_ok, file_hashes_ok, "trajectory SHA256 fields missing/malformed")

            run_data[run] = {"run": run, "seed": seed, "arm": arm, "trajectory": trajectory, "predictions": predictions, "capsule_paths": capsule_paths}

    stage191c_passed = not blockers and set(run_data) == set(RUNS)
    if not stage191c_passed:
        raise RuntimeError("Stage191-C replay equivalence failed; Stage191-D interpretation is forbidden")

    epoch_index = {(row["seed"], row["arm"], row["epoch"]): row for row in tables["epochs"]}
    delta_index: dict[tuple[int, int], dict[str, Any]] = {}
    for seed in SEEDS:
        for epoch in EPOCHS:
            baseline = epoch_index[seed, "baseline", epoch]
            intervention = epoch_index[seed, "intervention", epoch]
            row = {"seed": seed, "epoch": epoch, **{f"delta_{name}": intervention[name] - baseline[name] for name in METRICS}, "delta_selected_epoch": intervention["selected_epoch"] - baseline["selected_epoch"], "delta_replaced_selected_checkpoint": int(intervention["replaced_selected_checkpoint"]) - int(baseline["replaced_selected_checkpoint"])}
            tables["deltas"].append(row)
            delta_index[seed, epoch] = row

    late_by_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        d19, d20 = delta_index[seed, 19], delta_index[seed, 20]
        row = {
            "seed": seed,
            "delta_pred_SUPPORT_epoch19": d19["delta_pred_SUPPORT"], "delta_pred_SUPPORT_epoch20": d20["delta_pred_SUPPORT"],
            "sign_pred_SUPPORT_epoch19": sign(d19["delta_pred_SUPPORT"]), "sign_pred_SUPPORT_epoch20": sign(d20["delta_pred_SUPPORT"]),
            "delta_pred_NOT_ENTITLED_epoch19": d19["delta_pred_NOT_ENTITLED"], "delta_pred_NOT_ENTITLED_epoch20": d20["delta_pred_NOT_ENTITLED"],
            "sign_pred_NOT_ENTITLED_epoch19": sign(d19["delta_pred_NOT_ENTITLED"]), "sign_pred_NOT_ENTITLED_epoch20": sign(d20["delta_pred_NOT_ENTITLED"]),
            "delta_false_entitlement_epoch19": d19["delta_false_entitlement_total"], "delta_false_entitlement_epoch20": d20["delta_false_entitlement_total"],
            "sign_false_entitlement_epoch19": sign(d19["delta_false_entitlement_total"]), "sign_false_entitlement_epoch20": sign(d20["delta_false_entitlement_total"]),
            "abs_delta_pred_REFUTE_epoch19": abs(d19["delta_pred_REFUTE"]), "abs_delta_pred_REFUTE_epoch20": abs(d20["delta_pred_REFUTE"]),
            "delta_polarity_error_epoch19": d19["delta_polarity_error_total"], "delta_polarity_error_epoch20": d20["delta_polarity_error_total"],
        }
        row["pred_SUPPORT_sign_reversal"] = row["sign_pred_SUPPORT_epoch19"] * row["sign_pred_SUPPORT_epoch20"] == -1
        row["pred_NOT_ENTITLED_sign_reversal"] = row["sign_pred_NOT_ENTITLED_epoch19"] * row["sign_pred_NOT_ENTITLED_epoch20"] == -1
        row["false_entitlement_sign_reversal"] = row["sign_false_entitlement_epoch19"] * row["sign_false_entitlement_epoch20"] == -1
        row["phase_flip_seed_conditions_pass"] = row["pred_SUPPORT_sign_reversal"] and row["false_entitlement_sign_reversal"] and row["abs_delta_pred_REFUTE_epoch19"] <= 1 and row["abs_delta_pred_REFUTE_epoch20"] <= 1 and abs(row["delta_polarity_error_epoch19"]) <= 1 and abs(row["delta_polarity_error_epoch20"]) <= 1
        tables["late"].append(row)
        late_by_seed[seed] = row

    within_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    paired_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for run in RUNS:
        data = run_data[run]
        comparisons = (("epoch18_to_19", 18, 19), ("epoch19_to_20", 19, 20), ("selected_to_final", SELECTED[run], 20))
        for name, previous_epoch, next_epoch in comparisons:
            summary = transition_summary(data["predictions"][previous_epoch], data["predictions"][next_epoch])
            row = {"run": run, "seed": data["seed"], "arm": data["arm"], "comparison": name, "previous_epoch": previous_epoch, "next_epoch": next_epoch, **summary}
            tables["within"].append(row)
            within_lookup[run, name] = row
            for gold in LABELS:
                tables["within_gold"].append({"run": run, "seed": data["seed"], "arm": data["arm"], "comparison": name, "previous_epoch": previous_epoch, "next_epoch": next_epoch, "gold_label": gold, **transition_summary(data["predictions"][previous_epoch], data["predictions"][next_epoch], gold)})
    for seed in SEEDS:
        baseline = run_data[f"seed{seed}_baseline"]
        intervention = run_data[f"seed{seed}_intervention"]
        for epoch in EPOCHS:
            summary = transition_summary(baseline["predictions"][epoch], intervention["predictions"][epoch])
            row = {"seed": seed, "epoch": epoch, "previous_arm": "baseline", "next_arm": "intervention", **summary}
            tables["paired"].append(row)
            paired_lookup[seed, epoch] = row
            for gold in LABELS:
                tables["paired_gold"].append({"seed": seed, "epoch": epoch, "previous_arm": "baseline", "next_arm": "intervention", "gold_label": gold, **transition_summary(baseline["predictions"][epoch], intervention["predictions"][epoch], gold)})

    for seed in SEEDS:
        for series in INSTABILITY_SERIES:
            stats: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                values = [float(epoch_index[seed, arm, epoch][series]) for epoch in range(15, 21)]
                stats[arm] = series_stats(values, float(epoch_index[seed, arm, SELECTED[f"seed{seed}_{arm}"]][series]), float(epoch_index[seed, arm, 20][series]))
            tables["instability"].append({"seed": seed, "series": series, **{f"baseline_{name}": value for name, value in stats["baseline"].items()}, **{f"intervention_{name}": value for name, value in stats["intervention"].items()}, **{f"intervention_minus_baseline_{name}": stats["intervention"][name] - stats["baseline"][name] for name in stats["baseline"]}})

    state_integrity: dict[str, Any] = {}
    for run in RUNS:
        state_integrity[run] = analyze_capsules(run_data[run], tables["state"], tables["groups"], stage190_inventory_contract["by_run"][run])
    ownership_exhaustive_disjoint = all(state_integrity[run].get("ownership_exhaustive_disjoint") is True for run in RUNS)

    selected19_runs = [run for run in RUNS if run.endswith("intervention") and SELECTED[run] == 19]
    selected19_concentration = {run: within_lookup[run, "epoch19_to_20"]["exclusive_not_entitled_support_fraction_of_changed"] for run in selected19_runs}
    selected19_pass = selected19_runs == ["seed175_intervention", "seed176_intervention"] and all(value is not None and value >= 0.95 for value in selected19_concentration.values())
    phase_flip_pass = all(late_by_seed[seed]["phase_flip_seed_conditions_pass"] for seed in SEEDS) and selected19_pass
    redistribution_cells = {(seed, epoch): paired_lookup[seed, epoch] for seed in SEEDS for epoch in (19, 20)}
    redistribution_pass = (not phase_flip_pass and all(row["changed_rows"] > 0 and row["exclusive_not_entitled_support_fraction_of_changed"] is not None and row["exclusive_not_entitled_support_fraction_of_changed"] >= 0.95 and abs(delta_index[seed, epoch]["delta_pred_REFUTE"]) <= 1 and abs(delta_index[seed, epoch]["delta_polarity_error_total"]) <= 1 for (seed, epoch), row in redistribution_cells.items()))
    decision = CONFIRMED if phase_flip_pass else REDISTRIBUTION if redistribution_pass else INCONCLUSIVE
    selected_decision_matches_taxonomy = (phase_flip_pass and decision == CONFIRMED) or (not phase_flip_pass and redistribution_pass and decision == REDISTRIBUTION) or (not phase_flip_pass and not redistribution_pass and decision == INCONCLUSIVE)
    group_displacement_summary = {
        run: {
            group: {
                "all_epoch_steps_squared_l2_sum": sum(row["squared_l2_step_norm"] for row in tables["groups"] if row["run"] == run and row["parameter_group"] == group),
                "epoch15_to_20_steps_squared_l2_sum": sum(row["squared_l2_step_norm"] for row in tables["groups"] if row["run"] == run and row["parameter_group"] == group and 16 <= row["next_epoch"] <= 20),
            }
            for group in GROUPS
        }
        for run in RUNS
    }
    precommitted = [
        {"gate": "stage190_grouping_source_contract", "required": True, "observed": grouping_source_contract["source_contract_passed"], "passed": grouping_source_contract["source_contract_passed"]},
        {"gate": "stage190_exact_parameter_inventory_mapping", "required": True, "observed": stage190_inventory_contract["passed"], "passed": stage190_inventory_contract["passed"]},
        {"gate": "parameter_ownership_exhaustive_disjoint", "required": True, "observed": ownership_exhaustive_disjoint, "passed": ownership_exhaustive_disjoint},
        {"gate": "stage191c_equivalence_all_six", "required": True, "observed": stage191c_passed, "passed": stage191c_passed},
        {"gate": "support_delta_nonzero_opposite_sign_all_seeds", "required": "decision_alternative", "observed": {seed: late_by_seed[seed]["pred_SUPPORT_sign_reversal"] for seed in SEEDS}, "passed": None},
        {"gate": "false_entitlement_delta_nonzero_opposite_sign_all_seeds", "required": "decision_alternative", "observed": {seed: late_by_seed[seed]["false_entitlement_sign_reversal"] for seed in SEEDS}, "passed": None},
        {"gate": "refute_absolute_delta_at_most_one_all_late_cells", "required": "decision_alternative", "observed": {seed: [late_by_seed[seed]["abs_delta_pred_REFUTE_epoch19"], late_by_seed[seed]["abs_delta_pred_REFUTE_epoch20"]] for seed in SEEDS}, "passed": None},
        {"gate": "polarity_absolute_delta_at_most_one_all_late_cells", "required": "decision_alternative", "observed": {seed: [late_by_seed[seed]["delta_polarity_error_epoch19"], late_by_seed[seed]["delta_polarity_error_epoch20"]] for seed in SEEDS}, "passed": None},
        {"gate": "selected_epoch19_intervention_transition_concentration_at_least_095", "required": "decision_alternative", "observed": {"runs": ["seed175_intervention", "seed176_intervention"], "minimum": 0.95, "values": selected19_concentration, "condition_observed": selected19_pass}, "passed": None},
        {"gate": "phase_flip_condition_observed", "required": "decision_alternative", "observed": phase_flip_pass, "passed": None},
        {"gate": "redistribution_condition_observed", "required": "decision_alternative", "observed": redistribution_pass, "passed": None},
        {"gate": "selected_decision_matches_precommitted_taxonomy", "required": True, "observed": {"phase_flip_pass": phase_flip_pass, "redistribution_pass": redistribution_pass, "decision": decision}, "passed": selected_decision_matches_taxonomy},
    ]
    tables["precommit"].extend(precommitted)
    return {
        "stage": "Stage191-D", "decision": decision, "runnable": True, "blocking_reasons": [], "stage191c_equivalence_passed": True,
        "diagnostic_only": True, "training_performed": False, "model_constructed": False, "model_advancement_decision": False, "external_data_used": False, "clean_dev_ce_reduction_contract": dict(CLEAN_DEV_CE_REDUCTION_CONTRACT),
        "current_diagnostic_git_commit": args.current_diagnostic_git_commit, "diagnostic_source_commit_identity": diagnostic_source_identity, "stage191b_commit": STAGE191B_COMMIT, "stage191b_directory": str(stage191b), "stage185_sidecar_identity": {"path": str(authoritative_sidecar), "is_regular_file": authoritative_sidecar.is_file(), "semantic_sha256": observed_sidecar_semantic_sha256},
        "authoritative_stage190_diagnostic_commit": grouping_source_contract["authoritative_stage190_diagnostic_commit"], "authoritative_stage190_grouping_source_path": grouping_source_contract["authoritative_stage190_grouping_source_path"], "authoritative_stage190_grouping_commit_blob_sha256": grouping_source_contract["authoritative_stage190_grouping_commit_blob_sha256"], "observed_stage190_grouping_source_sha256": grouping_source_contract["observed_stage190_grouping_source_sha256"], "current_source_equals_frozen_commit_blob": grouping_source_contract["current_source_equals_frozen_commit_blob"], "unstaged_clean": grouping_source_contract["unstaged_clean"], "staged_clean": grouping_source_contract["staged_clean"], "source_contract_passed": grouping_source_contract["source_contract_passed"], "stage190_to_stage191d_alias_mapping": grouping_source_contract["alias_mapping"], "stage190_parameter_inventory_contract": stage190_inventory_summary,
        "six_run_identities": [{"run": run, "seed": run_data[run]["seed"], "arm": run_data[run]["arm"], "split_seed": 174, "selected_epoch": SELECTED[run]} for run in RUNS],
        "late_sign_reversal_results": tables["late"], "row_transition_concentration": {"selected_epoch19_intervention_epoch19_to_20": selected19_concentration, "paired_epochs19_20": {f"seed{seed}_epoch{epoch}": paired_lookup[seed, epoch]["exclusive_not_entitled_support_fraction_of_changed"] for seed in SEEDS for epoch in (19, 20)}},
        "selected_to_final_drift": [row for row in tables["instability"]], "state_capsule_integrity": state_integrity,
        "parameter_group_state_displacement": {"authoritative_source": grouping_source_contract, "original_stage190_group_names": list(STAGE190_ORIGINAL_GROUPS), "stage191d_reporting_groups": list(GROUPS), "alias_mapping": dict(STAGE190_TO_STAGE191D_ALIASES), "ownership_exhaustive_disjoint": ownership_exhaustive_disjoint, "descriptive_only": True, "rows": len(tables["groups"]), "summary": group_displacement_summary},
        "precommitted_gate_results": precommitted,
        "interpretation_restrictions": ["no causal parameter-group responsibility", "no statistical significance", "no generalization beyond these three paired seeds", "no external performance claim", "no model advancement"],
        "recommended_next_stage": "Stage192 design of a trajectory-stabilized entitlement mechanism or selection rule" if decision == CONFIRMED else "No Stage192 training is authorized; resolve or interpret the frozen diagnostic only.",
        "stage192_training_authorized": False, "exception": None,
    }


def blocked_report(args: argparse.Namespace, blockers: list[str], exception: BaseException | None) -> dict[str, Any]:
    exception_record = None if exception is None else {"type": type(exception).__name__, "message": str(exception), "traceback": traceback.format_exc()}
    reasons = list(blockers)
    try:
        diagnostic_identity = validate_diagnostic_source_identity(args.repo_root.resolve(), args.current_diagnostic_git_commit)
    except BaseException as identity_exc:
        diagnostic_identity = {"passed": False, "error_type": type(identity_exc).__name__, "error_message": str(identity_exc)}
    try:
        grouping_identity = validate_stage190_grouping_source(args.repo_root.resolve())
    except BaseException as grouping_exc:
        grouping_identity = {"authoritative_stage190_diagnostic_commit": STAGE190_DIAGNOSTIC_COMMIT, "authoritative_stage190_grouping_source_path": str((args.repo_root.resolve() / STAGE190_GROUPING_SOURCE_RELATIVE_PATH).resolve()), "authoritative_stage190_grouping_commit_blob_sha256": None, "observed_stage190_grouping_source_sha256": None, "current_source_equals_frozen_commit_blob": False, "unstaged_clean": False, "staged_clean": False, "source_contract_passed": False, "alias_mapping": dict(STAGE190_TO_STAGE191D_ALIASES), "error_type": type(grouping_exc).__name__, "error_message": str(grouping_exc)}
    if exception_record is not None:
        reasons.append(f"{exception_record['type']}: {exception_record['message']}")
    return {
        "stage": "Stage191-D", "decision": BLOCKED, "runnable": False, "blocking_reasons": reasons or ["Stage191-D did not complete"], "stage191c_equivalence_passed": False,
        "diagnostic_only": True, "training_performed": False, "model_constructed": False, "model_advancement_decision": False, "external_data_used": False, "clean_dev_ce_reduction_contract": dict(CLEAN_DEV_CE_REDUCTION_CONTRACT),
        "current_diagnostic_git_commit": args.current_diagnostic_git_commit, "diagnostic_source_commit_identity": diagnostic_identity, "stage191b_commit": STAGE191B_COMMIT,
        "authoritative_stage190_diagnostic_commit": grouping_identity.get("authoritative_stage190_diagnostic_commit", STAGE190_DIAGNOSTIC_COMMIT), "authoritative_stage190_grouping_source_path": grouping_identity.get("authoritative_stage190_grouping_source_path"), "authoritative_stage190_grouping_commit_blob_sha256": grouping_identity.get("authoritative_stage190_grouping_commit_blob_sha256"), "observed_stage190_grouping_source_sha256": grouping_identity.get("observed_stage190_grouping_source_sha256"), "current_source_equals_frozen_commit_blob": grouping_identity.get("current_source_equals_frozen_commit_blob", False), "unstaged_clean": grouping_identity.get("unstaged_clean", False), "staged_clean": grouping_identity.get("staged_clean", False), "source_contract_passed": grouping_identity.get("source_contract_passed", False), "stage190_to_stage191d_alias_mapping": grouping_identity.get("alias_mapping", dict(STAGE190_TO_STAGE191D_ALIASES)),
        "six_run_identities": [{"run": run, "seed": int(run[4:7]), "arm": run.split("_", 1)[1], "split_seed": 174, "selected_epoch": SELECTED[run]} for run in RUNS],
        "late_sign_reversal_results": [], "row_transition_concentration": None, "selected_to_final_drift": [], "state_capsule_integrity": {"passed": False}, "parameter_group_state_displacement": None,
        "interpretation_restrictions": ["Stage191-D interpretation forbidden because Stage191-C or capsule validation did not close", "no causal parameter-group responsibility", "no statistical significance", "no generalization beyond these three paired seeds", "no external performance claim", "no model advancement"],
        "recommended_next_stage": "Resolve the blocking Stage191-C equivalence or capsule-integrity failure; do not authorize Stage192 training.", "stage192_training_authorized": False, "exception": exception_record,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stage191-D trajectory phase-flip report", "", f"Decision: `{report['decision']}`", "",
        f"- Runnable: {str(report['runnable']).lower()}", f"- Stage191-C equivalence passed: {str(report['stage191c_equivalence_passed']).lower()}",
        "- Diagnostic only: true", "- Training performed: false", "- Model constructed: false", "- Model advancement decision: false", "- External data used: false", "- Stage192 training authorized: false",
        f"- Blocking reasons: {report['blocking_reasons'] if report['blocking_reasons'] else 'none'}", "",
        "## Interpretation", "",
    ]
    if report["decision"] == CONFIRMED:
        lines.append("The exact precommitted evidence confirms checkpoint-phase sensitivity with a late SUPPORT/NOT_ENTITLED phase flip across the three paired seeds. This is not causal or generalization evidence.")
    elif report["decision"] == REDISTRIBUTION:
        lines.append("Late paired changes are concentrated on the SUPPORT/NOT_ENTITLED boundary, but the complete precommitted phase-flip conjunction does not pass.")
    elif report["decision"] == INCONCLUSIVE:
        lines.append("The frozen trajectories do not satisfy either complete precommitted late-trajectory decision conjunction.")
    else:
        lines.append("Interpretation is blocked. No positive Stage191-D conclusion is emitted.")
    lines.extend(["", "Parameter-group displacement is descriptive only and does not identify causal responsibility. No statistical significance, external performance, broader generalization, or model advancement is claimed.", "", "## Recommended next stage", "", str(report["recommended_next_stage"]), ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        _, _, output = establish_safe_output(args)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 2
    output.mkdir(parents=False, exist_ok=True)
    tables = {name: [] for name in CSV_HEADERS}
    report: dict[str, Any]
    try:
        report = analyze(args, tables)
    except BaseException as exc:
        blockers = [row["blocking_reason"] for row in tables["equivalence"] if row.get("passed") is False and row.get("blocking_reason")]
        report = blocked_report(args, blockers, exc)
        tables["precommit"].append({"gate": "fail_closed_exception", "required": "no exception", "observed": {"type": type(exc).__name__, "message": str(exc)}, "passed": False})
    write_json(output / OUTPUTS["json"], report)
    (output / OUTPUTS["md"]).write_text(markdown_report(report), encoding="utf-8")
    for name, headers in CSV_HEADERS.items():
        write_csv(output / OUTPUTS[name], headers, tables[name])
    return 0 if report["runnable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
