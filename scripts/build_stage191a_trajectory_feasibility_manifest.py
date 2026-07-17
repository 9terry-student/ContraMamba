#!/usr/bin/env python3
"""Audit whether existing Stage189/190 artifacts support Stage191 trajectory analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import traceback
from pathlib import Path
from typing import Any, Iterable

SEEDS = (174, 175, 176)
ARMS = ("baseline", "intervention")
EPOCHS = tuple(range(1, 21))
SELECTED_EPOCHS = {
    "seed174_baseline": 20, "seed174_intervention": 20,
    "seed175_baseline": 20, "seed175_intervention": 19,
    "seed176_baseline": 20, "seed176_intervention": 19,
}
EXPECTED_OBJECTIVE_ROWS = {
    "margin_eligible": 605, "ce_eligible": 605, "ce_clean_dev_all": 720,
    "ce_clean_dev_support": 89, "neg_support_vs_not_entitled_margin": 89,
    "neg_support_vs_max_other_margin": 89, "neg_mean_frame_logit_compatible_fn": 13,
    "neg_mean_frame_logit_matched_controls": 14, "neg_mean_frame_logit_ineligible": 716,
    "neg_mean_frame_logit_unresolved": 119,
}
TRAINING_COMMIT = "bee2f5ad452d1d9f57b30f444d18835dbffdbecf"
STAGE190_COMMIT = "ac0b9032b94436ce8ac8134c650d389134faebd4"
TRAINER_SHA = "24b01c5799c762772fe1700204afae59f8566898f65e7f3eefa4ac57ac6f126f"
DATA_SHA = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
SIDECAR_SHA = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
READY189 = "STAGE189A_THREE_SEED_MARGIN_REPLICATION_AND_POSTHOC_REFERENCE_SPEC_READY"
READY190A = "STAGE190A_GRADIENT_CONFLICT_DIAGNOSTIC_MANIFEST_READY"
EXPORTED190B = "STAGE190B_GRADIENT_DIAGNOSTIC_EXPORTED"
CLOSED190C = "STAGE190C_MARGIN_GRADIENT_HEAD_LOCAL_OR_NONCONFLICTING"
BLOCKED = "STAGE191A_TRAJECTORY_FEASIBILITY_BLOCKED"
FULL = "STAGE191A_EXISTING_TRAJECTORY_FULL_READY"
METRIC_ONLY = "STAGE191A_EXISTING_TRAJECTORY_METRIC_ONLY_READY"
REPLAY = "STAGE191A_DETERMINISTIC_REPLAY_REQUIRED"

PATHS = {
    "margin_history": "runs.single.compatible_positive_margin.epoch_metrics",
    "clean_history": "runs.single.v7_epoch_diagnostic_history",
    "final_epoch": "runs.single.final_epoch",
    "selected_epoch": "runs.single.best_epoch",
    "selection_metric": "runs.single.select_metric",
    "selection_rules": "runs.single.audit_ledger.active_selection_rules",
    "clean_macro_f1": "runs.single.v7_epoch_diagnostic_history[].dev_final_macro_f1",
    "clean_support_recall": None,
    "clean_false_entitlement_total": None,
    "clean_polarity_error_total": None,
    "clean_prediction_counts": "runs.single.v7_epoch_diagnostic_history[].dev_prediction_distribution",
    "clean_dev_ce": None,
}

CLEAN_FIELDS = {
    "macro_f1": ("dev_final_macro_f1",),
    "support_recall": (),
    "false_entitlement_total": (),
    "polarity_error_total": (),
    "prediction_count_REFUTE": ("dev_prediction_distribution", "REFUTE"),
    "prediction_count_NOT_ENTITLED": ("dev_prediction_distribution", "NOT_ENTITLED"),
    "prediction_count_SUPPORT": ("dev_prediction_distribution", "SUPPORT"),
    "clean_dev_ce": (),
}

FULL_SIDECAR_TOPOLOGY = {
    "total_rows": 3600,
    "train_rows": 2880,
    "dev_rows": 720,
    "unique_row_ids": 3600,
}

TRAIN_COMPATIBLE_TOPOLOGY = {
    "train_rows": 1440,
    "incompatible_train_rows": 1440,
    "eligible_rows": 605,
    "eligible_pairs": 121,
    "eligible_families": 5,
    "eligible": 605,
    "ineligible": 716,
    "unresolved": 119,
}

CLEAN_PREDICTION_LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
PREDICTION_DISTRIBUTION_SEMANTICS = {
    "raw_prediction_distributions_are_sparse": True,
    "absent_known_labels_normalized_to_zero": True,
    "unknown_labels_invalid": True,
    "normalization_scope": [
        "prediction_count_REFUTE",
        "prediction_count_NOT_ENTITLED",
        "prediction_count_SUPPORT",
    ],
}
STAGE191_INPUT_INVENTORY = [
    "Stage189-A manifest report and six per-run manifests",
    "six Stage189 internal training reports",
    "six Stage189 internal run-provenance reports",
    "checkpoint file byte streams only within the six Stage189 internal run directories, for SHA256 inventory",
    "Stage190-A manifest report and six per-run manifests",
    "six Stage190-B internal reports",
    "Stage190-C internal report",
    "trainer, Stage189-D analyzer, and Stage190-C analyzer source text",
    "Git HEAD, commit blob, and staged/unstaged identity for this builder",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage189a-dir", type=Path, required=True)
    parser.add_argument("--stage190a-dir", type=Path, required=True)
    parser.add_argument("--stage190c-dir", type=Path, required=True)
    parser.add_argument("--current-diagnostic-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in headers})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def exact_int(value: Any) -> bool:
    return type(value) is int


def nested(root: Any, *keys: str) -> Any:
    value = root
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def git(repo: Path, *args: str, binary: bool = False) -> Any:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode("utf-8", errors="strict").strip()


def argv_value(argv: Any, option: str) -> str:
    if not isinstance(argv, list):
        raise ValueError(f"argv is not a list while resolving {option}")
    found: list[str] = []
    for index, token in enumerate(argv):
        if token == option and index + 1 < len(argv):
            found.append(str(argv[index + 1]))
        elif isinstance(token, str) and token.startswith(option + "="):
            found.append(token.split("=", 1)[1])
    if len(found) != 1:
        raise ValueError(f"expected exactly one {option}, found {found}")
    return found[0]


def resolve_run_dir(stage189a: Path, manifest: dict[str, Any], seed: int, arm: str) -> Path:
    raw = manifest.get("run_directory")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"seed{seed}_{arm}: Stage189-A manifest run_directory is absent")
    resolved = Path(raw).resolve()
    if not resolved.is_dir():
        raise ValueError(f"seed{seed}_{arm}: exact manifest run_directory is not a directory: {resolved}")
    return resolved

def source_contract(repo: Path) -> dict[str, Any]:
    trainer_path = repo / "scripts" / "train_controlled_v6b_minimal.py"
    stage189d_path = repo / "scripts" / "analyze_stage189d_three_seed_margin_replication.py"
    stage190c_path = repo / "scripts" / "analyze_stage190c_gradient_conflict.py"
    texts = {path.name: path.read_text(encoding="utf-8") for path in (trainer_path, stage189d_path, stage190c_path)}
    required = {
        trainer_path.name: [
            '"compatible_positive_margin": _compatible_positive_margin_report',
            '"epoch_metrics": _compatible_positive_margin_epoch_metrics',
            '"v7_epoch_diagnostic_history": _v7_epoch_history',
            '"dev_final_macro_f1": dev_metrics.get("final_macro_f1")',
            '"prediction_distribution": dict(sorted(pred_dist_overall.items()))',
            '"dev_prediction_distribution": dev_metrics.get("prediction_distribution")',
            '.get(label_name, 0)',
            '"final_epoch": epochs', '"best_epoch": best_epoch', '"select_metric": select_metric',
            '"active_selection_rules": _active_selection_rules', '"runs": reports',
        ],
        stage189d_path.name: ['aggregate.get("epoch_metrics")', 'report_run.get("best_epoch")'],
        stage190c_path.name: ['"intervention_support_conflict_by_seed":conflicts', '"model_advancement_decision":False'],
    }
    missing = {name: [item for item in needles if item not in texts[name]] for name, needles in required.items()}
    if any(missing.values()):
        raise ValueError(f"authoritative report-writer source contract missing: {missing}")
    return {"inspected_sources": [str(trainer_path), str(stage189d_path), str(stage190c_path)], "required_snippets_present": True, "authoritative_paths": PATHS, "prediction_distribution_semantics": PREDICTION_DISTRIBUTION_SEMANTICS}


def path_inventory(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.append(child)
            paths.extend(path_inventory(value[key], child))
    elif isinstance(value, list) and value:
        paths.extend(path_inventory(value[0], prefix + "[]"))
    return paths


def audit_margin(run_key: str, report: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    rows = nested(report, "runs", "single", "compatible_positive_margin", "epoch_metrics")
    if not isinstance(rows, list):
        raise TypeError(f"{run_key}: margin history is not a list")
    arm = run_key.rsplit("_", 1)[1]
    seen: list[int] = []
    output: list[dict[str, Any]] = []
    required = ("epoch", "enabled", "eligible_count", "active_count", "active_rate", "compatible_positive_margin_loss_raw", "compatible_positive_margin_loss_weighted", "eligible_frame_logit_sum", "mean_eligible_frame_logit", "zero_eligible_batch")
    for position, row in enumerate(rows, 1):
        missing = [key for key in required if not isinstance(row, dict) or key not in row]
        epoch = row.get("epoch") if isinstance(row, dict) else None
        seen.append(epoch)
        enabled_ok = isinstance(row, dict) and row.get("enabled") is (arm == "intervention")
        count_ok = isinstance(row, dict) and exact_int(row.get("eligible_count")) and row.get("eligible_count") == (605 if arm == "intervention" else 0) and exact_int(row.get("active_count")) and (0 <= row.get("active_count") <= 605 if arm == "intervention" else row.get("active_count") == 0)
        numeric_ok = isinstance(row, dict) and all(finite(row.get(key)) for key in ("compatible_positive_margin_loss_raw", "compatible_positive_margin_loss_weighted", "eligible_frame_logit_sum"))
        if arm == "intervention":
            nullable_ok = finite(row.get("active_rate")) and finite(row.get("mean_eligible_frame_logit")) and abs(row.get("active_rate") - row.get("active_count") / 605) <= 1e-12
        else:
            nullable_ok = row.get("active_rate") is None and row.get("mean_eligible_frame_logit") is None
        valid = not missing and exact_int(epoch) and enabled_ok and count_ok and numeric_ok and nullable_ok and type(row.get("zero_eligible_batch")) is bool
        output.append({"run": run_key, "source_path": PATHS["margin_history"], "source_position": position, "epoch": epoch, "enabled": row.get("enabled") if isinstance(row, dict) else None, "eligible_count": row.get("eligible_count") if isinstance(row, dict) else None, "active_count": row.get("active_count") if isinstance(row, dict) else None, "active_rate": row.get("active_rate") if isinstance(row, dict) else None, "raw_loss": row.get("compatible_positive_margin_loss_raw") if isinstance(row, dict) else None, "weighted_loss": row.get("compatible_positive_margin_loss_weighted") if isinstance(row, dict) else None, "eligible_frame_logit_sum": row.get("eligible_frame_logit_sum") if isinstance(row, dict) else None, "mean_eligible_frame_logit": row.get("mean_eligible_frame_logit") if isinstance(row, dict) else None, "zero_eligible_batch": row.get("zero_eligible_batch") if isinstance(row, dict) else None, "missing_fields": missing, "finite_and_schema_valid": valid, "authoritative_for_decision": True})
    complete = len(rows) == 20 and seen == list(EPOCHS) and len(set(seen)) == len(seen) and all(row["finite_and_schema_valid"] for row in output)
    return complete, output


def audit_clean(
    run_key: str,
    report: dict[str, Any],
) -> tuple[bool, bool, list[dict[str, Any]], dict[str, list[int]], list[str], dict[str, Any]]:
    try:
        rows = nested(report, "runs", "single", "v7_epoch_diagnostic_history")
    except KeyError:
        rows = []
    if not isinstance(rows, list):
        raise TypeError(f"{run_key}: clean history is not a list")

    history_present = bool(rows)
    schema_valid = True
    epochs: list[Any] = []
    valid_macro_epochs: set[int] = set()
    valid_distribution_epochs: set[int] = set()
    if history_present:
        for position, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                schema_valid = False
                epochs.append(None)
                continue
            epoch = row.get("epoch")
            epochs.append(epoch)
            if not exact_int(epoch) or epoch not in EPOCHS:
                schema_valid = False
                continue
            macro = row.get("dev_final_macro_f1")
            if finite(macro) and 0.0 <= float(macro) <= 1.0:
                valid_macro_epochs.add(epoch)
            else:
                schema_valid = False
            distribution = row.get("dev_prediction_distribution")
            raw_distribution_ok = (
                isinstance(distribution, dict)
                and set(distribution).issubset(set(CLEAN_PREDICTION_LABELS))
                and all(exact_int(value) and value >= 0 for value in distribution.values())
                and sum(distribution.values()) == 720
            )
            normalized_distribution = (
                {
                    label: distribution.get(label, 0)
                    for label in CLEAN_PREDICTION_LABELS
                }
                if raw_distribution_ok else {}
            )
            normalized_distribution_ok = (
                set(normalized_distribution) == set(CLEAN_PREDICTION_LABELS)
                and all(exact_int(normalized_distribution[label]) and normalized_distribution[label] >= 0
                        for label in CLEAN_PREDICTION_LABELS)
                and sum(normalized_distribution.values()) == 720
            )
            if raw_distribution_ok and normalized_distribution_ok:
                valid_distribution_epochs.add(epoch)
            else:
                schema_valid = False
    if len(rows) != 20 or epochs != list(EPOCHS) or len(set(epochs)) != len(epochs):
        schema_valid = False

    duplicate_epochs = sorted({epoch for epoch in epochs if exact_int(epoch) and epochs.count(epoch) > 1})
    observed_epoch_set = {epoch for epoch in epochs if exact_int(epoch)}
    history_missing_epochs = sorted(set(EPOCHS) - observed_epoch_set)
    available_by_metric = {
        "macro_f1": valid_macro_epochs,
        "prediction_count_REFUTE": valid_distribution_epochs,
        "prediction_count_NOT_ENTITLED": valid_distribution_epochs,
        "prediction_count_SUPPORT": valid_distribution_epochs,
        "support_recall": set(),
        "false_entitlement_total": set(),
        "polarity_error_total": set(),
        "clean_dev_ce": set(),
    }
    rows_out: list[dict[str, Any]] = []
    missing_by_metric: dict[str, list[int]] = {}
    for metric, field_path in CLEAN_FIELDS.items():
        available = available_by_metric[metric]
        absent = sorted(set(EPOCHS) - available)
        missing_by_metric[metric] = absent
        schema_path = PATHS.get("clean_" + metric)
        if metric.startswith("prediction_count_"):
            schema_path = PATHS["clean_prediction_counts"] + "." + metric.removeprefix("prediction_count_")
        rows_out.append({
            "run": run_key,
            "metric": metric,
            "authoritative_schema_path": schema_path,
            "present": not absent,
            "exact_row_count": len(available),
            "exact_epoch_set": available == set(EPOCHS),
            "missing_epochs": absent,
            "duplicate_epochs": duplicate_epochs,
            "required_fields": list(field_path),
            "missing_fields": [] if not absent else [".".join(field_path) if field_path else metric],
            "finite_value_gate": not absent,
            "authoritative_for_decision": True,
        })

    selection = {
        "final_epoch": nested(report, "runs", "single", "final_epoch"),
        "selected_epoch": nested(report, "runs", "single", "best_epoch"),
        "selection_metric": nested(report, "runs", "single", "select_metric"),
        "selection_rules": nested(report, "runs", "single", "audit_ledger", "active_selection_rules"),
    }
    rules = selection["selection_rules"]
    standard_rule = rules.get("standard_clean_dev") if isinstance(rules, dict) else None
    optional_rules = (
        "td_constrained_selection",
        "preservation_constrained_selection",
        "stage44b_anti_collapse_selection",
    )
    selection_ok = (
        selection["final_epoch"] == 20
        and exact_int(selection["selected_epoch"])
        and selection["selected_epoch"] in EPOCHS
        and selection["selection_metric"] == "final_macro_f1"
        and isinstance(standard_rule, dict)
        and standard_rule.get("enabled") is True
        and standard_rule.get("metric") == "final_macro_f1"
        and standard_rule.get("stage15_used") is False
        and all(isinstance(rules.get(name), dict) and rules[name].get("enabled") is False
                for name in optional_rules)
    )
    if not selection_ok:
        schema_valid = False
    missing_fields = [metric for metric, absent in missing_by_metric.items() if absent]
    complete = schema_valid and not missing_fields
    return schema_valid, complete, rows_out, missing_by_metric, missing_fields, selection
def checkpoint_inventory(
    run_key: str,
    run_dir: Path,
    selected: Path,
    final_path: Path | None,
    selected_epoch: int,
    final_epoch: int,
) -> tuple[bool, list[dict[str, Any]], list[int]]:
    candidates = sorted(
        path for path in run_dir.rglob("*")
        if path.is_file()
        and ("checkpoint" in path.name.lower() or path.suffix.lower() in {".pt", ".pth", ".ckpt"})
    )
    exact_epoch = re.compile(r"(?:^|[_-])epoch[_-]?(\d+)(?=[_.-]|$)", re.IGNORECASE)
    prelim: list[dict[str, Any]] = []
    for path in candidates:
        resolved = path.resolve()
        matches = exact_epoch.findall(path.name)
        filename_epoch = int(matches[0]) if len(matches) == 1 and int(matches[0]) in EPOCHS else None
        if resolved == selected:
            classification = "selected checkpoint"
            inferred_epoch = selected_epoch
            identity_source = "exact selected path plus runs.single.best_epoch"
        elif final_path is not None and resolved == final_path:
            classification = "final checkpoint"
            inferred_epoch = final_epoch
            identity_source = "exact final path plus runs.single.final_epoch"
        elif len(matches) == 1 and filename_epoch is not None:
            classification = "epoch-addressable checkpoint"
            inferred_epoch = filename_epoch
            identity_source = "one exact filename epoch token"
        else:
            classification = "unrelated checkpoint-like file"
            inferred_epoch = None
            identity_source = (
                "ambiguous filename epoch tokens" if len(matches) > 1
                else "no exact in-range filename epoch token"
            )
        prelim.append({
            "path": resolved,
            "classification": classification,
            "inferred_epoch": inferred_epoch,
            "identity_source": identity_source,
            "epoch_token_count": len(matches),
            "ambiguous_epoch_token": len(matches) > 1,
        })

    epoch_values = [
        row["inferred_epoch"] for row in prelim
        if row["classification"] == "epoch-addressable checkpoint"
    ]
    epoch_counts = {
        epoch: epoch_values.count(epoch)
        for epoch in EPOCHS
    }
    exact_epoch_coverage = (
        set(epoch_values) == set(EPOCHS)
        and all(epoch_counts[epoch] == 1 for epoch in EPOCHS)
    )
    ambiguous_candidates = [
        row for row in prelim
        if row["classification"] not in {"selected checkpoint", "final checkpoint"}
        and row["ambiguous_epoch_token"]
    ]
    complete = exact_epoch_coverage and not ambiguous_candidates
    rows = [{
        "run": run_key,
        "path": str(row["path"]),
        "sha256": sha256(row["path"]),
        "classification": row["classification"],
        "inferred_epoch": row["inferred_epoch"],
        "identity_source": row["identity_source"],
        "epoch_token_count": row["epoch_token_count"],
        "ambiguous_epoch_token": row["ambiguous_epoch_token"],
        "duplicate_epoch": (
            row["classification"] == "epoch-addressable checkpoint"
            and epoch_values.count(row["inferred_epoch"]) > 1
        ),
        "all_epochs_1_through_20_covered": complete,
    } for row in prelim]
    return complete, rows, sorted(set(epoch_values))

def main() -> int:
    args = parse_args()
    repo, stage189a, stage190a, stage190c, output = (args.repo_root.resolve(), args.stage189a_dir.resolve(), args.stage190a_dir.resolve(), args.stage190c_dir.resolve(), args.output_dir.resolve())
    output.mkdir(parents=True, exist_ok=True)
    identity_blockers: list[str] = []
    artifact_blockers: list[str] = []
    gates: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    run_matrix: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    margin_complete: dict[str, bool] = {}
    clean_complete: dict[str, bool] = {}
    checkpoint_complete: dict[str, bool] = {}
    missing_clean_fields: dict[str, list[str]] = {}
    missing_clean_epochs: dict[str, dict[str, list[int]]] = {}
    checkpoint_epochs: dict[str, list[int]] = {}
    selections: dict[str, Any] = {}
    inventory_paths: dict[str, list[str]] = {}
    exception_record: dict[str, str] | None = None
    source_inspection: dict[str, Any] = {}
    external_use_ledger: list[dict[str, Any]] = []
    exception_category = "identity_closure"

    def gate(name: str, required: Any, observed: Any, passed: bool, reason: str, category: str = "identity_closure") -> None:
        gates.append({"gate": name, "category": category, "required": required, "observed": observed, "passed": passed, "blocking_reason": "" if passed else reason})
        if not passed:
            (identity_blockers if category == "identity_closure" else artifact_blockers).append(reason)

    try:
        source_inspection = source_contract(repo)
        supplied_commit = args.current_diagnostic_git_commit.strip()
        head = git(repo, "rev-parse", "HEAD")
        script = (repo / "scripts" / "build_stage191a_trajectory_feasibility_manifest.py").resolve()
        relative = script.relative_to(repo).as_posix()
        working_bytes = script.read_bytes()
        blob_bytes = git(repo, "show", f"{supplied_commit}:{relative}", binary=True)
        unstaged = subprocess.run(["git", "diff", "--quiet", "--", relative], cwd=repo).returncode == 0
        staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative], cwd=repo).returncode == 0
        gate("current_head_identity", supplied_commit, head, head == supplied_commit, "current repository HEAD does not equal supplied Stage191 diagnostic commit")
        gate("stage191_script_commit_blob_identity", hashlib.sha256(blob_bytes).hexdigest(), hashlib.sha256(working_bytes).hexdigest(), working_bytes == blob_bytes, "Stage191-A working script bytes differ from supplied commit blob")
        gate("stage191_script_no_unstaged_diff", True, unstaged, unstaged, "Stage191-A script has an unstaged diff")
        gate("stage191_script_no_staged_diff", True, staged, staged, "Stage191-A script has a staged diff")
        identities.append({"artifact": "stage191a_script", "path": str(script), "expected_identity": hashlib.sha256(blob_bytes).hexdigest(), "observed_identity": hashlib.sha256(working_bytes).hexdigest(), "passed": working_bytes == blob_bytes and unstaged and staged})

        m189_path = stage189a / "stage189a_manifest_report.json"
        m189 = read_json(m189_path)
        gate("stage189a_decision", READY189, m189.get("decision"),
             m189.get("decision") == READY189, "Stage189-A manifest is not READY")
        required_training_identity = {
            "current_git_commit": TRAINING_COMMIT,
            "trainer_sha256": TRAINER_SHA,
            "dataset_sha256": DATA_SHA,
            "sidecar_semantic_sha256": SIDECAR_SHA,
        }
        observed_training_identity = {key: m189.get(key) for key in required_training_identity}
        gate("stage189_training_identity", required_training_identity,
             observed_training_identity,
             observed_training_identity == required_training_identity,
             "Stage189 training identity mismatch")
        observed_full_topology = nested(m189, "full_sidecar_topology")
        observed_train_topology = nested(m189, "sidecar_train_topology")
        gate("stage189_full_sidecar_topology", FULL_SIDECAR_TOPOLOGY,
             observed_full_topology, observed_full_topology == FULL_SIDECAR_TOPOLOGY,
             "Stage189 full sidecar topology mismatch")
        gate("stage189_sidecar_train_topology", TRAIN_COMPATIBLE_TOPOLOGY,
             observed_train_topology,
             observed_train_topology == TRAIN_COMPATIBLE_TOPOLOGY,
             "Stage189 train-compatible topology mismatch")
        identities.append({
            "artifact": "stage189a_manifest_report",
            "path": str(m189_path),
            "expected_identity": {
                **required_training_identity,
                "full_sidecar_topology": FULL_SIDECAR_TOPOLOGY,
                "sidecar_train_topology": TRAIN_COMPATIBLE_TOPOLOGY,
            },
            "observed_identity": {
                **observed_training_identity,
                "full_sidecar_topology": observed_full_topology,
                "sidecar_train_topology": observed_train_topology,
            },
            "passed": (
                m189.get("decision") == READY189
                and observed_training_identity == required_training_identity
                and observed_full_topology == FULL_SIDECAR_TOPOLOGY
                and observed_train_topology == TRAIN_COMPATIBLE_TOPOLOGY
            ),
        })
        m190a_path = stage190a / "stage190a_manifest_report.json"
        m190a = read_json(m190a_path)
        required_190a_report = {
            "decision": READY190A,
            "runnable": True,
            "blocking_reasons": [],
            "manifest_count": 6,
            "training_git_commit": TRAINING_COMMIT,
            "diagnostic_git_commit": STAGE190_COMMIT,
        }
        observed_190a_report = {key: m190a.get(key) for key in required_190a_report}
        gate("stage190a_report_contract", required_190a_report, observed_190a_report,
             observed_190a_report == required_190a_report,
             "Stage190-A manifest-report contract mismatch")
        identities.append({
            "artifact": "stage190a_manifest_report",
            "path": str(m190a_path),
            "expected_identity": required_190a_report,
            "observed_identity": observed_190a_report,
            "passed": observed_190a_report == required_190a_report,
        })
        closure = read_json(stage190c / "stage190c_gradient_conflict_report.json")
        gate("stage190c_decision", CLOSED190C, closure.get("decision"), closure.get("decision") == CLOSED190C, "Stage190-C decision mismatch")
        gate("stage190c_no_blockers", [], closure.get("blocking_reasons"), closure.get("blocking_reasons") == [], "Stage190-C has blockers")
        gate("stage190c_no_advancement", False, closure.get("model_advancement_decision"), closure.get("model_advancement_decision") is False, "Stage190-C model advancement must be false")
        conflicts = closure.get("intervention_support_conflict_by_seed") or {}
        conflict_ok = set(conflicts) == {str(seed) for seed in SEEDS} and all(conflicts[str(seed)] is False for seed in SEEDS)
        gate("stage190c_intervention_support_nonconflict", {str(seed): False for seed in SEEDS}, conflicts, conflict_ok, "Stage190-C intervention SUPPORT conflict is not false for every seed")
        identities.append({"artifact": "stage190c_report", "path": str(stage190c / "stage190c_gradient_conflict_report.json"), "expected_identity": CLOSED190C, "observed_identity": closure.get("decision"), "passed": closure.get("decision") == CLOSED190C and closure.get("blocking_reasons") == [] and closure.get("model_advancement_decision") is False and conflict_ok})

        run_dirs: set[Path] = set()
        selected_shas: set[str] = set()
        for seed in SEEDS:
            for arm in ARMS:
                exception_category = "identity_closure"
                run_key = f"seed{seed}_{arm}"
                arm_manifest = read_json(stage189a / f"stage189a_seed{seed}_{arm}_manifest.json")
                run_dir = resolve_run_dir(stage189a, arm_manifest, seed, arm)
                run_dirs.add(run_dir)
                report_path = (run_dir / "training_report.json").resolve()
                provenance_path = (run_dir / "run_provenance.json").resolve()
                expected_selected_path = (run_dir / "selected_checkpoint.pt").resolve()
                for required_path in (report_path, provenance_path, expected_selected_path):
                    if not required_path.is_file():
                        raise ValueError(f"{run_key}: missing exact internal artifact: {required_path}")
                stage189_selected_raw = arm_manifest.get("selected_checkpoint_path")
                stage189_selected_path = Path(stage189_selected_raw).resolve() if isinstance(stage189_selected_raw, str) and stage189_selected_raw else None
                gate(run_key + "_stage189_selected_path", str(expected_selected_path),
                     str(stage189_selected_path) if stage189_selected_path else None,
                     stage189_selected_path == expected_selected_path,
                     f"{run_key}: Stage189-A selected checkpoint path mismatch")
                current_hashes = {
                    "training_report": sha256(report_path),
                    "run_provenance": sha256(provenance_path),
                    "selected_checkpoint": sha256(expected_selected_path),
                }
                selected_shas.add(current_hashes["selected_checkpoint"])
                report = read_json(report_path)
                provenance = read_json(provenance_path)
                parsed = provenance.get("parsed_args") or {}
                split = provenance.get("split_seed_contract") or {}
                source_prov = provenance.get("source_provenance") or {}
                data_prov = (provenance.get("data_provenance") or {}).get("main_data") or {}
                finalization = provenance.get("finalization") or {}
                prov_selected = finalization.get("selected_checkpoint") or {}
                training_identity_ok = (
                    provenance.get("status") == "completed"
                    and arm_manifest.get("seed") == seed
                    and arm_manifest.get("arm") == arm
                    and arm_manifest.get("training_seed") == seed
                    and arm_manifest.get("split_seed") == 174
                    and arm_manifest.get("trainer_sha256") == TRAINER_SHA
                    and arm_manifest.get("dataset_sha256") == DATA_SHA
                    and source_prov.get("git_commit") == TRAINING_COMMIT
                    and source_prov.get("trainer_sha256") == TRAINER_SHA
                    and data_prov.get("sha256") == DATA_SHA
                    and parsed.get("seed") == seed
                    and parsed.get("split_seed") == 174
                    and split.get("resolved_split_seed") == 174
                    and split.get("clean_main_train_rows") == 2880
                    and split.get("clean_main_dev_rows") == 720
                    and parsed.get("epochs") == 20
                    and finalization.get("completed_epochs") == 20
                    and prov_selected.get("path") == str(expected_selected_path)
                    and prov_selected.get("sha256") == current_hashes["selected_checkpoint"]
                )
                gate(run_key + "_training_identity", True, training_identity_ok,
                     training_identity_ok,
                     f"{run_key}: Stage189 training/provenance identity mismatch")
                b_manifest = read_json(stage190a / f"stage190a_seed{seed}_{arm}_manifest.json")
                stage190_checkpoint_raw = b_manifest.get("checkpoint_path")
                stage190_checkpoint_path = Path(stage190_checkpoint_raw).resolve() if isinstance(stage190_checkpoint_raw, str) and stage190_checkpoint_raw else None
                external_contract = b_manifest.get("external_use_contract")
                runtime_contract = b_manifest.get("arm_runtime_contract")
                required_b_manifest = {
                    "runnable": True,
                    "blocking_reasons": [],
                    "seed": seed,
                    "arm": arm,
                    "training_seed": seed,
                    "split_seed": 174,
                    "training_git_commit": TRAINING_COMMIT,
                    "diagnostic_git_commit": STAGE190_COMMIT,
                    "checkpoint_sha256": current_hashes["selected_checkpoint"],
                    "checkpoint_path": str(expected_selected_path),
                    "fixed_split_identity": {"split_seed": 174, "train": 2880, "dev": 720},
                    "external_use_contract_passed": True,
                    "arm_runtime_contract_passed": True,
                }
                observed_b_manifest = {
                    "runnable": b_manifest.get("runnable"),
                    "blocking_reasons": b_manifest.get("blocking_reasons"),
                    "seed": b_manifest.get("seed"),
                    "arm": b_manifest.get("arm"),
                    "training_seed": b_manifest.get("training_seed"),
                    "split_seed": b_manifest.get("split_seed"),
                    "training_git_commit": b_manifest.get("training_git_commit"),
                    "diagnostic_git_commit": b_manifest.get("diagnostic_git_commit"),
                    "checkpoint_sha256": b_manifest.get("checkpoint_sha256"),
                    "checkpoint_path": str(stage190_checkpoint_path) if stage190_checkpoint_path else None,
                    "fixed_split_identity": b_manifest.get("fixed_split_identity"),
                    "external_use_contract_passed": external_contract.get("passed") if isinstance(external_contract, dict) else None,
                    "arm_runtime_contract_passed": runtime_contract.get("passed") if isinstance(runtime_contract, dict) else None,
                }
                b_manifest_ok = observed_b_manifest == required_b_manifest and isinstance(external_contract, dict) and isinstance(runtime_contract, dict) and stage190_checkpoint_path == expected_selected_path
                gate(run_key + "_stage190a_manifest_contract", required_b_manifest,
                     observed_b_manifest, b_manifest_ok,
                     f"{run_key}: Stage190-A per-run manifest contract mismatch")

                artifact_hashes = b_manifest.get("artifact_hashes")
                if not isinstance(artifact_hashes, dict):
                    artifact_hashes = {}
                expected_artifact_hashes = {
                    "training_report": artifact_hashes.get("training_report"),
                    "run_provenance": artifact_hashes.get("run_provenance"),
                    "selected_checkpoint": b_manifest.get("checkpoint_sha256"),
                }
                artifact_hash_ok = all(isinstance(value, str) and bool(value) for value in expected_artifact_hashes.values()) and expected_artifact_hashes == current_hashes
                gate(run_key + "_stage190a_current_artifact_hashes",
                     expected_artifact_hashes, current_hashes, artifact_hash_ok,
                     f"{run_key}: current internal artifact SHA256 mismatch")
                for artifact_name, artifact_path in (
                    ("training_report", report_path),
                    ("run_provenance", provenance_path),
                    ("selected_checkpoint", expected_selected_path),
                ):
                    identities.append({
                        "artifact": run_key + "_" + artifact_name,
                        "path": str(artifact_path),
                        "expected_identity": expected_artifact_hashes[artifact_name],
                        "observed_identity": current_hashes[artifact_name],
                        "passed": expected_artifact_hashes[artifact_name] == current_hashes[artifact_name],
                    })

                b_output = Path(argv_value(b_manifest.get("argv"), "--output-dir")).resolve()
                b_report = read_json(b_output / "stage190b_gradient_report.json")
                b_selected_raw = b_report.get("selected_checkpoint_path")
                b_selected_path = Path(b_selected_raw).resolve() if isinstance(b_selected_raw, str) and b_selected_raw else None
                b_ok = (
                    b_report.get("decision") == EXPORTED190B
                    and b_report.get("blocking_reasons") == []
                    and b_report.get("model_state_unchanged") is True
                    and isinstance(b_report.get("trainable_state_sha256_before"), str)
                    and bool(b_report.get("trainable_state_sha256_before"))
                    and b_report.get("trainable_state_sha256_before") == b_report.get("trainable_state_sha256_after")
                    and isinstance(b_report.get("buffer_state_sha256_before"), str)
                    and bool(b_report.get("buffer_state_sha256_before"))
                    and b_report.get("buffer_state_sha256_before") == b_report.get("buffer_state_sha256_after")
                    and b_report.get("external_data_used") is False
                    and b_report.get("evaluation_only") is True
                    and b_report.get("training_performed") is False
                    and b_report.get("optimizer_created") is False
                    and b_report.get("optimizer_step_performed") is False
                    and b_report.get("checkpoint_selection_performed") is False
                    and b_report.get("threshold_tuning_performed") is False
                    and b_selected_path == expected_selected_path
                    and b_report.get("selected_checkpoint_sha256") == current_hashes["selected_checkpoint"]
                    and b_report.get("training_seed") == seed
                    and b_report.get("split_seed") == 174
                    and b_report.get("arm") == arm
                    and b_report.get("label_mapping_contract_passed") is True
                    and b_report.get("finite_gradient_gates_passed") is True
                    and b_report.get("parameter_group_metric_expected_rows") == 54
                    and b_report.get("parameter_group_metric_observed_rows") == 54
                    and b_report.get("parameter_group_metric_grid_passed") is True
                    and b_report.get("directional_derivative_expected_rows") == 9
                    and b_report.get("directional_derivative_observed_rows") == 9
                    and b_report.get("directional_derivative_grid_passed") is True
                    and b_report.get("objective_row_counts") == EXPECTED_OBJECTIVE_ROWS
                    and b_report.get("expected_objective_row_counts") == EXPECTED_OBJECTIVE_ROWS
                    and b_report.get("training_git_commit") == TRAINING_COMMIT
                    and b_report.get("diagnostic_git_commit") == STAGE190_COMMIT
                )
                gate(run_key + "_stage190b_report_contract", True, b_ok, b_ok,
                     f"{run_key}: Stage190-B report contract mismatch")
                external_use_ledger.append({
                    "run": run_key,
                    "stage190a_external_use_contract_passed": external_contract.get("passed") if isinstance(external_contract, dict) else None,
                    "stage190a_arm_runtime_contract_passed": runtime_contract.get("passed") if isinstance(runtime_contract, dict) else None,
                    "stage190b_external_data_used": b_report.get("external_data_used"),
                })

                exception_category = "artifact_validity"
                margin_complete[run_key], rows = audit_margin(run_key, report)
                margin_rows.extend(rows)
                gate(run_key + "_margin_history", True, margin_complete[run_key], margin_complete[run_key], f"{run_key}: authoritative 20-epoch margin history is invalid", category="artifact_validity")
                clean_schema_valid, clean_complete[run_key], rows, missing_by_metric, missing_fields, selection = audit_clean(run_key, report)
                clean_rows.extend(rows)
                gate(run_key + "_clean_history_schema", True, clean_schema_valid, clean_schema_valid, f"{run_key}: existing clean-history rows are malformed", category="artifact_validity")
                missing_clean_epochs[run_key] = missing_by_metric
                missing_clean_fields[run_key] = missing_fields
                selections[run_key] = selection
                gate(run_key + "_selected_epoch_identity", SELECTED_EPOCHS[run_key], selection["selected_epoch"], selection["selected_epoch"] == SELECTED_EPOCHS[run_key], f"{run_key}: selected epoch identity mismatch")
                gate(run_key + "_final_epoch_identity", 20, selection["final_epoch"], selection["final_epoch"] == 20, f"{run_key}: final epoch identity mismatch")
                final_raw = (provenance.get("finalization") or {}).get("final_checkpoint_path")
                final_path = Path(final_raw).resolve() if final_raw else None
                checkpoint_complete[run_key], rows, checkpoint_epochs[run_key] = checkpoint_inventory(run_key, run_dir, expected_selected_path, final_path, selection["selected_epoch"], selection["final_epoch"])
                checkpoint_rows.extend(rows)
                inventory_paths[run_key] = path_inventory(report)
                run_matrix.append({"seed": seed, "arm": arm, "run": run_key, "run_directory": str(run_dir), "training_seed": parsed.get("seed"), "split_seed": parsed.get("split_seed"), "train_rows": split.get("clean_main_train_rows"), "dev_rows": split.get("clean_main_dev_rows"), "selected_epoch": selection["selected_epoch"], "final_epoch": selection["final_epoch"], "margin_history_complete": margin_complete[run_key], "clean_metric_history_complete": clean_complete[run_key], "epoch_checkpoint_history_complete": checkpoint_complete[run_key]})
        exception_category = "identity_closure"
        gate("exactly_six_run_directories", 6, len(run_dirs), len(run_dirs) == 6,
             "exactly six distinct Stage189 run directories did not resolve")
        gate("six_distinct_selected_checkpoints", 6, len(selected_shas), len(selected_shas) == 6,
             "selected checkpoint identities are not six exact distinct SHA256 values")
        expected_external_ledger = [{
            "run": f"seed{seed}_{arm}",
            "stage190a_external_use_contract_passed": True,
            "stage190a_arm_runtime_contract_passed": True,
            "stage190b_external_data_used": False,
        } for seed in SEEDS for arm in ARMS]
        gate("six_run_internal_only_contract", expected_external_ledger,
             external_use_ledger, external_use_ledger == expected_external_ledger,
             "six-run internal-only external-use ledger mismatch")
    except BaseException as exc:
        exception_record = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "category": exception_category}
        reason = f"blocking exception: {type(exc).__name__}: {exc}"
        (identity_blockers if exception_category == "identity_closure" else artifact_blockers).append(reason)

    blockers = identity_blockers + artifact_blockers
    identity_pass = bool([row for row in gates if row["category"] == "identity_closure"]) and all(row["passed"] for row in gates if row["category"] == "identity_closure") and any(row["gate"] == "six_run_internal_only_contract" and row["passed"] for row in gates) and not identity_blockers
    all_margin = len(margin_complete) == 6 and all(margin_complete.values())
    all_clean = len(clean_complete) == 6 and all(clean_complete.values())
    all_checkpoints = len(checkpoint_complete) == 6 and all(checkpoint_complete.values())
    artifact_sufficiency_reasons = [
        f"{run}: missing authoritative clean metrics: {', '.join(fields)}"
        for run, fields in missing_clean_fields.items() if fields
    ] + [
        f"{run}: incomplete epoch-addressable checkpoint coverage"
        for run, complete in checkpoint_complete.items() if not complete
    ]
    if blockers:
        decision, runnable, next_stage = BLOCKED, False, None
    elif all_margin and all_clean and all_checkpoints:
        decision, runnable, next_stage = FULL, True, "Stage191-B existing-artifact metric and gradient-trajectory analysis."
    elif all_margin and all_clean:
        decision, runnable, next_stage = METRIC_ONLY, True, "Stage191-B existing-artifact metric trajectory analysis. Do not authorize gradient-at-every-epoch claims."
    else:
        decision, runnable, next_stage = REPLAY, False, "Stage191-B deterministic replay specification. This decision does not itself authorize training."

    replay_required = decision == REPLAY
    report = {"stage": "Stage191-A", "decision": decision, "runnable": runnable, "blocking_reasons": blockers, "identity_closure_blocking_reasons": identity_blockers, "artifact_validity_blocking_reasons": artifact_blockers, "artifact_sufficiency_reasons": artifact_sufficiency_reasons, "recommended_next_stage": next_stage, "training_performed": False, "checkpoint_loaded": False, "model_loaded": False, "external_data_used": False, "training_authorized": False, "replay_execution_authorized": False, "model_advancement_decision": False, "replay_specification_required": replay_required, "next_stage_specification_authorized": replay_required, "training_git_commit": TRAINING_COMMIT, "stage190_diagnostic_git_commit": STAGE190_COMMIT, "current_stage191_diagnostic_git_commit": args.current_diagnostic_git_commit.strip(), "trainer_sha256": TRAINER_SHA, "dataset_sha256": DATA_SHA, "sidecar_semantic_sha256": SIDECAR_SHA, "exact_six_run_identity": run_matrix, "six_run_identity": run_matrix, "margin_history_complete_by_run": margin_complete, "clean_metric_history_complete_by_run": clean_complete, "epoch_checkpoint_history_complete_by_run": checkpoint_complete, "missing_clean_metric_fields_by_run": missing_clean_fields, "missing_clean_metric_epochs_by_run": missing_clean_epochs, "checkpoint_epochs_by_run": checkpoint_epochs, "selection_epoch_by_run": {key: value.get("selected_epoch") for key, value in selections.items()}, "final_epoch_by_run": {key: value.get("final_epoch") for key, value in selections.items()}, "authoritative_schema_paths": PATHS, "source_inspection": source_inspection, "prediction_distribution_semantics": PREDICTION_DISTRIBUTION_SEMANTICS, "selection_provenance_by_run": selections, "non_authoritative_inventory_paths": inventory_paths, "stage191_static_input_inventory": STAGE191_INPUT_INVENTORY, "six_run_external_use_ledger": external_use_ledger, "identity_and_closure_gates_passed": identity_pass, "fail_closed_exception": exception_record}
    write_json(output / "stage191a_trajectory_feasibility_report.json", report)
    write_csv(output / "stage191a_authoritative_input_identity.csv", ["artifact", "path", "expected_identity", "observed_identity", "passed"], identities)
    write_csv(output / "stage191a_run_matrix.csv", ["seed", "arm", "run", "run_directory", "training_seed", "split_seed", "train_rows", "dev_rows", "selected_epoch", "final_epoch", "margin_history_complete", "clean_metric_history_complete", "epoch_checkpoint_history_complete"], run_matrix)
    write_csv(output / "stage191a_margin_epoch_availability.csv", ["run", "source_path", "source_position", "epoch", "enabled", "eligible_count", "active_count", "active_rate", "raw_loss", "weighted_loss", "eligible_frame_logit_sum", "mean_eligible_frame_logit", "zero_eligible_batch", "missing_fields", "finite_and_schema_valid", "authoritative_for_decision"], margin_rows)
    write_csv(output / "stage191a_clean_metric_availability.csv", ["run", "metric", "authoritative_schema_path", "present", "exact_row_count", "exact_epoch_set", "missing_epochs", "duplicate_epochs", "required_fields", "missing_fields", "finite_value_gate", "authoritative_for_decision"], clean_rows)
    write_csv(output / "stage191a_checkpoint_inventory.csv", ["run", "path", "sha256", "classification", "inferred_epoch", "identity_source", "epoch_token_count", "ambiguous_epoch_token", "duplicate_epoch", "all_epochs_1_through_20_covered"], checkpoint_rows)
    write_csv(output / "stage191a_precommitted_gate.csv", ["gate", "category", "required", "observed", "passed", "blocking_reason"], gates)
    missing_lines = [f"- {run}: {', '.join(fields)}" for run, fields in missing_clean_fields.items() if fields]
    markdown = f"""# Stage191-A trajectory feasibility report

**Decision:** `{decision}`

- Runnable existing-artifact analysis: {str(runnable).lower()}
- Training performed/authorized: false / false
- Replay execution authorized: false
- Replay specification required: {str(replay_required).lower()}
- Checkpoint/model loaded: false
- External data used: false
- Model advancement: false
- Recommended next stage: `{next_stage}`

## Authoritative paths

- Margin: `{PATHS['margin_history']}`
- Clean outcomes: `{PATHS['clean_history']}`
- Selected/final epoch: `{PATHS['selected_epoch']}` / `{PATHS['final_epoch']}`
- Selection provenance: `{PATHS['selection_metric']}` and `{PATHS['selection_rules']}`

## Missing clean trajectory fields

{chr(10).join(missing_lines) if missing_lines else '- None.'}

## Blocking reasons

{chr(10).join('- ' + reason for reason in blockers) if blockers else '- None.'}

Top-level selected-checkpoint metrics, final predictions, Stage189-D deltas, interpolation, nearby epochs, and loss histories were not substituted for missing per-epoch outcomes. A replay-required decision authorizes only the next replay specification, not replay execution or training.
"""
    (output / "stage191a_trajectory_feasibility_report.md").write_text(markdown, encoding="utf-8")
    return 2 if decision == BLOCKED else 0


if __name__ == "__main__":
    raise SystemExit(main())
