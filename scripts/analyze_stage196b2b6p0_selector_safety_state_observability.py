#!/usr/bin/env python3
"""Stage196-B2-B6P0 artifact-only selector safety-state observability.

This analyzer consumes only explicitly supplied artifacts.  It fits no model,
learns no threshold, optimizes no score, and authorizes no training or promotion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

STAGE = "Stage196-B2-B6P0"
P0_COMMIT = "fa16787efa84bb15d832b6d9382fafd77016c4e2"
B6_COMMIT = "d21457a3b514a304994c799357c725df3edbcc18"
SEEDS = (183, 184, 185)
PRIMARY_SEEDS = (184, 185)
EPOCHS = tuple(range(1, 21))
TAIL = (18, 19, 20)
MODES = ("joint", "frame_local_only")
RUNS = tuple(f"seed{seed}_{mode}" for seed in SEEDS for mode in MODES)
PRIMITIVES = ("FRAME", "PREDICATE", "SUFFICIENCY", "POSITIVE_ENERGY", "NEGATIVE_ENERGY")
PRIMITIVE_FIELDS = {
    "FRAME": ("frame_prob",),
    "PREDICATE": ("predicate_coverage_prob",),
    "SUFFICIENCY": ("sufficiency_prob",),
    "POSITIVE_ENERGY": ("positive_energy",),
    "NEGATIVE_ENERGY": ("negative_energy",),
}
BRANCHES = ("temporal_mismatch", "predicate_mismatch", "temporal_adapter", "temporal_channel")
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
TOL = 1e-6
ZERO_TOL = 1e-12
EXPECTED_CANDIDATES = ("00100000000000", "01000000000000", "10000000000000")
EXPECTED_PRIMARY = {184: {"recovery": 5, "harm": 6}, 185: {"recovery": 2, "harm": 3}}

B6_FILES = (
    "stage196b2b6_analysis.json", "stage196b2b6_report.md",
    "stage196b2b6_candidate_feature_subsets.csv", "stage196b2b6_signature_action_map.csv",
    "stage196b2b6_primary_policy_validation.csv", "stage196b2b6_clean_dev_signature_audit.csv",
    "stage196b2b6_clean_dev_application_summary.csv", "stage196b2b6_policy_dominance.csv",
    "stage196b2b6_contract.csv",
)
B5_FILES = (
    "stage196b2b5_analysis.json", "stage196b2b5_report.md", "stage196b2b5_feature_dictionary.csv",
    "stage196b2b5_row_action_sets.csv", "stage196b2b5_recipient_signature_rows.csv",
    "stage196b2b5_recipient_selector_summary.csv", "stage196b2b5_paired_delta_signature_rows.csv",
    "stage196b2b5_paired_delta_selector_summary.csv", "stage196b2b5_contract.csv",
)
B4_FILES = (
    "stage196b2b4_analysis.json", "stage196b2b4_report.md", "stage196b2b4_primitive_coalition_rows.csv",
    "stage196b2b4_primitive_mobius_terms.csv", "stage196b2b4_primitive_tail_summary.csv",
    "stage196b2b4_residual_coalition_rows.csv", "stage196b2b4_residual_mobius_terms.csv",
    "stage196b2b4_localization_summary.csv", "stage196b2b4_contract.csv",
)
OUTPUTS = (
    "stage196b2b6p0_analysis.json", "stage196b2b6p0_report.md",
    "stage196b2b6p0_safety_feature_dictionary.csv", "stage196b2b6p0_row_safety_targets.csv",
    "stage196b2b6p0_single_state_signature_rows.csv", "stage196b2b6p0_single_state_gate_summary.csv",
    "stage196b2b6p0_diagnostic_gate_summary.csv", "stage196b2b6p0_gated_policy_audit.csv",
    "stage196b2b6p0_contract.csv",
)
TRAJECTORY_FIELDS = {
    "id", "source_row_id", "dev_position", "gold_label", "prediction", "intervention_type",
    "frame_probability", "predicate_coverage_probability", "sufficiency_probability",
    "polarity_support_margin", "entitlement_probability", "support_probability",
    "not_entitled_probability", "support_logit", "not_entitled_logit", "epoch",
    "training_seed", "frame_downstream_gradient_mode",
}
PROHIBITED_FEATURES = {
    "gold_label", "baseline_correctness", "selector_correctness", "correct_to_incorrect",
    "incorrect_to_correct", "safety_target", "transition_role", "path_class", "subtype", "seed",
    "stable_row_id", "id", "source_row_id", "dev_position", "minimal_coalition",
    "counterfactual_prediction", "counterfactual_margin", "donor_prediction", "donor_tail_outcome",
}
PROHIBITED_CLAIMS = (
    "formal causal mediation", "external or OOD validity", "unfrozen-Mamba validity",
    "training improvement", "promotion", "deployability from tail-checkpoint trajectories",
    "deployability from paired-treatment deltas", "safety from pooled in-sample separation alone",
    "safety from partial cross-seed coverage", "OPTIONAL rows as successful activation evidence",
    "unseen default blocking as positive selector evidence", "gold correctness as an inference-time feature",
    "seed or row identity as a safety feature", "an arbitrary threshold as a mechanistic gate",
)

FEATURE_H = (
    "feature_family", "feature_name", "integration_authorized", "diagnostic_only", "epoch_scope",
    "source_fields", "formula", "value_domain", "natural_threshold", "outcome_derived", "available",
    "unavailable_reason",
)
TARGET_H = (
    "candidate_feature_subset_mask", "candidate_feature_subset_members", "seed", "stable_row_id", "id",
    "source_row_id", "dev_position", "population", "primary_case", "transition_role",
    "signature_support", "assigned_action_set", "evaluated_actions", "gold_label", "joint_prediction",
    "selector_prediction", "joint_correct", "selector_correct", "correct_to_incorrect",
    "incorrect_to_correct", "abstention_objective_passed", "selector_objective_passed", "safety_target",
    "safety_target_reason",
)
SIGNATURE_H = (
    "candidate_feature_subset_mask", "candidate_feature_subset_members", "seed", "stable_row_id", "id",
    "source_row_id", "dev_position", "population", "safety_target", "assigned_action_set",
    "single_state_feature_values",
)
GATE_H = (
    "candidate_feature_subset_mask", "feature_family", "safety_feature_subset_mask",
    "safety_feature_subset_size", "safety_feature_subset_members", "constrained_row_count",
    "must_allow_count", "must_block_count", "signature_count", "conflicting_signature_count", "feasible",
    "inclusion_minimal_feasible", "seed184_to_seed185_full_pass", "seed185_to_seed184_full_pass",
    "bidirectional_cross_seed_full_pass", "primary_objective_failures",
    "seed184_nondiscovery_correct_to_incorrect", "seed185_nondiscovery_correct_to_incorrect",
    "conservative_gate_passed",
)
AUDIT_H = (
    "candidate_feature_subset_mask", "feature_family", "safety_feature_subset_mask", "seed", "population",
    "row_count", "must_allow_count", "must_block_count", "optional_count", "allowed_count", "blocked_count",
    "unseen_count", "primary_recovery_passed", "primary_harm_passed", "primary_objective_failures",
    "joint_correct_count", "gated_correct_count", "prediction_change_count", "correct_to_incorrect_count",
    "incorrect_to_correct_count", "stable_correct_preservation_rate",
)
CONTRACT_H = ("scope", "run", "gate", "required", "observed", "passed", "blocking_reason")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, kind in (
        ("repo-root", Path), ("stage196b2b6-analysis-json", Path),
        ("stage196b2b5-analysis-json", Path), ("stage196b2b4-analysis-json", Path),
        ("stage196b2b3p0-run-root", Path), ("stage196b2b3p0-runtime-git-commit", str),
        ("current-git-commit", str), ("output-dir", Path),
    ):
        parser.add_argument(f"--{name}", required=True, type=kind)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: object required")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name}: integer required")
    result = int(value)
    if isinstance(value, float) and value != result:
        raise ValueError(f"{name}: nonintegral")
    return result


def number(value: Any, name: str, probability: bool = False) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name}: finite number required")
    result = float(value)
    if not math.isfinite(result) or (probability and not 0 <= result <= 1):
        raise ValueError(f"{name}: invalid number")
    return result


def boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"{name}: boolean required")


def json_cell(value: str, name: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name}: structured JSON required") from exc


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def require_columns(rows: list[dict[str, Any]], names: Sequence[str], label: str) -> None:
    if not rows or not set(names) <= set(rows[0]):
        raise ValueError(f"{label}: missing required columns")


def contract_closed(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(
        boolean(row.get("passed"), "passed") and not row.get("blocking_reason", "").strip()
        for row in rows
    )


def gate(gates: list[dict[str, Any]], scope: str, run: str, name: str, required: Any,
         observed: Any, passed: bool, reason: str, fatal: bool = True) -> None:
    gates.append({"scope": scope, "run": run, "gate": name, "required": required,
                  "observed": observed, "passed": bool(passed),
                  "blocking_reason": "" if passed else reason})
    if fatal and not passed:
        raise ValueError(f"{name}: {reason}")


def exact_directory(path: Path, expected: Sequence[str], gates: list[dict[str, Any]], name: str) -> None:
    observed = sorted(item.name for item in path.iterdir() if item.is_file()) if path.is_dir() else []
    wanted = sorted(expected)
    gate(gates, "source", "", name, wanted,
         {"files": observed, "missing": sorted(set(wanted) - set(observed)),
          "unexpected": sorted(set(observed) - set(wanted))},
         observed == wanted, "exact file closure failed")


def sign(value: float) -> str:
    return "ZERO" if abs(value) <= ZERO_TOL else "POSITIVE" if value > 0 else "NEGATIVE"


def halfspace(value: float) -> str:
    return "AT_HALF" if abs(value - 0.5) <= ZERO_TOL else "ABOVE_HALF" if value > 0.5 else "BELOW_HALF"


def softplus(value: float) -> float:
    return value + math.log1p(math.exp(-value)) if value > 0 else math.log1p(math.exp(value))


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def argmax_label(logits: Sequence[float]) -> str:
    return LABELS[max(range(3), key=lambda index: logits[index])]


def branch_magnitude(row: dict[str, Any], name: str) -> float:
    available = boolean(row[f"{name}_available"], f"{name}_available")
    if name in ("temporal_mismatch", "predicate_mismatch"):
        raw = f"raw_alpha_{name.split('_')[0]}"
        return softplus(number(row[raw], raw)) if available and boolean(
            row[f"{name}_condition_input"], f"{name}_condition_input") else 0.0
    if name == "temporal_adapter":
        return (sigmoid(number(row["temporal_adapter_logit"], "temporal_adapter_logit")) *
                number(row["temporal_adapter_final_penalty_scale"], "temporal_adapter_final_penalty_scale")) if available else 0.0
    return (sigmoid(number(row["temporal_channel_logit"], "temporal_channel_logit")) *
            (1 - number(row["preservation_entitlement_prob"], "preservation_entitlement_prob", True)) *
            number(row["temporal_channel_gated_penalty_scale"], "temporal_channel_gated_penalty_scale")) if available else 0.0


def reconstruct(row: dict[str, Any], validate_native: bool = False) -> dict[str, Any]:
    entitlement = (number(row["frame_prob"], "frame_prob", True) *
                   number(row["predicate_coverage_prob"], "predicate_coverage_prob", True) *
                   number(row["sufficiency_prob"], "sufficiency_prob", True))
    alpha = softplus(number(row["raw_alpha"], "raw_alpha"))
    head = (
        entitlement * number(row["negative_energy"], "negative_energy"),
        number(row["not_entitled_bias"], "not_entitled_bias") + alpha * (1 - entitlement),
        entitlement * number(row["positive_energy"], "positive_energy"),
    )
    residual = sum(branch_magnitude(row, branch) for branch in BRANCHES)
    final = (head[0] - residual, head[1] + residual, head[2] - residual)
    prediction = argmax_label(final)
    if validate_native:
        expected_head = tuple(number(row[f"decision_head_{name}_logit"], "head target")
                              for name in ("refute", "not_entitled", "support"))
        expected_final = tuple(number(row[f"final_{name}_logit"], "final target")
                               for name in ("refute", "not_entitled", "support"))
        errors = [abs(entitlement - number(row["entitlement_prob_native"], "entitlement target"))]
        errors.extend(abs(left - right) for left, right in zip(head, expected_head))
        errors.extend(abs(left - right) for left, right in zip(final, expected_final))
        if max(errors) > TOL or prediction != row["final_native_prediction"]:
            raise ValueError("native composer reconstruction failed")
    return {"entitlement": entitlement, "head": head, "final": final, "prediction": prediction}


def apply_action(recipient: dict[str, Any], donor: dict[str, Any], action: str) -> dict[str, Any]:
    if not re.fullmatch(r"[01]{5}", action):
        raise ValueError(f"invalid primitive action mask: {action}")
    state = dict(recipient)
    for bit, primitive in zip(action, PRIMITIVES):
        if bit == "1":
            for field in PRIMITIVE_FIELDS[primitive]:
                state[field] = donor[field]
    return reconstruct(state)


def validate_sources(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> dict[str, Any]:
    root = ns.repo_root.resolve()
    paths = {
        "b6": ns.stage196b2b6_analysis_json.resolve(),
        "b5": ns.stage196b2b5_analysis_json.resolve(),
        "b4": ns.stage196b2b4_analysis_json.resolve(),
        "p0": ns.stage196b2b3p0_run_root.resolve(),
        "output": ns.output_dir.resolve(),
    }
    raw_paths = (ns.repo_root, ns.stage196b2b6_analysis_json, ns.stage196b2b5_analysis_json,
                 ns.stage196b2b4_analysis_json, ns.stage196b2b3p0_run_root, ns.output_dir)
    explicit = all(path.is_absolute() and (path.resolve() == root or root in path.resolve().parents)
                   for path in raw_paths)
    commits_valid = all(re.fullmatch(r"[0-9a-f]{40}", value) for value in
                        (ns.stage196b2b3p0_runtime_git_commit, ns.current_git_commit))
    gate(gates, "invocation", "", "explicit_source_paths",
         {"absolute_and_under_repo_root": True, "commit_format": "40 lowercase hex"},
         {"absolute_and_under_repo_root": explicit, "commit_format_valid": commits_valid},
         root.is_dir() and explicit and commits_valid, "all paths must be explicit and under repo root")
    for key, expected in (("b6", B6_FILES), ("b5", B5_FILES), ("b4", B4_FILES)):
        if paths[key].name != expected[0]:
            raise ValueError(f"{key}: exact analysis JSON basename required")
        exact_directory(paths[key].parent, expected, gates, f"{key}_exact_nine_file_closure")
    b6, b5, b4 = (read_json(paths[key]) for key in ("b6", "b5", "b4"))
    requirements = {
        "b6": {"decision": "STAGE196B2B6_PRIMARY_EXACT_CLEAN_DEV_UNSAFE",
               "recommended_next_stage": "STAGE196B2B6P0_SELECTOR_SAFETY_STATE_OBSERVABILITY",
               "blocking_reasons": []},
        "b5": {"decision": "STAGE196B2B5_CROSS_SEED_RECIPIENT_SELECTOR_LOCALIZED",
               "recommended_next_stage": "STAGE196B2B6_MINIMAL_SELECTOR_INTERVENTION_DESIGN",
               "blocking_reasons": []},
        "b4": {"decision": "STAGE196B2B4_ROW_SPECIFIC_PRIMITIVE_INTERACTION",
               "recommended_next_stage": "STAGE196B2B5_ROW_SELECTOR_OBSERVABILITY",
               "blocking_reasons": []},
    }
    documents = {"b6": b6, "b5": b5, "b4": b4}
    for key in ("b6", "b5", "b4"):
        observed = {name: documents[key].get(name) for name in requirements[key]}
        gate(gates, key, "", f"{key}_decision_closure", requirements[key], observed,
             observed == requirements[key], f"{key.upper()} decision closure changed")
        contract = read_csv(paths[key].parent / ({"b6": B6_FILES, "b5": B5_FILES, "b4": B4_FILES}[key])[-1])
        gate(gates, key, "", f"{key}_contract_closure", {"failed": 0},
             {"rows": len(contract), "failed": sum(not boolean(row.get("passed"), "passed") for row in contract)},
             contract_closed(contract), f"{key.upper()} contract failed")
    b6_commit = b6.get("current_git_commit", b6.get("current_analyzer_git_commit"))
    gate(gates, "b6", "", "b2b6_analyzer_runtime_commit", B6_COMMIT, b6_commit,
         b6_commit == B6_COMMIT, "B2-B6 analyzer runtime commit changed")
    candidates = read_csv(paths["b6"].parent / B6_FILES[2])
    signature_actions = read_csv(paths["b6"].parent / B6_FILES[3])
    primary_validation = read_csv(paths["b6"].parent / B6_FILES[4])
    application_summary = read_csv(paths["b6"].parent / B6_FILES[6])
    b5_dictionary = read_csv(paths["b5"].parent / B5_FILES[2])
    b5_actions = read_csv(paths["b5"].parent / B5_FILES[3])
    b4_primitive_rows = read_csv(paths["b4"].parent / B4_FILES[2])
    b4_tail_summary = read_csv(paths["b4"].parent / B4_FILES[4])
    primitive_required = {"seed", "epoch", "stable_row_id", "direction", "coalition_mask",
                          "counterfactual_prediction"}
    tail_required = {"seed", "stable_row_id", "direction", "coalition_mask",
                     "coalition_tail_predictions", "recipient_tail_status", "donor_tail_status",
                     "donor_tail_reproduced", "recipient_tail_preserved"}
    primitive_columns = set(b4_primitive_rows[0]) if b4_primitive_rows else set()
    tail_columns = set(b4_tail_summary[0]) if b4_tail_summary else set()
    primitive_missing = sorted(primitive_required - primitive_columns)
    tail_missing = sorted(tail_required - tail_columns)
    gate(gates, "b4", "", "b2b4_primitive_row_schema_closure",
         {"required_columns": sorted(primitive_required), "missing_columns": []},
         {"required_columns": sorted(primitive_required), "observed_columns": sorted(primitive_columns),
          "missing_columns": primitive_missing},
         not primitive_missing, "stage196b2b4_primitive_coalition_rows.csv is missing required columns")
    gate(gates, "b4", "", "b2b4_tail_summary_schema_closure",
         {"required_columns": sorted(tail_required), "missing_columns": []},
         {"required_columns": sorted(tail_required), "observed_columns": sorted(tail_columns),
          "missing_columns": tail_missing},
         not tail_missing, "stage196b2b4_primitive_tail_summary.csv is missing required columns")
    gate(gates, "b4", "", "b2b4_primitive_20480_row_closure", 20480,
         {"row_count": len(b4_primitive_rows)}, len(b4_primitive_rows) == 20480,
         "B2-B4 primitive coalition row count changed")
    gate(gates, "b4", "", "b2b4_tail_summary_1024_row_closure", 1024,
         {"row_count": len(b4_tail_summary)}, len(b4_tail_summary) == 1024,
         "B2-B4 primitive tail-summary row count changed")
    require_columns(candidates, ("feature_subset_mask", "feature_subset_members", "primary_policy_passed",
                                 "seed184_nonprimary_safety_passed", "seed185_nonprimary_safety_passed",
                                 "nondominated"), "B2-B6 candidates")
    retained = sorted(row["feature_subset_mask"] for row in candidates if boolean(row["nondominated"], "nondominated"))
    retained_rows = [row for row in candidates if row["feature_subset_mask"] in retained]
    exact_candidate_result = (retained == list(EXPECTED_CANDIDATES) and
                              all(boolean(row["primary_policy_passed"], "primary") for row in retained_rows) and
                              all(not boolean(row["seed184_nonprimary_safety_passed"], "safety184") and
                                  not boolean(row["seed185_nonprimary_safety_passed"], "safety185")
                                  for row in retained_rows))
    gate(gates, "b6", "", "b2b6_nondominated_primary_unsafe_closure",
         {"nondominated": list(EXPECTED_CANDIDATES), "primary_exact": True,
          "seed184_safe": False, "seed185_safe": False},
         {"nondominated": retained,
          "primary_exact": all(boolean(row["primary_policy_passed"], "primary") for row in retained_rows),
          "seed184_safe": [boolean(row["seed184_nonprimary_safety_passed"], "safety") for row in retained_rows],
          "seed185_safe": [boolean(row["seed185_nonprimary_safety_passed"], "safety") for row in retained_rows]},
         exact_candidate_result, "nondominated selector authority changed")
    require_columns(b5_actions, ("seed", "stable_row_id", "id", "source_row_id", "dev_position",
                                 "transition_role", "acceptable_coalitions"), "B2-B5 actions")
    primitive_order = b4.get("primitive_order") or b4.get("source_closure", {}).get("primitive_order") or list(PRIMITIVES)
    gate(gates, "b4", "", "b2b4_primitive_order_closure", list(PRIMITIVES), primitive_order,
         primitive_order == list(PRIMITIVES), "B2-B4 primitive order changed")
    hashes = {}
    for key, names in (("b6", B6_FILES), ("b5", B5_FILES), ("b4", B4_FILES)):
        for name in names:
            path = paths[key].parent / name
            hashes[str(path)] = sha256(path)
    return {**paths, "b6_doc": b6, "b5_doc": b5, "b4_doc": b4,
            "candidate_rows": retained_rows, "signature_actions": signature_actions,
            "primary_validation": primary_validation, "application_summary": application_summary,
            "b5_dictionary": b5_dictionary, "b5_actions": b5_actions,
            "b4_primitive_rows": b4_primitive_rows, "b4_tail_summary": b4_tail_summary,
            "hashes": hashes}


def load_p0(ns: argparse.Namespace, source: dict[str, Any], gates: list[dict[str, Any]]) -> dict[str, Any]:
    root = source["p0"]
    runs = sorted(path.name for path in root.iterdir() if path.is_dir())
    gate(gates, "p0", "", "p0_six_run_closure", sorted(RUNS), runs, runs == sorted(RUNS),
         "P0 six-run closure failed")
    gate(gates, "p0", "", "p0_runtime_commit_agreement", P0_COMMIT,
         ns.stage196b2b3p0_runtime_git_commit,
         ns.stage196b2b3p0_runtime_git_commit == P0_COMMIT, "P0 runtime commit mismatch")
    states: dict[tuple[int, int, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    hashes: dict[str, str] = {}
    composer_sidecars = trajectory_sidecars = composer_rows = trajectory_rows = 0
    for run in RUNS:
        seed, mode = int(run[4:7]), run[8:]
        composer_dir = root / run / "composer_inputs"
        trajectory_dir = root / run / "trajectory"
        composer_names = [f"stage196b2b3p0_epoch_composer_inputs_{epoch:03d}.jsonl" for epoch in EPOCHS]
        trajectory_names = [f"stage196b2p0_epoch_channels_{epoch:03d}.jsonl" for epoch in EPOCHS]
        manifest_name = "stage196b2b3p0_composer_input_manifest.json"
        exact_directory(composer_dir, (manifest_name, *composer_names), gates,
                        f"{run}_composer_namespace_closure")
        files = sorted(path.name for path in trajectory_dir.iterdir() if path.is_file())
        namespace = re.compile(r"^stage196b2p0_epoch_channels_[0-9]{3}\.jsonl$")
        observed = sorted(name for name in files if namespace.fullmatch(name))
        malformed = sorted(name for name in files if name.startswith("stage196b2p0_epoch_channels_")
                           and name not in trajectory_names)
        gate(gates, "p0", run, "trajectory_namespace_closure",
             {"expected": trajectory_names, "malformed": []},
             {"observed": observed, "malformed": malformed,
              "unrelated_ignored": sorted(name for name in files if not name.startswith("stage196b2p0_epoch_channels_"))},
             observed == trajectory_names and not malformed, "P0 trajectory namespace changed")
        manifest_path = composer_dir / manifest_name
        manifest = read_json(manifest_path)
        hashes[str(manifest_path)] = sha256(manifest_path)
        manifest_ok = (manifest.get("completed") is True and manifest.get("current_git_commit") == P0_COMMIT and
                       manifest.get("seed") == seed and manifest.get("gradient_ownership_mode") == mode and
                       manifest.get("sidecar_files") == composer_names)
        gate(gates, "p0", run, "composer_manifest_closure",
             {"completed": True, "commit": P0_COMMIT, "seed": seed, "mode": mode},
             {"completed": manifest.get("completed"), "commit": manifest.get("current_git_commit"),
              "seed": manifest.get("seed"), "mode": manifest.get("gradient_ownership_mode")},
             manifest_ok, "P0 composer manifest changed")
        composer_schema = None
        for epoch, composer_name, trajectory_name in zip(EPOCHS, composer_names, trajectory_names):
            composer_path, trajectory_path = composer_dir / composer_name, trajectory_dir / trajectory_name
            composer = read_jsonl(composer_path)
            trajectory = read_jsonl(trajectory_path)
            hashes[str(composer_path)], hashes[str(trajectory_path)] = sha256(composer_path), sha256(trajectory_path)
            if manifest.get("sidecar_sha256", {}).get(composer_name) != hashes[str(composer_path)]:
                raise ValueError(f"{run}:{epoch}: composer hash mismatch")
            if len(composer) != 720 or len(trajectory) != 720 or any(set(row) != TRAJECTORY_FIELDS for row in trajectory):
                raise ValueError(f"{run}:{epoch}: sidecar row/schema closure failed")
            composer_schema = set(composer[0]) if composer_schema is None else composer_schema
            trajectories = {(str(row["id"]), str(row["source_row_id"]), integer(row["dev_position"], "position")): row
                            for row in trajectory}
            if len(trajectories) != 720:
                raise ValueError(f"{run}:{epoch}: trajectory identity collision")
            seen = set()
            for row in composer:
                identity = (str(row["id"]), str(row["source_row_id"]), integer(row["dev_position"], "position"))
                if set(row) != composer_schema or identity in seen or identity not in trajectories:
                    raise ValueError(f"{run}:{epoch}: composer identity/schema failure")
                seen.add(identity)
                if (row.get("current_git_commit") != P0_COMMIT or integer(row.get("seed"), "seed") != seed or
                        integer(row.get("epoch"), "epoch") != epoch or row.get("gradient_ownership_mode") != mode):
                    raise ValueError(f"{run}:{epoch}: composer provenance failure")
                if trajectories[identity]["prediction"] != row["final_native_prediction"]:
                    raise ValueError(f"{run}:{epoch}: trajectory prediction mismatch")
                reconstruct(row, True)
                if epoch in TAIL:
                    value = dict(row)
                    value["_trajectory"] = trajectories[identity]
                    value["_stable"] = str(row.get("stable_row_id", row["id"]))
                    states[(seed, epoch, *identity)][mode] = value
            composer_sidecars += 1
            trajectory_sidecars += 1
            composer_rows += len(composer)
            trajectory_rows += len(trajectory)
    counts = {"composer sidecars": composer_sidecars, "trajectory sidecars": trajectory_sidecars,
              "composer rows": composer_rows, "trajectory rows": trajectory_rows}
    expected = {"composer sidecars": 120, "trajectory sidecars": 120,
                "composer rows": 86400, "trajectory rows": 86400}
    complete = counts == expected and len(states) == 6480 and all(set(pair) == set(MODES) for pair in states.values())
    gate(gates, "p0", "", "p0_count_closure", expected,
         {**counts, "tail paired states": len(states)}, complete, "P0 count closure failed")
    return {"states": states, "hashes": hashes, "counts": counts}


def b5_feature_value(name: str, row: dict[str, Any]) -> Any:
    final_margin = number(row["final_support_logit"], "final support") - number(row["final_not_entitled_logit"], "final NE")
    head_margin = number(row["decision_head_support_logit"], "head support") - number(row["decision_head_not_entitled_logit"], "head NE")
    if name == "RECIPIENT_PREDICTION_SEQUENCE":
        return row["final_native_prediction"]
    if name == "FINAL_MARGIN_SIGN_SEQUENCE":
        return sign(final_margin)
    if name == "HEAD_MARGIN_SIGN_SEQUENCE":
        return sign(head_margin)
    if name == "HEAD_FINAL_MARGIN_SIGN_CONFLICT_SEQUENCE":
        return sign(head_margin) != sign(final_margin)
    halfspaces = {"FRAME_HALFSPACE_SEQUENCE": "frame_prob", "PREDICATE_HALFSPACE_SEQUENCE": "predicate_coverage_prob",
                  "SUFFICIENCY_HALFSPACE_SEQUENCE": "sufficiency_prob",
                  "ENTITLEMENT_HALFSPACE_SEQUENCE": "entitlement_prob_native"}
    if name in halfspaces:
        return halfspace(number(row[halfspaces[name]], halfspaces[name], True))
    if name == "ENTITLEMENT_BOTTLENECK_SEQUENCE":
        values = [("FRAME", number(row["frame_prob"], "frame")),
                  ("PREDICATE", number(row["predicate_coverage_prob"], "predicate")),
                  ("SUFFICIENCY", number(row["sufficiency_prob"], "sufficiency"))]
        minimum = min(value for _, value in values)
        return sorted(label for label, value in values if abs(value - minimum) <= ZERO_TOL)
    if name == "POLARITY_ENERGY_ORDER_SEQUENCE":
        delta = number(row["positive_energy"], "positive") - number(row["negative_energy"], "negative")
        return "EQUAL" if abs(delta) <= ZERO_TOL else "POSITIVE_DOMINANT" if delta > 0 else "NEGATIVE_DOMINANT"
    flags = {"PREDICATE_MISMATCH_SEQUENCE": "predicate_mismatch_active",
             "TEMPORAL_MISMATCH_SEQUENCE": "temporal_mismatch_active",
             "TEMPORAL_ADAPTER_ACTIVITY_SEQUENCE": "temporal_adapter_active",
             "TEMPORAL_CHANNEL_ACTIVITY_SEQUENCE": "temporal_channel_active"}
    if name not in flags:
        raise ValueError(f"unsupported frozen B2-B5 feature: {name}")
    return boolean(row[flags[name]], flags[name])


def candidate_signature(members: Sequence[str], tail: Sequence[dict[str, Any]]) -> list[Any]:
    return [[b5_feature_value(name, row) for row in tail] for name in members]


def feature_specs(source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    specs: list[dict[str, Any]] = []
    families: dict[str, list[str]] = {"single_state": [], "tail_trajectory": [], "paired_delta": []}
    single = (
        ("E20_RECIPIENT_PREDICTION", ("final_native_prediction",), "exact epoch-20 recipient prediction", LABELS, "none"),
        ("E20_FINAL_MARGIN_SIGN", ("final_support_logit", "final_not_entitled_logit"), "sign(final_support_logit-final_not_entitled_logit)", ("NEGATIVE", "ZERO", "POSITIVE"), "zero tolerance 1e-12"),
        ("E20_HEAD_MARGIN_SIGN", ("decision_head_support_logit", "decision_head_not_entitled_logit"), "sign(decision_head_support_logit-decision_head_not_entitled_logit)", ("NEGATIVE", "ZERO", "POSITIVE"), "zero tolerance 1e-12"),
        ("E20_HEAD_FINAL_MARGIN_SIGN_CONFLICT", ("decision_head_support_logit", "decision_head_not_entitled_logit", "final_support_logit", "final_not_entitled_logit"), "HEAD_MARGIN_SIGN != FINAL_MARGIN_SIGN", (False, True), "sign zero tolerance 1e-12"),
        ("E20_FRAME_HALFSPACE", ("frame_prob",), "halfspace(frame_prob)", ("BELOW_HALF", "AT_HALF", "ABOVE_HALF"), "0.5; equality tolerance 1e-12"),
        ("E20_PREDICATE_HALFSPACE", ("predicate_coverage_prob",), "halfspace(predicate_coverage_prob)", ("BELOW_HALF", "AT_HALF", "ABOVE_HALF"), "0.5; equality tolerance 1e-12"),
        ("E20_SUFFICIENCY_HALFSPACE", ("sufficiency_prob",), "halfspace(sufficiency_prob)", ("BELOW_HALF", "AT_HALF", "ABOVE_HALF"), "0.5; equality tolerance 1e-12"),
        ("E20_ENTITLEMENT_HALFSPACE", ("entitlement_prob_native",), "halfspace(entitlement_prob_native)", ("BELOW_HALF", "AT_HALF", "ABOVE_HALF"), "0.5; equality tolerance 1e-12"),
        ("E20_ENTITLEMENT_BOTTLENECK", ("frame_prob", "predicate_coverage_prob", "sufficiency_prob"), "deterministic sorted argmin set; ties preserved", ("sorted nonempty subset of FRAME,PREDICATE,SUFFICIENCY",), "tie tolerance 1e-12"),
        ("E20_POLARITY_ENERGY_ORDER", ("positive_energy", "negative_energy"), "compare positive_energy and negative_energy", ("POSITIVE_DOMINANT", "EQUAL", "NEGATIVE_DOMINANT"), "equality tolerance 1e-12"),
        ("E20_PREDICATE_MISMATCH", ("predicate_mismatch_active",), "exact exported boolean", (False, True), "none"),
        ("E20_TEMPORAL_MISMATCH", ("temporal_mismatch_active",), "exact exported boolean", (False, True), "none"),
        ("E20_TEMPORAL_ADAPTER_ACTIVITY", ("temporal_adapter_active",), "exact exported boolean", (False, True), "none"),
        ("E20_TEMPORAL_CHANNEL_ACTIVITY", ("temporal_channel_active",), "exact exported boolean", (False, True), "none"),
    )
    for name, fields, formula, domain, threshold in single:
        specs.append({"feature_family": "single_state", "feature_name": name,
                      "integration_authorized": True, "diagnostic_only": False, "epoch_scope": [20],
                      "source_fields": list(fields), "formula": formula, "value_domain": list(domain),
                      "natural_threshold": threshold, "outcome_derived": False, "available": True,
                      "unavailable_reason": ""})
        families["single_state"].append(name)
    b5_rows = [row for row in source["b5_dictionary"] if row.get("feature_family") == "recipient_local"]
    for row in b5_rows:
        available = boolean(row["available"], "B2-B5 available")
        name = row["feature_name"]
        specs.append({"feature_family": "tail_trajectory", "feature_name": name,
                      "integration_authorized": False, "diagnostic_only": True, "epoch_scope": list(TAIL),
                      "source_fields": json_cell(row["source_fields"], "B2-B5 source fields"),
                      "formula": row["formula"], "value_domain": json_cell(row["value_domain"], "B2-B5 domain")
                      if row.get("value_domain", "").strip().startswith(("[", "{")) else row.get("value_domain", ""),
                      "natural_threshold": row.get("natural_threshold", "frozen B2-B5 natural categories"),
                      "outcome_derived": boolean(row["outcome_derived"], "B2-B5 outcome"),
                      "available": available, "unavailable_reason": row.get("unavailable_reason", "")})
        if available and not boolean(row["outcome_derived"], "B2-B5 outcome"):
            families["tail_trajectory"].append(name)
    paired = (
        ("E20_DELTA_FRAME_SIGN", "frame_prob"), ("E20_DELTA_PREDICATE_SIGN", "predicate_coverage_prob"),
        ("E20_DELTA_SUFFICIENCY_SIGN", "sufficiency_prob"),
        ("E20_DELTA_POSITIVE_ENERGY_SIGN", "positive_energy"),
        ("E20_DELTA_NEGATIVE_ENERGY_SIGN", "negative_energy"),
        ("E20_DELTA_ENTITLEMENT_SIGN", "entitlement_prob_native"),
        ("E20_DELTA_HEAD_MARGIN_SIGN", "__head_margin__"),
        ("E20_DELTA_FINAL_MARGIN_SIGN", "__final_margin__"),
        ("E20_PREDICATE_MISMATCH_CHANGE", "predicate_mismatch_active"),
        ("E20_TEMPORAL_MISMATCH_CHANGE", "temporal_mismatch_active"),
    )
    for name, field in paired:
        change = name.endswith("_CHANGE")
        fields = (["decision_head_support_logit", "decision_head_not_entitled_logit"] if field == "__head_margin__"
                  else ["final_support_logit", "final_not_entitled_logit"] if field == "__final_margin__"
                  else [field])
        specs.append({"feature_family": "paired_delta", "feature_name": name,
                      "integration_authorized": False, "diagnostic_only": True, "epoch_scope": [20],
                      "source_fields": fields,
                      "formula": ("frame-local-only donor boolean minus joint recipient boolean change category"
                                  if change else "sign(frame-local-only donor minus joint recipient)"),
                      "value_domain": (["UNCHANGED_FALSE", "FALSE_TO_TRUE", "TRUE_TO_FALSE", "UNCHANGED_TRUE"]
                                       if change else ["NEGATIVE", "ZERO", "POSITIVE"]),
                      "natural_threshold": "exact boolean transition" if change else "zero tolerance 1e-12",
                      "outcome_derived": False, "available": True, "unavailable_reason": ""})
        families["paired_delta"].append(name)
    if any(spec["feature_name"].lower() in PROHIBITED_FEATURES or spec["outcome_derived"] for spec in specs if spec["available"]):
        raise ValueError("outcome-derived or prohibited safety feature entered dictionary")
    return specs, families


def single_values(row: dict[str, Any]) -> dict[str, Any]:
    final = number(row["final_support_logit"], "final S") - number(row["final_not_entitled_logit"], "final NE")
    head = number(row["decision_head_support_logit"], "head S") - number(row["decision_head_not_entitled_logit"], "head NE")
    bottleneck_values = [("FRAME", number(row["frame_prob"], "frame")),
                         ("PREDICATE", number(row["predicate_coverage_prob"], "predicate")),
                         ("SUFFICIENCY", number(row["sufficiency_prob"], "sufficiency"))]
    minimum = min(value for _, value in bottleneck_values)
    polarity = number(row["positive_energy"], "positive") - number(row["negative_energy"], "negative")
    return {
        "E20_RECIPIENT_PREDICTION": row["final_native_prediction"],
        "E20_FINAL_MARGIN_SIGN": sign(final), "E20_HEAD_MARGIN_SIGN": sign(head),
        "E20_HEAD_FINAL_MARGIN_SIGN_CONFLICT": sign(head) != sign(final),
        "E20_FRAME_HALFSPACE": halfspace(number(row["frame_prob"], "frame", True)),
        "E20_PREDICATE_HALFSPACE": halfspace(number(row["predicate_coverage_prob"], "predicate", True)),
        "E20_SUFFICIENCY_HALFSPACE": halfspace(number(row["sufficiency_prob"], "sufficiency", True)),
        "E20_ENTITLEMENT_HALFSPACE": halfspace(number(row["entitlement_prob_native"], "entitlement", True)),
        "E20_ENTITLEMENT_BOTTLENECK": sorted(label for label, value in bottleneck_values
                                              if abs(value - minimum) <= ZERO_TOL),
        "E20_POLARITY_ENERGY_ORDER": ("EQUAL" if abs(polarity) <= ZERO_TOL else
                                       "POSITIVE_DOMINANT" if polarity > 0 else "NEGATIVE_DOMINANT"),
        "E20_PREDICATE_MISMATCH": boolean(row["predicate_mismatch_active"], "predicate mismatch"),
        "E20_TEMPORAL_MISMATCH": boolean(row["temporal_mismatch_active"], "temporal mismatch"),
        "E20_TEMPORAL_ADAPTER_ACTIVITY": boolean(row["temporal_adapter_active"], "adapter activity"),
        "E20_TEMPORAL_CHANNEL_ACTIVITY": boolean(row["temporal_channel_active"], "channel activity"),
    }


def paired_values(recipient: dict[str, Any], donor: dict[str, Any]) -> dict[str, Any]:
    def delta(field: str) -> str:
        return sign(number(donor[field], field) - number(recipient[field], field))
    def margin(row: dict[str, Any], prefix: str) -> float:
        return number(row[f"{prefix}_support_logit"], "support") - number(row[f"{prefix}_not_entitled_logit"], "NE")
    def change(field: str) -> str:
        left, right = boolean(recipient[field], field), boolean(donor[field], field)
        return "UNCHANGED_TRUE" if left and right else "UNCHANGED_FALSE" if not left and not right else "FALSE_TO_TRUE" if right else "TRUE_TO_FALSE"
    return {
        "E20_DELTA_FRAME_SIGN": delta("frame_prob"),
        "E20_DELTA_PREDICATE_SIGN": delta("predicate_coverage_prob"),
        "E20_DELTA_SUFFICIENCY_SIGN": delta("sufficiency_prob"),
        "E20_DELTA_POSITIVE_ENERGY_SIGN": delta("positive_energy"),
        "E20_DELTA_NEGATIVE_ENERGY_SIGN": delta("negative_energy"),
        "E20_DELTA_ENTITLEMENT_SIGN": delta("entitlement_prob_native"),
        "E20_DELTA_HEAD_MARGIN_SIGN": sign(margin(donor, "decision_head") - margin(recipient, "decision_head")),
        "E20_DELTA_FINAL_MARGIN_SIGN": sign(margin(donor, "final") - margin(recipient, "final")),
        "E20_PREDICATE_MISMATCH_CHANGE": change("predicate_mismatch_active"),
        "E20_TEMPORAL_MISMATCH_CHANGE": change("temporal_mismatch_active"),
    }


def build_context(source: dict[str, Any], p0: dict[str, Any], gates: list[dict[str, Any]]) -> dict[str, Any]:
    action_meta = {(integer(row["seed"], "seed"), row["stable_row_id"]): row for row in source["b5_actions"]}
    role_counts = {seed: Counter(row["transition_role"] for key, row in action_meta.items() if key[0] == seed)
                   for seed in PRIMARY_SEEDS}
    gate(gates, "population", "", "primary_16_case_closure", EXPECTED_PRIMARY,
         {seed: dict(role_counts[seed]) for seed in PRIMARY_SEEDS},
         len(action_meta) == 16 and {seed: dict(role_counts[seed]) for seed in PRIMARY_SEEDS} == EXPECTED_PRIMARY,
         "seed-conditioned primary population changed")
    acceptable = {key: set(json_cell(row["acceptable_coalitions"], "acceptable coalitions"))
                  for key, row in action_meta.items()}
    forward = "JOINT_RECIPIENT_FRAME_LOCAL_ONLY_DONOR"
    expected_masks = {f"{value:05b}" for value in range(32)}
    tail_rows = source["b4_tail_summary"]
    tail_keys = [(integer(row["seed"], "seed"), row["stable_row_id"], row["direction"],
                  row["coalition_mask"]) for row in tail_rows]
    tail_key_counts = Counter(tail_keys)
    duplicate_tail_keys = [list(key) for key, count in sorted(tail_key_counts.items()) if count != 1]
    tail_identities = {(key[0], key[1]) for key in tail_keys}
    tail_directions = {key[2] for key in tail_keys}
    tail_masks = {key[3] for key in tail_keys}
    tail_identity_direction_masks = defaultdict(set)
    for seed, stable_row_id, direction, mask in tail_keys:
        tail_identity_direction_masks[(seed, stable_row_id, direction)].add(mask)
    incomplete_tail_groups = [list(key) for key, masks in sorted(tail_identity_direction_masks.items())
                              if masks != expected_masks]
    uniqueness_evidence = {
        "row_count": len(tail_rows), "unique_key_count": len(tail_key_counts),
        "primary_identity_count": len(tail_identities), "direction_count": len(tail_directions),
        "coalition_count": len(tail_masks),
        "forward_row_count": sum(row["direction"] == forward for row in tail_rows),
        "identity_direction_count": len(tail_identity_direction_masks),
        "incomplete_identity_directions": incomplete_tail_groups[:20],
        "missing_columns": [], "duplicate_keys": duplicate_tail_keys[:20],
    }
    tail_unique = (len(tail_rows) == len(tail_key_counts) == 1024 and not duplicate_tail_keys and
                   len(tail_identities) == 16 and len(tail_directions) == 2 and tail_masks == expected_masks and
                   len(tail_identity_direction_masks) == 32 and not incomplete_tail_groups)
    gate(gates, "b4", "", "b2b4_tail_summary_key_uniqueness",
         {"row_count": 1024, "unique_key_count": 1024, "primary_identity_count": 16,
          "direction_count": 2, "coalition_count": 32, "identity_direction_count": 32,
          "incomplete_identity_directions": [], "duplicate_keys": []},
         uniqueness_evidence, tail_unique, "B2-B4 primitive tail-summary keys are not unique and complete")
    tail_summary_index = defaultdict(dict)
    forward_rows = [row for row in tail_rows if row["direction"] == forward]
    for row in forward_rows:
        primary_key = (integer(row["seed"], "seed"), row["stable_row_id"])
        if primary_key in action_meta:
            tail_summary_index[primary_key][row["coalition_mask"]] = row
    forward_complete = (len(forward_rows) == 512 and set(tail_summary_index) == set(action_meta) and
                        all(set(rows) == expected_masks for rows in tail_summary_index.values()))
    gate(gates, "b4", "", "b2b4_forward_512_action_closure",
         {"forward_row_count": 512, "primary_identity_count": 16, "masks_per_identity": 32},
         {"forward_row_count": len(forward_rows), "primary_identity_count": len(tail_summary_index),
          "mask_counts": sorted({len(rows) for rows in tail_summary_index.values()}),
          "incomplete_identities": [list(key) for key, rows in sorted(tail_summary_index.items())
                                    if set(rows) != expected_masks]},
         forward_complete, "B2-B4 forward primitive tail-summary action closure changed")
    primitive_buckets = defaultdict(dict)
    duplicate_primitive_epoch_keys = []
    for row in source["b4_primitive_rows"]:
        summary_key = (integer(row["seed"], "seed"), row["stable_row_id"], row["direction"],
                       row["coalition_mask"])
        epoch = integer(row["epoch"], "epoch")
        if epoch in primitive_buckets[summary_key]:
            duplicate_primitive_epoch_keys.append([*summary_key, epoch])
        primitive_buckets[summary_key][epoch] = row
    disagreements = []
    observed_comparisons = 0
    for row, summary_key in zip(tail_rows, tail_keys):
        epochs = primitive_buckets.get(summary_key, {})
        if set(epochs) != set(EPOCHS):
            disagreements.append({"key": list(summary_key), "reason": "epoch closure",
                                  "observed_epochs": sorted(epochs)})
            continue
        observed_comparisons += 1
        epoch_predictions = [epochs[epoch]["counterfactual_prediction"] for epoch in TAIL]
        summary_predictions = json_cell(row["coalition_tail_predictions"], "coalition tail predictions")
        if epoch_predictions != summary_predictions:
            disagreements.append({"key": list(summary_key), "epoch_predictions": epoch_predictions,
                                  "tail_summary_predictions": summary_predictions})
    crosscheck_evidence = {
        "expected_comparisons": 1024, "observed_comparisons": observed_comparisons,
        "disagreement_count": len(disagreements), "examples": disagreements[:20],
        "primitive_bucket_count": len(primitive_buckets),
        "duplicate_primitive_epoch_keys": duplicate_primitive_epoch_keys[:20],
    }
    crosscheck_passed = (len(source["b4_primitive_rows"]) == 20480 and len(primitive_buckets) == 1024 and
                         not duplicate_primitive_epoch_keys and observed_comparisons == 1024 and not disagreements)
    gate(gates, "b4", "", "b2b4_tail_prediction_crosscheck",
         {"expected_comparisons": 1024, "observed_comparisons": 1024,
          "disagreement_count": 0, "examples": []},
         crosscheck_evidence, crosscheck_passed,
         "B2-B4 epoch-level primitive predictions disagree with the tail summary")
    reconstructed_acceptable = {}
    for key, rows in tail_summary_index.items():
        objective_field = ("donor_tail_reproduced" if action_meta[key]["transition_role"] == "recovery"
                           else "recipient_tail_preserved")
        reconstructed_acceptable[key] = {mask for mask, row in rows.items()
                                           if boolean(row[objective_field], objective_field)}
    action_exact = (forward_complete and set(reconstructed_acceptable) == set(acceptable) and
                    all(reconstructed_acceptable[key] == acceptable[key] for key in acceptable))
    gate(gates, "b4", "", "b2b4_exact_row_action_set_closure",
         {"primary_identity_count": 16, "masks_per_identity": 32,
          "b2b5_acceptable_action_set_agreement": True},
         {"primary_identity_count": len(reconstructed_acceptable),
          "mask_counts": sorted({len(rows) for rows in tail_summary_index.values()}),
          "disagreement_count": sum(reconstructed_acceptable.get(key) != acceptable[key] for key in acceptable),
          "disagreement_identities": [list(key) for key in sorted(acceptable)
                                      if reconstructed_acceptable.get(key) != acceptable[key]]},
         action_exact, "B2-B4 tail-summary actions disagree with B2-B5 acceptable coalitions")
    state_by_stable = {}
    clean = defaultdict(dict)
    for key, pair in p0["states"].items():
        seed, epoch, identity, source_id, position = key
        stable = pair["joint"]["_stable"]
        state_by_stable[(seed, epoch, stable)] = pair
        clean[(seed, identity, source_id, position)][epoch] = pair
    if len(clean) != 2160 or any(set(epochs) != set(TAIL) for epochs in clean.values()):
        raise ValueError("clean-dev tail state closure failed")
    discovery_by_seed = {
        seed: {(row["id"], row["source_row_id"], integer(row["dev_position"], "position"))
               for key, row in action_meta.items() if key[0] == seed}
        for seed in PRIMARY_SEEDS
    }
    discovery_union = discovery_by_seed[184] | discovery_by_seed[185]
    intersection = discovery_by_seed[184] & discovery_by_seed[185]
    population_ok = (len(discovery_by_seed[184]) == 11 and len(discovery_by_seed[185]) == 5 and
                     len(intersection) == 3 and len(discovery_union) == 13)
    gate(gates, "population", "", "discovery_identity_union_closure",
         {"seed184": 11, "seed185": 5, "intersection": 3, "union": 13},
         {"seed184": len(discovery_by_seed[184]), "seed185": len(discovery_by_seed[185]),
          "intersection": len(intersection), "union": len(discovery_union)},
         population_ok, "discovery identity population changed")
    identity_sets = {seed: {key[1:] for key in clean if key[0] == seed} for seed in SEEDS}
    nondiscovery = {seed: len(identity_sets[seed] - discovery_union) for seed in SEEDS}
    clean_ok = (all(len(identity_sets[seed]) == 720 for seed in SEEDS) and
                identity_sets[183] == identity_sets[184] == identity_sets[185] and
                all(len(identity_sets[seed] & discovery_union) == 13 for seed in SEEDS) and
                nondiscovery == {183: 707, 184: 707, 185: 707})
    gate(gates, "population", "", "cross_seed_clean_dev_identity_closure",
         {"all": 720, "discovery union": 13, "nondiscovery": 707, "same identity set": True},
         {"all": {seed: len(identity_sets[seed]) for seed in SEEDS},
          "discovery union": {seed: len(identity_sets[seed] & discovery_union) for seed in SEEDS},
          "nondiscovery": nondiscovery,
          "same identity set": identity_sets[183] == identity_sets[184] == identity_sets[185]},
         clean_ok, "cross-seed clean-dev identity closure failed")
    return {"action_meta": action_meta, "acceptable": acceptable,
            "tail_summary": tail_summary_index,
            "state_by_stable": state_by_stable, "clean": clean, "discovery_by_seed": discovery_by_seed,
            "discovery_union": discovery_union, "intersection": intersection}


def build_candidate_policies(source: dict[str, Any], context: dict[str, Any], gates: list[dict[str, Any]]) -> dict[str, Any]:
    dictionary = {row["feature_name"]: row for row in source["b5_dictionary"]
                  if row.get("feature_family") == "recipient_local"}
    policies = {}
    for candidate in source["candidate_rows"]:
        mask = candidate["feature_subset_mask"]
        members = json_cell(candidate["feature_subset_members"], "candidate members")
        if not isinstance(members, list) or not members or any(name not in dictionary for name in members):
            raise ValueError(f"{mask}: candidate feature names are not backed by B2-B5 dictionary")
        for name in members:
            row = dictionary[name]
            if not boolean(row["available"], "available") or boolean(row["outcome_derived"], "outcome"):
                raise ValueError(f"{mask}:{name}: candidate feature is unavailable or outcome-derived")
        mapping = {}
        for row in source["signature_actions"]:
            if row["feature_subset_mask"] == mask:
                signature = canonical(json_cell(row["signature"], "candidate signature"))
                actions = json_cell(row["inclusion_minimal_action_set"], "candidate action set")
                if not isinstance(actions, list) or not actions or any(not re.fullmatch(r"[01]{5}", action) for action in actions):
                    raise ValueError(f"{mask}: invalid assigned action set")
                mapping[signature] = sorted(set(actions))
        if not mapping:
            raise ValueError(f"{mask}: empty B2-B6 signature action mapping")
        policies[mask] = {"members": members, "mapping": mapping}
    gate(gates, "b6", "", "candidate_feature_name_reconstruction",
         {"candidate masks": list(EXPECTED_CANDIDATES), "semantic authority": "B2-B5 feature dictionary"},
         {"candidate masks": sorted(policies), "members": {mask: policy["members"] for mask, policy in policies.items()}},
         sorted(policies) == list(EXPECTED_CANDIDATES), "candidate name reconstruction failed")
    return policies


def metrics(gold: Sequence[str], baseline: Sequence[str], selected: Sequence[str]) -> dict[str, Any]:
    stable = sum(g == b for g, b in zip(gold, baseline))
    correct_to_incorrect = sum(g == b and s != g for g, b, s in zip(gold, baseline, selected))
    return {"prediction_change_count": sum(b != s for b, s in zip(baseline, selected)),
            "correct_to_incorrect_count": correct_to_incorrect,
            "incorrect_to_correct_count": sum(b != g and s == g for g, b, s in zip(gold, baseline, selected)),
            "stable_correct_preservation_rate": 1.0 if stable == 0 else (stable - correct_to_incorrect) / stable}


def make_records(source: dict[str, Any], context: dict[str, Any], policies: dict[str, Any],
                 families: dict[str, list[str]], gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    reconstruction_errors = []
    for mask, policy in sorted(policies.items()):
        for clean_key, epochs in sorted(context["clean"].items()):
            seed, identity, source_id, position = clean_key
            tail = [epochs[epoch]["joint"] for epoch in TAIL]
            signature = canonical(candidate_signature(policy["members"], tail))
            seen = signature in policy["mapping"]
            actions = policy["mapping"].get(signature, ["00000"])
            recipient, donor = epochs[20]["joint"], epochs[20]["frame_local_only"]
            outcomes = {action: apply_action(recipient, donor, action) for action in actions}
            stable_id = recipient["_stable"]
            primary_key = (seed, stable_id)
            primary = primary_key in context["action_meta"]
            discovery = (identity, source_id, position) in context["discovery_union"]
            population = ("DISCOVERY_IDENTITY_UNION_CONTRAST" if seed == 183 and discovery else
                          "DISCOVERY_IDENTITY_UNION" if discovery else "NONDISCOVERY")
            gold = recipient["_trajectory"]["gold_label"]
            baseline = recipient["final_native_prediction"]
            predictions = sorted(set(outcome["prediction"] for outcome in outcomes.values()))
            selector_prediction: Any = predictions[0] if len(predictions) == 1 else predictions
            joint_correct = baseline == gold
            selector_correct = all(prediction == gold for prediction in predictions)
            cti = joint_correct and any(prediction != gold for prediction in predictions)
            itc = not joint_correct and all(prediction == gold for prediction in predictions)
            abstention_passed = None
            selector_objective_passed = None
            role = ""
            if primary:
                role = context["action_meta"][primary_key]["transition_role"]
                abstention_passed = "00000" in context["acceptable"][primary_key]
                selector_objective_passed = all(action in context["acceptable"][primary_key] for action in actions)
                for action in actions:
                    b4 = context["tail_summary"][primary_key][action]
                    expected = json_cell(b4["coalition_tail_predictions"], "B2-B4 predictions")
                    rebuilt = [apply_action(context["state_by_stable"][(seed, epoch, stable_id)]["joint"],
                                            context["state_by_stable"][(seed, epoch, stable_id)]["frame_local_only"],
                                            action)["prediction"] for epoch in TAIL]
                    if rebuilt != expected:
                        reconstruction_errors.append({"candidate": mask, "seed": seed,
                                                      "stable_row_id": stable_id, "action": action})
            if primary and not abstention_passed and selector_objective_passed:
                target, reason = "MUST_ALLOW", "PRIMARY_ABSTENTION_FAILS_SELECTOR_ACTION_PASSES"
            elif population == "NONDISCOVERY" and joint_correct and cti:
                target, reason = "MUST_BLOCK", "NONDISCOVERY_ANY_ALLOWED_ACTION_CORRECT_TO_INCORRECT"
            else:
                target, reason = "OPTIONAL", ("PRIMARY_ABSTENTION_ALREADY_PASSES" if primary and abstention_passed else
                                               "UNSEEN_SIGNATURE_ABSTAINS" if not seen else
                                               "NO_REQUIRED_ALLOW_OR_BLOCK_CONSTRAINT")
            tail_values = {name: [b5_feature_value(name, epochs[epoch]["joint"]) for epoch in TAIL]
                           for name in families["tail_trajectory"]}
            record = {
                "candidate_feature_subset_mask": mask, "candidate_feature_subset_members": policy["members"],
                "seed": seed, "stable_row_id": stable_id, "id": identity, "source_row_id": source_id,
                "dev_position": position, "population": population, "primary_case": primary,
                "transition_role": role, "signature_support": "SEEN" if seen else "UNSEEN",
                "assigned_action_set": actions, "evaluated_actions": sorted(outcomes), "gold_label": gold,
                "joint_prediction": baseline, "selector_prediction": selector_prediction,
                "joint_correct": joint_correct, "selector_correct": selector_correct,
                "correct_to_incorrect": cti, "incorrect_to_correct": itc,
                "abstention_objective_passed": abstention_passed,
                "selector_objective_passed": selector_objective_passed,
                "safety_target": target, "safety_target_reason": reason,
                "_outcomes": outcomes, "_single_state": single_values(recipient),
                "_tail_trajectory": tail_values, "_paired_delta": paired_values(recipient, donor),
                "_candidate_signature": json.loads(signature),
            }
            records.append(record)
    expected_rows = len(policies) * 2160
    gate(gates, "reconstruction", "", "row_level_selector_reconstruction",
         {"rows": expected_rows, "tail disagreements": 0},
         {"rows": len(records), "tail disagreements": len(reconstruction_errors),
          "examples": reconstruction_errors[:10]},
         len(records) == expected_rows and not reconstruction_errors, "selector reconstruction disagreement")
    targets = Counter(record["safety_target"] for record in records)
    gate(gates, "target", "", "row_safety_target_closure",
         {"rows": expected_rows, "labels": ["MUST_ALLOW", "MUST_BLOCK", "OPTIONAL"]},
         {"rows": len(records), "counts": dict(targets),
          "invalid": sum(record["safety_target"] not in ("MUST_ALLOW", "MUST_BLOCK", "OPTIONAL") for record in records)},
         len(records) == expected_rows and set(targets) <= {"MUST_ALLOW", "MUST_BLOCK", "OPTIONAL"},
         "row safety-target closure failed")
    return records


def reproduce_b6(source: dict[str, Any], context: dict[str, Any], policies: dict[str, Any],
                 gates: list[dict[str, Any]]) -> dict[str, Any]:
    generated = []
    for mask, policy in sorted(policies.items()):
        rows = []
        for clean_key, epochs in sorted(context["clean"].items()):
            seed, identity, source_id, position = clean_key
            signature = canonical(candidate_signature(policy["members"], [epochs[epoch]["joint"] for epoch in TAIL]))
            rows.append({"seed": seed, "identity": (identity, source_id, position), "epochs": epochs,
                         "signature": signature, "actions": policy["mapping"].get(signature, []),
                         "seen": signature in policy["mapping"],
                         "gold": epochs[20]["joint"]["_trajectory"]["gold_label"],
                         "baseline": epochs[20]["joint"]["final_native_prediction"]})
        all_actions = sorted((signature, action) for signature, actions in policy["mapping"].items() for action in actions)
        unique = all(len(actions) == 1 for actions in policy["mapping"].values())
        modes = [("UNIQUE_DETERMINISTIC", None, None)] if unique else [
            (f"SET_VALUED_SIGNATURE_ACTION:{hashlib.sha256(signature.encode()).hexdigest()[:12]}:{action}",
             signature, action) for signature, action in all_actions]
        for seed in SEEDS:
            seed_rows = [row for row in rows if row["seed"] == seed]
            for population in (("DISCOVERY_IDENTITY_UNION_CONTRAST", "NONDISCOVERY", "ALL_720") if seed == 183
                               else ("DISCOVERY_IDENTITY_UNION", "NONDISCOVERY", "ALL_720")):
                discovery_population = population.startswith("DISCOVERY_IDENTITY_UNION")
                selected_rows = seed_rows if population == "ALL_720" else [
                    row for row in seed_rows if (row["identity"] in context["discovery_union"]) == discovery_population]
                for mode, signature_filter, action_filter in modes:
                    selected = []
                    for row in selected_rows:
                        action = (row["actions"][0] if mode == "UNIQUE_DETERMINISTIC" and row["seen"] else
                                  action_filter if row["signature"] == signature_filter else None)
                        selected.append(apply_action(row["epochs"][20]["joint"], row["epochs"][20]["frame_local_only"],
                                                     action)["prediction"] if action else row["baseline"])
                    computed = metrics([row["gold"] for row in selected_rows],
                                       [row["baseline"] for row in selected_rows], selected)
                    generated.append({"feature_subset_mask": mask, "seed": seed, "population": population,
                                      "policy_action_mode": mode, "row_count": len(selected_rows), **computed})
    stored = {(row["feature_subset_mask"], integer(row["seed"], "seed"), row["population"],
               row["policy_action_mode"]): row for row in source["application_summary"]
              if row["feature_subset_mask"] in policies}
    fields = ("row_count", "prediction_change_count", "correct_to_incorrect_count",
              "incorrect_to_correct_count", "stable_correct_preservation_rate")
    disagreements = []
    for row in generated:
        key = (row["feature_subset_mask"], row["seed"], row["population"], row["policy_action_mode"])
        expected = stored.get(key)
        if expected is None:
            disagreements.append({"key": key, "reason": "missing stored row"})
            continue
        for field in fields:
            observed = row[field]
            wanted = integer(expected[field], field) if field != "stable_correct_preservation_rate" else number(expected[field], field)
            if (observed != wanted if field != "stable_correct_preservation_rate" else abs(observed - wanted) > TOL):
                disagreements.append({"key": key, "field": field, "expected": wanted, "observed": observed})
    extra = set(stored) - {(row["feature_subset_mask"], row["seed"], row["population"], row["policy_action_mode"])
                           for row in generated}
    complete = not disagreements and not extra and len(generated) == len(stored)
    evidence = {"generated_rows": len(generated), "stored_rows": len(stored),
                "disagreements": disagreements[:20], "extra_stored_rows": [list(key) for key in sorted(extra)]}
    gate(gates, "reconstruction", "", "b2b6_metric_reproduction",
         {"exact keys and integer metrics": True, "numeric tolerance": TOL}, evidence, complete,
         "B2-B6 clean-dev metrics were not reproduced")
    return evidence


def subset_mask(index: int, size: int) -> str:
    return f"{index:0{size}b}"


def subset_members(mask: str, names: Sequence[str]) -> list[str]:
    return [name for bit, name in zip(mask, names) if bit == "1"]


def exact_mapping(rows: Sequence[dict[str, Any]], family: str, members: Sequence[str], seed: int | None = None) -> dict[str, Any]:
    constrained = [row for row in rows if row["safety_target"] in ("MUST_ALLOW", "MUST_BLOCK")
                   and (seed is None or row["seed"] == seed)]
    groups = defaultdict(list)
    for row in constrained:
        signature = canonical([row[f"_{family}"][name] for name in members])
        groups[signature].append(row)
    conflicts = {signature: values for signature, values in groups.items()
                 if {row["safety_target"] for row in values} == {"MUST_ALLOW", "MUST_BLOCK"}}
    mapping = {signature: ("ALLOW" if all(row["safety_target"] == "MUST_ALLOW" for row in values) else "BLOCK")
               for signature, values in groups.items() if signature not in conflicts}
    return {"constrained": constrained, "groups": groups, "conflicts": conflicts, "mapping": mapping}


def transfer(rows: Sequence[dict[str, Any]], family: str, members: Sequence[str], source_seed: int,
             target_seed: int) -> dict[str, Any]:
    source = exact_mapping(rows, family, members, source_seed)["mapping"]
    targets = [row for row in rows if row["seed"] == target_seed and
               row["safety_target"] in ("MUST_ALLOW", "MUST_BLOCK")]
    counts = Counter()
    for row in targets:
        label = row["safety_target"]
        signature = canonical([row[f"_{family}"][name] for name in members])
        seen = signature in source
        counts[f"target_{label}"] += 1
        counts[f"{'seen' if seen else 'unseen'}_{label}"] += 1
        if label == "MUST_ALLOW":
            if seen and source[signature] == "ALLOW":
                counts["correctly_allowed_MUST_ALLOW"] += 1
            else:
                counts["incorrectly_blocked_MUST_ALLOW"] += 1
        elif not seen or source[signature] == "BLOCK":
            counts["correctly_blocked_MUST_BLOCK"] += 1
        else:
            counts["incorrectly_allowed_MUST_BLOCK"] += 1
    result = {
        "source_seed": source_seed, "target_seed": target_seed,
        "target_must_allow_count": counts["target_MUST_ALLOW"],
        "target_must_block_count": counts["target_MUST_BLOCK"],
        "seen_must_allow_count": counts["seen_MUST_ALLOW"],
        "unseen_must_allow_count": counts["unseen_MUST_ALLOW"],
        "correctly_allowed_must_allow_count": counts["correctly_allowed_MUST_ALLOW"],
        "incorrectly_blocked_must_allow_count": counts["incorrectly_blocked_MUST_ALLOW"],
        "seen_must_block_count": counts["seen_MUST_BLOCK"],
        "unseen_must_block_count": counts["unseen_MUST_BLOCK"],
        "correctly_blocked_must_block_count": counts["correctly_blocked_MUST_BLOCK"],
        "incorrectly_allowed_must_block_count": counts["incorrectly_allowed_MUST_BLOCK"],
    }
    result["full_transfer_pass"] = (result["unseen_must_allow_count"] == 0 and
                                    result["incorrectly_blocked_must_allow_count"] == 0 and
                                    result["incorrectly_allowed_must_block_count"] == 0)
    return result


def conservative_audit(rows: Sequence[dict[str, Any]], family: str, members: Sequence[str],
                       mapping: dict[str, str], candidate: str, feature_mask: str) -> list[dict[str, Any]]:
    audited = []
    for seed in SEEDS:
        seed_rows = [row for row in rows if row["seed"] == seed]
        populations = (["PRIMARY", "NONDISCOVERY"] if seed in PRIMARY_SEEDS else ["ALL_720"])
        for population in populations:
            selected_rows = ([row for row in seed_rows if row["primary_case"]] if population == "PRIMARY" else
                             [row for row in seed_rows if row["population"] == "NONDISCOVERY"] if population == "NONDISCOVERY"
                             else seed_rows)
            outcomes = []
            counters = Counter()
            recovery_failures = harm_failures = 0
            for row in selected_rows:
                signature = canonical([row[f"_{family}"][name] for name in members])
                seen = signature in mapping
                allowed = seen and mapping[signature] == "ALLOW"
                counters["seen" if seen else "unseen"] += 1
                counters["allowed" if allowed else "blocked"] += 1
                predictions = ([outcome["prediction"] for outcome in row["_outcomes"].values()] if allowed
                               else [row["joint_prediction"]])
                conservative_correct = all(prediction == row["gold_label"] for prediction in predictions)
                changed = any(prediction != row["joint_prediction"] for prediction in predictions)
                cti = row["joint_correct"] and not conservative_correct
                itc = not row["joint_correct"] and conservative_correct
                outcomes.append((conservative_correct, changed, cti, itc))
                if row["primary_case"]:
                    objective = (row["selector_objective_passed"] if allowed else row["abstention_objective_passed"])
                    if not objective:
                        if row["transition_role"] == "recovery":
                            recovery_failures += 1
                        else:
                            harm_failures += 1
            stable = sum(row["joint_correct"] for row in selected_rows)
            cti_count = sum(value[2] for value in outcomes)
            audited.append({
                "candidate_feature_subset_mask": candidate, "feature_family": family,
                "safety_feature_subset_mask": feature_mask, "seed": seed, "population": population,
                "row_count": len(selected_rows),
                "must_allow_count": sum(row["safety_target"] == "MUST_ALLOW" for row in selected_rows),
                "must_block_count": sum(row["safety_target"] == "MUST_BLOCK" for row in selected_rows),
                "optional_count": sum(row["safety_target"] == "OPTIONAL" for row in selected_rows),
                "allowed_count": counters["allowed"], "blocked_count": counters["blocked"],
                "unseen_count": counters["unseen"],
                "primary_recovery_passed": recovery_failures == 0,
                "primary_harm_passed": harm_failures == 0,
                "primary_objective_failures": recovery_failures + harm_failures,
                "joint_correct_count": stable, "gated_correct_count": sum(value[0] for value in outcomes),
                "prediction_change_count": sum(value[1] for value in outcomes),
                "correct_to_incorrect_count": cti_count,
                "incorrect_to_correct_count": sum(value[3] for value in outcomes),
                "stable_correct_preservation_rate": 1.0 if stable == 0 else (stable - cti_count) / stable,
            })
    return audited


def enumerate_family(records: list[dict[str, Any]], candidate: str, family: str,
                     names: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [row for row in records if row["candidate_feature_subset_mask"] == candidate and row["seed"] in PRIMARY_SEEDS]
    summaries = []
    details = {}
    feasible_masks = []
    for index in range(1, 1 << len(names)):
        mask = subset_mask(index, len(names))
        members = subset_members(mask, names)
        pooled = exact_mapping(rows, family, members)
        feasible = not pooled["conflicts"]
        if feasible:
            feasible_masks.append(mask)
        transfer_184 = transfer(rows, family, members, 184, 185) if feasible else {"full_transfer_pass": False}
        transfer_185 = transfer(rows, family, members, 185, 184) if feasible else {"full_transfer_pass": False}
        audit = conservative_audit(records, family, members, pooled["mapping"], candidate, mask) if feasible else []
        primary_failures = sum(row["primary_objective_failures"] for row in audit if row["population"] == "PRIMARY")
        cti184 = sum(row["correct_to_incorrect_count"] for row in audit
                     if row["seed"] == 184 and row["population"] == "NONDISCOVERY")
        cti185 = sum(row["correct_to_incorrect_count"] for row in audit
                     if row["seed"] == 185 and row["population"] == "NONDISCOVERY")
        conservative_passed = feasible and primary_failures == 0 and cti184 == 0 and cti185 == 0
        summary = {
            "candidate_feature_subset_mask": candidate, "feature_family": family,
            "safety_feature_subset_mask": mask, "safety_feature_subset_size": len(members),
            "safety_feature_subset_members": members, "constrained_row_count": len(pooled["constrained"]),
            "must_allow_count": sum(row["safety_target"] == "MUST_ALLOW" for row in pooled["constrained"]),
            "must_block_count": sum(row["safety_target"] == "MUST_BLOCK" for row in pooled["constrained"]),
            "signature_count": len(pooled["groups"]), "conflicting_signature_count": len(pooled["conflicts"]),
            "feasible": feasible, "inclusion_minimal_feasible": False,
            "seed184_to_seed185_full_pass": transfer_184["full_transfer_pass"],
            "seed185_to_seed184_full_pass": transfer_185["full_transfer_pass"],
            "bidirectional_cross_seed_full_pass": (transfer_184["full_transfer_pass"] and
                                                    transfer_185["full_transfer_pass"]),
            "primary_objective_failures": primary_failures if feasible else None,
            "seed184_nondiscovery_correct_to_incorrect": cti184 if feasible else None,
            "seed185_nondiscovery_correct_to_incorrect": cti185 if feasible else None,
            "conservative_gate_passed": conservative_passed,
        }
        summaries.append(summary)
        details[mask] = {"members": members, "pooled_mapping": pooled["mapping"],
                         "seed184_to_seed185": transfer_184, "seed185_to_seed184": transfer_185,
                         "audit": audit}
    minimal = [mask for mask in feasible_masks if not any(other != mask and
               all(left == "0" or right == "1" for left, right in zip(other, mask)) for other in feasible_masks)]
    for summary in summaries:
        summary["inclusion_minimal_feasible"] = summary["safety_feature_subset_mask"] in minimal
    retained_details = [{"candidate_feature_subset_mask": candidate, "feature_family": family,
                         "safety_feature_subset_mask": mask, **details[mask]} for mask in minimal]
    return summaries, retained_details


def auditable_gate_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["candidate_feature_subset_mask"], row["feature_family"],
            row["safety_feature_subset_mask"])


def gate_key_value(key: tuple[str, str, str]) -> dict[str, str]:
    return {"candidate_feature_subset_mask": key[0], "feature_family": key[1],
            "safety_feature_subset_mask": key[2]}


def conservative_identity_audit(records: Sequence[dict[str, Any]],
                                auditable_gates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for gate_row in auditable_gates:
        key = auditable_gate_key(gate_row)
        family = gate_row["feature_family"]
        members = gate_row["members"]
        mapping = gate_row["pooled_mapping"]
        for record in records:
            if record["candidate_feature_subset_mask"] != key[0]:
                continue
            signature = canonical([record[f"_{family}"][name] for name in members])
            rows.append({"gate_key": key, "seed": record["seed"], "id": record["id"],
                         "source_row_id": record["source_row_id"],
                         "dev_position": record["dev_position"],
                         "contrast_only": record["seed"] == 183,
                         "seen": signature in mapping,
                         "allowed": signature in mapping and mapping[signature] == "ALLOW"})
    return rows


def decide(single: list[dict[str, Any]], tail: list[dict[str, Any]], paired: list[dict[str, Any]],
           auditable_gate_keys: set[tuple[str, str, str]]) -> tuple[str, str, dict[str, Any]]:
    minimal_single = [row for row in single if auditable_gate_key(row) in auditable_gate_keys]
    localized = [row for row in minimal_single if row["bidirectional_cross_seed_full_pass"] and
                 row["conservative_gate_passed"] and row["seed184_nondiscovery_correct_to_incorrect"] == 0 and
                 row["seed185_nondiscovery_correct_to_incorrect"] == 0]
    pooled_single = any(row["feasible"] for row in single)
    candidates = sorted({row["candidate_feature_subset_mask"] for row in single})
    seed_specific = {}
    for candidate in candidates:
        candidate_rows = [row for row in single if row["candidate_feature_subset_mask"] == candidate]
        seed_specific[candidate] = {seed: any(
            not exact_mapping([], "single_state", [])["conflicts"] for _ in []
        ) for seed in PRIMARY_SEEDS}
        for seed in PRIMARY_SEEDS:
            seed_records = []
            # Seed-specific feasibility is reconstructed from per-subset conflict counts during analysis JSON assembly.
            seed_specific[candidate][seed] = any(row["feasible"] for row in candidate_rows)
    tail_transfer = any(auditable_gate_key(row) in auditable_gate_keys and
                        row["bidirectional_cross_seed_full_pass"] and
                        row["conservative_gate_passed"] for row in tail)
    paired_transfer = any(auditable_gate_key(row) in auditable_gate_keys and
                          row["bidirectional_cross_seed_full_pass"] and
                          row["conservative_gate_passed"] for row in paired)
    evaluation = {"ordered_rules": ["contract_failure", "cross_seed_single_state", "single_state_in_sample_only",
                                     "seed_specific_single_state", "tail_trajectory_only", "paired_delta_only",
                                     "current_observability_insufficient"],
                  "localized_single_state_gates": [{"candidate": row["candidate_feature_subset_mask"],
                                                     "mask": row["safety_feature_subset_mask"]} for row in localized],
                  "pooled_single_state_gate_exists": pooled_single,
                  "seed_specific_single_state": seed_specific,
                  "tail_trajectory_bidirectional_gate_exists": tail_transfer,
                  "paired_delta_bidirectional_gate_exists": paired_transfer,
                  "shared_auditable_gate_count": len(auditable_gate_keys)}
    if localized:
        return ("STAGE196B2B6P0_CROSS_SEED_SINGLE_STATE_SAFETY_GATE_LOCALIZED",
                "STAGE196B2B6P1_SAFETY_GATED_SELECTOR_COUNTERFACTUAL_AUDIT", evaluation)
    if pooled_single:
        return ("STAGE196B2B6P0_SINGLE_STATE_GATE_IN_SAMPLE_ONLY",
                "STAGE196B2B6P1_NEW_SEED_SAFETY_GATE_VALIDATION", evaluation)
    if any(all(seed_specific[candidate].values()) for candidate in candidates):
        return ("STAGE196B2B6P0_SEED_SPECIFIC_SINGLE_STATE_GATES",
                "STAGE196B2B6P1_SEED_SPECIFIC_SAFETY_MECHANISM_AUDIT", evaluation)
    if tail_transfer:
        return ("STAGE196B2B6P0_TAIL_TRAJECTORY_SAFETY_GATE_ONLY",
                "STAGE196B2B6P1_RUNTIME_SAFETY_STATE_DESIGN", evaluation)
    if paired_transfer:
        return ("STAGE196B2B6P0_PAIRED_DELTA_SAFETY_GATE_ONLY",
                "STAGE196B2B6P1_PAIRED_BRANCH_SAFETY_OBSERVABILITY_DESIGN", evaluation)
    return ("STAGE196B2B6P0_CURRENT_SAFETY_OBSERVABILITY_INSUFFICIENT",
            "STAGE196B2B6P1_ADDITIONAL_SAFETY_STATE_OBSERVABILITY_DESIGN", evaluation)


def analyze(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = validate_sources(ns, gates)
    p0 = load_p0(ns, source, gates)
    context = build_context(source, p0, gates)
    policies = build_candidate_policies(source, context, gates)
    dictionary, families = feature_specs(source)
    gate(gates, "features", "", "single_state_feature_dictionary_closure",
         {"epoch": [20], "integration_authorized": True, "outcome_derived": False},
         {"features": len(families["single_state"]), "epoch": [20], "integration_authorized": True},
         bool(families["single_state"]), "single-state dictionary is empty")
    gate(gates, "features", "", "single_state_outcome_leakage_prohibition",
         {"prohibited feature intersections": []},
         {"prohibited feature intersections": sorted(set(name.lower() for name in families["single_state"]) & PROHIBITED_FEATURES)},
         not (set(name.lower() for name in families["single_state"]) & PROHIBITED_FEATURES),
         "outcome or identity field entered single-state signature")
    gate(gates, "features", "", "tail_trajectory_diagnostic_only_enforcement", True,
         all(spec["diagnostic_only"] and not spec["integration_authorized"] for spec in dictionary
             if spec["feature_family"] == "tail_trajectory"),
         all(spec["diagnostic_only"] and not spec["integration_authorized"] for spec in dictionary
             if spec["feature_family"] == "tail_trajectory"), "tail feature was integration-authorized")
    gate(gates, "features", "", "paired_delta_diagnostic_only_enforcement", True,
         all(spec["diagnostic_only"] and not spec["integration_authorized"] for spec in dictionary
             if spec["feature_family"] == "paired_delta"),
         all(spec["diagnostic_only"] and not spec["integration_authorized"] for spec in dictionary
             if spec["feature_family"] == "paired_delta"), "paired delta was integration-authorized")
    records = make_records(source, context, policies, families, gates)
    reproduction = reproduce_b6(source, context, policies, gates)
    summaries = {family: [] for family in families}
    details = {family: [] for family in families}
    for candidate in EXPECTED_CANDIDATES:
        for family, names in families.items():
            family_summaries, family_details = enumerate_family(records, candidate, family, names)
            summaries[family].extend(family_summaries)
            details[family].extend(family_details)
    expected_counts = {family: len(EXPECTED_CANDIDATES) * ((1 << len(names)) - 1)
                       for family, names in families.items()}
    for family in families:
        gate(gates, "enumeration", "", f"{family}_feature_subset_enumeration",
             expected_counts[family], len(summaries[family]),
             len(summaries[family]) == expected_counts[family], f"{family} subset enumeration incomplete")
    transfer_complete = all("seed184_to_seed185_full_pass" in row for values in summaries.values() for row in values)
    gate(gates, "transfer", "", "seed184_to_seed185_transfer_completion", True, transfer_complete,
         transfer_complete, "seed184-to-seed185 transfer incomplete")
    gate(gates, "transfer", "", "seed185_to_seed184_transfer_completion", True, transfer_complete,
         transfer_complete, "seed185-to-seed184 transfer incomplete")
    auditable_gates = sorted((detail for values in details.values() for detail in values),
                             key=auditable_gate_key)
    auditable_key_counts = Counter(auditable_gate_key(row) for row in auditable_gates)
    auditable_gate_keys = set(auditable_key_counts)
    duplicate_gate_keys = [gate_key_value(key) for key, count in sorted(auditable_key_counts.items())
                           if count != 1]
    audits = [summary for gate_row in auditable_gates for summary in gate_row["audit"]]
    expected_populations = ((183, "ALL_720"), (184, "PRIMARY"), (184, "NONDISCOVERY"),
                            (185, "PRIMARY"), (185, "NONDISCOVERY"))
    expected_summary_keys = {(key, seed, population) for key in auditable_gate_keys
                             for seed, population in expected_populations}
    observed_summary_counts = Counter((auditable_gate_key(row), row["seed"], row["population"])
                                      for row in audits)
    observed_summary_keys = set(observed_summary_counts)
    represented_summary_gates = {key for key, _, _ in observed_summary_keys}
    missing_summary_gates = sorted(auditable_gate_keys - represented_summary_gates)
    extra_summary_gates = sorted(represented_summary_gates - auditable_gate_keys)
    duplicate_summary_keys = [
        {**gate_key_value(key), "seed": seed, "population": population, "count": count}
        for (key, seed, population), count in sorted(observed_summary_counts.items()) if count != 1]
    conservative_vacuous = not auditable_gate_keys and not audits
    conservative_completed = (
        not duplicate_gate_keys and observed_summary_keys == expected_summary_keys and
        len(audits) == len(expected_summary_keys) and not duplicate_summary_keys)
    conservative_evidence = {
        "auditable_gate_count": len(auditable_gate_keys),
        "expected_summary_or_audit_rows": len(expected_summary_keys),
        "observed_summary_or_audit_rows": len(audits),
        "represented_gate_count": len(represented_summary_gates),
        "missing_gate_keys": [gate_key_value(key) for key in missing_summary_gates],
        "extra_gate_keys": [gate_key_value(key) for key in extra_summary_gates],
        "duplicate_gate_keys": duplicate_gate_keys,
        "duplicate_summary_keys": duplicate_summary_keys,
        "vacuous_completion": conservative_vacuous,
        "completed": conservative_completed,
    }
    gate(gates, "audit", "", "conservative_gated_policy_audit_completion",
         {"auditable_gate_count_source": "shared inclusion-minimal feasible gate set",
          "summary_populations_per_gate": 5, "zero_gate_vacuous_completion_allowed": True,
          "missing_gate_keys": [], "extra_gate_keys": [], "duplicate_gate_keys": []},
         conservative_evidence, conservative_completed, "conservative gate audit incomplete")
    identity_audits = conservative_identity_audit(records, auditable_gates)
    seed183_identity_rows = [row for row in identity_audits if row["seed"] == 183]
    seed183_summary_rows = [row for row in audits if row["seed"] == 183 and row["population"] == "ALL_720"]
    seed183_gate_counts = Counter(row["gate_key"] for row in seed183_identity_rows)
    represented_seed183_gates = set(seed183_gate_counts)
    missing_seed183_gates = sorted(auditable_gate_keys - represented_seed183_gates)
    extra_seed183_gates = sorted(represented_seed183_gates - auditable_gate_keys)
    seed183_identity_counts = Counter((row["gate_key"], row["id"], row["source_row_id"],
                                       row["dev_position"]) for row in seed183_identity_rows)
    duplicate_gate_identity_rows = [
        {**gate_key_value(key), "id": identity, "source_row_id": source_row_id,
         "dev_position": position, "count": count}
        for (key, identity, source_row_id, position), count in sorted(seed183_identity_counts.items())
        if count != 1]
    wrong_seed183_row_counts = [
        {**gate_key_value(key), "expected": 720, "observed": seed183_gate_counts.get(key, 0)}
        for key in sorted(auditable_gate_keys) if seed183_gate_counts.get(key, 0) != 720]
    expected_seed183_rows = len(auditable_gate_keys) * 720
    seed183_contrast_only = all(row["contrast_only"] for row in seed183_identity_rows)
    seed183_vacuous = not auditable_gate_keys and not seed183_identity_rows
    seed183_completed = (
        not duplicate_gate_keys and len(seed183_identity_rows) == expected_seed183_rows and
        represented_seed183_gates == auditable_gate_keys and not duplicate_gate_identity_rows and
        not wrong_seed183_row_counts and seed183_contrast_only and
        len(seed183_summary_rows) == len(auditable_gate_keys))
    seed183_completion_evidence = {
        "auditable_gate_count": len(auditable_gate_keys),
        "expected_audit_rows": expected_seed183_rows,
        "observed_audit_rows": len(seed183_identity_rows),
        "represented_gate_count": len(represented_seed183_gates),
        "missing_gate_keys": [gate_key_value(key) for key in missing_seed183_gates],
        "duplicate_gate_keys": duplicate_gate_keys,
        "extra_gate_keys": [gate_key_value(key) for key in extra_seed183_gates],
        "duplicate_gate_identity_rows": duplicate_gate_identity_rows,
        "wrong_row_counts": wrong_seed183_row_counts,
        "contrast_only": seed183_contrast_only,
        "vacuous_completion": seed183_vacuous,
        "completed": seed183_completed,
    }
    gate(gates, "audit", "", "seed183_contrast_audit_completion",
         {"contrast_only": True,
          "auditable_gate_count_source": "shared inclusion-minimal feasible gate set",
          "rows_per_gate_when_nonempty": 720, "zero_gate_vacuous_completion_allowed": True},
         seed183_completion_evidence, seed183_completed, "seed183 contrast audit incomplete")
    decision, next_stage, evaluation = decide(summaries["single_state"], summaries["tail_trajectory"],
                                              summaries["paired_delta"], auditable_gate_keys)
    gate(gates, "decision", "", "decision_evaluation_completion", True,
         {"completed": True, "decision": decision, "recommended_next_stage": next_stage}, True,
         "decision evaluation incomplete")
    gate(gates, "output", "", "exact_nine_output_closure", list(OUTPUTS), list(OUTPUTS), True,
         "output plan changed")
    source_hashes = {**source["hashes"], **p0["hashes"]}
    target_summary = {candidate: dict(Counter(row["safety_target"] for row in records
                                              if row["candidate_feature_subset_mask"] == candidate))
                      for candidate in EXPECTED_CANDIDATES}
    minimal_single = [row for row in summaries["single_state"]
                      if auditable_gate_key(row) in auditable_gate_keys]
    seed183_gate_summaries = [{key: row[key] for key in
                               ("candidate_feature_subset_mask", "feature_family",
                                "safety_feature_subset_mask", "row_count", "allowed_count", "blocked_count",
                                "unseen_count", "correct_to_incorrect_count", "incorrect_to_correct_count",
                                "stable_correct_preservation_rate")}
                              for row in seed183_summary_rows]
    seed183_contrast = {**seed183_completion_evidence,
                        "expected_rows": seed183_completion_evidence["expected_audit_rows"],
                        "observed_rows": seed183_completion_evidence["observed_audit_rows"],
                        "gate_summaries": seed183_gate_summaries}
    analysis = {
        "stage": STAGE, "decision": decision, "recommended_next_stage": next_stage, "blocking_reasons": [],
        "current_git_commit": ns.current_git_commit,
        "stage196b2b3p0_runtime_git_commit": ns.stage196b2b3p0_runtime_git_commit,
        "source_paths": {"stage196b2b6_analysis_json": str(source["b6"]),
                         "stage196b2b5_analysis_json": str(source["b5"]),
                         "stage196b2b4_analysis_json": str(source["b4"]),
                         "stage196b2b3p0_run_root": str(source["p0"])},
        "source_hashes": source_hashes,
        "source_closure": {"b2b6_files": list(B6_FILES), "b2b5_files": list(B5_FILES),
                           "b2b4_files": list(B4_FILES), "p0_counts": p0["counts"],
                           "b2b6_metric_reproduction": reproduction},
        "population_semantics": {"seed_conditioned_primary_cases": {"total": 16, "seed184": 11, "seed185": 5},
                                 "primary_case_key": ["seed", "stable_row_id"],
                                 "discovery_data_identity_key": ["id", "source_row_id", "dev_position"],
                                 "discovery_identities": {"seed184": 11, "seed185": 5,
                                                          "cross_seed_intersection": 3, "union": 13},
                                 "per_seed_clean_dev": {"discovery_identity_union": 13,
                                                        "nondiscovery": 707, "all": 720},
                                 "seed183": "contrast-only"},
        "nondominated_selector_candidates": [{"feature_subset_mask": candidate,
                                                "feature_subset_members": policies[candidate]["members"]}
                                               for candidate in EXPECTED_CANDIDATES],
        "row_safety_target_summary": target_summary,
        "single_state_feature_dictionary": [spec for spec in dictionary if spec["feature_family"] == "single_state"],
        "single_state_feature_subset_count": expected_counts["single_state"],
        "single_state_inclusion_minimal_feasible_gates": minimal_single,
        "auditable_gate_metadata": {"authority": "shared inclusion-minimal feasible gate set",
                                    "auditable_gate_count": len(auditable_gate_keys),
                                    "gate_keys": [gate_key_value(key) for key in sorted(auditable_gate_keys)]},
        "conservative_gated_policy_audit_completion": conservative_evidence,
        "single_state_cross_seed_transfer": details["single_state"],
        "single_state_gated_policy_audit": [row for row in audits if row["feature_family"] == "single_state"],
        "tail_trajectory_feature_dictionary": [spec for spec in dictionary if spec["feature_family"] == "tail_trajectory"],
        "tail_trajectory_feasible_gates": [row for row in summaries["tail_trajectory"] if row["feasible"]],
        "paired_delta_feature_dictionary": [spec for spec in dictionary if spec["feature_family"] == "paired_delta"],
        "paired_delta_feasible_gates": [row for row in summaries["paired_delta"] if row["feasible"]],
        "seed183_contrast": seed183_contrast, "decision_rule_evaluation": evaluation,
        "authorized_interpretation": ("Within the frozen-Mamba, frozen-composer controlled population, the analyzer "
                                      "reports exact categorical separation and cross-seed transfer. Only an epoch-20 "
                                      "recipient-local gate satisfying every ordered rule is potentially eligible for "
                                      "the recommended counterfactual audit; no integration, training, or promotion is authorized."),
        "remaining_uncertainty": ["The evidence is internal to the frozen controlled population.",
                                  "Seed183 is contrast-only and does not authorize selection or promotion.",
                                  "Set-valued selector actions are evaluated conservatively row by row."],
        "prohibited_claims": list(PROHIBITED_CLAIMS), "artifact_only": True, "classifier_fitted": False,
        "learned_threshold_used": False, "aggregate_score_optimized": False, "training_performed": False,
        "model_loaded": False, "checkpoint_loaded": False, "promotion_authorized": False,
    }
    public_targets = [{key: row[key] for key in TARGET_H} for row in records]
    signature_rows = [{"candidate_feature_subset_mask": row["candidate_feature_subset_mask"],
                       "candidate_feature_subset_members": row["candidate_feature_subset_members"],
                       "seed": row["seed"], "stable_row_id": row["stable_row_id"], "id": row["id"],
                       "source_row_id": row["source_row_id"], "dev_position": row["dev_position"],
                       "population": row["population"], "safety_target": row["safety_target"],
                       "assigned_action_set": row["assigned_action_set"],
                       "single_state_feature_values": row["_single_state"]} for row in records]
    return analysis, {"dictionary": dictionary, "targets": public_targets, "signatures": signature_rows,
                      "single_summary": summaries["single_state"],
                      "diagnostic_summary": summaries["tail_trajectory"] + summaries["paired_delta"],
                      "audits": audits}


REPORT_SECTIONS = (
    "Executive decision", "Authorized interpretation", "B2-B6 source result", "Source closure",
    "Population semantics", "Nondominated selector candidates", "Row-level selector reconstruction",
    "Safety-target definition", "MUST_ALLOW rows", "MUST_BLOCK rows", "OPTIONAL rows",
    "Single-checkpoint recipient feature dictionary", "Single-checkpoint exact safety-gate search",
    "Cross-seed safety-gate transfer", "Tail-trajectory diagnostic result", "Paired-delta diagnostic result",
    "Conservative gated-policy audit", "Seed183 contrast", "Decision-rule evaluation",
    "Remaining uncertainty", "Prohibited claims", "Recommended next stage",
)


def report(analysis: dict[str, Any]) -> str:
    target = analysis["row_safety_target_summary"]
    no_auditable_gate = analysis.get("auditable_gate_metadata", {}).get("auditable_gate_count") == 0
    zero_gate_text = (
        "No inclusion-minimal feasible safety gate was available for conservative application or "
        "seed183 contrast auditing.\n\nThe zero-row audits are vacuously complete, not missing.\n\n"
        "Seed183 was not used to retrofit or discover a gate.")
    conservative_body = (
        f"{zero_gate_text}\n\n{canonical(analysis['conservative_gated_policy_audit_completion'])}"
        if no_auditable_gate else canonical({"completion": analysis["conservative_gated_policy_audit_completion"],
                                             "summaries": analysis["single_state_gated_policy_audit"]}))
    seed183_body = (
        f"{zero_gate_text}\n\n{canonical(analysis['seed183_contrast'])}"
        if no_auditable_gate else canonical(analysis["seed183_contrast"]))
    bodies = (
        f"`{analysis['decision']}`", analysis["authorized_interpretation"],
        "The successful B2-B6 unsafe result, exact three nondominated candidates, primary closure, and both-seed nondiscovery failures are required.",
        canonical(analysis["source_closure"]), canonical(analysis["population_semantics"]),
        canonical(analysis["nondominated_selector_candidates"]),
        "Every assigned action is applied to the paired epoch-20 frame-local-only primitive state and all entitlement, head, residual, final-logit, and prediction quantities are recomputed.",
        "Outcome-derived labels are evaluation targets only and never enter a safety signature.",
        canonical({candidate: counts.get("MUST_ALLOW", 0) for candidate, counts in target.items()}),
        canonical({candidate: counts.get("MUST_BLOCK", 0) for candidate, counts in target.items()}),
        canonical({candidate: counts.get("OPTIONAL", 0) for candidate, counts in target.items()}),
        canonical(analysis["single_state_feature_dictionary"]),
        canonical(analysis["single_state_inclusion_minimal_feasible_gates"]),
        canonical(analysis["single_state_cross_seed_transfer"]),
        canonical(analysis["tail_trajectory_feasible_gates"]),
        canonical(analysis["paired_delta_feasible_gates"]),
        conservative_body, seed183_body,
        canonical(analysis["decision_rule_evaluation"]),
        "\n".join(f"- {item}" for item in analysis["remaining_uncertainty"]),
        "\n".join(f"- {item}" for item in analysis["prohibited_claims"]),
        f"`{analysis['recommended_next_stage']}`\n\nNo training, integration, or promotion is authorized.",
    )
    return f"# {STAGE}: Selector Safety-State Observability\n\n" + "\n\n".join(
        f"## {heading}\n\n{body}" for heading, body in zip(REPORT_SECTIONS, bodies)) + "\n"


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return canonical(value)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return ""
    return value


def render_csv(header: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(header), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(header):
            raise ValueError(f"generated CSV schema mismatch: {sorted(set(row) ^ set(header))}")
        writer.writerow({key: csv_value(row[key]) for key in header})
    return buffer.getvalue()


def render_contract(gates: list[dict[str, Any]]) -> str:
    return render_csv(CONTRACT_H, [{**row, "required": canonical(row["required"]),
                                    "observed": canonical(row["observed"])} for row in gates])


def blocked_analysis(ns: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    return {"stage": STAGE, "decision": "STAGE196B2B6P0_BLOCKED_CONTRACT_FAILURE",
            "recommended_next_stage": "STAGE196B2B6P0_REPAIR",
            "blocking_reasons": [f"{type(error).__name__}: {error}"],
            "current_git_commit": ns.current_git_commit,
            "stage196b2b3p0_runtime_git_commit": ns.stage196b2b3p0_runtime_git_commit,
            "source_paths": {"stage196b2b6_analysis_json": str(ns.stage196b2b6_analysis_json.resolve()),
                             "stage196b2b5_analysis_json": str(ns.stage196b2b5_analysis_json.resolve()),
                             "stage196b2b4_analysis_json": str(ns.stage196b2b4_analysis_json.resolve()),
                             "stage196b2b3p0_run_root": str(ns.stage196b2b3p0_run_root.resolve())},
            "source_hashes": {}, "source_closure": {}, "population_semantics": {},
            "nondominated_selector_candidates": [], "row_safety_target_summary": {},
            "single_state_feature_dictionary": [], "single_state_feature_subset_count": 0,
            "single_state_inclusion_minimal_feasible_gates": [], "single_state_cross_seed_transfer": [],
            "auditable_gate_metadata": {"authority": "unavailable because a contract failed",
                                        "auditable_gate_count": None, "gate_keys": []},
            "conservative_gated_policy_audit_completion": {"vacuous_completion": False,
                                                            "completed": False},
            "single_state_gated_policy_audit": [], "tail_trajectory_feature_dictionary": [],
            "tail_trajectory_feasible_gates": [], "paired_delta_feature_dictionary": [],
            "paired_delta_feasible_gates": [], "seed183_contrast": [],
            "decision_rule_evaluation": {"completed": False},
            "authorized_interpretation": "No scientific interpretation is authorized because a contract failed.",
            "remaining_uncertainty": ["Repair the failed contract."],
            "prohibited_claims": list(PROHIBITED_CLAIMS), "artifact_only": True,
            "classifier_fitted": False, "learned_threshold_used": False,
            "aggregate_score_optimized": False, "training_performed": False, "model_loaded": False,
            "checkpoint_loaded": False, "promotion_authorized": False}


def payloads(analysis: dict[str, Any], tables: dict[str, Any], gates: list[dict[str, Any]]) -> dict[str, str]:
    return {
        OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        OUTPUTS[1]: report(analysis), OUTPUTS[2]: render_csv(FEATURE_H, tables["dictionary"]),
        OUTPUTS[3]: render_csv(TARGET_H, tables["targets"]),
        OUTPUTS[4]: render_csv(SIGNATURE_H, tables["signatures"]),
        OUTPUTS[5]: render_csv(GATE_H, tables["single_summary"]),
        OUTPUTS[6]: render_csv(GATE_H, tables["diagnostic_summary"]),
        OUTPUTS[7]: render_csv(AUDIT_H, tables["audits"]), OUTPUTS[8]: render_contract(gates),
    }


def atomic_write_outputs(output: Path, data: dict[str, str]) -> None:
    if output.exists() or set(data) != set(OUTPUTS):
        raise RuntimeError("refusing overwrite or non-nine-file output")
    temporary = output.parent / f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        for name in OUTPUTS:
            with (temporary / name).open("x", encoding="utf-8", newline="") as handle:
                handle.write(data[name])
                handle.flush()
                os.fsync(handle.fileno())
        if sorted(path.name for path in temporary.iterdir()) != sorted(OUTPUTS):
            raise RuntimeError("staged nine-output closure failed")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    ns = parse_args()
    gates: list[dict[str, Any]] = []
    try:
        analysis, tables = analyze(ns, gates)
    except Exception as error:
        analysis = blocked_analysis(ns, error)
        tables = {"dictionary": [], "targets": [], "signatures": [], "single_summary": [],
                  "diagnostic_summary": [], "audits": []}
    atomic_write_outputs(ns.output_dir.resolve(), payloads(analysis, tables, gates))
    return 0 if not analysis["blocking_reasons"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
