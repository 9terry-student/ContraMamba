#!/usr/bin/env python3
"""Stage196-B2-B5 artifact-only exact row-selector observability analysis.

No classifier, fitted threshold, score optimization, model, or checkpoint is
used.  Selector feasibility is set intersection over the frozen primitive
coalition lattice.
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
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


STAGE = "Stage196-B2-B5"
P0_COMMIT = "fa16787efa84bb15d832b6d9382fafd77016c4e2"
B2B4_COMMIT = "8664fef0527a461ea8e8644bbf04770f526d4569"
PRIMARY_SEEDS = (184, 185)
ALL_SEEDS = (183, 184, 185)
EPOCHS = tuple(range(1, 21))
TAIL = (18, 19, 20)
MODES = ("joint", "frame_local_only")
RUNS = tuple(f"seed{s}_{m}" for s in ALL_SEEDS for m in MODES)
FORWARD = "JOINT_RECIPIENT_FRAME_LOCAL_ONLY_DONOR"
REVERSE = "FRAME_LOCAL_ONLY_RECIPIENT_JOINT_DONOR"
PRIMITIVES = ("FRAME", "PREDICATE", "SUFFICIENCY", "POSITIVE_ENERGY", "NEGATIVE_ENERGY")
MASKS = tuple(f"{value:05b}" for value in range(32))
EXPECTED_PRIMARY = {184: {"recovery": 5, "harm": 6}, 185: {"recovery": 2, "harm": 3}}
TOL = 1e-6
ZERO_TOL = 1e-12

B2B4_FILES = (
    "stage196b2b4_analysis.json", "stage196b2b4_report.md",
    "stage196b2b4_primitive_coalition_rows.csv", "stage196b2b4_primitive_mobius_terms.csv",
    "stage196b2b4_primitive_tail_summary.csv", "stage196b2b4_residual_coalition_rows.csv",
    "stage196b2b4_residual_mobius_terms.csv", "stage196b2b4_localization_summary.csv",
    "stage196b2b4_contract.csv",
)
B2B2_FILES = (
    "stage196b2b2_analysis.json", "stage196b2b2_report.md",
    "stage196b2b2_row_path_summary.csv", "stage196b2b2_epoch_paired_paths.csv",
    "stage196b2b2_group_path_summary.csv", "stage196b2b2_event_order_summary.csv",
    "stage196b2b2_intervention_type_paths.csv", "stage196b2b2_contrast_summary.csv",
    "stage196b2b2_contract.csv",
)
OUTPUTS = (
    "stage196b2b5_analysis.json", "stage196b2b5_report.md",
    "stage196b2b5_feature_dictionary.csv", "stage196b2b5_row_action_sets.csv",
    "stage196b2b5_recipient_signature_rows.csv", "stage196b2b5_recipient_selector_summary.csv",
    "stage196b2b5_paired_delta_signature_rows.csv", "stage196b2b5_paired_delta_selector_summary.csv",
    "stage196b2b5_contract.csv",
)
TRAJECTORY_FIELDS = {
    "id", "source_row_id", "dev_position", "gold_label", "prediction", "intervention_type",
    "frame_probability", "predicate_coverage_probability", "sufficiency_probability",
    "polarity_support_margin", "entitlement_probability", "support_probability",
    "not_entitled_probability", "support_logit", "not_entitled_logit", "epoch",
    "training_seed", "frame_downstream_gradient_mode",
}
PROHIBITED_SELECTOR_FIELDS = {
    "seed", "training_seed", "stable_row_id", "id", "source_row_id", "dev_position",
    "transition_role", "path_class", "subtype", "minimal_coalition",
    "minimal_coalition_labels", "donor_tail_reproduced", "recipient_tail_preserved",
    "counterfactual_prediction", "counterfactual_margin",
}
PROHIBITED_CLAIMS = (
    "formal causal mediation", "external or OOD validity", "unfrozen-Mamba validity",
    "training improvement", "promotion", "a universal selector from in-sample-only partitions",
    "deployability from paired-treatment delta features", "selector validity from outcome-derived fields",
    "success based on partial coverage", "success based on average accuracy",
    "success based on learned thresholds", "row identity as a selector",
    "path class or subtype as an inference-time selector",
)

FEATURE_H = (
    "feature_family", "feature_name", "deployment_authorized", "diagnostic_only", "source_fields",
    "formula", "tail_aggregation", "value_domain", "natural_threshold", "outcome_derived",
    "available", "unavailable_reason",
)
ACTION_H = (
    "seed", "stable_row_id", "id", "source_row_id", "dev_position", "transition_role",
    "intervention_type", "path_class", "subtype", "direction", "tail_epochs",
    "recipient_tail_status", "donor_tail_status", "acceptable_coalitions",
    "acceptable_coalition_count", "inclusion_minimal_acceptable_coalitions",
    "minimal_coalition_count", "empty_coalition_acceptable", "full_coalition_acceptable",
)
SIGNATURE_H = (
    "feature_family", "feature_subset_mask", "feature_subset_size", "feature_subset_members",
    "seed", "stable_row_id", "transition_role", "signature", "acceptable_coalitions",
    "signature_action_intersection", "signature_feasible", "source_seed", "target_seed",
    "transfer_status",
)
SUMMARY_H = (
    "feature_family", "feature_subset_mask", "feature_subset_size", "feature_subset_members",
    "row_count", "signature_count", "feasible", "inclusion_minimal_feasible",
    "mixed_role_signature_count", "cross_seed_signature_count",
    "seed184_to_seed185_seen", "seed184_to_seed185_unseen", "seed184_to_seed185_incompatible",
    "seed184_to_seed185_full_pass", "seed185_to_seed184_seen", "seed185_to_seed184_unseen",
    "seed185_to_seed184_incompatible", "seed185_to_seed184_full_pass", "pooled_full_pass",
    "bidirectional_cross_seed_full_pass",
)
CONTRACT_H = ("scope", "run", "gate", "required", "observed", "passed", "blocking_reason")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, kind in (
        ("repo-root", Path), ("stage196b2b4-analysis-json", Path),
        ("stage196b2b2-analysis-json", Path), ("stage196b2b3p0-run-root", Path),
        ("stage196b2b3p0-runtime-git-commit", str), ("current-git-commit", str),
        ("output-dir", Path),
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
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError(f"{path}:{line_number}: JSON object required")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def cell(value: Any) -> Any:
    if type(value) is not str:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def integer(value: Any, name: str) -> int:
    value = cell(value)
    if type(value) is not int:
        raise ValueError(f"{name}: integer required")
    return value


def number(value: Any, name: str, probability: bool = False) -> float:
    value = cell(value)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name}: finite number required")
    result = float(value)
    if probability and not 0.0 <= result <= 1.0:
        raise ValueError(f"{name}: probability outside [0,1]")
    return result


def boolean(value: Any, name: str) -> bool:
    value = cell(value)
    if type(value) is not bool:
        raise ValueError(f"{name}: boolean required")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def gate(rows: list[dict[str, Any]], scope: str, run: str, name: str,
         required: Any, observed: Any, passed: bool, reason: str, fatal: bool = True) -> None:
    # required and observed deliberately retain JSON types until contract rendering.
    rows.append({"scope": scope, "run": run, "gate": name, "required": required,
                 "observed": observed, "passed": bool(passed),
                 "blocking_reason": "" if passed else reason})
    if fatal and not passed:
        raise ValueError(f"{name}: {reason}")


def exact_files(directory: Path, expected: Sequence[str], gates: list[dict[str, Any]], name: str) -> None:
    observed = sorted(path.name for path in directory.iterdir() if path.is_file())
    required = sorted(expected)
    evidence = {"files": observed, "missing": sorted(set(required) - set(observed)),
                "unexpected": sorted(set(observed) - set(required))}
    gate(gates, "source", "", name, required, evidence, observed == required, "exact file closure failed")


def contract_closed(rows: list[dict[str, str]], expected: int | None = None) -> bool:
    return (expected is None or len(rows) == expected) and bool(rows) and all(
        boolean(row.get("passed", ""), "contract passed") and not row.get("blocking_reason", "").strip()
        for row in rows
    )


def require_columns(rows: list[dict[str, Any]], names: Sequence[str], label: str) -> None:
    missing = set(names) - (set(rows[0]) if rows else set())
    if not rows or missing:
        raise ValueError(f"{label}: missing rows/columns {sorted(missing)}")


def sign(value: float) -> str:
    if abs(value) <= ZERO_TOL:
        return "ZERO"
    return "NEGATIVE" if value < 0.0 else "POSITIVE"


def halfspace(value: float) -> str:
    if abs(value - 0.5) <= ZERO_TOL:
        return "AT_HALF"
    return "BELOW_HALF" if value < 0.5 else "ABOVE_HALF"


def order_energy(positive: float, negative: float) -> str:
    delta = positive - negative
    if abs(delta) <= ZERO_TOL:
        return "EQUAL"
    return "POSITIVE_DOMINANT" if delta > 0.0 else "NEGATIVE_DOMINANT"


def minimal_masks(masks: Iterable[str]) -> list[str]:
    ordered = sorted(set(masks))
    return [mask for mask in ordered if not any(
        other != mask and all(a == "0" or b == "1" for a, b in zip(other, mask))
        for other in ordered
    )]


def validate_b2b4(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> dict[str, Any]:
    path = ns.stage196b2b4_analysis_json.resolve()
    if path.name != B2B4_FILES[0]:
        raise ValueError("B2-B4 analysis basename is not exact")
    exact_files(path.parent, B2B4_FILES, gates, "b2b4_exact_nine_file_closure")
    analysis = read_json(path)
    required = {"decision": "STAGE196B2B4_ROW_SPECIFIC_PRIMITIVE_INTERACTION",
                "recommended_next_stage": "STAGE196B2B5_ROW_SELECTOR_OBSERVABILITY",
                "blocking_reasons": []}
    observed = {key: analysis.get(key) for key in required}
    gate(gates, "source", "", "b2b4_decision_closure", required, observed,
         observed == required, "B2-B4 scientific authority mismatch")
    artifact_commit = analysis.get("current_git_commit")
    gate(gates, "provenance", "", "b2b4_analyzer_runtime_commit", B2B4_COMMIT, artifact_commit,
         artifact_commit == B2B4_COMMIT, "B2-B4 analyzer runtime commit mismatch")
    contract = read_csv(path.parent / B2B4_FILES[-1])
    gate(gates, "source", "", "b2b4_contract_closure", True,
         {"gate_count": len(contract), "passed_count": sum(boolean(r.get("passed", ""), "passed") for r in contract)},
         contract_closed(contract), "B2-B4 contract gate failure")
    primitive = read_csv(path.parent / B2B4_FILES[2])
    pmobius = read_csv(path.parent / B2B4_FILES[3])
    tails = read_csv(path.parent / B2B4_FILES[4])
    residual = read_csv(path.parent / B2B4_FILES[5])
    rmobius = read_csv(path.parent / B2B4_FILES[6])
    required_counts = {"primitive_coalition_rows": 20480, "primitive_mobius_rows": 20480,
                       "primitive_tail_summaries": 1024, "residual_coalition_rows": 20480,
                       "residual_mobius_rows": 20480}
    observed_counts = {"primitive_coalition_rows": len(primitive), "primitive_mobius_rows": len(pmobius),
                       "primitive_tail_summaries": len(tails), "residual_coalition_rows": len(residual),
                       "residual_mobius_rows": len(rmobius)}
    gate(gates, "source", "", "b2b4_row_closure", required_counts, observed_counts,
         observed_counts == required_counts, "B2-B4 row count closure failed")
    identities = {(r["seed"], r["stable_row_id"]) for r in primitive}
    directions = {(r["seed"], r["epoch"], r["stable_row_id"], r["direction"]) for r in primitive}
    coalitions = sorted({r["coalition_mask"] for r in primitive})
    lattice_obs = {"primary_identities": len(identities), "directional_states": len(directions),
                   "primitive_coalitions": len(coalitions),
                   "residual_coalitions": len({r["coalition_mask"] for r in residual})}
    lattice_req = {"primary_identities": 16, "directional_states": 640,
                   "primitive_coalitions": 32, "residual_coalitions": 32}
    gate(gates, "source", "", "b2b4_lattice_closure", lattice_req, lattice_obs,
         lattice_obs == lattice_req and coalitions == list(MASKS), "B2-B4 lattice closure failed")
    mobius_columns = [name for name in pmobius[0] if name.startswith("interaction_sum_error_")]
    pmax = max(abs(number(row[name], name)) for row in pmobius for name in mobius_columns)
    rcolumns = [name for name in rmobius[0] if name.startswith("interaction_sum_error_")]
    rmax = max(abs(number(row[name], name)) for row in rmobius for name in rcolumns)
    gate(gates, "source", "", "b2b4_mobius_closure", {"maximum_error": TOL},
         {"primitive_maximum_error": pmax, "residual_maximum_error": rmax},
         pmax <= TOL and rmax <= TOL, "B2-B4 Mobius reconstruction error")
    controls = analysis.get("residual_positive_controls", {})
    control_ok = controls.get("empty", {}).get("passed") is True and controls.get("full", {}).get("passed") is True
    gate(gates, "source", "", "b2b4_empty_full_controls", True,
         {"empty": controls.get("empty", {}).get("passed"), "full": controls.get("full", {}).get("passed")},
         control_ok, "B2-B4 empty/full controls failed")
    require_columns(primitive, ("seed", "epoch", "stable_row_id", "direction", "coalition_mask",
                                "recipient_prediction", "donor_prediction", "counterfactual_prediction",
                                "donor_prediction_reproduced", "recipient_prediction_preserved"), "primitive rows")
    require_columns(tails, ("seed", "stable_row_id", "id", "source_row_id", "dev_position",
                            "transition_role", "intervention_type", "path_class", "subtype", "direction",
                            "coalition_mask", "tail_epochs", "recipient_tail_status", "donor_tail_status",
                            "donor_tail_reproduced", "recipient_tail_preserved"), "primitive tails")
    hashes = {str(path.parent / name): sha256(path.parent / name) for name in B2B4_FILES}
    return {"analysis": analysis, "path": path, "primitive": primitive, "tails": tails,
            "source_hashes": hashes, "counts": observed_counts,
            "mobius": {"primitive_maximum_error": pmax, "residual_maximum_error": rmax}}


def validate_b2b2(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> dict[str, Any]:
    path = ns.stage196b2b2_analysis_json.resolve()
    if path.name != B2B2_FILES[0]:
        raise ValueError("B2-B2 analysis basename is not exact")
    exact_files(path.parent, B2B2_FILES, gates, "b2b2_exact_nine_file_closure")
    analysis = read_json(path)
    required = {"decision": "STAGE196B2B2_SEED_SPECIFIC_MULTIPATH_EFFECT",
                "recommended_next_stage": "STAGE196B2B3_NO_PROMOTION_INFERENCE_ONLY_COMPONENT_SWAP_PROBE"}
    observed = {key: analysis.get(key) for key in required}
    gate(gates, "source", "", "b2b2_decision_closure", required, observed,
         observed == required, "B2-B2 authority mismatch")
    contract = read_csv(path.parent / B2B2_FILES[-1])
    gate(gates, "source", "", "b2b2_155_of_155_contract_closure", {"gates": 155, "passed": 155},
         {"gates": len(contract), "passed": sum(boolean(r.get("passed", ""), "passed") for r in contract)},
         contract_closed(contract, 155), "B2-B2 contract closure failed")
    static = read_csv(path.parent / B2B2_FILES[2])
    epochs = read_csv(path.parent / B2B2_FILES[3])
    require_columns(static, ("seed", "stable_row_id", "id", "source_row_id", "dev_position",
                             "transition_role", "intervention_type", "path_class"), "B2-B2 identities")
    counts = {seed: Counter() for seed in PRIMARY_SEEDS}
    primary: dict[tuple[int, str], dict[str, str]] = {}
    for row in static:
        seed = integer(row["seed"], "seed")
        role = row["transition_role"]
        key = (seed, row["stable_row_id"])
        if seed not in PRIMARY_SEEDS or role not in ("recovery", "harm") or key in primary:
            raise ValueError("B2-B2 primary identity/role failure")
        primary[key] = row
        counts[seed][role] += 1
    normalized = {seed: dict(counts[seed]) for seed in PRIMARY_SEEDS}
    epoch_keys = Counter((integer(row["seed"], "seed"), row["stable_row_id"]) for row in epochs)
    tail_epochs = sorted({integer(row["epoch"], "epoch") for row in epochs if integer(row["epoch"], "epoch") in TAIL})
    closure = len(static) == 16 and len(epochs) == 320 and normalized == EXPECTED_PRIMARY and set(epoch_keys.values()) == {20}
    gate(gates, "population", "", "b2b2_population_closure",
         {"identities": 16, "epoch_rows": 320, "tail_epochs": list(TAIL), "seed_roles": EXPECTED_PRIMARY},
         {"identities": len(static), "epoch_rows": len(epochs), "tail_epochs": tail_epochs,
          "seed_roles": normalized, "contrast_seed": "excluded from selector decisions"},
         closure and tail_epochs == list(TAIL), "B2-B2 primary population changed")
    hashes = {str(path.parent / name): sha256(path.parent / name) for name in B2B2_FILES}
    return {"analysis": analysis, "path": path, "primary": primary, "epochs": epochs,
            "source_hashes": hashes}


def validate_p0(ns: argparse.Namespace, primary: dict[tuple[int, str], dict[str, str]],
                gates: list[dict[str, Any]]) -> tuple[dict[tuple[int, int, str], dict[str, dict[str, Any]]], dict[str, str]]:
    root = ns.stage196b2b3p0_run_root.resolve()
    observed_runs = sorted(path.name for path in root.iterdir() if path.is_dir())
    gate(gates, "p0", "", "p0_six_run_closure", sorted(RUNS), observed_runs,
         observed_runs == sorted(RUNS), "P0 exact run closure failed")
    gate(gates, "p0", "", "p0_runtime_commit_agreement", P0_COMMIT,
         ns.stage196b2b3p0_runtime_git_commit, ns.stage196b2b3p0_runtime_git_commit == P0_COMMIT,
         "P0 runtime commit mismatch")
    states: dict[tuple[int, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    hashes: dict[str, str] = {}
    composer_rows = trajectory_rows = composer_sidecars = trajectory_sidecars = 0
    for run in RUNS:
        seed, mode = int(run[4:7]), run[8:]
        run_dir = root / run
        composer_dir, trajectory_dir = run_dir / "composer_inputs", run_dir / "trajectory"
        composer_names = [f"stage196b2b3p0_epoch_composer_inputs_{epoch:03d}.jsonl" for epoch in EPOCHS]
        manifest_name = "stage196b2b3p0_composer_input_manifest.json"
        exact_files(composer_dir, (manifest_name, *composer_names), gates, f"{run}_composer_namespace")
        manifest = read_json(composer_dir / manifest_name)
        manifest_ok = (manifest.get("completed") is True and manifest.get("current_git_commit") == P0_COMMIT
                       and manifest.get("seed") == seed and manifest.get("gradient_ownership_mode") == mode
                       and manifest.get("sidecar_files") == composer_names)
        gate(gates, "p0", run, "composer_manifest_closure", True,
             {"completed": manifest.get("completed"), "commit": manifest.get("current_git_commit"),
              "seed": manifest.get("seed"), "mode": manifest.get("gradient_ownership_mode")},
             manifest_ok, "P0 composer manifest closure failed")
        trajectory_names = [f"stage196b2p0_epoch_channels_{epoch:03d}.jsonl" for epoch in EPOCHS]
        pattern = re.compile(r"^stage196b2p0_epoch_channels_[0-9]{3}\.jsonl$")
        files = sorted(path.name for path in trajectory_dir.iterdir() if path.is_file())
        observed_namespace = sorted(name for name in files if pattern.fullmatch(name))
        malformed = sorted(name for name in files if name.startswith("stage196b2p0_epoch_channels_")
                           and name not in trajectory_names)
        evidence = {"expected": trajectory_names, "observed": observed_namespace,
                    "malformed_namespace_like": malformed,
                    "unrelated_files_ignored": sorted(set(files) - set(observed_namespace) - set(malformed))}
        gate(gates, "p0", run, "trajectory_namespace_closure",
             {"expected": trajectory_names, "malformed_namespace_like": []}, evidence,
             observed_namespace == trajectory_names and not malformed,
             "P0 trajectory namespace closure failed")
        schema: set[str] | None = None
        for epoch, composer_name, trajectory_name in zip(EPOCHS, composer_names, trajectory_names):
            composer_path, trajectory_path = composer_dir / composer_name, trajectory_dir / trajectory_name
            composer, trajectory = read_jsonl(composer_path), read_jsonl(trajectory_path)
            hashes[str(composer_path)] = sha256(composer_path)
            hashes[str(trajectory_path)] = sha256(trajectory_path)
            expected_hash = manifest.get("sidecar_sha256", {}).get(composer_name)
            if expected_hash != hashes[str(composer_path)]:
                raise ValueError(f"{run}:{epoch}: composer sidecar hash mismatch")
            if len(composer) != 720 or len(trajectory) != 720 or any(set(row) != TRAJECTORY_FIELDS for row in trajectory):
                raise ValueError(f"{run}:{epoch}: sidecar row/schema closure failed")
            if schema is None:
                schema = set(composer[0])
            trajectory_index = {(str(row["id"]), str(row["source_row_id"]), integer(row["dev_position"], "position")): row
                                for row in trajectory}
            if len(trajectory_index) != 720:
                raise ValueError(f"{run}:{epoch}: trajectory identity collision")
            seen = set()
            for row in composer:
                if set(row) != schema:
                    raise ValueError(f"{run}:{epoch}: composer schema drift")
                identity = (str(row["id"]), str(row["source_row_id"]), integer(row["dev_position"], "position"))
                if identity in seen or identity not in trajectory_index:
                    raise ValueError(f"{run}:{epoch}: composer identity closure failed")
                seen.add(identity)
                if (row.get("current_git_commit") != P0_COMMIT or integer(row.get("seed"), "seed") != seed
                        or integer(row.get("epoch"), "epoch") != epoch or row.get("gradient_ownership_mode") != mode):
                    raise ValueError(f"{run}:{epoch}: composer provenance mismatch")
                stable = str(row.get("stable_row_id", row["id"]))
                key = (seed, stable)
                if key in primary:
                    meta = primary[key]
                    if (str(row["id"]), str(row["source_row_id"]), integer(row["dev_position"], "position")) != (
                            meta["id"], meta["source_row_id"], integer(meta["dev_position"], "position")):
                        raise ValueError("P0/B2-B2 primary identity mismatch")
                    state_key = (seed, epoch, stable)
                    if mode in states[state_key]:
                        raise ValueError("duplicate primary P0 state")
                    states[state_key][mode] = row
            composer_rows += len(composer)
            trajectory_rows += len(trajectory)
            composer_sidecars += 1
            trajectory_sidecars += 1
        hashes[str(composer_dir / manifest_name)] = sha256(composer_dir / manifest_name)
    counts = {"composer_rows": composer_rows, "trajectory_rows": trajectory_rows,
              "composer_sidecars": composer_sidecars, "trajectory_sidecars": trajectory_sidecars}
    required = {"composer_rows": 86400, "trajectory_rows": 86400,
                "composer_sidecars": 120, "trajectory_sidecars": 120}
    gate(gates, "p0", "", "p0_row_and_sidecar_closure", required, counts,
         counts == required, "P0 count closure failed")
    expected_states = len(primary) * len(EPOCHS)
    paired = len(states) == expected_states and all(set(value) == set(MODES) for value in states.values())
    gate(gates, "p0", "", "primary_paired_state_closure", {"states": 320, "modes": list(MODES)},
         {"states": len(states), "bad_mode_sets": sum(set(value) != set(MODES) for value in states.values())},
         paired, "primary paired P0 states incomplete")
    return states, hashes


def action_sets(b4: dict[str, Any], primary: dict[tuple[int, str], dict[str, str]],
                gates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[int, str], set[str]], list[dict[str, Any]]]:
    forward_tails = [row for row in b4["tails"] if row["direction"] == FORWARD]
    buckets: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in forward_tails:
        key = (integer(row["seed"], "seed"), row["stable_row_id"])
        if key in primary:
            buckets[key].append(row)
    if set(buckets) != set(primary) or any(len(rows) != 32 for rows in buckets.values()):
        raise ValueError("forward primitive tail target is not 16 x 32")
    output, acceptable = [], {}
    for key in sorted(primary):
        meta, rows = primary[key], sorted(buckets[key], key=lambda row: row["coalition_mask"])
        if [row["coalition_mask"] for row in rows] != list(MASKS):
            raise ValueError("row primitive mask closure failed")
        role = meta["transition_role"]
        chosen = {row["coalition_mask"] for row in rows if boolean(
            row["donor_tail_reproduced"] if role == "recovery" else row["recipient_tail_preserved"], "tail objective")}
        if not chosen:
            raise ValueError(f"{key}: empty acceptable action set")
        acceptable[key] = chosen
        recipient_statuses = {row["recipient_tail_status"] for row in rows}
        donor_statuses = {row["donor_tail_status"] for row in rows}
        if len(recipient_statuses) != 1 or len(donor_statuses) != 1:
            raise ValueError("tail reference status drift")
        output.append({
            "seed": key[0], "stable_row_id": key[1], "id": meta["id"],
            "source_row_id": meta["source_row_id"], "dev_position": integer(meta["dev_position"], "position"),
            "transition_role": role, "intervention_type": meta["intervention_type"],
            "path_class": meta["path_class"], "subtype": rows[0]["subtype"], "direction": FORWARD,
            "tail_epochs": list(TAIL), "recipient_tail_status": next(iter(recipient_statuses)),
            "donor_tail_status": next(iter(donor_statuses)), "acceptable_coalitions": sorted(chosen),
            "acceptable_coalition_count": len(chosen),
            "inclusion_minimal_acceptable_coalitions": minimal_masks(chosen),
            "minimal_coalition_count": len(minimal_masks(chosen)),
            "empty_coalition_acceptable": "00000" in chosen, "full_coalition_acceptable": "11111" in chosen,
        })
    roles = Counter(row["transition_role"] for row in output)
    gate(gates, "target", "", "primary_forward_16_row_closure", {"rows": 16, "direction": FORWARD},
         {"rows": len(output), "direction_values": sorted({row["direction"] for row in output}), "roles": dict(roles)},
         len(output) == 16 and {row["direction"] for row in output} == {FORWARD}, "primary selector target closure failed")
    gate(gates, "target", "", "row_acceptable_action_set_closure", True,
         {"rows": len(acceptable), "all_nonempty": all(acceptable.values())},
         len(acceptable) == 16 and all(acceptable.values()), "acceptable action closure failed")
    for role in ("recovery", "harm"):
        subset = [row for row in output if row["transition_role"] == role]
        gate(gates, "target", "", f"{role}_acceptable_set_nonempty_closure", True,
             {"row_count": len(subset), "empty_count": sum(row["acceptable_coalition_count"] == 0 for row in subset)},
             bool(subset) and all(row["acceptable_coalition_count"] > 0 for row in subset),
             f"{role} acceptable action set empty")
    epoch_buckets: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in b4["primitive"]:
        key = (integer(row["seed"], "seed"), row["stable_row_id"])
        if key in primary and row["direction"] == FORWARD and row["coalition_mask"] in acceptable[key]:
            epoch_buckets[(key[0], key[1], row["coalition_mask"])].append(row)
    stability = []
    for (seed, stable, mask), rows in sorted(epoch_buckets.items()):
        rows.sort(key=lambda row: integer(row["epoch"], "epoch"))
        role = primary[(seed, stable)]["transition_role"]
        flags = [boolean(row["donor_prediction_reproduced"] if role == "recovery" else row["recipient_prediction_preserved"], "epoch objective") for row in rows]
        satisfying = [integer(row["epoch"], "epoch") for row, passed in zip(rows, flags) if passed]
        stability.append({"seed": seed, "stable_row_id": stable, "transition_role": role,
                          "coalition_mask": mask, "epochs_satisfying_objective": satisfying,
                          "first_satisfying_epoch": satisfying[0] if satisfying else None,
                          "last_satisfying_epoch": satisfying[-1] if satisfying else None,
                          "tail3_stable": all(flags[epoch - 1] for epoch in TAIL),
                          "all20_stable": all(flags),
                          "number_of_state_transitions": sum(a != b for a, b in zip(flags, flags[1:]))})
    expected = sum(len(value) for value in acceptable.values())
    gate(gates, "diagnostic", "", "epoch_stability_audit_completion", {"coalitions": expected, "epochs_each": 20},
         {"coalitions": len(stability), "bad_epoch_counts": sum(len(epoch_buckets[(r["seed"], r["stable_row_id"], r["coalition_mask"])]) != 20 for r in stability)},
         len(stability) == expected and all(len(epoch_buckets[(r["seed"], r["stable_row_id"], r["coalition_mask"])]) == 20 for r in stability),
         "epoch stability audit incomplete")
    return output, acceptable, stability


def feature_specs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recipient = [
        ("RECIPIENT_PREDICTION_SEQUENCE", ("final_native_prediction",), "exact final_native_prediction at epochs 18,19,20", ["REFUTE", "NOT_ENTITLED", "SUPPORT"]),
        ("FINAL_MARGIN_SIGN_SEQUENCE", ("final_support_logit", "final_not_entitled_logit"), "sign(final_support_logit - final_not_entitled_logit), zero iff abs(delta) <= 1e-12", ["NEGATIVE", "ZERO", "POSITIVE"]),
        ("HEAD_MARGIN_SIGN_SEQUENCE", ("decision_head_support_logit", "decision_head_not_entitled_logit"), "sign(decision_head_support_logit - decision_head_not_entitled_logit), zero iff abs(delta) <= 1e-12", ["NEGATIVE", "ZERO", "POSITIVE"]),
        ("HEAD_FINAL_MARGIN_SIGN_CONFLICT_SEQUENCE", ("decision_head_support_logit", "decision_head_not_entitled_logit", "final_support_logit", "final_not_entitled_logit"), "HEAD_SIGN != FINAL_SIGN at each tail epoch", [False, True]),
        ("FRAME_HALFSPACE_SEQUENCE", ("frame_prob",), "halfspace(frame_prob; threshold=0.5, AT_HALF iff abs(x-0.5)<=1e-12)", ["BELOW_HALF", "AT_HALF", "ABOVE_HALF"]),
        ("PREDICATE_HALFSPACE_SEQUENCE", ("predicate_coverage_prob",), "halfspace(predicate_coverage_prob; threshold=0.5, AT_HALF iff abs(x-0.5)<=1e-12)", ["BELOW_HALF", "AT_HALF", "ABOVE_HALF"]),
        ("SUFFICIENCY_HALFSPACE_SEQUENCE", ("sufficiency_prob",), "halfspace(sufficiency_prob; threshold=0.5, AT_HALF iff abs(x-0.5)<=1e-12)", ["BELOW_HALF", "AT_HALF", "ABOVE_HALF"]),
        ("ENTITLEMENT_HALFSPACE_SEQUENCE", ("entitlement_prob_native",), "halfspace(entitlement_prob_native; threshold=0.5, AT_HALF iff abs(x-0.5)<=1e-12)", ["BELOW_HALF", "AT_HALF", "ABOVE_HALF"]),
        ("ENTITLEMENT_BOTTLENECK_SEQUENCE", ("frame_prob", "predicate_coverage_prob", "sufficiency_prob"), "sorted set of argmin names over frame_prob,predicate_coverage_prob,sufficiency_prob; ties preserved within 1e-12", ["sorted nonempty subsets of FRAME,PREDICATE,SUFFICIENCY"]),
        ("POLARITY_ENERGY_ORDER_SEQUENCE", ("positive_energy", "negative_energy"), "compare positive_energy with negative_energy; EQUAL iff abs(delta)<=1e-12", ["POSITIVE_DOMINANT", "EQUAL", "NEGATIVE_DOMINANT"]),
        ("PREDICATE_MISMATCH_SEQUENCE", ("predicate_mismatch_active",), "raw exported predicate_mismatch_active flag", [False, True]),
        ("TEMPORAL_MISMATCH_SEQUENCE", ("temporal_mismatch_active",), "raw exported temporal_mismatch_active flag", [False, True]),
        ("TEMPORAL_ADAPTER_ACTIVITY_SEQUENCE", ("temporal_adapter_active",), "raw exported temporal_adapter_active flag", [False, True]),
        ("TEMPORAL_CHANNEL_ACTIVITY_SEQUENCE", ("temporal_channel_active",), "raw exported temporal_channel_active flag", [False, True]),
    ]
    paired = [
        ("DELTA_FRAME_SIGN_SEQUENCE", ("frame_prob",), "sign(donor.frame_prob - recipient.frame_prob), zero iff abs(delta)<=1e-12"),
        ("DELTA_PREDICATE_SIGN_SEQUENCE", ("predicate_coverage_prob",), "sign(donor.predicate_coverage_prob - recipient.predicate_coverage_prob), zero iff abs(delta)<=1e-12"),
        ("DELTA_SUFFICIENCY_SIGN_SEQUENCE", ("sufficiency_prob",), "sign(donor.sufficiency_prob - recipient.sufficiency_prob), zero iff abs(delta)<=1e-12"),
        ("DELTA_POSITIVE_ENERGY_SIGN_SEQUENCE", ("positive_energy",), "sign(donor.positive_energy - recipient.positive_energy), zero iff abs(delta)<=1e-12"),
        ("DELTA_NEGATIVE_ENERGY_SIGN_SEQUENCE", ("negative_energy",), "sign(donor.negative_energy - recipient.negative_energy), zero iff abs(delta)<=1e-12"),
        ("DELTA_ENTITLEMENT_SIGN_SEQUENCE", ("entitlement_prob_native",), "sign(donor.entitlement_prob_native - recipient.entitlement_prob_native), zero iff abs(delta)<=1e-12"),
        ("DELTA_HEAD_MARGIN_SIGN_SEQUENCE", ("decision_head_support_logit", "decision_head_not_entitled_logit"), "sign(donor head margin - recipient head margin), zero iff abs(delta)<=1e-12"),
        ("DELTA_FINAL_MARGIN_SIGN_SEQUENCE", ("final_support_logit", "final_not_entitled_logit"), "sign(donor final margin - recipient final margin), zero iff abs(delta)<=1e-12"),
        ("PREDICATE_MISMATCH_CHANGE_SEQUENCE", ("predicate_mismatch_active",), "exact donor versus recipient raw predicate_mismatch_active state pair"),
        ("TEMPORAL_MISMATCH_CHANGE_SEQUENCE", ("temporal_mismatch_active",), "exact donor versus recipient raw temporal_mismatch_active state pair"),
    ]
    def rows(items: Sequence[tuple[Any, ...]], family: str) -> list[dict[str, Any]]:
        output = []
        for item in items:
            name, fields, formula = item[:3]
            domain = item[3] if len(item) == 4 else (["NEGATIVE", "ZERO", "POSITIVE"] if name.startswith("DELTA_") else ["FALSE_TO_FALSE", "FALSE_TO_TRUE", "TRUE_TO_FALSE", "TRUE_TO_TRUE"])
            output.append({"feature_family": family, "feature_name": name,
                           "deployment_authorized": family == "recipient_local",
                           "diagnostic_only": family == "paired_delta", "source_fields": list(fields),
                           "formula": formula, "tail_aggregation": "ordered exact sequence at epochs 18,19,20; no vote",
                           "value_domain": domain,
                           "natural_threshold": 0.5 if "HALFSPACE" in name else (0.0 if "SIGN" in name or "ORDER" in name else None),
                           "outcome_derived": False, "available": False, "unavailable_reason": "not evaluated"})
        return output
    return rows(recipient, "recipient_local"), rows(paired, "paired_delta")


def build_features(states: dict[tuple[int, int, str], dict[str, dict[str, Any]]],
                   primary: dict[tuple[int, str], dict[str, str]], gates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[tuple[int, str], tuple[Any, ...]]]]:
    recipient_specs, paired_specs = feature_specs()
    values: dict[str, dict[tuple[int, str], tuple[Any, ...]]] = {}
    all_rows = [arm for pair in states.values() for arm in pair.values()]

    def available(spec: dict[str, Any]) -> bool:
        return all(all(field in row and row[field] is not None for field in spec["source_fields"]) for row in all_rows)

    def recipient_value(name: str, row: dict[str, Any]) -> Any:
        if name == "RECIPIENT_PREDICTION_SEQUENCE": return row["final_native_prediction"]
        if name == "FINAL_MARGIN_SIGN_SEQUENCE": return sign(number(row["final_support_logit"], "final support") - number(row["final_not_entitled_logit"], "final NE"))
        if name == "HEAD_MARGIN_SIGN_SEQUENCE": return sign(number(row["decision_head_support_logit"], "head support") - number(row["decision_head_not_entitled_logit"], "head NE"))
        if name == "HEAD_FINAL_MARGIN_SIGN_CONFLICT_SEQUENCE":
            return recipient_value("HEAD_MARGIN_SIGN_SEQUENCE", row) != recipient_value("FINAL_MARGIN_SIGN_SEQUENCE", row)
        if name.endswith("_HALFSPACE_SEQUENCE"):
            field = {"FRAME_HALFSPACE_SEQUENCE": "frame_prob", "PREDICATE_HALFSPACE_SEQUENCE": "predicate_coverage_prob",
                     "SUFFICIENCY_HALFSPACE_SEQUENCE": "sufficiency_prob", "ENTITLEMENT_HALFSPACE_SEQUENCE": "entitlement_prob_native"}[name]
            return halfspace(number(row[field], field, True))
        if name == "ENTITLEMENT_BOTTLENECK_SEQUENCE":
            entries = [("FRAME", number(row["frame_prob"], "frame", True)),
                       ("PREDICATE", number(row["predicate_coverage_prob"], "predicate", True)),
                       ("SUFFICIENCY", number(row["sufficiency_prob"], "sufficiency", True))]
            minimum = min(value for _, value in entries)
            return tuple(sorted(label for label, value in entries if abs(value - minimum) <= ZERO_TOL))
        if name == "POLARITY_ENERGY_ORDER_SEQUENCE": return order_energy(number(row["positive_energy"], "positive"), number(row["negative_energy"], "negative"))
        field = {"PREDICATE_MISMATCH_SEQUENCE": "predicate_mismatch_active", "TEMPORAL_MISMATCH_SEQUENCE": "temporal_mismatch_active",
                 "TEMPORAL_ADAPTER_ACTIVITY_SEQUENCE": "temporal_adapter_active", "TEMPORAL_CHANNEL_ACTIVITY_SEQUENCE": "temporal_channel_active"}[name]
        return boolean(row[field], field)

    def paired_value(name: str, recipient: dict[str, Any], donor: dict[str, Any]) -> Any:
        direct = {"DELTA_FRAME_SIGN_SEQUENCE": "frame_prob", "DELTA_PREDICATE_SIGN_SEQUENCE": "predicate_coverage_prob",
                  "DELTA_SUFFICIENCY_SIGN_SEQUENCE": "sufficiency_prob", "DELTA_POSITIVE_ENERGY_SIGN_SEQUENCE": "positive_energy",
                  "DELTA_NEGATIVE_ENERGY_SIGN_SEQUENCE": "negative_energy", "DELTA_ENTITLEMENT_SIGN_SEQUENCE": "entitlement_prob_native"}
        if name in direct:
            field = direct[name]
            return sign(number(donor[field], field) - number(recipient[field], field))
        if name in ("DELTA_HEAD_MARGIN_SIGN_SEQUENCE", "DELTA_FINAL_MARGIN_SIGN_SEQUENCE"):
            prefix = "decision_head" if name.startswith("DELTA_HEAD") else "final"
            rmargin = number(recipient[f"{prefix}_support_logit"], "support") - number(recipient[f"{prefix}_not_entitled_logit"], "NE")
            dmargin = number(donor[f"{prefix}_support_logit"], "support") - number(donor[f"{prefix}_not_entitled_logit"], "NE")
            return sign(dmargin - rmargin)
        field = "predicate_mismatch_active" if name.startswith("PREDICATE") else "temporal_mismatch_active"
        return f"{str(boolean(recipient[field], field)).upper()}_TO_{str(boolean(donor[field], field)).upper()}"

    for specs in (recipient_specs, paired_specs):
        for spec in specs:
            spec["available"] = available(spec)
            spec["unavailable_reason"] = "" if spec["available"] else "one or more exact exported source fields are absent or null; no substitution allowed"
            if not spec["available"]:
                continue
            feature_values = {}
            for identity in sorted(primary):
                sequence = []
                for epoch in TAIL:
                    pair = states[(identity[0], epoch, identity[1])]
                    sequence.append(recipient_value(spec["feature_name"], pair["joint"]) if spec["feature_family"] == "recipient_local"
                                    else paired_value(spec["feature_name"], pair["joint"], pair["frame_local_only"]))
                feature_values[identity] = tuple(sequence)
            values[spec["feature_name"]] = feature_values
    dictionary = recipient_specs + paired_specs
    recipient_available = [r["feature_name"] for r in recipient_specs if r["available"]]
    paired_available = [r["feature_name"] for r in paired_specs if r["available"]]
    leakage = all(not set(r["source_fields"]) & PROHIBITED_SELECTOR_FIELDS and r["outcome_derived"] is False for r in dictionary)
    gate(gates, "features", "", "recipient_feature_dictionary_closure", True,
         {"semantic_features": len(recipient_specs), "available": recipient_available}, bool(recipient_available),
         "no recipient-local semantic feature is available")
    gate(gates, "features", "", "recipient_outcome_leakage_prohibition", True,
         {"passed": leakage, "prohibited_selector_fields": sorted(PROHIBITED_SELECTOR_FIELDS)}, leakage,
         "selector feature dictionary contains prohibited input")
    gate(gates, "features", "", "paired_delta_dictionary_closure", 10,
         {"semantic_features": len(paired_specs), "available": paired_available}, len(paired_specs) == 10,
         "paired delta dictionary incomplete")
    diagnostic_enforced = all(r["diagnostic_only"] is True and r["deployment_authorized"] is False for r in paired_specs)
    gate(gates, "features", "", "paired_delta_diagnostic_only_enforcement", True, diagnostic_enforced,
         diagnostic_enforced, "paired delta feature was deployment-authorized")
    return dictionary, values


def signature_for(identity: tuple[int, str], members: Sequence[str], values: dict[str, dict[tuple[int, str], tuple[Any, ...]]]) -> tuple[Any, ...]:
    return tuple(values[name][identity] for name in members)


def transfer(source_seed: int, target_seed: int, members: Sequence[str], identities: Sequence[tuple[int, str]],
             values: dict[str, dict[tuple[int, str], tuple[Any, ...]]], acceptable: dict[tuple[int, str], set[str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mapping: dict[tuple[Any, ...], set[str]] = {}
    for identity in identities:
        if identity[0] != source_seed: continue
        signature = signature_for(identity, members, values)
        mapping[signature] = set(acceptable[identity]) if signature not in mapping else mapping[signature] & acceptable[identity]
    rows, seen = [], 0
    for identity in identities:
        if identity[0] != target_seed: continue
        signature = signature_for(identity, members, values)
        source_actions = mapping.get(signature)
        if source_actions is None:
            status, intersection = "UNSEEN", set()
        else:
            seen += 1
            intersection = source_actions & acceptable[identity]
            status = "COMPATIBLE" if intersection else "INCOMPATIBLE"
        rows.append({"identity": identity, "signature": signature, "source_actions": source_actions or set(),
                     "intersection": intersection, "status": status})
    target_count = len(rows)
    unseen = sum(row["status"] == "UNSEEN" for row in rows)
    incompatible = sum(row["status"] == "INCOMPATIBLE" for row in rows)
    compatible = sum(row["status"] == "COMPATIBLE" for row in rows)
    return {"source_seed": source_seed, "target_seed": target_seed, "target_row_count": target_count,
            "seen_signature_row_count": seen, "unseen_signature_row_count": unseen,
            "compatible_seen_rows": compatible, "incompatible_seen_rows": incompatible,
            "coverage_rate": seen / target_count, "compatibility_rate_on_seen_rows": compatible / seen if seen else None,
            "full_transfer_pass": unseen == 0 and incompatible == 0}, rows


def audit_family(family: str, feature_names: Sequence[str], values: dict[str, dict[tuple[int, str], tuple[Any, ...]]],
                 primary: dict[tuple[int, str], dict[str, str]], acceptable: dict[tuple[int, str], set[str]],
                 gates: list[dict[str, Any]]) -> dict[str, Any]:
    identities = sorted(primary)
    summaries, signature_rows, detail = [], [], {}
    feasible_masks = []
    expected_subsets = (1 << len(feature_names)) - 1
    for raw_mask in range(1, 1 << len(feature_names)):
        mask = f"{raw_mask:0{len(feature_names)}b}"
        members = [name for bit, name in zip(mask, feature_names) if bit == "1"]
        groups: dict[tuple[Any, ...], list[tuple[int, str]]] = defaultdict(list)
        for identity in identities:
            groups[signature_for(identity, members, values)].append(identity)
        intersections = {sig: set.intersection(*(acceptable[identity] for identity in rows)) for sig, rows in groups.items()}
        feasible = all(intersections.values())
        if feasible: feasible_masks.append(mask)
        transfers = {}
        transfer_rows = {}
        for source, target in ((184, 185), (185, 184)):
            transfers[(source, target)], transfer_rows[(source, target)] = transfer(source, target, members, identities, values, acceptable)
        mixed = sum({primary[i]["transition_role"] for i in rows} == {"recovery", "harm"} for rows in groups.values())
        cross = sum({i[0] for i in rows} == {184, 185} for rows in groups.values())
        for signature, rows in sorted(groups.items(), key=lambda item: json.dumps(item[0], sort_keys=True)):
            intersection = sorted(intersections[signature])
            for identity in rows:
                signature_rows.append({"feature_family": family, "feature_subset_mask": mask,
                    "feature_subset_size": len(members), "feature_subset_members": list(members),
                    "seed": identity[0], "stable_row_id": identity[1],
                    "transition_role": primary[identity]["transition_role"], "signature": signature,
                    "acceptable_coalitions": sorted(acceptable[identity]),
                    "signature_action_intersection": intersection, "signature_feasible": bool(intersection),
                    "source_seed": None, "target_seed": None, "transfer_status": "POOLED"})
        for (source, target), rows in transfer_rows.items():
            for row in rows:
                identity = row["identity"]
                signature_rows.append({"feature_family": family, "feature_subset_mask": mask,
                    "feature_subset_size": len(members), "feature_subset_members": list(members),
                    "seed": identity[0], "stable_row_id": identity[1],
                    "transition_role": primary[identity]["transition_role"], "signature": row["signature"],
                    "acceptable_coalitions": sorted(acceptable[identity]),
                    "signature_action_intersection": sorted(row["intersection"]),
                    "signature_feasible": bool(row["intersection"]), "source_seed": source,
                    "target_seed": target, "transfer_status": row["status"]})
        a, b = transfers[(184, 185)], transfers[(185, 184)]
        summaries.append({"feature_family": family, "feature_subset_mask": mask,
            "feature_subset_size": len(members), "feature_subset_members": list(members),
            "row_count": len(identities), "signature_count": len(groups), "feasible": feasible,
            "inclusion_minimal_feasible": False, "mixed_role_signature_count": mixed,
            "cross_seed_signature_count": cross, "seed184_to_seed185_seen": a["seen_signature_row_count"],
            "seed184_to_seed185_unseen": a["unseen_signature_row_count"],
            "seed184_to_seed185_incompatible": a["incompatible_seen_rows"],
            "seed184_to_seed185_full_pass": a["full_transfer_pass"],
            "seed185_to_seed184_seen": b["seen_signature_row_count"],
            "seed185_to_seed184_unseen": b["unseen_signature_row_count"],
            "seed185_to_seed184_incompatible": b["incompatible_seen_rows"],
            "seed185_to_seed184_full_pass": b["full_transfer_pass"], "pooled_full_pass": feasible,
            "bidirectional_cross_seed_full_pass": a["full_transfer_pass"] and b["full_transfer_pass"]})
        detail[mask] = {"feature_subset_members": list(members), "feasible": feasible,
                        "signature_count": len(groups), "maximum_rows_per_signature": max(map(len, groups.values())),
                        "full_row_coverage": sum(map(len, groups.values())) == len(identities),
                        "every_signature_mixed_role": all({primary[i]["transition_role"] for i in rows} == {"recovery", "harm"} for rows in groups.values()),
                        "every_signature_cross_seed": all({i[0] for i in rows} == {184, 185} for rows in groups.values()),
                        "signature_action_intersections": [minimal_masks(value) for _, value in sorted(intersections.items(), key=lambda item: json.dumps(item[0], sort_keys=True))],
                        "transfers": {f"{s}_to_{t}": transfers[(s, t)] for s, t in transfers}}
    minimal_feasible = [mask for mask in feasible_masks if not any(
        other != mask and all(a == "0" or b == "1" for a, b in zip(other, mask)) for other in feasible_masks)]
    minimal_set = set(minimal_feasible)
    for row in summaries:
        row["inclusion_minimal_feasible"] = row["feature_subset_mask"] in minimal_set
    seed_results = {}
    for seed in PRIMARY_SEEDS:
        seed_identities = [identity for identity in identities if identity[0] == seed]
        feasible = []
        for raw_mask in range(1, 1 << len(feature_names)):
            mask = f"{raw_mask:0{len(feature_names)}b}"
            members = [name for bit, name in zip(mask, feature_names) if bit == "1"]
            groups: dict[tuple[Any, ...], list[tuple[int, str]]] = defaultdict(list)
            for identity in seed_identities: groups[signature_for(identity, members, values)].append(identity)
            if all(set.intersection(*(acceptable[i] for i in rows)) for rows in groups.values()): feasible.append(mask)
        minimal = [mask for mask in feasible if not any(other != mask and all(a == "0" or b == "1" for a, b in zip(other, mask)) for other in feasible)]
        seed_results[str(seed)] = {"row_count": len(seed_identities), "feasible_subset_count": len(feasible),
                                   "inclusion_minimal_feasible_subsets": [{"feature_subset_mask": mask,
                                       "feature_subset_members": [name for bit, name in zip(mask, feature_names) if bit == "1"]} for mask in minimal]}
    gate(gates, "selector", "", f"{family}_feature_subset_lattice_closure", expected_subsets,
         {"enumerated": len(summaries), "available_features": len(feature_names)},
         len(summaries) == expected_subsets, "feature subset lattice incomplete")
    gate(gates, "selector", "", f"{family}_selector_audit_completion", True,
         {"pooled_subsets": len(summaries), "signature_rows": len(signature_rows)}, bool(summaries),
         "selector audit incomplete")
    return {"feature_names": list(feature_names), "feature_subset_count": expected_subsets,
            "summaries": summaries, "signature_rows": signature_rows, "detail": detail,
            "feasible_subset_count": len(feasible_masks),
            "inclusion_minimal_feasible_subsets": [{"feature_subset_mask": mask,
                "feature_subset_members": [name for bit, name in zip(mask, feature_names) if bit == "1"],
                "bidirectional_cross_seed_full_pass": next(row["bidirectional_cross_seed_full_pass"] for row in summaries if row["feature_subset_mask"] == mask)}
                for mask in minimal_feasible], "seed_results": seed_results}


def decide(recipient: dict[str, Any], paired: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    pooled = recipient["feasible_subset_count"] > 0
    cross = any(row["bidirectional_cross_seed_full_pass"] for row in recipient["summaries"] if row["inclusion_minimal_feasible"])
    both_seed = all(recipient["seed_results"][str(seed)]["feasible_subset_count"] > 0 for seed in PRIMARY_SEEDS)
    paired_exact = paired["feasible_subset_count"] > 0
    evaluation = {"ordered_rules": ["cross_seed_recipient", "recipient_in_sample_only", "both_seed_specific", "paired_delta_only", "insufficient"],
                  "recipient_pooled_exact": pooled, "recipient_minimal_bidirectional_cross_seed": cross,
                  "both_seed_specific_exact": both_seed, "paired_delta_exact": paired_exact,
                  "partial_coverage_can_pass": False}
    if cross:
        return "STAGE196B2B5_CROSS_SEED_RECIPIENT_SELECTOR_LOCALIZED", "STAGE196B2B6_MINIMAL_SELECTOR_INTERVENTION_DESIGN", evaluation
    if pooled:
        return "STAGE196B2B5_RECIPIENT_SELECTOR_OBSERVABLE_IN_SAMPLE_ONLY", "STAGE196B2B6_NEW_SEED_SELECTOR_VALIDATION", evaluation
    if both_seed:
        return "STAGE196B2B5_SEED_SPECIFIC_RECIPIENT_SELECTORS", "STAGE196B2B6_SEED_SPECIFIC_SELECTOR_AUDIT", evaluation
    if paired_exact:
        return "STAGE196B2B5_PAIRED_DELTA_SELECTOR_ONLY", "STAGE196B2B5P0_RECIPIENT_SELECTOR_STATE_OBSERVABILITY_DESIGN", evaluation
    return "STAGE196B2B5_CURRENT_OBSERVABILITY_INSUFFICIENT", "STAGE196B2B5P0_ADDITIONAL_SELECTOR_STATE_OBSERVABILITY_DESIGN", evaluation


def analyze(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    repo, output = ns.repo_root.resolve(), ns.output_dir.resolve()
    inputs = (ns.stage196b2b4_analysis_json.resolve(), ns.stage196b2b2_analysis_json.resolve(), ns.stage196b2b3p0_run_root.resolve())
    valid_paths = repo.is_dir() and all(under(path, repo) for path in (*inputs, output))
    gate(gates, "path", "", "explicit_paths_below_repo", True,
         {"repo": str(repo), "inputs": [str(path) for path in inputs], "output": str(output)}, valid_paths,
         "path is absent or escapes repo")
    separated = not output.exists() and all(output != path and output not in path.parents and path not in output.parents for path in inputs)
    gate(gates, "path", "", "fresh_separate_output", True, {"exists": output.exists(), "separated": separated},
         separated, "output exists or overlaps an input")
    commit_format = re.fullmatch(r"[0-9a-f]{40}", ns.current_git_commit or "") is not None
    gate(gates, "provenance", "", "current_git_commit_format", "40 lowercase hex", ns.current_git_commit,
         commit_format, "current commit format invalid")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    gate(gates, "provenance", "", "current_git_commit_equals_head", ns.current_git_commit, head,
         head == ns.current_git_commit, "current commit differs from HEAD")
    b4 = validate_b2b4(ns, gates)
    b2 = validate_b2b2(ns, gates)
    states, p0_hashes = validate_p0(ns, b2["primary"], gates)
    actions, acceptable, stability = action_sets(b4, b2["primary"], gates)
    dictionary, feature_values = build_features(states, b2["primary"], gates)
    recipient_names = [row["feature_name"] for row in dictionary if row["feature_family"] == "recipient_local" and row["available"]]
    paired_names = [row["feature_name"] for row in dictionary if row["feature_family"] == "paired_delta" and row["available"]]
    recipient = audit_family("recipient_local", recipient_names, feature_values, b2["primary"], acceptable, gates)
    paired = audit_family("paired_delta", paired_names, feature_values, b2["primary"], acceptable, gates)
    for seed in PRIMARY_SEEDS:
        gate(gates, "selector", "", f"seed{seed}_selector_audit_completion", True,
             recipient["seed_results"][str(seed)], True, "")
    gate(gates, "selector", "", "pooled_selector_audit_completion", True,
         {"feature_subsets": recipient["feature_subset_count"]}, True, "")
    gate(gates, "selector", "", "cross_seed_transfer_completion", True,
         {"directions": ["184_to_185", "185_to_184"], "subsets": recipient["feature_subset_count"]}, True, "")
    decision, next_stage, evaluation = decide(recipient, paired)
    gate(gates, "decision", "", "decision_evaluation_completion", True, evaluation, True, "")
    gate(gates, "output", "", "exact_nine_output_plan", list(OUTPUTS), list(OUTPUTS), True, "")
    source_paths = {"stage196b2b4_analysis_json": str(b4["path"]), "stage196b2b2_analysis_json": str(b2["path"]),
                    "stage196b2b3p0_run_root": str(ns.stage196b2b3p0_run_root.resolve())}
    analysis = {
        "stage": STAGE, "decision": decision, "recommended_next_stage": next_stage, "blocking_reasons": [],
        "current_git_commit": ns.current_git_commit, "stage196b2b3p0_runtime_git_commit": ns.stage196b2b3p0_runtime_git_commit,
        "source_paths": source_paths, "source_hashes": {**b4["source_hashes"], **b2["source_hashes"], **p0_hashes},
        "source_closure": {"b2b4_files": list(B2B4_FILES), "b2b2_files": list(B2B2_FILES), "p0_runs": list(RUNS),
                           "b2b4_counts": b4["counts"], "b2b4_mobius": b4["mobius"]},
        "primary_population": {"identity_count": 16, "epoch_diagnostic_count": 320, "direction": FORWARD,
                               "reverse_direction_usage": "symmetry diagnostic only; excluded from selector target",
                               "tail_epochs": list(TAIL), "seed_counts": EXPECTED_PRIMARY,
                               "seed183": "contrast-only; excluded from selector decisions"},
        "action_set_definition": {"primitive_order": list(PRIMITIVES), "coalition_masks": list(MASKS),
                                  "recovery": "all coalitions reproducing donor predictions exactly at epochs 18,19,20",
                                  "harm": "all coalitions preserving recipient predictions exactly at epochs 18,19,20",
                                  "signature_rule": "set intersection must be nonempty"},
        "row_action_set_summary": actions,
        "recipient_feature_dictionary": [row for row in dictionary if row["feature_family"] == "recipient_local"],
        "recipient_feature_subset_count": recipient["feature_subset_count"],
        "recipient_inclusion_minimal_feasible_subsets": recipient["inclusion_minimal_feasible_subsets"],
        "recipient_seed184_results": recipient["seed_results"]["184"],
        "recipient_seed185_results": recipient["seed_results"]["185"],
        "recipient_pooled_results": {"feasible_subset_count": recipient["feasible_subset_count"],
                                     "subset_results": recipient["detail"]},
        "recipient_cross_seed_transfer": {mask: value["transfers"] for mask, value in recipient["detail"].items()},
        "paired_delta_feature_dictionary": [row for row in dictionary if row["feature_family"] == "paired_delta"],
        "paired_delta_feature_subset_count": paired["feature_subset_count"],
        "paired_delta_inclusion_minimal_feasible_subsets": paired["inclusion_minimal_feasible_subsets"],
        "paired_delta_cross_seed_transfer": {mask: value["transfers"] for mask, value in paired["detail"].items()},
        "epoch_stability": stability, "decision_rule_evaluation": evaluation,
        "authorized_interpretation": "Within the frozen-Mamba six-run artifact population, exact observable signatures may be assessed only as deterministic primitive-coalition set-intersection rules under the stated decision tier.",
        "remaining_uncertainty": ["observed clean-dev identities and seeds only", "paired deltas require both treatment states and are diagnostic only",
                                  "tail selector targets do not establish all-epoch stability", "no conclusion applies to unfrozen Mamba training"],
        "prohibited_claims": list(PROHIBITED_CLAIMS), "artifact_only": True, "classifier_fitted": False,
        "learned_threshold_used": False, "score_optimization_used": False, "training_performed": False,
        "model_loaded": False, "checkpoint_loaded": False, "promotion_authorized": False,
    }
    return analysis, {"dictionary": dictionary, "actions": actions, "recipient": recipient, "paired": paired}


REPORT_SECTIONS = (
    "Executive decision", "Authorized interpretation", "B2-B4 source result", "Source closure",
    "Selector target definition", "Recovery action sets", "Harm preservation action sets",
    "Recipient-local feature dictionary", "Outcome-leakage audit", "Recipient-local exact selector search",
    "Seed184 selector result", "Seed185 selector result", "Pooled selector result", "Cross-seed transfer result",
    "Paired-treatment diagnostic selector result", "Epoch-level stability", "Decision-rule evaluation",
    "Remaining uncertainty", "Prohibited claims", "Recommended next stage",
)


def report(analysis: dict[str, Any]) -> str:
    actions = analysis.get("row_action_set_summary", [])
    bodies = (
        f"`{analysis['decision']}`", analysis["authorized_interpretation"],
        json.dumps({"decision": analysis.get("source_closure", {}).get("b2b4_counts"), "authority_commit": B2B4_COMMIT}, sort_keys=True),
        json.dumps(analysis.get("source_closure", {}), sort_keys=True), json.dumps(analysis.get("action_set_definition", {}), sort_keys=True),
        json.dumps([row for row in actions if row["transition_role"] == "recovery"], sort_keys=True),
        json.dumps([row for row in actions if row["transition_role"] == "harm"], sort_keys=True),
        json.dumps(analysis.get("recipient_feature_dictionary", []), sort_keys=True),
        "Seed, identity keys, transition role, path class, subtype, B2-B4 outcome labels, and counterfactual outcomes are excluded from selector signatures.",
        json.dumps(analysis.get("recipient_inclusion_minimal_feasible_subsets", []), sort_keys=True),
        json.dumps(analysis.get("recipient_seed184_results", {}), sort_keys=True),
        json.dumps(analysis.get("recipient_seed185_results", {}), sort_keys=True),
        json.dumps({"feasible_subset_count": analysis.get("recipient_pooled_results", {}).get("feasible_subset_count")}, sort_keys=True),
        json.dumps(analysis.get("recipient_cross_seed_transfer", {}), sort_keys=True),
        json.dumps({"diagnostic_only": True, "deployment_authorized": False,
                    "minimal_feasible": analysis.get("paired_delta_inclusion_minimal_feasible_subsets", [])}, sort_keys=True),
        json.dumps(analysis.get("epoch_stability", []), sort_keys=True),
        json.dumps(analysis.get("decision_rule_evaluation", {}), sort_keys=True),
        "\n".join(f"- {value}" for value in analysis.get("remaining_uncertainty", [])),
        "\n".join(f"- {value}" for value in analysis.get("prohibited_claims", [])),
        f"`{analysis['recommended_next_stage']}`\n\nNo training or promotion is authorized.",
    )
    return f"# {STAGE}: Row-Selector Observability Analysis\n\n" + "\n\n".join(
        f"## {heading}\n\n{body}" for heading, body in zip(REPORT_SECTIONS, bodies)) + "\n"


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)): return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is True: return "true"
    if value is False: return "false"
    if value is None: return ""
    return value


def render_csv(header: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(header), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(header): raise ValueError(f"generated CSV schema mismatch: {sorted(set(row) ^ set(header))}")
        writer.writerow({key: csv_value(row[key]) for key in header})
    return buffer.getvalue()


def render_contract(rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(CONTRACT_H), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        rendered = dict(row)
        rendered["required"] = json.dumps(row["required"], sort_keys=True, separators=(",", ":"))
        rendered["observed"] = json.dumps(row["observed"], sort_keys=True, separators=(",", ":"))
        rendered["passed"] = csv_value(row["passed"])
        writer.writerow(rendered)
    return buffer.getvalue()


def blocked(ns: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    return {"stage": STAGE, "decision": "STAGE196B2B5_BLOCKED_CONTRACT_FAILURE",
            "recommended_next_stage": "STAGE196B2B5_REPAIR", "blocking_reasons": [f"{type(error).__name__}: {error}"],
            "current_git_commit": ns.current_git_commit,
            "stage196b2b3p0_runtime_git_commit": ns.stage196b2b3p0_runtime_git_commit,
            "source_paths": {"stage196b2b4_analysis_json": str(ns.stage196b2b4_analysis_json.resolve()),
                             "stage196b2b2_analysis_json": str(ns.stage196b2b2_analysis_json.resolve()),
                             "stage196b2b3p0_run_root": str(ns.stage196b2b3p0_run_root.resolve())},
            "source_hashes": {}, "source_closure": {}, "primary_population": {}, "action_set_definition": {},
            "row_action_set_summary": [], "recipient_feature_dictionary": [], "recipient_feature_subset_count": 0,
            "recipient_inclusion_minimal_feasible_subsets": [], "recipient_seed184_results": {},
            "recipient_seed185_results": {}, "recipient_pooled_results": {}, "recipient_cross_seed_transfer": {},
            "paired_delta_feature_dictionary": [], "paired_delta_feature_subset_count": 0,
            "paired_delta_inclusion_minimal_feasible_subsets": [], "paired_delta_cross_seed_transfer": {},
            "epoch_stability": [], "decision_rule_evaluation": {"completed": False},
            "authorized_interpretation": "No scientific interpretation is authorized because a contract failed.",
            "remaining_uncertainty": ["Repair the failed source or analysis contract."],
            "prohibited_claims": list(PROHIBITED_CLAIMS), "artifact_only": True, "classifier_fitted": False,
            "learned_threshold_used": False, "score_optimization_used": False, "training_performed": False,
            "model_loaded": False, "checkpoint_loaded": False, "promotion_authorized": False}


def render(analysis: dict[str, Any], tables: dict[str, Any], gates: list[dict[str, Any]]) -> dict[str, str]:
    return {OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True) + "\n", OUTPUTS[1]: report(analysis),
            OUTPUTS[2]: render_csv(FEATURE_H, tables["dictionary"]), OUTPUTS[3]: render_csv(ACTION_H, tables["actions"]),
            OUTPUTS[4]: render_csv(SIGNATURE_H, tables["recipient"]["signature_rows"]),
            OUTPUTS[5]: render_csv(SUMMARY_H, tables["recipient"]["summaries"]),
            OUTPUTS[6]: render_csv(SIGNATURE_H, tables["paired"]["signature_rows"]),
            OUTPUTS[7]: render_csv(SUMMARY_H, tables["paired"]["summaries"]),
            OUTPUTS[8]: render_contract(gates)}


def atomic_write_outputs(output: Path, payloads: dict[str, str]) -> None:
    if output.exists() or set(payloads) != set(OUTPUTS):
        raise RuntimeError("refusing overwrite or non-nine-file output")
    temporary = output.parent / f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        for name in OUTPUTS:
            path = temporary / name
            with path.open("x", encoding="utf-8", newline="") as handle:
                handle.write(payloads[name]); handle.flush(); os.fsync(handle.fileno())
        observed = sorted(path.name for path in temporary.iterdir())
        if observed != sorted(OUTPUTS):
            raise RuntimeError("staged output closure is not exactly nine files")
        os.replace(temporary, output)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


def main() -> int:
    ns = parse_args()
    gates: list[dict[str, Any]] = []
    empty_family = {"signature_rows": [], "summaries": []}
    tables = {"dictionary": [], "actions": [], "recipient": empty_family, "paired": empty_family}
    try:
        analysis, tables = analyze(ns, gates)
    except Exception as error:
        analysis = blocked(ns, error)
        if not any(not row["passed"] for row in gates):
            gate(gates, "analysis", "", "unhandled_contract_failure", True,
                 {"error_type": type(error).__name__, "message": str(error)}, False, str(error), fatal=False)
        gate(gates, "output", "", "exact_nine_output_plan", list(OUTPUTS), list(OUTPUTS), True, "", fatal=False)
    atomic_write_outputs(ns.output_dir.resolve(), render(analysis, tables, gates))
    return 3 if analysis["decision"] == "STAGE196B2B5_BLOCKED_CONTRACT_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
